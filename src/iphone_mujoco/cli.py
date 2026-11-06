"""Runnable camera, robot, record and replay workflows."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from PIL import Image

from . import __version__
from .camera import CameraCalibration, MujocoRGBDCamera
from .profiles import phone_mjcf, write_phone_module


def demo_calibration():
    """Synthetic teaching profile. Obtain the real K with `calibrate`."""
    return CameraCalibration(width=640, height=480, fx=520., fy=515., cx=316., cy=242.)


def look_at(eye=(1.1, -1.1, 1.), target=(0., 0., .4)):
    forward = np.asarray(target, dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    pose[:3, 3] = eye
    return pose


def _scene(args, calibration=None):
    from .robots import RobotSpec, load_scene
    if args.robot in ("ur5e", "panda"):
        from .robots import fetch_menagerie
        fetch_menagerie(args.robot, cache_dir=getattr(args, "model_cache", None))
    # get_robot/load_scene own registered and downloaded model resolution.
    selected = RobotSpec.from_json(args.robot) if str(args.robot).endswith(".json") else args.robot
    return load_scene(selected, phone_xml=phone_mjcf(calibration) if calibration else None,
                      model_cache=getattr(args, "model_cache", None))


def _viewer(stack, model, data, headless):
    if headless:
        return None
    import mujoco.viewer
    viewer = stack.enter_context(mujoco.viewer.launch_passive(model, data))
    viewer.cam.lookat[:] = [0., 0., .4]
    viewer.cam.distance = 2.2
    viewer.cam.azimuth = 125
    viewer.cam.elevation = -22
    return viewer


def _save_frame(frame, directory: Path, prefix="camera"):
    directory.mkdir(parents=True, exist_ok=True)
    frame.calibration.save(directory / "calibration.json")
    frame.calibration.save(directory / f"{prefix}_calibration.json")
    (directory / f"{prefix}_frame.json").write_text(json.dumps({
        "timestamp": frame.timestamp, "frame_id": frame.frame_id, "source": frame.source,
        "world_from_camera": None if frame.world_from_camera is None else frame.world_from_camera.tolist(),
        "metadata": frame.metadata,
    }, indent=2, default=str) + "\n")
    if frame.rgb is not None:
        Image.fromarray(frame.rgb).save(directory / f"{prefix}_rgb.png")
    if frame.depth_m is not None:
        np.save(directory / f"{prefix}_depth_m.npy", frame.depth_m, allow_pickle=False)
        frame.point_cloud(stride=2).save_ply(directory / f"{prefix}_points.ply")


def simulate(args):
    from .control import CartesianTeleoperator
    calibration = CameraCalibration.load(args.profile) if args.profile else demo_calibration()
    model, data, spec = _scene(args, calibration)
    controller = CartesianTeleoperator(model, data, phone_to_world=np.eye(3),
                                      translation_scale=getattr(args, "translation_scale", 1.), **spec.controller_kwargs)
    output = Path(args.output) if args.output else None
    frame_count = 0
    with ExitStack() as stack:
        viewer = _viewer(stack, model, data, args.headless)
        camera = stack.enter_context(MujocoRGBDCamera(model, calibration))
        camera.set_world_pose(data, look_at(), move_body=True)
        publisher = None
        writer = None
        if args.publish:
            from .virtual_device import VirtualPhonePublisher
            publisher = stack.enter_context(VirtualPhonePublisher(args.publish))
            print(f"Virtual camera {args.publish}; receivers must select robotics world coordinates.", flush=True)
        if output:
            from .recording import RecordingWriter
            writer = stack.enter_context(RecordingWriter(output / "recording"))
        anchor = np.eye(4)
        start = time.monotonic()
        controller.update(anchor, receive_time=start, session_id="scripted-demo")
        next_frame = 0.
        max_steps = args.steps or (max(1, int(args.duration / model.opt.timestep)) if args.duration > 0 else 0)
        step_index = 0
        while (not max_steps or step_index < max_steps) and (viewer is None or viewer.is_running()):
            now = time.monotonic()
            pose = anchor.copy()
            pose[:3, 3] = [.025 * np.sin(data.time), .025 * (np.cos(data.time) - 1), .02 * np.sin(data.time * .7)]
            controller.update(pose, receive_time=now, session_id="scripted-demo")
            controller.step(now=now, dt=model.opt.timestep)
            mujoco.mj_step(model, data)
            if data.time >= next_frame:
                frame = camera.render(data, timestamp=time.time())
                frame.metadata["simulation_time"] = float(data.time)
                frame.metadata["timestamp_clock"] = "host_wall"
                frame.metadata["calibration_source"] = str(args.profile) if args.profile else "synthetic-demo"
                if writer:
                    writer.write(frame)
                if publisher:
                    publisher.publish(frame)
                if output and frame_count == 0:
                    _save_frame(frame, output)
                frame_count += 1
                next_frame += 1 / args.fps
                if viewer:
                    viewer.sync()
            step_index += 1
            if not args.fast:
                delay = start + data.time - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, .02))
        result = {"robot": spec.name, "physics_steps": step_index, "camera_frames": frame_count,
                  "simulation_seconds": float(data.time), "calibration_source": "device-profile" if args.profile else "synthetic-demo"}
        if output:
            (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


def calibrate(args):
    from .streams import AsMagicSource
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with AsMagicSource(args.ip, mode="camera", pose_convention=args.pose_convention) as source:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            sample = source.poll(timeout_s=.1)
            if sample and sample.rgbd is not None:
                sample.rgbd.calibration.save(output)
                metadata = {"source": "asMagic-live", "ip": args.ip, "sender_timestamp": sample.timestamp,
                            "session_id": sample.session_id, "frame_id": sample.frame_id,
                            "pose_convention": args.pose_convention, "stream_metadata": sample.metadata}
                Path(str(output) + ".source.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
                print(f"Saved received camera intrinsics to {output}")
                return
        raise TimeoutError(f"No calibrated RGB frame from {args.ip}; open Stream and enable camera sending")


def record(args):
    from .recording import RecordingWriter
    from .streams import AsMagicSource
    count = 0
    start = time.monotonic()
    with AsMagicSource(args.ip, mode=args.mode, pose_convention=args.pose_convention) as source, RecordingWriter(args.output) as writer:
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            sample = source.poll(timeout_s=.05)
            if sample:
                writer.write(sample)
                count += 1
                if sample.rgbd is not None and count == 1:
                    _save_frame(sample.rgbd, Path(args.output) / "preview")
        print(json.dumps({"recorded_samples": count, "source": source.stats, "output": args.output}, indent=2))
    if count == 0:
        raise TimeoutError("No frames received; verify the iPhone IP, Wi-Fi, Local Network permission and selected asMagic page")


def live_camera(args):
    """Place a MuJoCo camera from each real frame and retain both observations."""
    from .recording import RecordingWriter
    from .streams import AsMagicSource
    from .geometry import validate_transform
    alignment = validate_transform(np.asarray(json.loads(Path(args.world_transform).read_text()))) if args.world_transform else None
    count = 0
    start = time.monotonic()
    with ExitStack() as stack:
        source = stack.enter_context(AsMagicSource(args.ip, mode="camera", pose_convention=args.pose_convention))
        output = Path(args.output) if args.output else None
        measured = stack.enter_context(RecordingWriter(output / "real")) if output else None
        virtual = stack.enter_context(RecordingWriter(output / "simulation")) if output else None
        camera = viewer = model = data = None
        previous_calibration = None
        previous_session = None
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            sample = source.poll(timeout_s=.02)
            if sample is None or sample.rgbd is None:
                if viewer and not viewer.is_running():
                    break
                continue
            real = sample.rgbd
            if real.world_from_camera is None:
                continue
            if previous_session is not None and sample.session_id != previous_session:
                if args.world_transform:
                    raise RuntimeError("iPhone coordinate session changed; capture a new world alignment before continuing")
                alignment = None
            previous_session = sample.session_id
            if model is None:
                model, data, spec = _scene(args, real.calibration)
                viewer = _viewer(stack, model, data, args.headless)
            if alignment is None:
                # An explicit initial relative registration, not a claim that the
                # arbitrary ARKit origin matches the virtual laboratory origin.
                alignment = look_at() @ np.linalg.inv(real.world_from_camera)
            if previous_calibration != real.calibration:
                if camera:
                    camera.update_calibration(real.calibration)
                else:
                    camera = stack.enter_context(MujocoRGBDCamera(model, real.calibration))
                previous_calibration = real.calibration
            camera.set_world_pose(data, alignment @ real.world_from_camera, move_body=True)
            mujoco.mj_step(model, data)
            simulated = camera.render(data, timestamp=real.timestamp)
            simulated.metadata.update({"paired_frame_id": real.frame_id, "paired_source": "asMagic-live",
                                       "paired_session_id": sample.session_id,
                                       "timestamp_clock": real.metadata.get("timestamp_clock", "sender_source"),
                                       "world_from_sender_alignment": alignment.tolist(),
                                       "registration": "supplied-transform" if args.world_transform else "relative-first-frame"})
            if measured:
                measured.write(sample)
                virtual.write(simulated)
            if output and (count == 0 or (real.rgb is not None and not (output / "real_rgb.png").exists())):
                _save_frame(real, output, "real")
                _save_frame(simulated, output, "simulation")
                (output / "world_from_stream.json").write_text(json.dumps(alignment.tolist(), indent=2) + "\n")
            if viewer:
                viewer.sync()
                if not viewer.is_running():
                    break
            count += 1
        print(json.dumps({"paired_camera_frames": count, "source": source.stats,
                          "registration": "supplied-transform" if args.world_transform else "relative-first-frame"}, indent=2))
    if count == 0:
        raise TimeoutError("No calibrated camera poses received; open Stream, select Y up with no custom transform, and send RGB/depth")


def teleop(args, replay=False):
    from .control import CartesianTeleoperator
    from .recording import RecordingWriter, ReplaySource
    from .streams import AsMagicSource
    model, data, spec = _scene(args)
    controller = CartesianTeleoperator(model, data, phone_to_world=np.eye(3),
                                      translation_scale=args.translation_scale, **spec.controller_kwargs)
    count = 0
    start = time.monotonic()
    previous_calibrate = False
    previous_record = False
    previous_up = previous_down = False
    recording_enabled = True
    with ExitStack() as stack:
        viewer = _viewer(stack, model, data, args.headless)
        source = stack.enter_context(ReplaySource(args.recording, realtime=True, speed=args.speed) if replay else
                                     AsMagicSource(args.ip, mode="teleop"))
        writer = stack.enter_context(RecordingWriter(args.output, metadata={"robot": spec.name, "robot_source": spec.source})) if args.output else None
        telemetry = stack.enter_context((Path(args.output) / "robot-state.jsonl").open("w")) if args.output else None
        print("Hold Square to move. Release to clutch. Cross reanchors. Waiting for phone pose…", flush=True)
        next_tick = time.monotonic()
        latest = None
        status = None
        was_home = False
        initial_qpos = data.qpos[controller.qpos_ids].copy()
        initial_command = data.ctrl[controller.actuator_ids].copy()
        maximum_joint_motion = np.zeros(len(initial_qpos))
        maximum_command_motion = np.zeros(len(initial_command))
        mode_steps = {}
        next_telemetry = start
        release_hint_shown = False
        while (args.duration <= 0 or time.monotonic() - start < args.duration) and (viewer is None or viewer.is_running()):
            now = time.monotonic()
            sample = source.poll(timeout_s=0)
            if sample is not None:
                latest = sample
                count += 1
                if sample.metadata.get("move_release_required") and sample.metadata.get("raw_buttons", {}).get("move") and not release_hint_shown:
                    print("Connected. Release Square once, then hold Square to move.", flush=True)
                    release_hint_shown = True
                if sample.buttons.record and not previous_record:
                    recording_enabled = not recording_enabled
                previous_record = sample.buttons.record
                if sample.buttons.speed_up and not previous_up:
                    controller.change_speed(min(3., controller.speed_scale * 1.25))
                if sample.buttons.speed_down and not previous_down:
                    controller.change_speed(max(.05, controller.speed_scale / 1.25))
                previous_up, previous_down = sample.buttons.speed_up, sample.buttons.speed_down
                if writer and recording_enabled:
                    writer.write(sample)
                if sample.pose is not None:
                    controller.update(sample.pose, receive_time=sample.received_at, gripper=sample.gripper,
                                      session_id=sample.session_id,
                                      reset=sample.buttons.calibrate and not previous_calibrate,
                                      clutch=sample.clutch)
                previous_calibrate = sample.buttons.calibrate
            is_home = bool(latest and latest.buttons.home and not latest.pose_stale and not latest.control_stale and now - latest.received_at < .3)
            if was_home and not is_home:
                controller.hold(reanchor=True)
            if is_home:
                status = controller.go_home(now=now, dt=model.opt.timestep)
            else:
                status = controller.step(now=now, dt=model.opt.timestep)
            was_home = is_home
            mujoco.mj_step(model, data)
            maximum_joint_motion = np.maximum(maximum_joint_motion, np.abs(data.qpos[controller.qpos_ids] - initial_qpos))
            maximum_command_motion = np.maximum(maximum_command_motion, np.abs(data.ctrl[controller.actuator_ids] - initial_command))
            mode_steps[status.mode] = mode_steps.get(status.mode, 0) + 1
            if telemetry and now >= next_telemetry:
                telemetry.write(json.dumps({"received_at": now, "simulation_time": float(data.time),
                                            "mode": status.mode, "qpos": data.qpos.tolist(),
                                            "ctrl": data.ctrl.tolist(), "tcp_pose": controller.tcp_pose().tolist(),
                                            "target_pose": None if controller.target_pose is None else controller.target_pose.tolist(),
                                            "position_error_m": status.position_error,
                                            "rotation_error_rad": status.rotation_error}, allow_nan=False) + "\n")
                next_telemetry = now + .02
            if viewer:
                viewer.sync()
            if replay and source.exhausted:
                controller.hold(reanchor=True)
                break
            next_tick += model.opt.timestep
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, .02))
            elif delay < -.1:
                next_tick = time.monotonic()
        result = {"robot": spec.name, "received_samples": count,
                  "control_state": status.mode if status else "waiting", "control_steps_by_mode": mode_steps,
                  "max_joint_motion_from_start": maximum_joint_motion.tolist(),
                  "max_command_motion_from_start": maximum_command_motion.tolist(),
                  "source": getattr(source, "stats", {})}
        if args.output:
            (Path(args.output) / "robot-run.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2, default=str))
    if count == 0:
        raise TimeoutError("No control frames received; enable Send in the asMagic Robot control page")


def export_phone(args):
    calibration = CameraCalibration.load(args.profile) if args.profile else demo_calibration()
    write_phone_module(args.output, calibration, body_name=args.body_name,
                       camera_name=args.camera_name, mocap=args.mocap, overwrite=args.overwrite)
    print(f"Include {args.output} inside your MJCF worldbody or robot body. Camera: {args.camera_name}")
    print(f"RGB/depth calibration: {Path(args.output).with_suffix('.calibration.json')}")


def doctor(args):
    versions = {}
    for name in ("mujoco", "numpy", "asmagic", "pyzmq"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    print(json.dumps({"iphone_mujoco": __version__, "python": platform.python_version(),
                      "platform": platform.platform(), "packages": versions,
                      "live_sdk_available": versions["asmagic"] != "not installed" and versions["pyzmq"] != "not installed",
                      "graphics": {"status": "not_probed", "requested_backend": os.environ.get("MUJOCO_GL", "auto"),
                                   "context_required_for_headless_rgbd": True,
                                   "linux_display_configured": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) if sys.platform.startswith("linux") else None},
                      "graphics_hint": "macOS viewers: mjpython -m iphone_mujoco phone-lab. Linux headless RGB-D: configure supported EGL, or Xvfb with GLFW. --headless disables the viewer, not rendering.",
                      "live_hint": "Real iPhone streams use the existing asmagic SDK; install this package with its [live] extra."}, indent=2))


def inspect_recording(args):
    from .inspection import inspect_episode
    result = inspect_episode(args.recording)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print(f"Episode: {result['directory']}")
    print(f"Format: {result['format']} v{result['version']}")
    print(f"Frames: {result['frame_count']}; sessions: {result['session_count']}")
    print("Sources: " + (", ".join(f"{name}={count}" for name, count in result["sources"].items()) or "none"))
    for name in ("rgb", "depth"):
        shapes = ", ".join("x".join(map(str, item["shape"])) + f" ({item['frames']} frames)" for item in result[f"{name}_shapes"])
        print(f"{name.upper()}: {shapes or 'absent'}")
    print("Missing: " + ", ".join(f"{name}={count}" for name, count in result["missing"].items()))
    duration = result["recorded_receipt_duration_s"]
    print("Recorded receipt duration: " + (f"{duration:.6g} s" if duration is not None else "unavailable"))
    print(f"Calibration variants: {len(result['calibrations'])}; metric depth range: {result['depth']['minimum_m']} .. {result['depth']['maximum_m']} m")
    for warning in result["validation"]["warnings"]:
        print(f"Timing warning: {warning}")
    print("Structure: valid. " + result["clock_note"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="iPhone RGB-D camera modules and robot experiments in MuJoCo")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Show installed runtime versions").set_defaults(func=doctor)
    inspect = sub.add_parser("inspect-episode", help="Validate and summarize a recorded episode without graphics or network")
    inspect.add_argument("recording", help="Episode directory containing manifest.json and frames/")
    inspect.add_argument("--json", action="store_true", help="Print the complete machine-readable summary")
    inspect.set_defaults(func=inspect_recording)
    lab = sub.add_parser("phone-lab", help="Open the visible virtual iPhone with RGB-D previews and tabletop experiments")
    lab.add_argument("--profile", help="Saved runtime iPhone calibration; otherwise use a labeled synthetic example")
    lab.add_argument("--duration", type=float, default=0., help="Wall seconds; 0 keeps the window open")
    lab.add_argument("--fps", type=float, default=10., help="Maximum preview refresh rate")
    lab.add_argument("--headless", action="store_true", help="Save a single rendered RGB-D example without opening a viewer")
    lab.add_argument("--output", help="Snapshot directory; P saves a new capture inside it")
    lab.add_argument("--scan", action="store_true", help="Move the virtual phone slowly across the scene")
    from .phone_lab import run_phone_lab
    lab.set_defaults(func=run_phone_lab)
    export_lab = sub.add_parser("export-lab", help="Export a standalone MuJoCo scene plus reusable iPhone module")
    export_lab.add_argument("--profile")
    export_lab.add_argument("--output", required=True, help="New or empty export directory")
    def export_lab_command(args):
        from .phone_scene import export_phone_lab
        calibration = CameraCalibration.load(args.profile) if args.profile else None
        print(export_phone_lab(args.output, calibration))
    export_lab.set_defaults(func=export_lab_command)
    models = sub.add_parser("models", help="List or download the supported robot models")
    models.add_argument("--fetch", choices=("panda", "ur5e"))
    models.add_argument("--model-cache")
    def models_command(args):
        from .robots import fetch_menagerie, list_robots
        if args.fetch:
            print(fetch_menagerie(args.fetch, cache_dir=args.model_cache))
        else:
            print(list_robots())
    models.set_defaults(func=models_command)
    def robot_options(command):
        command.add_argument("--robot", default="panda", help="panda (default), ur5e, or custom adapter JSON")
        command.add_argument("--model-cache")
        command.add_argument("--headless", action="store_true")
        command.add_argument("--duration", type=float, default=10., help="Seconds; 0 keeps running")
        command.add_argument("--output", help="Record into a new directory")
        command.add_argument("--translation-scale", type=float, default=1., help="Robot/phone translation ratio; try 0.25 for smaller targets")
    sim = sub.add_parser("simulate", help="Run a robot with a calibrated virtual iPhone camera")
    robot_options(sim)
    sim.add_argument("--profile", help="JSON intrinsics captured with calibrate")
    sim.add_argument("--steps", type=int, default=0)
    sim.add_argument("--fps", type=float, default=15.)
    sim.add_argument("--fast", action="store_true", help="Run without wall-clock pacing")
    sim.add_argument("--publish", metavar="ENDPOINT", help="e.g. tcp://127.0.0.1:18000")
    sim.set_defaults(func=simulate)
    wrist = sub.add_parser("capture-wrist", help="Collect synchronized wrist RGB-D and robot states using iPhone calibration")
    wrist.add_argument("--robot", default="panda", help="panda (default), ur5e, or custom adapter JSON")
    wrist.add_argument("--model-cache")
    wrist.add_argument("--profile", required=True, help="Calibration JSON from calibrate or the synthetic export-lab example")
    wrist.add_argument("--mount", help="Measured parent-link <- CV camera mount JSON; default is a demo mount")
    wrist.add_argument("--show-phone", action="store_true", help="Show an approximate massless phone shell at the optical mount")
    wrist.add_argument("--output", required=True, help="New dataset directory")
    wrist.add_argument("--duration", type=float, default=10., help="Simulation seconds; 0 runs until stopped")
    wrist.add_argument("--fps", type=float, default=30., help="Frames per simulation second")
    wrist.add_argument("--depth-scale", type=float, default=.0001, help="Meters per UInt16 depth code")
    wrist.add_argument("--headless", action="store_true")
    wrist.add_argument("--fast", action="store_true", help="Generate offline without wall-clock pacing")
    wrist.add_argument("--motion", choices=("demo", "hold"), default="demo")
    wrist.add_argument("--translation-scale", type=float, default=.25)
    wrist.add_argument("--publish", metavar="ENDPOINT", help="Optional existing asMagic camera wire output")
    wrist_input = wrist.add_mutually_exclusive_group()
    wrist_input.add_argument("--ip", help="Drive the robot with live asMagic Robot control input")
    wrist_input.add_argument("--input-episode", help="Drive the robot with a previously recorded teleoperation episode")
    from .wrist import capture_command
    wrist.set_defaults(func=capture_command)
    cal = sub.add_parser("calibrate", help="Save actual per-frame iPhone intrinsics")
    cal.add_argument("--ip", required=True)
    cal.add_argument("--output", required=True)
    cal.add_argument("--timeout", type=float, default=15.)
    cal.add_argument("--pose-convention", choices=("arkit", "robotics"), default="arkit")
    cal.set_defaults(func=calibrate)
    rec = sub.add_parser("record", help="Record live camera or Robot control streams")
    rec.add_argument("--ip", required=True)
    rec.add_argument("--mode", choices=("camera", "teleop"), default="camera")
    rec.add_argument("--pose-convention", choices=("arkit", "robotics"), default="arkit")
    rec.add_argument("--duration", type=float, default=10.)
    rec.add_argument("--output", required=True)
    rec.set_defaults(func=record)
    cam = sub.add_parser("camera", help="Move a virtual camera with live iPhone poses; save paired real/sim RGB-D")
    robot_options(cam)
    cam.add_argument("--ip", required=True)
    cam.add_argument("--pose-convention", choices=("arkit", "robotics"), default="arkit")
    cam.add_argument("--world-transform", help="JSON 4x4 mapping stream Z-up world into the virtual world")
    cam.set_defaults(func=live_camera)
    live = sub.add_parser("teleop", help="Control a simulated robot from the asMagic Robot control page")
    robot_options(live)
    live.add_argument("--ip", required=True)
    live.set_defaults(func=teleop)
    replay_parser = sub.add_parser("replay", help="Replay recorded teleoperation into a selected robot")
    robot_options(replay_parser)
    replay_parser.add_argument("recording")
    replay_parser.add_argument("--speed", type=float, default=1.)
    replay_parser.set_defaults(func=lambda args: teleop(args, replay=True))
    export = sub.add_parser("export-phone", help="Export a portable MJCF phone include")
    export.add_argument("--profile")
    export.add_argument("--output", required=True)
    export.add_argument("--body-name", default="iphone")
    export.add_argument("--camera-name", default="iphone_camera")
    export.add_argument("--mocap", action="store_true", help="Movable phone, directly under worldbody")
    export.add_argument("--overwrite", action="store_true", help="Replace this generated XML and calibration sidecar")
    export.set_defaults(func=export_phone)
    args = parser.parse_args(argv)
    for name in ("duration", "fps", "speed", "translation_scale", "depth_scale", "timeout"):
        value = getattr(args, name, None)
        if value is None:
            continue
        if not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
        if (name == "duration" and value < 0) or (name != "duration" and value <= 0):
            qualifier = "nonnegative" if name == "duration" else "positive"
            parser.error(f"--{name.replace('_', '-')} must be {qualifier}")
    if getattr(args, "steps", 0) < 0:
        parser.error("--steps must be nonnegative")
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, TimeoutError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
