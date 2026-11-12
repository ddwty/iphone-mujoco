"""Camera attachment, actuated scene stepping and synchronized RGB-D episodes."""
from __future__ import annotations

from contextlib import ExitStack
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from .camera import MujocoRGBDCamera
from .phone_scene import load_phone_lab
from .recording import RecordingWriter
from .virtual_device import quantize_depth
from .wrist import WristCameraMount, wrist_camera_mjcf


def _scene_configuration(calibration, configuration):
    from .configuration import CameraConfiguration, configure_camera
    if configuration is not None:
        if calibration is not None and calibration != configuration.calibration:
            raise ValueError("Configuration and camera calibration must match")
        return configuration
    if calibration is None:
        return configure_camera()
    return CameraConfiguration("iphone-17-pro", calibration, "user_supplied_intrinsics",
                               metadata={"geometry_selection": "default_scene_device"})


def load_camera_scene(scenario, calibration=None, *, scene=None, parent_body=None,
                      mount=None, configuration=None, model_cache=None):
    """Load the inspection bench, TIAGo, or a camera attached to your MJCF.

    Custom models keep relative asset/include resolution, joints, controls and
    the first keyframe. Mobile motion is owned by DifferentialDriveController.
    """
    configuration = _scene_configuration(calibration, configuration)
    calibration = configuration.calibration
    if scenario == "tabletop":
        model, data = load_phone_lab(calibration, configuration=configuration)
        return model, data, "iphone_camera"
    if scenario == "mobile":
        from .mobile import load_mobile_scene
        model, data, _ = load_mobile_scene(calibration, configuration=configuration, model_cache=model_cache)
        return model, data, "iphone_camera"
    if scenario != "custom":
        raise ValueError("Choose tabletop, mobile or custom")
    if scene is None or not isinstance(mount, WristCameraMount):
        raise ValueError("Custom scenes require an MJCF path and a WristCameraMount")
    if parent_body is not None and parent_body != mount.parent_body:
        raise ValueError("--parent-body must match mount.parent_body")
    spec = mujoco.MjSpec.from_file(str(Path(scene).expanduser().resolve()))
    parent = spec.worldbody if mount.parent_body == "world" else spec.body(mount.parent_body)
    if parent is None:
        raise ValueError(f"Camera parent body is missing: {mount.parent_body}")
    if spec.camera(mount.camera_name) is not None:
        raise ValueError(f"Camera already exists: {mount.camera_name}")
    phone = wrist_camera_mjcf(calibration, mount, show_phone=True, configuration=configuration)
    document = ET.fromstring(phone)
    child = mujoco.MjSpec.from_string(
        phone if document.tag == "mujoco" else f"<mujoco><worldbody>{phone}</worldbody></mujoco>")
    spec.attach(child, prefix="", frame=parent.add_frame())
    model = spec.compile()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, calibration.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, calibration.height)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, mount.camera_name


def mobile_command(time_s, motion="route", linear_speed=.18, angular_speed=.35):
    """Return a repeatable base-frame twist in meters/s and radians/s."""
    if not np.isfinite([time_s, linear_speed, angular_speed]).all() or min(time_s, linear_speed, angular_speed) < 0:
        raise ValueError("Route time and speed limits must be finite and nonnegative")
    if motion == "hold":
        return 0., 0.
    if motion == "straight":
        return linear_speed, 0.
    if motion != "route":
        raise ValueError("Motion must be route, straight or hold")
    if time_s < 2.5:
        return linear_speed, 0.
    if time_s < 5.5:
        return linear_speed * .65, angular_speed
    if time_s < 7.5:
        return linear_speed, 0.
    return 0., 0.


def advance_scene(scenario, model, data, time_s, *, controller=None,
                  motion="route", linear_speed=.18, angular_speed=.35, sequence_start=0.):
    """Advance real physics; built-in trajectories command actuators only."""
    if not np.isfinite(time_s) or time_s < data.time - 1e-9:
        raise ValueError("Scene time must be finite and nondecreasing")
    if scenario == "mobile" and controller is None:
        raise ValueError("Mobile stepping requires a DifferentialDriveController")
    while data.time < time_s - 1e-9:
        if scenario == "mobile":
            controller.command(*mobile_command(max(0., float(data.time)-sequence_start), motion, linear_speed, angular_speed))
            controller.step()
        else:
            if scenario == "tabletop":
                data.ctrl[model.actuator("inspection_slide_drive").id] = .065 * np.sin(.7 * data.time)
            mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def world_from_body(model, data, name):
    body = model.body(name).id
    pose = np.eye(4)
    pose[:3, :3] = data.xmat[body].reshape(3, 3)
    pose[:3, 3] = data.xpos[body]
    return pose.tolist()


class SceneRGBDRecorder:
    """Capture RGB, optical Z depth and scene state from one physics state."""
    def __init__(self, model, data, calibration, directory, *, camera_name="iphone_camera", metadata=None):
        self.model, self.data = model, data
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(f"Choose a new or empty output directory: {self.directory}")
        self.metadata = dict(metadata or {})
        self.metadata.update(source="mujoco-simulation", timestamp_clock="simulation",
                             camera_name=camera_name, depth_meters_per_unit=.0001,
                             depth_invalid_code=0, depth_saturated_reserved_code=65535,
                             depth_definition="optical_axis_z")
        self.metadata["rgb"] = {"width": calibration.width, "height": calibration.height,
                                "order": "RGB", "K": calibration.matrix.tolist()}
        kd = calibration.depth_calibration()
        self.metadata["depth"] = {"width": kd.width, "height": kd.height,
                                  "K": kd.matrix.tolist(), "definition": "optical_axis_z",
                                  "meters_per_unit": .0001}
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in ("rgb", "depth"):
            (self.directory / name).mkdir()
        calibration.save(self.directory / "calibration.json")
        (self.directory / "camera_info.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        self._resources = ExitStack()
        try:
            self._camera = self._resources.enter_context(MujocoRGBDCamera(model, calibration, camera_name, max_depth_m=6.55345))
            self._writer = self._resources.enter_context(RecordingWriter(self.directory / "episode", metadata=self.metadata))
            self._states = self._resources.enter_context((self.directory / "states.jsonl").open("w"))
        except Exception:
            self._resources.close()
            raise
        self.count, self.closed = 0, False

    def capture(self, extra_state=None):
        if self.closed:
            raise RuntimeError("Scene recorder is closed")
        frame = self._camera.render(self.data)
        raw = quantize_depth(frame.depth_m)
        frame.depth_m = raw.astype(np.float32) * .0001
        frame.depth_m[raw == 0] = np.nan
        frame.frame_id = self.count
        state = dict(extra_state or {})
        state.update(index=self.count, simulation_time=float(self.data.time),
                     qpos=self.data.qpos.tolist(), qvel=self.data.qvel.tolist(), ctrl=self.data.ctrl.tolist(),
                     world_from_camera=frame.world_from_camera.tolist())
        for name in ("qpos", "qvel", "ctrl"):
            if not np.isfinite(state[name]).all():
                raise RuntimeError("Nonfinite scene state; refusing to record")
        frame.metadata.update(self.metadata, scene_state=state, simulation_time=float(self.data.time))
        self._writer.write(frame)
        stem = f"{self.count:08d}.png"
        Image.fromarray(frame.rgb).save(self.directory / "rgb" / stem)
        Image.fromarray(raw).save(self.directory / "depth" / stem)
        self._states.write(json.dumps(state, allow_nan=False) + "\n")
        if self.count == 0:
            frame.point_cloud(world=True, stride=2).save_ply(self.directory / "preview_points.ply")
        self.count += 1
        return frame

    def close(self):
        self._resources.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def collect_camera_episode(scenario, directory, *, configuration=None, calibration=None,
                           duration=8., fps=15., headless=False, scene=None, parent_body=None,
                           mount=None, model_cache=None, motion="route", linear_speed=.18, angular_speed=.35):
    """Record an actuated inspection/navigation scene or an existing MJCF."""
    if not np.isfinite([duration, fps]).all() or min(duration, fps) <= 0:
        raise ValueError("Duration and FPS must be finite and positive")
    configuration = _scene_configuration(calibration, configuration)
    calibration = configuration.calibration
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Choose a new or empty output directory: {directory}")
    mobile_spec = None
    if scenario == "mobile":
        from .mobile import load_mobile_scene
        model, data, mobile_spec = load_mobile_scene(calibration, configuration=configuration, model_cache=model_cache)
        camera_name = mobile_spec.camera_name
    else:
        model, data, camera_name = load_camera_scene(
            scenario, calibration, scene=scene, parent_body=parent_body, mount=mount,
            configuration=configuration, model_cache=model_cache)
    if fps > 1 / model.opt.timestep:
        raise ValueError("FPS cannot exceed the physics rate")
    controller = None
    metadata = {"scenario": scenario, "calibration_source": configuration.calibration_source if configuration else "caller_supplied",
                "motion_model": "source_scene_dynamics" if scenario == "custom" else "actuated_linear_stage"}
    if scenario == "mobile":
        from .mobile import DifferentialDriveController
        controller = DifferentialDriveController(model, data, mobile_spec)
        metadata.update(robot="tiago", robot_source=controller.spec.source,
                        motion_model="differential_drive_wheel_contact_dynamics",
                        odometry_source="wheel_encoder_integration", ground_truth_source="mujoco_body_pose",
                        wheel_joint_names=list(controller.spec.wheel_joint_names),
                        wheel_actuator_names=list(controller.spec.wheel_actuator_names),
                        wheel_radius_m=controller.spec.radius_m, wheel_track_m=controller.spec.track_m,
                        route=motion, robot_metadata=mobile_spec.metadata)
    if configuration:
        metadata["camera_configuration"] = configuration.to_dict()
    if mount:
        from .wrist import configured_wrist_mount
        metadata["mount"] = configured_wrist_mount(mount, configuration).to_dict()
    start = time.monotonic()
    epoch = float(data.time)
    metadata["simulation_start_time"] = epoch
    with ExitStack() as stack:
        recorder = stack.enter_context(SceneRGBDRecorder(model, data, calibration, directory,
                                                         camera_name=camera_name, metadata=metadata))
        if configuration:
            configuration.save(Path(directory) / "device.json")
        viewer = None
        if not headless:
            import mujoco.viewer
            from .phone_lab import update_rgbd_preview
            viewer = stack.enter_context(mujoco.viewer.launch_passive(model, data))
            viewer.cam.lookat[:] = [.65, 0., .8] if scenario == "mobile" else [0., -.14, .76]
            viewer.cam.distance = 3.8 if scenario == "mobile" else 2.
            viewer.cam.azimuth, viewer.cam.elevation = 140, -22
        for index in range(math.ceil(duration * fps)):
            if viewer is not None and not viewer.is_running():
                break
            advance_scene(scenario, model, data, epoch + min((index + 1) / fps, duration), controller=controller,
                          motion=motion, linear_speed=linear_speed, angular_speed=angular_speed, sequence_start=epoch)
            state = controller.state() if controller else {}
            if scenario == "tabletop":
                state["world_from_target"] = world_from_body(model, data, "inspection_part")
            frame = recorder.capture(state)
            if viewer:
                update_rgbd_preview(viewer, frame)
                viewer.sync()
                time.sleep(max(0., min(.05, start + float(data.time)-epoch - time.monotonic())))
        result = dict(metadata, frames=recorder.count, simulation_seconds=float(data.time)-epoch,
                      simulation_end_time=float(data.time),
                      wall_seconds=time.monotonic() - start)
        (Path(directory) / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def capture_mobile_command(args):
    from .cli import _camera_configuration
    from .configuration import configure_camera
    configuration = _camera_configuration(args) or configure_camera()
    result = collect_camera_episode("mobile", args.output, configuration=configuration,
                                    duration=args.duration, fps=args.fps, headless=args.headless,
                                    model_cache=args.model_cache, motion=args.motion,
                                    linear_speed=args.linear_speed, angular_speed=args.angular_speed)
    print(json.dumps({key: result[key] for key in ("robot", "frames", "simulation_seconds")}, indent=2))
