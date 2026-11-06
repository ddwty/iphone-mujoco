"""Collect iPhone-format RGB-D from a tabletop, mobile base or your own MJCF.

macOS viewers use mjpython. Headless capture still requires OpenGL.
The built-in scenes use scripted target/base motion and synthetic calibration.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image

from iphone_mujoco import CameraCalibration, MujocoRGBDCamera
from iphone_mujoco.phone_lab import update_rgbd_preview
from iphone_mujoco.cli import look_at
from iphone_mujoco.phone_scene import demo_calibration, load_phone_lab
from iphone_mujoco.recording import RecordingWriter
from iphone_mujoco.virtual_device import quantize_depth
from iphone_mujoco.wrist import WristCameraMount, wrist_camera_mjcf


def _mobile_scene(calibration):
    mount = WristCameraMount(
        "mobile_base", look_at(eye=(0., .20, .62), target=(0., 1.8, .32)),
        camera_name="iphone_camera", measured=False,
    )
    phone = wrist_camera_mjcf(calibration, mount, show_phone=True)
    xml = f"""<mujoco model="Scripted mobile RGB-D camera">
      <option timestep=".002"/>
      <visual><global offwidth="640" offheight="480"/>
        <headlight ambient=".4 .4 .4" diffuse=".6 .6 .6"/>
        <map znear=".005" zfar="12"/>
      </visual>
      <statistic center="0 .1 .4" extent="2.8"/>
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="128" height="128"
          rgb1=".17 .20 .24" rgb2=".25 .29 .33"/>
        <material name="floor" texture="grid" texrepeat="10 10"/>
      </asset>
      <worldbody>
        <light pos="0 -1 4" dir="0 .3 -1" diffuse=".85 .85 .85"/>
        <geom name="floor" type="plane" size="4 4 .1" material="floor"/>
        <geom type="box" pos="-1.2 .3 .40" size=".16 .28 .40" rgba=".19 .48 .82 1"/>
        <geom type="box" pos="1.15 .9 .5" size=".15 .40 .5" rgba=".85 .47 .18 1"/>
        <geom type="box" pos="-.35 1.4 .25" size=".22 .17 .25" rgba=".85 .24 .19 1"/>
        <geom type="cylinder" pos=".45 .65 .30" size=".14 .30" rgba=".16 .67 .44 1"/>
        <geom type="sphere" pos="-.7 2.0 .20" size=".20" rgba=".83 .72 .22 1"/>
        <geom type="box" pos="0 2.6 .8" size="1.5 .04 .8" rgba=".58 .62 .67 1"/>
        <body name="mobile_base" mocap="true" pos="0 -1.3 0">
          <geom type="box" pos="0 0 .16" size=".22 .27 .09" rgba=".12 .42 .64 1"/>
          <geom type="cylinder" pos="-.235 .13 .10" size=".10 .028"
            quat=".70710678 0 .70710678 0" rgba=".08 .09 .11 1"/>
          <geom type="cylinder" pos=".235 .13 .10" size=".10 .028"
            quat=".70710678 0 .70710678 0" rgba=".08 .09 .11 1"/>
          <geom type="cylinder" pos="-.235 -.13 .10" size=".10 .028"
            quat=".70710678 0 .70710678 0" rgba=".08 .09 .11 1"/>
          <geom type="cylinder" pos=".235 -.13 .10" size=".10 .028"
            quat=".70710678 0 .70710678 0" rgba=".08 .09 .11 1"/>
          <geom type="cylinder" fromto="0 .16 .25 0 .16 .57"
            size=".018" rgba=".25 .28 .33 1"/>
          {phone}
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model), mount.camera_name


def build_scene(scenario, calibration, *, scene=None, parent_body=None, mount=None):
    """Return model, state and camera name for a reproducible experiment.

    A custom scene uses MjSpec to retain original include/mesh resolution.
    Its camera follows the selected body's kinematics at the supplied fixed
    parent-link <- CV-camera transform. Existing scene bodies stay in place.
    Initialization uses the scene's first keyframe when present, otherwise its
    default state. The collector then advances the source dynamics and controls.
    """
    if scenario == "tabletop":
        model, data = load_phone_lab(calibration)
        return model, data, "iphone_camera"
    if scenario == "mobile":
        model, data, camera_name = _mobile_scene(calibration)
        mujoco.mj_forward(model, data)
        return model, data, camera_name
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
        raise ValueError(f"Camera already exists; choose another mount.camera_name: {mount.camera_name}")
    phone = wrist_camera_mjcf(calibration, mount, show_phone=True)
    child = mujoco.MjSpec.from_string(f"<mujoco><worldbody>{phone}</worldbody></mujoco>")
    frame = parent.add_frame()
    spec.attach(child, prefix="", frame=frame)
    model = spec.compile()
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, mount.camera_name


def advance_scene(scenario, model, data, time_s):
    """Integrate to a simulation time and apply built-in scripted motion.

    Only the demonstration target or mocap base is positioned by this function.
    Custom scenes advance through their own MuJoCo dynamics and controls.
    """
    if not np.isfinite(time_s) or time_s < data.time - 1e-9:
        raise ValueError("Scene time must be finite and nondecreasing")
    while data.time < time_s - 1e-9:
        mujoco.mj_step(model, data)
    if scenario == "tabletop":
        target = model.body("blue_sphere").id
        model.body_pos[target] = [.16 + .15 * np.sin(1.5 * data.time),
                                  -.055 + .08 * np.sin(.75 * data.time), .805]
    elif scenario == "mobile":
        mocap = int(model.body_mocapid[model.body("mobile_base").id])
        yaw = .22 * np.sin(.8 * data.time)
        data.mocap_pos[mocap] = [.6 * np.sin(.6 * data.time),
                                -1.3 + .25 * np.sin(.3 * data.time), 0.]
        data.mocap_quat[mocap] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)


def _world_from_body(model, data, name):
    body = model.body(name).id
    pose = np.eye(4)
    pose[:3, :3] = data.xmat[body].reshape(3, 3)
    pose[:3, 3] = data.xpos[body]
    return pose.tolist()


def collect(args):
    calibration = CameraCalibration.load(args.profile) if args.profile else demo_calibration()
    mount = WristCameraMount.load(args.mount) if args.mount else None
    model, data, camera_name = build_scene(args.scenario, calibration, scene=args.scene,
                                         parent_body=args.parent_body, mount=mount)
    if args.fps > 1 / model.opt.timestep:
        raise ValueError("FPS must not exceed the simulation physics rate")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "rgb").mkdir(exist_ok=True)
    (output / "depth").mkdir(exist_ok=True)
    calibration.save(output / "calibration.json")
    if mount is not None:
        mount.save(output / "mount.json")
    metadata = {"scenario": args.scenario, "source": "mujoco-simulation",
                "calibration_source": "caller_profile" if args.profile else "synthetic_example",
                "camera_name": camera_name, "timestamp_clock": "simulation",
                "depth_meters_per_unit": .0001, "depth_invalid_code": 0,
                "depth_saturated_reserved_code": 65535,
                "motion_model": {"tabletop": "scripted_target_path",
                                 "mobile": "scripted_kinematic_base_path",
                                 "custom": "source_scene_dynamics"}[args.scenario],
                "rgb": {"width": calibration.width, "height": calibration.height, "order": "RGB",
                        "K": calibration.matrix.tolist()},
                "depth": {"width": calibration.depth_calibration().width,
                          "height": calibration.depth_calibration().height,
                          "K": calibration.depth_calibration().matrix.tolist(),
                          "definition": "optical_axis_z", "meters_per_unit": .0001}}
    (output / "camera_info.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()
    frame_count = 0
    with ExitStack() as stack:
        camera = stack.enter_context(MujocoRGBDCamera(model, calibration, camera_name,
                                                     max_depth_m=6.55345))
        writer = stack.enter_context(RecordingWriter(output / "episode", metadata=metadata))
        states = stack.enter_context((output / "states.jsonl").open("w"))
        viewer = None
        if not args.headless:
            import mujoco.viewer
            viewer = stack.enter_context(mujoco.viewer.launch_passive(model, data))
            viewer.cam.lookat[:] = [0., 0., .6]
            viewer.cam.distance = 2.6 if args.scenario == "tabletop" else 4.2
            viewer.cam.azimuth, viewer.cam.elevation = 130, -28
        for index in range(math.ceil(args.duration * args.fps)):
            if viewer is not None and not viewer.is_running():
                break
            advance_scene(args.scenario, model, data, min((index + 1) / args.fps, args.duration))
            frame = camera.render(data)
            raw_depth = quantize_depth(frame.depth_m)
            frame.depth_m = raw_depth.astype(np.float32) * .0001
            frame.depth_m[raw_depth == 0] = np.nan
            frame.frame_id = index
            state = {"index": index, "simulation_time": float(data.time),
                     "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(), "ctrl": data.ctrl.tolist(),
                     "world_from_camera": frame.world_from_camera.tolist()}
            if args.scenario == "mobile":
                state["world_from_base"] = _world_from_body(model, data, "mobile_base")
            elif args.scenario == "tabletop":
                state["world_from_target"] = _world_from_body(model, data, "blue_sphere")
            frame.metadata.update(metadata, scene_state=state, simulation_time=float(data.time))
            writer.write(frame)
            stem = f"{index:08d}.png"
            Image.fromarray(frame.rgb).save(output / "rgb" / stem)
            Image.fromarray(raw_depth).save(output / "depth" / stem)
            states.write(json.dumps(state, allow_nan=False) + "\n")
            if index == 0:
                frame.point_cloud(world=True, stride=2).save_ply(output / "preview_points.ply")
            frame_count += 1
            if viewer is not None:
                update_rgbd_preview(viewer, frame)
                viewer.sync()
                time.sleep(max(0., min(.1, started + data.time - time.monotonic())))
    result = dict(metadata, frames=frame_count, simulation_seconds=float(data.time),
                  wall_seconds=time.monotonic() - started)
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "frames": frame_count,
                      "simulation_seconds": float(data.time)}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("tabletop", "mobile", "custom"), default="tabletop")
    parser.add_argument("--profile", help="Runtime camera calibration JSON; default is synthetic")
    parser.add_argument("--scene", help="Your MJCF XML, required for custom")
    parser.add_argument("--parent-body", help="Must match the body named in the mount JSON")
    parser.add_argument("--mount", help="WristCameraMount JSON with parent_from_camera in CV optical axes")
    parser.add_argument("--duration", type=float, default=4.)
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--output", required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite([args.duration, args.fps]).all() or args.duration <= 0 or args.fps <= 0:
        parser.error("Duration and FPS must be finite and positive")
    if args.scenario == "custom" and (not args.scene or not args.mount):
        parser.error("Custom scenes require --scene and --mount")
    if args.scenario != "custom" and (args.scene or args.mount or args.parent_body):
        parser.error("--scene, --mount and --parent-body are for custom scenes")
    try:
        collect(args)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
