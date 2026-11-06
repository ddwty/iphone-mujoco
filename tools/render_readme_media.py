"""Render the README GIFs from actual MuJoCo camera observations.

Run from a source checkout with a working OpenGL context. Robot assets must
already be present when selecting Panda or UR5e. Outputs default to recordings/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from iphone_mujoco import MujocoRGBDCamera
from iphone_mujoco.phone_lab import _frustum, depth_preview
from iphone_mujoco.phone_scene import (
    VIEWER_SETTINGS, default_world_from_camera, demo_calibration, load_phone_lab,
)
from iphone_mujoco.robots import load_scene
from iphone_mujoco.wrist import default_wrist_mount, wrist_camera_mjcf, _place_demo_targets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from scene_camera import advance_scene, build_scene


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def panel(overview, frame, title, subtitle):
    """Resize only the presentation. Sensor arrays keep their native raster."""
    canvas = Image.new("RGB", (816, 508), "#101923")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), title, font=font(20), fill="#f1f6fc")
    draw.text((16, 36), subtitle, font=font(12), fill="#99b4cc")
    draw.text((685, 16), f"t = {frame.timestamp:04.1f} s", font=font(14), fill="#6cdddf")
    canvas.paste(Image.fromarray(overview), (16, 64))
    canvas.paste(Image.fromarray(frame.rgb).resize((256, 192), Image.Resampling.LANCZOS), (544, 64))
    depth = depth_preview(frame.depth_m, near=.15, far=2.)
    canvas.paste(Image.fromarray(depth).resize((256, 192), Image.Resampling.NEAREST), (544, 284))
    draw.text((544, 46), "RGB  /  640 x 480", font=font(12), fill="#dce8f4")
    draw.text((544, 266), "DEPTH  /  0.15 - 2.00 m", font=font(12), fill="#dce8f4")
    position = frame.world_from_camera[:3, 3]
    draw.text((16, 459), "CAMERA WORLD XYZ  " + "  ".join(f"{x:+.3f}" for x in position) + " m",
              font=font(13), fill="#a1bdcf")
    draw.text((16, 487), "MUJOCO SIMULATION  |  synthetic K  |  ideal optical Z  |  synchronized views",
              font=font(12), fill="#6cdddf")
    return canvas


def render_overview(renderer, model, data, view, frame):
    renderer.update_scene(data, camera=view)
    # Reuse the lab's display-only frustum, then append scene geometry.
    _frustum(renderer.scene, frame)
    mujoco.mjv_addGeoms(model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
                       mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)
    return renderer.render().copy()


def overview_camera(settings):
    view = mujoco.MjvCamera()
    for name, value in settings.items():
        if name == "lookat":
            view.lookat[:] = value
        else:
            setattr(view, name, value)
    return view


def lab_frames(count):
    calibration = demo_calibration()
    model, data = load_phone_lab(calibration)
    view = overview_camera(VIEWER_SETTINGS)
    frames = []
    with MujocoRGBDCamera(model, calibration, max_depth_m=6.55345, hidden_geom_groups=(3,)) as sensor, \
            mujoco.Renderer(model, height=384, width=512) as overview:
        for time_s in np.linspace(0, 6, count, endpoint=False):
            pose = default_world_from_camera()
            pose[0, 3] += .12 * np.sin(time_s * np.pi / 3)
            sensor.set_world_pose(data, pose, move_body=True)
            data.time = time_s
            frame = sensor.render(data)
            image = render_overview(overview, model, data, view, frame)
            frames.append(panel(image, frame, "iPhone MuJoCo / camera laboratory",
                                "Move the phone. Observe RGB, depth and occlusion from the same pose."))
    return frames


def wrist_frames(robot, count):
    calibration = demo_calibration()
    mount = default_wrist_mount(robot)
    model, data, spec = load_scene(robot, phone_xml=wrist_camera_mjcf(calibration, mount, show_phone=True),
                                   phone_parent_body=mount.parent_body)
    _place_demo_targets(model, data, mount.camera_name)
    model.vis.headlight.ambient[:] = .4
    home = np.array([data.qpos[model.joint(name).qposadr[0]] for name in spec.joint_names])
    actuators = [model.actuator(name).id for name in spec.actuator_names]
    view = overview_camera({"lookat": [.12, 0., .35], "distance": 1.55,
                            "azimuth": 132., "elevation": -25.})
    frames = []
    with MujocoRGBDCamera(model, calibration, mount.camera_name, max_depth_m=6.55345,
                         hidden_geom_groups=(3,)) as sensor, \
            mujoco.Renderer(model, height=384, width=512) as overview:
        for time_s in np.linspace(.002, 3.2, count):
            while data.time < time_s - 1e-9:
                phase = np.arange(len(home)) * .4
                desired = home + .08 * (np.sin(phase + data.time * .8) - np.sin(phase))
                data.ctrl[actuators] = desired
                mujoco.mj_step(model, data)
            frame = sensor.render(data)
            image = render_overview(overview, model, data, view, frame)
            name = "Franka Panda" if robot == "panda" else "Universal Robots UR5e"
            frames.append(panel(image, frame, "iPhone MuJoCo / " + name,
                                "Rigid link mount. RGB-D follows forward kinematics and joint motion."))
    return frames


def mobile_frames(count):
    calibration = demo_calibration()
    model, data, camera_name = build_scene("mobile", calibration)
    view = overview_camera({"lookat": [0., 0., .25], "distance": 3.6,
                            "azimuth": 132., "elevation": -32.})
    frames = []
    with MujocoRGBDCamera(model, calibration, camera_name, max_depth_m=6.55345,
                         hidden_geom_groups=(3,)) as sensor, \
            mujoco.Renderer(model, height=384, width=512) as overview:
        for time_s in np.linspace(0, 6, count, endpoint=False):
            advance_scene("mobile", model, data, time_s)
            frame = sensor.render(data)
            image = render_overview(overview, model, data, view, frame)
            frames.append(panel(image, frame, "iPhone MuJoCo / mobile camera",
                                "Scripted mobile base. Record optical pose, base pose and RGB-D."))
    return frames


def save_animation(frames, path, fps):
    # One palette across the clip avoids flicker and bounds download size.
    samples = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
    atlas = Image.new("RGB", (408, 254 * len(samples)))
    for row, index in enumerate(samples):
        atlas.paste(frames[index].resize((408, 254)), (0, row * 254))
    palette = atlas.quantize(colors=112)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=round(1000 / fps),
                    loop=0, optimize=True, disposal=2)
    return {"file": path.name, "frames": len(frames), "width": 816, "height": 508,
            "playback_fps": fps, "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("recordings/readme-media"))
    parser.add_argument("--robots", nargs="+", choices=("panda", "ur5e"),
                        default=["panda", "ur5e"])
    parser.add_argument("--frames", type=int, default=32, help="Frames per lab/mobile clip")
    parser.add_argument("--robot-frames", type=int, default=18, help="Frames per robot in the wrist montage")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    if not 4 <= args.frames <= 120 or not 4 <= args.robot_frames <= 60 or not 1 <= args.fps <= 30:
        parser.error("Use 4-120 clip frames, 4-60 robot frames and 1-30 playback FPS")
    paths = [args.output / name for name in ("phone-lab.gif", "robot-wrist.gif", "mobile-camera.gif")]
    if any(path.exists() for path in paths):
        parser.error("Animation already exists. Choose a new output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    lab = save_animation(lab_frames(args.frames), paths[0], args.fps)
    print(json.dumps(lab), flush=True)
    wrist = []
    for robot in args.robots:
        wrist.extend(wrist_frames(robot, args.robot_frames))
    print(json.dumps(save_animation(wrist, paths[1], args.fps)), flush=True)
    print(json.dumps(save_animation(mobile_frames(args.frames), paths[2], args.fps)), flush=True)


if __name__ == "__main__":
    main()
