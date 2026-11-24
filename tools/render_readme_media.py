"""Render reproducible inspection and robot-camera demonstrations from MuJoCo.

The installed package owns scenes, physics and capture. This tool only selects
viewpoints and lays out the rendered RGB-D observations for the README.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from iphone_mujoco import MujocoRGBDCamera
from iphone_mujoco.configuration import configure_camera
from iphone_mujoco.experiments import advance_scene
from iphone_mujoco.mobile import DifferentialDriveController, load_mobile_scene
from iphone_mujoco.phone_lab import _frustum, depth_preview
from iphone_mujoco.phone_scene import VIEWER_SETTINGS, load_phone_lab
from iphone_mujoco.robots import load_scene
from iphone_mujoco.spatial import SpatialOverlay
from iphone_mujoco.wrist import default_wrist_mount, wrist_camera_mjcf, _place_demo_targets


def font(size):
    for name in ("Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def panel(overview, frame, title, subtitle, *, state=None):
    """Presentation resizing never changes recorded sensor rasters or K."""
    canvas = Image.new("RGB", (1120, 760), "#eef1f3")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 15), title, font=font(24), fill="#1c2c36")
    draw.text((24, 49), subtitle, font=font(13), fill="#51646f")
    draw.text((976, 20), f"{frame.timestamp:05.2f} s", font=font(17), fill="#315968")
    canvas.paste(Image.fromarray(overview), (24, 78))
    canvas.paste(Image.fromarray(frame.rgb).resize((320, 240), Image.Resampling.LANCZOS), (752, 78))
    depth = depth_preview(frame.depth_m, near=.2, far=4.5 if state else 2.)
    canvas.paste(Image.fromarray(depth).resize((320, 240), Image.Resampling.NEAREST), (752, 366))
    draw.text((752, 56), "RGB", font=font(13), fill="#223741")
    draw.text((752, 344), "OPTICAL Z / METERS", font=font(13), fill="#223741")
    for left, top, width, height in ((24, 78, 704, 528), (752, 78, 320, 240), (752, 366, 320, 240)):
        draw.rectangle((left-1, top-1, left+width, top+height), outline="#c6cdd1")
    position = frame.world_from_camera[:3, 3]
    pose_text = "OPTICAL XYZ   " + "   ".join(f"{value:+.3f}" for value in position) + " m"
    draw.text((24, 625), pose_text, font=font(14), fill="#354b57")
    spatial = frame.metadata["spatial_frames"]
    device_position = np.asarray(spatial["world_from_01"])[:3, 3]
    draw.text((24, 651), "DEVICE 01      " + "   ".join(f"{value:+.3f}" for value in device_position) + " m",
              font=font(14), fill="#354b57")
    if state:
        twist = state["command_twist"]
        wheels = state["wheel_velocity_rad_s"]
        draw.text((24, 678), f"BASE COMMAND   v {twist['linear_mps']:+.2f} m/s   w {twist['angular_rps']:+.2f} rad/s"
                  f"     WHEELS   L {wheels[0]:+.2f}   R {wheels[1]:+.2f} rad/s",
                  font=font(13), fill="#315968")
    else:
        draw.text((24, 678), "iPhone 17 Pro  /  simulated RGB-D  /  same-state camera pose",
                  font=font(13), fill="#51646f")
    k = frame.calibration
    draw.text((752, 625), f"RGB {k.width} x {k.height}", font=font(13), fill="#51646f")
    draw.text((752, 650), f"DEPTH {frame.depth_m.shape[1]} x {frame.depth_m.shape[0]}",
              font=font(13), fill="#51646f")
    draw.text((24, 720), "asTriad: W world / D01 device / C optical     |     asDevice: screen-plane 01 / portrait",
              font=font(14), fill="#315968")
    for x, axis, color in ((890, "X", "#bd2222"), (950, "Y", "#16812d"), (1010, "Z", "#2356c4")):
        draw.text((x, 718), axis, font=font(17), fill=color)
    canvas.info["simulation_time"] = float(frame.timestamp)
    canvas.info["spatial_frames"] = spatial
    return canvas


def render_spatial_view(renderer, model, data, view, frame, spatial, *, detail=False):
    renderer.update_scene(data, camera=view)
    _frustum(renderer.scene, frame)
    mujoco.mjv_addGeoms(model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
                       mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)
    spatial.annotate(frame)
    spatial.append(renderer.scene, frame)
    overview = Image.fromarray(renderer.render().copy())
    draw = ImageDraw.Draw(overview)
    # Public asTriad anchors; ordinary perspective projection into the overview.
    camera = mujoco.mjv_averageCamera(renderer.scene.camera[0], renderer.scene.camera[1])
    forward = np.asarray(camera.forward)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, camera.up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    span = camera.frustum_top - camera.frustum_bottom
    half_width = camera.frustum_width or .5 * overview.width / overview.height * span
    for triad in spatial.triads:
        for label in triad.label_positions(renderer.scene):
            if not detail and triad.name != "W" and label.text in ("X", "Y", "Z"):
                continue
            delta = np.asarray(label.position) - camera.pos
            depth = np.dot(delta, forward)
            if depth <= 0:
                continue
            scale = 1 if camera.orthographic else camera.frustum_near / depth
            x = (np.dot(delta, right) * scale - camera.frustum_center + half_width) / (2 * half_width)
            y = (np.dot(delta, up) * scale - camera.frustum_bottom) / span
            if 0 <= x <= 1 and 0 <= y <= 1:
                color = tuple(round(255 * channel) for channel in label.color)
                draw.text((x * overview.width, (1-y) * overview.height), label.text,
                          font=font(13), fill=color, stroke_width=1, stroke_fill="#18212b", anchor="mm")
    return np.asarray(overview)


def render_overview(renderer, model, data, view, frame, spatial):
    overview = Image.fromarray(render_spatial_view(renderer, model, data, view, frame, spatial))
    device = spatial.placement.world_from_device(frame.world_from_camera)
    close_view = overview_camera({"lookat": device[:3, 3], "distance": .42,
                                  "azimuth": view.azimuth, "elevation": -20.})
    close = Image.fromarray(render_spatial_view(renderer, model, data, close_view, frame, spatial, detail=True))
    overview.paste(close.resize((232, 174), Image.Resampling.LANCZOS), (12, 32))
    draw = ImageDraw.Draw(overview)
    draw.rectangle((11, 11, 244, 32), fill="#18212b")
    draw.text((18, 14), "asDevice 01 / optical C", font=font(13), fill="#eef1f3")
    draw.rectangle((11, 11, 244, 206), outline="#9aaab3")
    return np.asarray(overview)


def overview_camera(settings):
    view = mujoco.MjvCamera()
    for name, value in settings.items():
        if name == "lookat":
            view.lookat[:] = value
        else:
            setattr(view, name, value)
    return view


def prepare_overview(model):
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 704)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 528)


def lab_frames(count):
    config = configure_camera(region="cn")
    model, data = load_phone_lab(configuration=config)
    spatial = SpatialOverlay(config)
    prepare_overview(model)
    view = overview_camera(VIEWER_SETTINGS)
    frames = []
    with MujocoRGBDCamera(model, config.calibration, max_depth_m=6.55345, hidden_geom_groups=(3,)) as sensor, \
            mujoco.Renderer(model, height=528, width=704) as overview:
        for time_s in np.linspace(.002, 8, count):
            advance_scene("tabletop", model, data, time_s)
            frame = sensor.render(data)
            frames.append(panel(render_overview(overview, model, data, view, frame, spatial), frame,
                                "Optical inspection bench", "Clamped iPhone / servo-driven linear stage / metric calibration fixtures"))
    return frames


def wrist_frames(robot, count):
    config = configure_camera(region="cn")
    mount = default_wrist_mount(robot)
    model, data, spec = load_scene(robot, phone_xml=wrist_camera_mjcf(config.calibration, mount, show_phone=True,
                                                                    configuration=config), phone_parent_body=mount.parent_body)
    _place_demo_targets(model, data, mount.camera_name)
    spatial = SpatialOverlay(config)
    prepare_overview(model)
    home = np.array([data.qpos[model.joint(name).qposadr[0]] for name in spec.joint_names])
    actuators = [model.actuator(name).id for name in spec.actuator_names]
    view = overview_camera({"lookat": [.12, .06, .16], "distance": 2., "azimuth": 132., "elevation": -23.})
    frames = []
    with MujocoRGBDCamera(model, config.calibration, mount.camera_name, max_depth_m=6.55345,
                         hidden_geom_groups=(3,)) as sensor, mujoco.Renderer(model, height=528, width=704) as overview:
        for time_s in np.linspace(.002, 4, count):
            while data.time < time_s - 1e-9:
                phase = np.arange(len(home)) * .4
                data.ctrl[actuators] = home + .065 * (np.sin(phase + data.time * .7) - np.sin(phase))
                mujoco.mj_step(model, data)
            frame = sensor.render(data)
            close = len(frames) >= count * .6
            if close:
                view.lookat[:] = frame.world_from_camera[:3, 3]
                view.distance = .46
                view.azimuth = 220. if robot == "ur5e" else 135.
                view.elevation = -25.
            name = "Franka Emika Panda" if robot == "panda" else "Universal Robots UR5e"
            frames.append(panel(render_overview(overview, model, data, view, frame, spatial), frame, name,
                                "Wrist-mounted iPhone / rigid support / synchronized joint and RGB-D observations"))
    return frames


def mobile_frames(count):
    config = configure_camera(region="cn")
    model, data, spec = load_mobile_scene(configuration=config)
    spatial = SpatialOverlay(config)
    prepare_overview(model)
    drive = DifferentialDriveController(model, data, spec)
    view = overview_camera({"lookat": [0., 0., .8], "distance": 2.9, "azimuth": 215., "elevation": -18.})
    frames, epoch = [], float(data.time)
    with MujocoRGBDCamera(model, config.calibration, max_depth_m=6.55345, hidden_geom_groups=(3,)) as sensor, \
            mujoco.Renderer(model, height=528, width=704) as overview:
        for elapsed in np.linspace(.002, 8.5, count):
            advance_scene("mobile", model, data, epoch + elapsed, controller=drive, sequence_start=epoch)
            state = drive.state()
            base = np.asarray(state["world_from_base"])
            view.lookat[:] = base[:3, 3] + [.10, 0., .8]
            frame = sensor.render(data)
            frames.append(panel(render_overview(overview, model, data, view, frame, spatial), frame,
                                "PAL Robotics TIAGo", "Differential wheel drive / head-mounted iPhone / encoder and RGB-D capture", state=state))
    return frames


def lens_frames():
    frames = []
    for zoom in (.5, 1., 2., 4., 8.):
        config = configure_camera(zoom=zoom, region="cn")
        model, data = load_phone_lab(configuration=config)
        spatial = SpatialOverlay(config)
        prepare_overview(model)
        view = overview_camera(VIEWER_SETTINGS)
        with MujocoRGBDCamera(model, config.calibration, max_depth_m=6.55345, hidden_geom_groups=(3,)) as sensor, \
                mujoco.Renderer(model, height=528, width=704) as overview:
            frame = sensor.render(data)
            focal = config.metadata["camera_mode"]["equivalent_focal_mm"]
            shot = panel(render_overview(overview, model, data, view, frame, spatial), frame,
                         f"iPhone 17 Pro / {zoom:g}x / {config.lens.replace('_', ' ')}",
                         f"{focal:g} mm equivalent / fixed phone body / selected physical lens and crop")
            frames.extend([shot] * 8)
    return frames


def save_animation(frames, path, fps, *, poster_fraction=.25):
    samples = np.linspace(0, len(frames)-1, min(8, len(frames)), dtype=int)
    atlas = Image.new("RGB", (560, 380 * len(samples)))
    for row, index in enumerate(samples):
        atlas.paste(frames[index].resize((560, 380)), (0, row * 380))
    palette = atlas.quantize(colors=160)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    times = [frame.info.get("simulation_time", 0.) for frame in frames]
    intervals = np.diff(times)
    positive = intervals[intervals > 1e-6]
    default_delay = float(np.median(positive)) if len(positive) else 1/fps
    durations = [max(10, round(1000 * (value if value > 1e-6 else default_delay))) for value in intervals]
    durations.append(max(10, round(1000*default_delay)))
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=durations,
                    loop=0, optimize=True, disposal=2)
    frames[min(len(frames)-1, int(len(frames)*poster_fraction))].save(path.with_suffix(".png"))
    provenance = {
        "generator": "tools/render_readme_media.py", "pose_source": "simulation",
        "packages": {name: version(name) for name in ("iphone-mujoco", "asTriad", "asDevice", "mujoco")},
        "overview_frames": ["W", "D01", "C"], "sensor_overlays": False,
        "samples": [{"simulation_time": frame.info["simulation_time"],
                     "spatial_frames": frame.info["spatial_frames"]} for frame in frames],
        "sha256": {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                   for item in (path, path.with_suffix(".png"))},
    }
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"file": path.name, "frames": len(frames), "width": 1120, "height": 760,
            "duration_seconds": sum(durations)/1000, "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("recordings/readme-media"))
    parser.add_argument("--robots", nargs="+", choices=("panda", "ur5e"), default=["panda", "ur5e"])
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--robot-frames", type=int, default=28)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    if not 4 <= args.frames <= 120 or not 4 <= args.robot_frames <= 60 or not 1 <= args.fps <= 30:
        parser.error("Use 4-120 frames, 4-60 robot frames and 1-30 playback FPS")
    paths = [args.output/name for name in ("phone-lab.gif", "robot-wrist.gif", "mobile-camera.gif", "camera-lenses.gif")]
    if any(path.with_suffix(suffix).exists() for path in paths for suffix in (".gif", ".png", ".json")):
        parser.error("Choose a new output directory; media already exists")
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(save_animation(lab_frames(args.frames), paths[0], args.fps)), flush=True)
    wrist = []
    for robot in args.robots:
        wrist.extend(wrist_frames(robot, args.robot_frames))
    print(json.dumps(save_animation(wrist, paths[1], args.fps, poster_fraction=.85)), flush=True)
    print(json.dumps(save_animation(mobile_frames(args.frames), paths[2], args.fps)), flush=True)
    print(json.dumps(save_animation(lens_frames(), paths[3], args.fps)), flush=True)


if __name__ == "__main__":
    main()
