"""One native MuJoCo window: visible phone, RGB-D views and dataset snapshots."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import queue
import time
import uuid

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import CameraCalibration, MujocoRGBDCamera
from .geometry import CV_TO_MUJOCO_CAMERA
from .phone_scene import VIEWER_SETTINGS, default_world_from_camera, demo_calibration, load_phone_lab
from .recording import RecordingWriter
from .virtual_device import quantize_depth


def depth_preview(depth_m, near=.15, far=2.) -> np.ndarray:
    """Fixed, labeled metric range; display colors never replace raw depth."""
    if not np.isfinite([near, far]).all() or near < 0 or far <= near:
        raise ValueError("Depth preview requires finite 0 <= near < far")
    depth = np.asarray(depth_m)
    valid = np.isfinite(depth) & (depth > 0)
    unit = np.clip((np.where(valid, depth, near) - near) / (far - near), 0, 1)
    # A compact viridis-like display palette, independent of matplotlib/OpenCV.
    stops = np.array([[68, 1, 84], [64, 67, 135], [41, 120, 142],
                      [34, 167, 132], [122, 209, 81], [253, 231, 37]], dtype=float)
    rgb = np.stack([np.interp(unit, np.linspace(0, 1, len(stops)), stops[:, channel])
                    for channel in range(3)], axis=-1).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _panel(image, width, title, footer):
    image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    height = max(1, round(width * image.height / image.width))
    scale = max(1., width / 360)
    header, bottom = round(26 * scale), round(24 * scale)
    panel = Image.new("RGB", (width, height + header + bottom), (18, 24, 32))
    panel.paste(image.resize((width, height), Image.Resampling.NEAREST), (0, header))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default(size=round(14 * scale))
    except TypeError:
        font = ImageFont.load_default()
    draw.text((round(9*scale), round(5*scale)), title, fill=(225, 238, 245), font=font)
    draw.text((round(9*scale), header + height + round(4*scale)), footer, fill=(175, 195, 208), font=font)
    return np.asarray(panel)


def update_rgbd_preview(viewer, frame, enabled=True):
    """Display the captured RGB and metric-depth preview in a native viewer."""
    viewport = viewer.viewport
    if viewport is None or viewport.width <= 0 or viewport.height <= 0:
        return
    calibration = frame.calibration
    width = max(120, min(600, int(viewport.width * .30), int((viewport.height-30)/1.85)))
    rgb = _panel(frame.rgb, width, "VIRTUAL iPHONE / RGB", f"{calibration.width} x {calibration.height}  |  simulated")
    depth = _panel(depth_preview(frame.depth_m), width, "DEPTH / meters", "0.15m near / 2.0m far | black invalid")
    left = viewport.left + viewport.width - width - 12
    bottom = viewport.bottom + 12
    images = [(mujoco.MjrRect(left, bottom, depth.shape[1], depth.shape[0]), depth),
              (mujoco.MjrRect(left, bottom+depth.shape[0]+10, rgb.shape[1], rgb.shape[0]), rgb)]
    viewer.set_images(images if enabled else [])


def _optical_pose(model, data):
    camera = model.camera("iphone_camera").id
    pose = np.eye(4)
    pose[:3, :3] = data.cam_xmat[camera].reshape(3, 3)
    pose[:3, 3] = data.cam_xpos[camera]
    return pose @ CV_TO_MUJOCO_CAMERA


def _move_phone(pose, key):
    result = pose.copy()
    # GLFW constants are stable integer key codes; callbacks only enqueue them.
    translation = {263: (-.025, 0, 0), 262: (.025, 0, 0),
                   265: (0, .025, 0), 264: (0, -.025, 0),
                   ord("Q"): (0, 0, -.025), ord("E"): (0, 0, .025)}
    if key in translation:
        result[:3, 3] += translation[key]
        result[:3, 3] = np.clip(result[:3, 3], [-1.2, -1.6, .35], [1.2, 1.3, 2.1])
    elif key in (ord("J"), ord("L"), ord("I"), ord("K")):
        angle = np.deg2rad(5.) * (1 if key in (ord("J"), ord("I")) else -1)
        c, s = np.cos(angle), np.sin(angle)
        if key in (ord("J"), ord("L")):
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            result[:3, :3] = rotation @ result[:3, :3]
        else:
            rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            result[:3, :3] = result[:3, :3] @ rotation
    return result


def _frustum(scene, frame):
    """Overview-only lens rays; they are absent from sensor RGB and depth."""
    scene.ngeom = 0
    k = frame.calibration
    length = .24
    pixels = np.array([[0, 0], [k.width-1, 0], [k.width-1, k.height-1], [0, k.height-1]])
    cv = np.column_stack(((pixels[:, 0]-k.cx)/k.fx*length,
                          (pixels[:, 1]-k.cy)/k.fy*length, np.full(4, length)))
    pose = frame.world_from_camera
    corners = cv @ pose[:3, :3].T + pose[:3, 3]
    lines = [(pose[:3, 3], point) for point in corners]
    lines += [(corners[i], corners[(i+1) % 4]) for i in range(4)]
    for start, end in lines:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                           np.eye(3).ravel(), np.array([.15, .85, .95, .8]))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2., start, end)
        scene.ngeom += 1


def _snapshot(frame, directory, overview=None):
    """A portable, replayable RGB-D capture without overwriting earlier shots."""
    encoded = quantize_depth(frame.depth_m)
    metric = encoded.astype(np.float32) * .0001
    metric[encoded == 0] = np.nan
    frame = replace(frame, depth_m=metric, metadata=dict(frame.metadata,
                    depth_meters_per_unit=.0001, depth_model="ideal_optical_axis_quantized"))
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=False)
    with RecordingWriter(directory / "episode") as writer:
        writer.write(frame)
    frame.calibration.save(directory / "calibration.json")
    Image.fromarray(frame.rgb).save(directory / "rgb.png")
    np.save(directory / "depth_m.npy", frame.depth_m, allow_pickle=False)
    Image.fromarray(encoded).save(directory / "depth_uint16.png")
    Image.fromarray(depth_preview(frame.depth_m)).save(directory / "depth_preview.png")
    frame.point_cloud(stride=2).save_ply(directory / "points.ply")
    frame.point_cloud(world=True, stride=2).save_ply(directory / "points_world.ply")
    if overview is not None:
        Image.fromarray(overview).save(directory / "scene.png")
    (directory / "capture.json").write_text(json.dumps({
        "source": "MuJoCo virtual iPhone", "frame_id": frame.frame_id,
        "simulation_time": frame.timestamp, "depth_meters_per_unit": .0001,
        "world_from_camera": frame.world_from_camera.tolist(),
        "depth_preview_range_m": [.15, 2.], "metadata": frame.metadata,
    }, indent=2) + "\n")


def run_phone_lab(args, *, initial_keys=()):
    """Interactive offline lab; no physical phone, robot download or network needed."""
    if not np.isfinite([args.duration, args.fps]).all() or args.duration < 0 or args.fps <= 0:
        raise ValueError("Duration must be finite and nonnegative; preview FPS must be finite and positive")
    calibration = CameraCalibration.load(args.profile) if args.profile else demo_calibration()
    model, data = load_phone_lab(calibration)
    # The raw portable XML keeps group 2 for standard-viewer visibility. This
    # application explicitly shows group 5 only in its overview, hiding cosmetics
    # from the sensor without hiding any other model's visual meshes.
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name.startswith("iphone_"):
            model.geom_group[index] = 5
    commands = queue.SimpleQueue()
    for key in initial_keys:
        commands.put(key)
    rendered = captures = 0
    automatic = bool(args.scan)
    show_frustum = show_images = True
    output = Path(args.output) if args.output else Path.cwd() / "recordings" / ("phone-lab-" + time.strftime("%Y%m%d-%H%M%S"))
    started = time.monotonic()
    last_capture_path = "Press P to save RGB-D"
    frame = None
    with ExitStack() as stack:
        viewer = None
        if not args.headless:
            from mujoco import viewer as mj_viewer
            viewer = stack.enter_context(mj_viewer.launch_passive(
                model, data, key_callback=commands.put, show_left_ui=False, show_right_ui=False))
            with viewer.lock():
                viewer.opt.geomgroup[5] = 1
                for name, value in VIEWER_SETTINGS.items():
                    if name == "lookat":
                        viewer.cam.lookat[:] = value
                    else:
                        setattr(viewer.cam, name, value)
        sensor = stack.enter_context(MujocoRGBDCamera(model, calibration,
                                      max_depth_m=6.55345, hidden_geom_groups=(5,)))
        # A separate context makes headless overview exports contain the visible
        # iPhone. It never changes the main RGB-D camera's intrinsics or pose.
        overview = stack.enter_context(mujoco.Renderer(model, height=480, width=640)) if args.headless else None
        overview_option = mujoco.MjvOption()
        overview_option.geomgroup[5] = 1
        session = "phone-lab-" + uuid.uuid4().hex
        started = last_physics_wall = time.monotonic()
        print("Virtual iPhone lab ready. Arrows move; Q/E height; J/L yaw; I/K tilt; 1 overview; 2 lens; P capture; R reset; B scan; Esc close.", flush=True)
        try:
            while (args.duration <= 0 or time.monotonic() - started < args.duration) and (viewer is None or viewer.is_running()):
                iteration_started = time.monotonic()
                capture = bool(args.headless and rendered == 0)
                lock = viewer.lock() if viewer else ExitStack()
                with lock:
                    while not commands.empty():
                        key = commands.get()
                        if key == ord("P"):
                            capture = True
                        elif key == ord("R"):
                            automatic = False
                            sensor.set_world_pose(data, default_world_from_camera(), move_body=True)
                        elif key == ord("B"):
                            automatic = not automatic
                        elif key == ord("F"):
                            show_frustum = not show_frustum
                        elif key == ord("V"):
                            show_images = not show_images
                        elif viewer and key in (ord("1"), ord("2")):
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE if key == ord("1") else mujoco.mjtCamera.mjCAMERA_FIXED
                            if key == ord("2"):
                                viewer.cam.fixedcamid = model.camera("iphone_camera").id
                        else:
                            pose = _move_phone(_optical_pose(model, data), key)
                            if not np.array_equal(pose, _optical_pose(model, data)):
                                automatic = False
                                sensor.set_world_pose(data, pose, move_body=True)
                    if automatic:
                        pose = default_world_from_camera()
                        elapsed = time.monotonic() - started
                        pose[0, 3] += .12 * np.sin(elapsed * .6)
                        sensor.set_world_pose(data, pose, move_body=True)
                    steps = max(1, min(125, round((iteration_started-last_physics_wall) / model.opt.timestep)))
                    mujoco.mj_step(model, data, nstep=steps)
                    last_physics_wall = iteration_started
                    frame = sensor.render(data)
                    frame.frame_id = rendered
                    frame.metadata.update({"session_id": session, "simulation_time": float(data.time),
                                           "calibration_source": str(args.profile) if args.profile else "synthetic_demo",
                                           "scene": "phone_lab", "phone_geometry": "approximate_visual_only"})
                    if viewer and viewer.user_scn is not None:
                        if show_frustum:
                            _frustum(viewer.user_scn, frame)
                        else:
                            viewer.user_scn.ngeom = 0
                rendered += 1
                if capture:
                    scene_image = None
                    if overview:
                        overview.update_scene(data, camera="lab_overview", scene_option=overview_option)
                        scene_image = overview.render().copy()
                    index = captures
                    path = output / f"capture-{index:04d}"
                    while path.exists():
                        index += 1
                        path = output / f"capture-{index:04d}"
                    _snapshot(frame, path, scene_image)
                    captures += 1
                    last_capture_path = f"Saved {path.name} (see terminal)"
                    print(f"Saved: {path}", flush=True)
                if viewer:
                    update_rgbd_preview(viewer, frame, enabled=show_images)
                    valid = np.isfinite(frame.depth_m)
                    depth_text = f"{np.nanmin(frame.depth_m):.2f}-{np.nanmax(frame.depth_m):.2f} m" if valid.any() else "no return"
                    viewer.set_texts([(mujoco.mjtFont.mjFONT_SHADOW, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                       "iPhone MuJoCo | Virtual iPhone\nArrows  Move phone\nQ / E    Down / Up\nJ / L    Yaw\nI / K    Tilt\n1 / 2    Overview / Lens\nP         Save RGB-D\nR / B    Reset / Scan\nV / F    Views / Frustum",
                                       f"Frame {rendered}\nRGB {calibration.width}x{calibration.height}\nDepth {frame.depth_m.shape[1]}x{frame.depth_m.shape[0]}\n{depth_text}\n\n{last_capture_path}")])
                    viewer.sync()
                if args.headless:
                    break
                time.sleep(max(0., 1 / args.fps - (time.monotonic()-iteration_started)))
        finally:
            print(json.dumps({"rendered_frames": rendered, "captures": captures,
                              "profile": str(args.profile) if args.profile else "synthetic_demo",
                              "output": str(output.resolve())}, indent=2), flush=True)
