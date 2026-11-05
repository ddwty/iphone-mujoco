"""Portable camera laboratory with a visible, explicitly approximate iPhone.

The scene is entirely MJCF primitives and built-in textures. It needs neither
robot downloads nor a phone connection. Runtime intrinsics are supplied by the
caller; the packaged scene uses clearly labeled synthetic intrinsics.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .camera import CameraCalibration
from .geometry import CV_TO_MUJOCO_CAMERA
from .profiles import phone_mjcf, write_phone_module


SCENE_PATH = Path(__file__).parent / "assets" / "scenes" / "phone_lab.xml"
VIEWER_SETTINGS = {"lookat": [0., -.14, .76], "distance": 1.95,
                   "azimuth": 135., "elevation": -23.}


def demo_calibration() -> CameraCalibration:
    """Synthetic example; never a factory or measured iPhone calibration."""
    return CameraCalibration(640, 480, 544., 544., 319.5, 239.5, 256, 192)


def default_world_from_camera() -> np.ndarray:
    """CV optical pose looking from the phone stand toward the test objects."""
    eye = np.array([0., -.80, 1.05])
    forward = np.array([0., .04, .84]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    pose[:3, 3] = eye
    return pose


def _phone_body(calibration: CameraCalibration) -> ET.Element:
    body = ET.fromstring(phone_mjcf(calibration, mocap=True))
    # Keep the portable file visible in MuJoCo's default groups. The interactive
    # application can move these named cosmetics to a sensor-hidden group.
    for geom in body.iter("geom"):
        geom.set("group", "2")
    camera = body.find("camera[@name='iphone_camera']")
    body_from_camera = np.eye(4)
    body_from_camera[:3, 3] = np.fromstring(camera.get("pos", "0 0 0"), sep=" ")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.fromstring(camera.get("quat", "1 0 0 0"), sep=" "))
    body_from_camera[:3, :3] = rotation.reshape(3, 3)
    desired_body = default_world_from_camera() @ CV_TO_MUJOCO_CAMERA @ np.linalg.inv(body_from_camera)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(desired_body[:3, :3]).ravel())
    body.set("pos", " ".join(f"{value:.17g}" for value in desired_body[:3, 3]))
    body.set("quat", " ".join(f"{value:.17g}" for value in quaternion))
    return body


def _scene_tree(calibration: CameraCalibration) -> ET.Element:
    root = ET.parse(SCENE_PATH).getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Phone lab asset lacks worldbody")
    previous = world.find("body[@name='iphone']")
    if previous is not None:
        world.remove(previous)
    world.append(_phone_body(calibration))
    visual = root.find("visual/global")
    if visual is not None:
        visual.set("offwidth", str(max(640, calibration.width)))
        visual.set("offheight", str(max(480, calibration.height)))
    return root


def load_phone_lab(calibration: CameraCalibration | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the tabletop, test targets, stand and mocap iPhone camera module."""
    root = _scene_tree(calibration or demo_calibration())
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def export_phone_lab(directory: str | Path, calibration: CameraCalibration | None = None) -> Path:
    """Export an independent MJCF scene with a relative phone include.

    The exported scene can be opened by MuJoCo without this Python package.
    The movable phone include belongs directly inside worldbody. For a camera
    rigidly attached to a robot, generate phone_mjcf(K, mocap=False) instead.
    Existing nonempty directories are rejected to preserve user experiments.
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Export directory must be new or empty: {directory}")
    calibration = calibration or demo_calibration()
    root = _scene_tree(calibration)
    world = root.find("worldbody")
    phone = world.find("body[@name='iphone']")
    world.remove(phone)
    ET.SubElement(world, "include", file="iphone.xml")
    include = ET.Element("mujocoinclude")
    include.append(phone)
    directory.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.indent(include)
    ET.ElementTree(root).write(directory / "scene.xml", encoding="unicode")
    ET.ElementTree(include).write(directory / "iphone.xml", encoding="unicode")
    calibration.save(directory / "calibration.json")
    write_phone_module(directory / "iphone_fixed.xml", calibration)
    info = {"format": "asmagic-mujoco-phone-lab", "camera_name": "iphone_camera",
            "phone_body": "iphone", "phone_control": "mocap; direct child of worldbody",
            "camera_pose_convention": "world_z_up_from_cv_optical",
            "table_top_height_m": .755, "checker_square_m": .035,
            "phone_geometry": "approximate visual only; collision disabled",
            "fixed_robot_module": "iphone_fixed.xml",
            "phone_mass_inertia": "not modeled", "lens_mount": "approximate",
            "sensor_model": "ideal pinhole RGB-D; physical sensor effects not modeled",
            "calibration_source": "synthetic_demo" if calibration == demo_calibration() else "caller_supplied",
            "default_world_from_camera": default_world_from_camera().tolist()}
    (directory / "scene_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (directory / "README.md").write_text(
        "# Virtual iPhone camera laboratory\n\n"
        "Open scene.xml as a complete MuJoCo scene. No mesh downloads or absolute paths are needed.\n"
        "Example: python -m mujoco.viewer --mjcf scene.xml\n\n"
        "iphone.xml is the movable world-level phone used by this scene.\n"
        "For a fixed camera on your robot, include iphone_fixed.xml inside the robot body; "
        "load iphone_fixed.calibration.json for RGB/depth dimensions.\n"
        "The full interactive RGB/depth panels and P-to-save workflow are available through "
        "the iphone-mujoco package: mjpython -m iphone_mujoco phone-lab.\n\n"
        "Calibration source: " + info["calibration_source"] + ". "
        "This is ideal pinhole RGB-D. Phone geometry/lens mount are approximate; "
        "mass, inertia, LiDAR noise and physical sensor effects are not measured or simulated.\n",
        encoding="utf-8")
    return directory / "scene.xml"
