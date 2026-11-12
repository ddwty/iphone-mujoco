"""Portable optical inspection bench with metric fixtures and a clamped iPhone."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .camera import CameraCalibration
from .profiles import configured_phone_mjcf, write_phone_module


SCENE_PATH = Path(__file__).parent / "assets" / "scenes" / "phone_lab.xml"
VIEWER_SETTINGS = {"lookat": [0., -.19, .69], "distance": 2.10,
                   "azimuth": 135., "elevation": -23.}

LAB_METRICS = {"table_top_height_m": .755, "fixture_plate_height_m": .763,
               "checker_square_m": .035, "checker_square_count": [8, 5],
               "ruler_major_spacing_m": .100, "ruler_minor_spacing_m": .010,
               "depth_step_heights_m": [.040, .080, .120],
               "inspection_slide_range_m": [-.080, .080],
               "inspection_slide_joint": "inspection_slide",
               "inspection_slide_actuator": "inspection_slide_drive"}


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


def _lab_configuration(calibration: CameraCalibration, configuration=None):
    if configuration is not None:
        return configuration
    from .configuration import CameraConfiguration
    source = "synthetic_demo" if calibration == demo_calibration() else "caller_supplied_intrinsics"
    return CameraConfiguration("iphone-17-pro", calibration, source,
                               metadata={"geometry_selection": "default_lab_device",
                                         "depth_source": "simulation"})


def _add_phone_clamp(body: ET.Element, configuration) -> None:
    """Add a lower phone cradle, screw jaw and ball-head connection."""
    from .devices import get_device
    device = get_device(configuration.device_id)
    half_x, half_y, half_z = device.width_m / 2, device.height_m / 2, device.thickness_m / 2
    plate_y = -half_y + .035
    common = {"mass": "0", "density": "0", "contype": "0", "conaffinity": "0", "group": "2"}

    def geom(suffix, kind="box", rgba=".095 .11 .13 1", **attrs):
        return ET.SubElement(body, "geom", name="iphone_mount_" + suffix, type=kind, rgba=rgba, **common, **attrs)

    geom("backplate", pos=f"0 {plate_y} {half_z + .003}", size=f"{half_x + .006} .022 .003")
    geom("lower_shelf", pos=f"0 {-half_y - .003} 0", size=f"{half_x + .006} .003 {half_z + .006}")
    for side, sign in (("left", -1), ("right", 1)):
        geom("jaw_" + side, pos=f"{sign * (half_x + .003)} {plate_y} 0", size=".003 .024 .008")
        geom("pad_" + side, pos=f"{sign * (half_x + .0006)} {plate_y} 0",
             size=f".0006 .019 {half_z + .001}", rgba=".025 .03 .035 1")
    geom("adjustment_screw", "cylinder", fromto=f"{half_x + .003} {plate_y} 0 {half_x + .022} {plate_y} 0",
         size=".003", rgba=".45 .48 .51 1")
    geom("adjustment_knob", "cylinder", fromto=f"{half_x + .020} {plate_y} 0 {half_x + .028} {plate_y} 0",
         size=".011")
    ball = [0., plate_y, half_z + .033]
    geom("stem", "cylinder", fromto=f"0 {plate_y} {half_z + .006} 0 {plate_y} {ball[2]}",
         size=".008", rgba=".42 .45 .48 1")
    geom("ball", "sphere", pos=" ".join(map(str, ball)), size=".014", rgba=".30 .33 .36 1")


def _phone_body(calibration: CameraCalibration, configuration=None) -> ET.Element:
    configuration = _lab_configuration(calibration, configuration)
    module = ET.fromstring(configured_phone_mjcf(configuration, mocap=True))
    body = module.find("worldbody/body")
    # Keep the portable file visible in MuJoCo's default groups. The interactive
    # application can move these named cosmetics to a sensor-hidden group.
    for geom in body.iter("geom"):
        geom.set("group", "2")
    from .devices import get_device
    # The lab placement is anchored to the main camera, keeping the phone and
    # its stand in place when a different lens or digital crop is selected.
    desired_body = default_world_from_camera() @ np.linalg.inv(
        get_device(configuration.device_id).phone_from_camera("wide"))
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(desired_body[:3, :3]).ravel())
    body.set("pos", " ".join(f"{value:.17g}" for value in desired_body[:3, 3]))
    body.set("quat", " ".join(f"{value:.17g}" for value in quaternion))
    _add_phone_clamp(body, configuration)
    return body


def _camera_stand(phone: ET.Element) -> ET.Element:
    """Build a bench-edge clamp and two-axis rail ending at the phone ball head."""
    position = np.fromstring(phone.get("pos"), sep=" ")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.fromstring(phone.get("quat"), sep=" "))
    ball = np.fromstring(phone.find("geom[@name='iphone_mount_ball']").get("pos"), sep=" ")
    target = rotation.reshape(3, 3) @ ball + position
    stand = ET.Element("body", name="camera_stand")
    common = {"mass": "0", "density": "0", "contype": "0", "conaffinity": "0", "group": "2"}

    def geom(name, kind="box", material="black_anodized", **attrs):
        return ET.SubElement(stand, "geom", name=name, type=kind, material=material, **common, **attrs)

    mast_x, mast_y, bottom = -.43, -.32, .771
    rail_y, rail_z = float(target[1] - .04), float(target[2])
    geom("stand_top_clamp", pos=f"{mast_x} -.34 .763", size=".067 .070 .008")
    geom("stand_clamp_spine", pos=f"{mast_x} -.412 .727", size=".027 .012 .043")
    geom("stand_lower_clamp", pos=f"{mast_x} -.365 .694", size=".040 .047 .010")
    geom("stand_clamp_screw", "cylinder", "steel_material", fromto=f"{mast_x} -.36 .668 {mast_x} -.36 .706", size=".008")
    geom("stand_clamp_handle", "cylinder", pos=f"{mast_x} -.36 .668", size=".023 .006")
    geom("stand_column", material="extrusion_material", pos=f"{mast_x} {mast_y} {(bottom + rail_z) / 2}",
         size=f".020 .020 {(rail_z - bottom) / 2}")
    for sign in (-1, 1):
        geom(f"stand_column_slot_{sign}", material="slot_material",
             pos=f"{mast_x + sign * .0202} {mast_y} {(bottom + rail_z) / 2}",
             size=f".0004 .004 {(rail_z - bottom) / 2 - .01}")
    geom("stand_fore_aft_rail", material="extrusion_material",
         pos=f"{mast_x} {(mast_y + rail_y) / 2} {rail_z}", size=f".018 {(mast_y - rail_y) / 2} .018")
    geom("stand_fore_aft_slot", material="slot_material",
         pos=f"{mast_x + .0182} {(mast_y + rail_y) / 2} {rail_z}", size=f".0004 {(mast_y - rail_y) / 2 - .01} .003")
    geom("stand_cross_rail", material="extrusion_material", pos=f"{(mast_x + target[0]) / 2} {rail_y} {rail_z}",
         size=f"{(target[0] - mast_x) / 2} .018 .018")
    geom("stand_cross_rail_slot", material="slot_material", pos=f"{(mast_x + target[0]) / 2} {rail_y - .0182} {rail_z}",
         size=f"{(target[0] - mast_x) / 2 - .01} .0004 .003")
    geom("stand_height_carriage", pos=f"{mast_x} {mast_y} {rail_z}", size=".027 .028 .027")
    geom("stand_elbow_carriage", pos=f"{mast_x} {rail_y} {rail_z}", size=".027 .027 .027")
    for index, (x, y) in enumerate(((mast_x, mast_y), (mast_x, rail_y))):
        geom(f"stand_lock_screw_{index}", "cylinder", "steel_material", fromto=f"{x + .025} {y} {rail_z} {x + .042} {y} {rail_z}", size=".004")
        geom(f"stand_lock_knob_{index}", "cylinder", fromto=f"{x + .038} {y} {rail_z} {x + .048} {y} {rail_z}", size=".015")
    geom("stand_ball_head_socket", "cylinder", "steel_material",
         fromto=f"{target[0]} {rail_y} {rail_z} {target[0]} {target[1]} {rail_z}", size=".010")
    for index, (dx, dy) in enumerate(((-.05, -.045), (.05, -.045), (-.05, .045), (.05, .045))):
        geom(f"stand_base_bolt_{index}", "cylinder", "steel_material", pos=f"{mast_x + dx} {-.34 + dy} .773", size=".004 .002")
    return stand


def _scene_tree(calibration: CameraCalibration, configuration=None) -> ET.Element:
    configuration = _lab_configuration(calibration, configuration)
    root = ET.parse(SCENE_PATH).getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Phone lab asset lacks worldbody")
    for name in ("iphone", "camera_stand"):
        previous = world.find(f"body[@name='{name}']")
        if previous is not None:
            world.remove(previous)
    module = ET.fromstring(configured_phone_mjcf(configuration))
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    # The packaged standalone scene already contains its default phone meshes.
    # Replace their names before adding a different configured device.
    for previous in list(assets):
        if previous.get("name", "").startswith("iphone_"):
            assets.remove(previous)
    assets.extend(module.find("asset"))
    phone = _phone_body(calibration, configuration)
    world.append(_camera_stand(phone))
    world.append(phone)
    visual = root.find("visual/global")
    if visual is not None:
        visual.set("offwidth", str(max(640, calibration.width)))
        visual.set("offheight", str(max(480, calibration.height)))
    return root


def load_phone_lab(calibration: CameraCalibration | None = None, *, configuration=None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the tabletop, test targets, stand and mocap iPhone camera module."""
    if configuration and calibration and calibration != configuration.calibration:
        raise ValueError("Configuration and camera calibration must match")
    root = _scene_tree(configuration.calibration if configuration else calibration or demo_calibration(), configuration)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def export_phone_lab(directory: str | Path, calibration: CameraCalibration | None = None, *, configuration=None) -> Path:
    """Export an independent MJCF scene with a relative phone include.

    The exported scene can be opened by MuJoCo without this Python package.
    The movable phone include belongs directly inside worldbody. For a camera
    rigidly attached to a robot, generate phone_mjcf(K, mocap=False) instead.
    Existing nonempty directories are rejected to preserve user experiments.
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Export directory must be new or empty: {directory}")
    if configuration and calibration and calibration != configuration.calibration:
        raise ValueError("Configuration and camera calibration must match")
    calibration = configuration.calibration if configuration else calibration or demo_calibration()
    configuration = _lab_configuration(calibration, configuration)
    root = _scene_tree(calibration, configuration)
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
    write_phone_module(directory / "iphone_fixed.xml", calibration, configuration=configuration)
    info = {"format": "asmagic-mujoco-phone-lab", "camera_name": "iphone_camera",
            "phone_body": "iphone", "phone_control": "mocap; direct child of worldbody",
            "camera_pose_convention": "world_z_up_from_cv_optical",
            **LAB_METRICS,
            "phone_geometry": "Apple dimensional drawing; interpolated corner outline",
            "fixed_robot_module": "iphone_fixed.xml",
            "phone_mass_inertia": "visual mode; payload disabled", "lens_mount": "mechanical lens-cover center",
            "sensor_model": "ideal pinhole RGB-D; physical sensor effects not modeled",
            "calibration_source": "synthetic_demo" if calibration == demo_calibration() else "caller_supplied",
            "default_world_from_camera": default_world_from_camera().tolist()}
    if configuration:
        configuration.save(directory / "device.json")
        info.update(camera_configuration=configuration.to_dict(),
                    phone_geometry="Apple dimensional drawing; interpolated corner outline",
                    phone_mass_inertia="visual mode; payload disabled", lens_mount="mechanical lens-cover center",
                    calibration_source=configuration.calibration_source)
    (directory / "scene_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (directory / "README.md").write_text(
        "# Virtual iPhone camera laboratory\n\n"
        "Open scene.xml as a complete MuJoCo scene. No mesh downloads or absolute paths are needed.\n"
        "Example: python -m mujoco.viewer --mjcf scene.xml\n\n"
        "iphone.xml is the movable world-level phone used by this scene.\n" +
        ("For a fixed camera on your robot, include iphone_fixed.assets.xml inside the scene asset element, "
         if configuration else "") +
        "include iphone_fixed.xml inside the robot body; "
        "load iphone_fixed.calibration.json for RGB/depth dimensions.\n"
        "The full interactive RGB/depth panels and P-to-save workflow are available through "
        "the iphone-mujoco package: mjpython -m iphone_mujoco phone-lab.\n\n"
        "Calibration source: " + info["calibration_source"] + ". "
        "The camera renders registered pinhole RGB-D with optical-axis Z depth.\n",
        encoding="utf-8")
    return directory / "scene.xml"
