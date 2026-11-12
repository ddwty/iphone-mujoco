"""Wheel-driven PAL Robotics TIAGo with a rigid iPhone RGB-D camera."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .camera import CameraCalibration
from .configuration import CameraConfiguration, configure_camera
from .geometry import CV_TO_MUJOCO_CAMERA
from .profiles import configured_phone_mjcf
from .robots import fetch_menagerie


_COMMIT = "370c43965323c0944b433b202aac5058b5f907a9"
_SOURCE = f"https://github.com/google-deepmind/mujoco_menagerie/tree/{_COMMIT}/pal_tiago"
_DRIVE_SOURCE = ("https://github.com/pal-robotics/pmb2_robot/blob/"
                 "e6584136f15356b8f4944c25d70fb8689e3355af/"
                 "pmb2_controller_configuration/config/mobile_base_controller_public_sim.yaml")
SCENE_PATH = Path(__file__).parent / "assets" / "scenes" / "mobile_lab.xml"


@dataclass(frozen=True)
class MobileRobotSpec:
    name: str = "tiago"
    base_body: str = "base_link"
    wheel_joint_names: tuple[str, str] = ("wheel_left_joint", "wheel_right_joint")
    wheel_actuator_names: tuple[str, str] = ("wheel_left_joint_vel", "wheel_right_joint_vel")
    radius_m: float = .0985
    track_m: float = .4044
    camera_name: str = "iphone_camera"
    camera_parent_body: str = "head_2_link"
    source: str = f"menagerie@{_COMMIT}"
    metadata: dict = field(default_factory=dict)


def _numbers(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in np.asarray(values).ravel())


def _merge(root: ET.Element, addition: ET.Element) -> None:
    for child in addition:
        if child.tag in {"include", "keyframe"}:
            continue
        existing = root.find(child.tag)
        if existing is None:
            root.append(deepcopy(child))
        elif child.tag in {"option", "statistic", "compiler"}:
            existing.attrib.update(child.attrib)
            existing.extend(deepcopy(list(child)))
        else:
            existing.extend(deepcopy(list(child)))


def _attach_phone(root: ET.Element, configuration: CameraConfiguration) -> None:
    module = ET.fromstring(configured_phone_mjcf(configuration, body_name="iphone"))
    assets = root.find("asset")
    assets.extend(module.find("asset"))
    head = root.find(".//body[@name='head_2_link']")
    if head is None:
        raise ValueError("Upstream TIAGo is missing its head sensor mount")
    phone = module.find("worldbody/body")
    # Head-local +X is forward, +Y up, +Z robot-right at the home pose.
    # The fixed phone body faces out of the original head camera's front panel.
    rotation = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    position = np.array([.155, .075, 0.])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    phone.set("pos", _numbers(position))
    phone.set("quat", _numbers(quaternion))
    # A backplate touches the screen-side surface, with two edge jaws. The
    # support is illustrative geometry and does not add an unmeasured payload.
    from .devices import get_device
    device = get_device(configuration.device_id)
    half_width, half_depth = device.width_m / 2, device.thickness_m / 2
    common = {"mass": "0", "density": "0", "contype": "0", "conaffinity": "0",
              "group": "2", "rgba": ".27 .30 .33 1"}
    plate = np.array([0., -.03, half_depth + .004])
    ET.SubElement(phone, "geom", name="iphone_mobile_backplate", type="box",
                  pos=_numbers(plate), size=_numbers([half_width + .004, .018, .004]), **common)
    for side, sign in (("left", -1), ("right", 1)):
        ET.SubElement(phone, "geom", name="iphone_mobile_clamp_" + side, type="box",
                      pos=_numbers([sign * (half_width + .003), plate[1], 0]),
                      size=_numbers([.003, .021, half_depth + .004]), **common)
    anchor = np.array([.10, .045, 0.])
    plate_in_head = rotation @ plate + position
    ET.SubElement(head, "geom", name="iphone_mobile_support", type="capsule",
                  fromto=_numbers(np.r_[anchor, plate_in_head]), size=".009", **common)
    head.append(phone)


def load_mobile_scene(calibration: CameraCalibration | None = None, *,
                      configuration: CameraConfiguration | None = None,
                      model_cache=None) -> tuple[mujoco.MjModel, mujoco.MjData, MobileRobotSpec]:
    """Load pinned TIAGo assets and the mobile laboratory, then settle physics.

    The original wheel velocity and arm/head/torso position servos are retained.
    The source home control targets initialize scalar position joints once;
    the free base starts 5 mm above the z=0 floor with a valid quaternion.
    The 0.5 s warmup advances real simulation time and remains in ``data.time``.
    Phone and clamp geometry are rigid, massless, and contact-disabled.
    """
    if configuration is not None and calibration is not None and calibration != configuration.calibration:
        raise ValueError("Configuration and mobile camera calibration must match")
    if configuration is None:
        configuration = (configure_camera("iphone-17-pro") if calibration is None else
                         CameraConfiguration("iphone-17-pro", calibration,
                                             "user_supplied_intrinsics", metadata={"identity_source": "user_supplied"}))
    directory = fetch_menagerie("tiago", cache_dir=model_cache)
    root = ET.parse(directory / "tiago.xml").getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((directory / "assets").resolve()))
    # Keep the phone/clamp body frames addressable instead of fusing them into
    # the head; robot joint physics and upstream mesh assets remain unchanged.
    compiler.set("fusestatic", "false")
    wrapper = ET.parse(directory / "tiago_position.xml").getroot()
    home_control = np.fromstring(wrapper.find("keyframe/key[@name='home']").get("ctrl"), sep=" ")
    _merge(root, wrapper)
    environment = ET.parse(SCENE_PATH).getroot()
    # Upstream robot defaults disable contype. Scene collisions must explicitly
    # override that inherited default, including wheel contact with the floor.
    for geom in environment.iter("geom"):
        geom.set("contype", geom.get("contype", "1"))
        geom.set("conaffinity", geom.get("conaffinity", "1"))
        geom.set("mass", "0")
    _merge(root, environment)
    _attach_phone(root, configuration)
    root.find("visual/global").set("offwidth", str(max(1280, configuration.calibration.width)))
    root.find("visual/global").set("offheight", str(max(960, configuration.calibration.height)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    reference = model.joint("reference")
    address = reference.qposadr[0]
    data.qpos[address:address + 7] = [0., 0., .005, 1., 0., 0., 0.]
    if len(home_control) != model.nu:
        raise ValueError("Upstream TIAGo home control count changed")
    data.ctrl[:] = home_control
    for actuator in range(model.nu):
        if model.actuator_biasprm[actuator, 1] < 0:
            joint = int(model.actuator_trnid[actuator, 0])
            data.qpos[model.jnt_qposadr[joint]] = data.ctrl[actuator] / model.actuator_gear[actuator, 0]
    mujoco.mj_forward(model, data)
    warmup_steps = round(.5 / model.opt.timestep)
    mujoco.mj_step(model, data, nstep=warmup_steps)
    mujoco.mj_forward(model, data)
    spec = MobileRobotSpec(metadata={
        "repository": "google-deepmind/mujoco_menagerie", "commit": _COMMIT,
        "model_url": _SOURCE, "license": "Apache-2.0", "license_path": str(directory / "LICENSE"),
        "drive_parameters_source": _DRIVE_SOURCE,
        "wheel_radius_definition": "PAL PMB2 nominal controller radius; original wheel mesh contacts retained",
        "warmup_seconds": float(data.time), "initial_free_base_height_m": .005,
        "home_initialization": "original position-servo targets; identity free-base quaternion",
        "camera_configuration": configuration.to_dict(), "phone_payload": "massless_visual",
        "wheel_odometry_frame": "initial settled base at controller creation; encoder-derived",
        "camera_mount": "head_2_link rigid supported clamp; illustrative, not measured hardware extrinsics",
        "adapter_changes": ["absolute mesh paths", "preserved fixed body frames", "floor at z=0",
                            "home controls initialize position joints once", "contact-enabled scene geoms"],
    })
    return model, data, spec


class WheelOdometry:
    """Integrate measured wheel angles with differential-drive arc kinematics."""

    def __init__(self, radius_m: float, track_m: float):
        if not np.isfinite([radius_m, track_m]).all() or min(radius_m, track_m) <= 0:
            raise ValueError("Wheel radius and track must be finite and positive")
        self.radius_m, self.track_m = float(radius_m), float(track_m)
        self.x_m = self.y_m = self.yaw_rad = 0.
        self._previous = None

    def update(self, left_position_rad: float, right_position_rad: float) -> None:
        position = np.asarray([left_position_rad, right_position_rad], dtype=float)
        if not np.isfinite(position).all():
            raise ValueError("Wheel positions must be finite")
        if self._previous is not None:
            left, right = (position - self._previous) * self.radius_m
            distance, angle = (right + left) / 2, (right - left) / self.track_m
            local_x = distance * np.sinc(angle / np.pi)
            local_y = distance * (1 - math.cos(angle)) / angle if abs(angle) > 1e-10 else 0.
            cosine, sine = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
            self.x_m += cosine * local_x - sine * local_y
            self.y_m += sine * local_x + cosine * local_y
            self.yaw_rad += angle
        self._previous = position.copy()

    def state(self) -> dict[str, float]:
        return {"x_m": float(self.x_m), "y_m": float(self.y_m), "yaw_rad": float(self.yaw_rad)}


class DifferentialDriveController:
    """Bounded wheel velocity control; each step advances real MuJoCo physics."""

    def __init__(self, model, data, spec: MobileRobotSpec, *,
                 linear_limit: float = .25, angular_limit: float = .6):
        if not np.isfinite([linear_limit, angular_limit]).all() or min(linear_limit, angular_limit) <= 0:
            raise ValueError("Velocity limits must be finite and positive")
        self.model, self.data, self.spec = model, data, spec
        self.linear_limit, self.angular_limit = float(linear_limit), float(angular_limit)
        self.linear_acceleration = .5
        self.angular_acceleration = 1.05
        self._joint_ids = np.array([model.joint(name).id for name in spec.wheel_joint_names])
        self._qpos_addresses = model.jnt_qposadr[self._joint_ids]
        self._qvel_addresses = model.jnt_dofadr[self._joint_ids]
        self._actuator_ids = np.array([model.actuator(name).id for name in spec.wheel_actuator_names])
        self._gears = model.actuator_gear[self._actuator_ids, 0].copy()
        if np.any(self._gears <= 0):
            raise ValueError("Wheel actuators require positive scalar transmission gears")
        self._requested = np.zeros(2)
        self._applied = np.zeros(2)
        self._wheel_target = np.zeros(2)
        self.odometry = WheelOdometry(spec.radius_m, spec.track_m)
        self.odometry.update(*data.qpos[self._qpos_addresses])

    def command(self, linear_mps: float, angular_rps: float) -> None:
        values = np.asarray([linear_mps, angular_rps], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Velocity command must be finite")
        self._requested = np.clip(values, [-self.linear_limit, -self.angular_limit],
                                   [self.linear_limit, self.angular_limit])

    def step(self, dt: float | None = None) -> None:
        timestep = float(self.model.opt.timestep)
        duration = timestep if dt is None else float(dt)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Step duration must be finite and positive")
        count = round(duration / timestep)
        if count < 1 or not math.isclose(duration, count * timestep, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Step duration must be an integer multiple of the physics timestep")
        for _ in range(count):
            limits = np.array([self.linear_acceleration, self.angular_acceleration]) * timestep
            self._applied += np.clip(self._requested - self._applied, -limits, limits)
            linear, angular = self._applied
            self._wheel_target = np.array([linear - angular * self.spec.track_m / 2,
                                           linear + angular * self.spec.track_m / 2]) / self.spec.radius_m
            control = self._wheel_target * self._gears
            ranges = self.model.actuator_ctrlrange[self._actuator_ids]
            control = np.clip(control, ranges[:, 0], ranges[:, 1])
            self._wheel_target = control / self._gears
            self.data.ctrl[self._actuator_ids] = control
            mujoco.mj_step(self.model, self.data)
            self.odometry.update(*self.data.qpos[self._qpos_addresses])

    def state(self) -> dict:
        # mj_step integrates qpos after its position stage. Refresh kinematics
        # so poses and encoders describe the same completed simulation step.
        mujoco.mj_forward(self.model, self.data)
        def body_pose(name):
            index = self.model.body(name).id
            result = np.eye(4)
            result[:3, :3] = self.data.xmat[index].reshape(3, 3)
            result[:3, 3] = self.data.xpos[index]
            return result
        camera = self.model.camera(self.spec.camera_name).id
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = self.data.cam_xmat[camera].reshape(3, 3)
        camera_pose[:3, 3] = self.data.cam_xpos[camera]
        return {
            "command_twist": {"linear_mps": float(self._requested[0]), "angular_rps": float(self._requested[1])},
            "applied_twist": {"linear_mps": float(self._applied[0]), "angular_rps": float(self._applied[1])},
            "wheel_position_rad": self.data.qpos[self._qpos_addresses].tolist(),
            "wheel_velocity_rad_s": self.data.qvel[self._qvel_addresses].tolist(),
            "wheel_target_rad_s": self._wheel_target.tolist(),
            "wheel_odometry": self.odometry.state(),
            "world_from_base": body_pose(self.spec.base_body).tolist(),
            "world_from_camera": (camera_pose @ CV_TO_MUJOCO_CAMERA).tolist(),
            "simulation_time": float(self.data.time),
        }
