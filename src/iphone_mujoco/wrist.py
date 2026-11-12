"""Rigid wrist-mounted RGB-D collection with the real iPhone's data conventions."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING
import uuid
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from .camera import CameraCalibration, MujocoRGBDCamera, RGBDFrame, camera_mjcf
from .geometry import CV_TO_MUJOCO_CAMERA, validate_transform
from .profiles import phone_mjcf
from .recording import RecordingWriter
from .streams import LiveSample
from .virtual_device import quantize_depth

if TYPE_CHECKING:
    from .configuration import CameraConfiguration


@dataclass(frozen=True)
class WristCameraMount:
    """Robot-link <- CV optical-camera transform in meters; measurement is explicit.

    The optional support frame locates a visual flange collar; its local +Z
    points out of the flange. It does not alter the camera pose or robot load.
    """
    parent_body: str
    parent_from_camera: np.ndarray
    camera_name: str = "wrist_camera"
    measured: bool = False
    parent_from_support: np.ndarray | None = None

    def __post_init__(self):
        if not self.parent_body or not self.camera_name:
            raise ValueError("Camera and parent body names must be nonempty")
        transform = validate_transform(self.parent_from_camera).copy()
        transform.setflags(write=False)
        object.__setattr__(self, "parent_from_camera", transform)
        if self.parent_from_support is not None:
            support = validate_transform(self.parent_from_support).copy()
            support.setflags(write=False)
            object.__setattr__(self, "parent_from_support", support)

    def to_dict(self):
        values = {"parent_body": self.parent_body, "parent_from_camera": self.parent_from_camera.tolist(),
                  "camera_name": self.camera_name, "measured": self.measured,
                  "camera_axes": "CV: X right, Y down, Z forward", "translation_units": "meters"}
        if self.parent_from_support is not None:
            values["parent_from_support"] = self.parent_from_support.tolist()
        return values

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        return cls(values["parent_body"], np.asarray(values["parent_from_camera"]),
                   values.get("camera_name", "wrist_camera"), values.get("measured", False),
                   values.get("parent_from_support"))


def default_wrist_mount(robot: str) -> WristCameraMount:
    """Demonstration mounts, not measured real iPhone-to-robot extrinsics."""
    pose = np.eye(4)
    support = None
    if robot == "panda":
        pose[:3, 3] = [.09, 0., .06]
        support = np.eye(4)
        support[:3, 3] = [0., 0., .02]
        parent = "hand"
    elif robot == "ur5e":
        pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        # The Menagerie attachment_site is on the wrist's +Y axis, 100 mm
        # from its origin. Keep the lens in that TCP plane, beside the flange.
        pose[:3, 3] = [.105, .10, 0.]
        support = pose.copy()
        support[:3, 3] = [0., .10, 0.]
        parent = "wrist_3_link"
    else:
        raise ValueError("A custom robot requires an explicit wrist mount JSON")
    return WristCameraMount(parent, pose, parent_from_support=support)


def _phone_shell_bounds(phone: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    """Read shell bounds from generated geometry, including rounded components."""
    bounds = []
    for geom in phone.findall("geom"):
        if "_shell" not in geom.get("name", ""):
            continue
        size = np.fromstring(geom.get("size", ""), sep=" ")
        kind = geom.get("type", "sphere")
        if kind in ("box", "ellipsoid") and len(size) == 3:
            half = size
        elif kind == "sphere" and len(size):
            half = np.repeat(size[0], 3)
        elif kind in ("capsule", "cylinder") and len(size):
            endpoints = np.fromstring(geom.get("fromto", ""), sep=" ")
            if len(endpoints) == 6:
                points = endpoints.reshape(2, 3)
                bounds.append((points.min(axis=0) - size[0], points.max(axis=0) + size[0]))
                continue
            if len(size) < 2:
                continue
            half = np.array([size[0], size[0], size[1] + (size[0] if kind == "capsule" else 0.)])
        else:
            continue
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.fromstring(geom.get("quat", "1 0 0 0"), sep=" "))
        half = np.abs(rotation.reshape(3, 3)) @ half
        position = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
        bounds.append((position - half, position + half))
    if not bounds:
        raise ValueError("Phone support requires shell geometry with explicit dimensions")
    return np.min([item[0] for item in bounds], axis=0), np.max([item[1] for item in bounds], axis=0)


def _append_phone_support(body: ET.Element, phone: ET.Element, mount: WristCameraMount,
                          optical_from_phone: np.ndarray) -> None:
    """Connect the flange collar to a small clamp on the phone's lower edge."""
    lower, upper = _phone_shell_bounds(phone)
    half_width = (upper[0] - lower[0]) / 2
    plate_center = np.array([(upper[0] + lower[0]) / 2, lower[1] + .035, upper[2] + .003])
    appearance = {"rgba": ".32 .35 .40 1", "mass": "0", "density": "0",
                  "contype": "0", "conaffinity": "0", "group": "2"}
    prefix = mount.camera_name + "_support"
    ET.SubElement(phone, "geom", name=prefix + "_plate", type="box",
                  pos=" ".join(map(str, plate_center)),
                  size=f"{half_width + .004} .016 .003", **appearance)
    for side, edge in (("left", lower[0] - .003), ("right", upper[0] + .003)):
        ET.SubElement(phone, "geom", name=prefix + "_clamp_" + side, type="box",
                      pos=f"{edge} {plate_center[1]} {upper[2]}", size=".003 .018 .009", **appearance)
    optical_from_support = np.linalg.inv(mount.parent_from_camera) @ mount.parent_from_support
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(optical_from_support[:3, :3]).ravel())
    support = ET.SubElement(body, "body", name=prefix,
                            pos=" ".join(map(str, optical_from_support[:3, 3])),
                            quat=" ".join(map(str, quaternion)))
    collar_center = np.array([0., 0., -.006, 1.])
    ET.SubElement(support, "geom", name=prefix + "_collar", type="cylinder",
                  pos="0 0 -.006", size=".0355 .006", **appearance)
    start = (optical_from_support @ collar_center)[:3]
    end = (optical_from_phone @ np.r_[plate_center, 1.])[:3]
    ET.SubElement(body, "geom", name=prefix + "_brace", type="capsule",
                  fromto=" ".join(map(str, np.r_[start, end])), size=".006", **appearance)


def _configured_phone_from_camera(configuration: CameraConfiguration) -> np.ndarray:
    from .devices import get_device
    measured = configuration.metadata.get("phone_from_camera")
    if measured is not None:
        return validate_transform(np.asarray(measured, dtype=float))
    return get_device(configuration.device_id).phone_from_camera(configuration.lens)


def configured_wrist_mount(mount: WristCameraMount,
                           configuration: CameraConfiguration | None) -> WristCameraMount:
    """Resolve a selected lens while keeping the wide-reference phone pose fixed."""
    if configuration is None:
        return mount
    from .devices import get_device
    device = get_device(configuration.device_id)
    parent_from_phone = mount.parent_from_camera @ np.linalg.inv(device.phone_from_camera("wide"))
    return WristCameraMount(
        mount.parent_body, parent_from_phone @ _configured_phone_from_camera(configuration),
        camera_name=mount.camera_name, measured=mount.measured,
        parent_from_support=mount.parent_from_support,
    )


def wrist_camera_mjcf(calibration: CameraCalibration, mount: WristCameraMount, *,
                      show_phone: bool = False, configuration: CameraConfiguration | None = None,
                      payload: bool = False) -> str:
    """Return a rigid optical rig, optionally with a selected phone and payload.

    With a configuration, ``mount`` locates the wide lens. Selecting another
    lens preserves the phone body pose and moves the optical center within it.
    The configured payload uses the device model's documented inertial estimate.
    Legacy phone cosmetics remain massless.
    """
    if configuration is not None and calibration != configuration.calibration:
        raise ValueError("Wrist calibration must match the selected camera configuration")
    if payload and configuration is None:
        raise ValueError("A phone payload requires a device camera configuration")
    effective_mount = configured_wrist_mount(mount, configuration)
    optical = effective_mount.parent_from_camera
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(optical[:3, :3]).ravel())
    body = ET.Element("body", name=mount.camera_name + "_mount",
                      pos=" ".join(map(str, optical[:3, 3])), quat=" ".join(map(str, quaternion)))
    camera = ET.fromstring(camera_mjcf(calibration, name=mount.camera_name))
    # The parent rig uses CV axes; MuJoCo cameras look down local -Z.
    camera.set("quat", "0 1 0 0")
    body.append(camera)
    assets = None
    if configuration is not None and (show_phone or payload):
        from .devices import get_device, phone_geometry
        device = get_device(configuration.device_id)
        assets, phone = phone_geometry(device, body_name=mount.camera_name + "_phone",
                                       region=configuration.region, payload=payload, visual_group=2)
        optical_from_phone = np.linalg.inv(_configured_phone_from_camera(configuration))
        if not show_phone:
            for child in list(phone):
                if child.tag != "inertial":
                    phone.remove(child)
            assets = None
    elif show_phone:
        phone = ET.fromstring(phone_mjcf(calibration,
                                        body_name=mount.camera_name + "_phone",
                                        camera_name=mount.camera_name + "_visual_camera"))
        visual_camera = phone.find("camera")
        phone_from_optical = np.eye(4)
        phone_from_optical[:3, 3] = np.fromstring(visual_camera.get("pos", "0 0 0"), sep=" ")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.fromstring(visual_camera.get("quat", "1 0 0 0"), sep=" "))
        phone_from_optical[:3, :3] = rotation.reshape(3, 3)
        optical_from_phone = np.linalg.inv(phone_from_optical @ CV_TO_MUJOCO_CAMERA)
        phone.remove(visual_camera)
    else:
        phone = None
    if phone is not None:
        mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(optical_from_phone[:3, :3]).ravel())
        phone.set("pos", " ".join(map(str, optical_from_phone[:3, 3])))
        phone.set("quat", " ".join(map(str, quaternion)))
        if show_phone and effective_mount.parent_from_support is not None:
            _append_phone_support(body, phone, effective_mount, optical_from_phone)
        body.append(phone)
    if configuration is not None:
        document = ET.Element("mujoco")
        if assets is not None and len(assets):
            document.append(assets)
        ET.SubElement(document, "worldbody").append(body)
        return ET.tostring(document, encoding="unicode")
    return ET.tostring(body, encoding="unicode")


class WristRGBDRecorder:
    """Save same-state RGB, metric/UInt16 depth, robot state and optical poses.

    The PNGs are convenient dataset exports. episode/ uses the same NPZ/JSON
    format and RGBDFrame API as real iPhone recordings. All sensor frames are
    keyed by one index and simulation timestamp; no physics steps occur between
    rendering RGB and depth or reading joints.
    """
    def __init__(self, model, data, spec, mount: WristCameraMount,
                 calibration: CameraCalibration, directory, *, depth_scale: float = 0.0001,
                 configuration: CameraConfiguration | None = None):
        quantize_depth(np.zeros((1, 1)), depth_scale)
        self.model, self.data, self.spec = model, data, spec
        self.mount, self.calibration = mount, calibration
        if configuration is not None and configuration.calibration != calibration:
            raise ValueError("Wrist calibration must match the selected camera configuration")
        self.configuration = configuration.to_dict() if configuration is not None else None
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(f"Recording directory must be new or empty: {self.directory}")
        parent = model.body(mount.parent_body).id
        camera = model.camera(mount.camera_name).id
        body = int(model.cam_bodyid[camera])
        if int(model.body_parentid[body]) != parent or model.body_jntnum[body] or model.body_mocapid[body] >= 0:
            raise ValueError("Wrist camera must be a rigid child of the configured parent body")
        self.depth_scale = float(depth_scale)
        self.session_id = "wrist-sim-" + uuid.uuid4().hex
        self.count = 0
        self._closed = False
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in ("rgb", "depth"):
            (self.directory / name).mkdir()
        calibration.save(self.directory / "calibration.json")
        mount.save(self.directory / "mount.json")
        info = {"schema_version": 1, "source": "mujoco-simulation", "robot": spec.name,
                "robot_source": spec.source, "session_id": self.session_id,
                "rgb": {"order": "RGB", "dtype": "uint8", "width": calibration.width,
                        "height": calibration.height, "K": calibration.matrix.tolist()},
                "depth": {"dtype": "uint16", "width": calibration.depth_width or calibration.width,
                          "height": calibration.depth_height or calibration.height,
                          "K": calibration.depth_calibration().matrix.tolist(),
                          "meters_per_unit": depth_scale, "invalid_code": 0,
                          "saturation_reserved_code": 65535, "definition": "optical_axis_z"},
                "timestamp_clock": "simulation", "simulation_timestep": float(model.opt.timestep),
                "sensor_model": "ideal_pinhole_quantized_depth; no LiDAR noise or measured confidence",
                "mount": mount.to_dict()}
        if self.configuration is not None:
            info["camera_configuration"] = self.configuration
            configuration.save(self.directory / "device.json")
            from .devices import get_device
            info["phone_hardware"] = get_device(configuration.device_id).to_dict()
            phone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mount.camera_name + "_phone")
            info["phone_payload"] = {
                "enabled": bool(phone_id >= 0 and model.body_mass[phone_id] > 0),
                "mass_kg": float(model.body_mass[phone_id]) if phone_id >= 0 else 0.,
                "inertia_model": "uniform_box_estimate_at_body_center",
                "mount_mass_kg": 0.,
            }
        (self.directory / "camera_info.json").write_text(json.dumps(info, indent=2) + "\n")
        self._camera = MujocoRGBDCamera(model, calibration, mount.camera_name,
                                       max_depth_m=65534.5 * self.depth_scale)
        try:
            self._writer = RecordingWriter(self.directory / "episode", metadata=info)
            self._states = (self.directory / "states.jsonl").open("w")
        except Exception:
            self._camera.close()
            raise

    def capture(self) -> RGBDFrame:
        if self._closed:
            raise RuntimeError("Wrist recorder is closed")
        frame = self._camera.render(self.data)
        frame.frame_id = self.count
        encoded = quantize_depth(frame.depth_m, self.depth_scale)
        frame.depth_m = encoded.astype(np.float32) * self.depth_scale
        frame.depth_m[encoded == 0] = np.nan
        state = {"index": self.count, "simulation_time": float(self.data.time),
                 "qpos": self.data.qpos.tolist(), "qvel": self.data.qvel.tolist(),
                 "ctrl": self.data.ctrl.tolist(), "world_from_camera": frame.world_from_camera.tolist()}
        if not all(np.isfinite(np.asarray(state[name])).all() for name in ("qpos", "qvel", "ctrl")):
            raise RuntimeError("Simulation state is nonfinite; refusing to record")
        frame.metadata.update({"session_id": self.session_id, "simulation_time": float(self.data.time),
                               "camera_mount": self.mount.to_dict(), "depth_meters_per_unit": self.depth_scale,
                               "depth_model": "ideal_optical_axis_quantized", "confidence_model": "not_simulated",
                               "robot_state": state})
        if self.configuration is not None:
            frame.metadata["camera_configuration"] = self.configuration
        stem = f"{self.count:08d}"
        Image.fromarray(frame.rgb).save(self.directory / "rgb" / (stem + ".png"))
        Image.fromarray(encoded).save(self.directory / "depth" / (stem + ".png"))
        self._writer.write(LiveSample(timestamp=frame.timestamp, source_timestamp=frame.timestamp,
                                      received_at=time.monotonic(), rgbd=frame, pose=frame.world_from_camera,
                                      session_id=self.session_id, frame_id=self.count,
                                      metadata={"source": "simulation", "robot_state": state}))
        self._states.write(json.dumps(state, allow_nan=False) + "\n")
        if self.count == 0:
            frame.point_cloud(stride=2).save_ply(self.directory / "preview_points.ply")
        self.count += 1
        return frame

    def close(self):
        if not self._closed:
            self._camera.close()
            self._writer.close()
            self._states.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _place_demo_targets(model, data, camera_name):
    """Place complete inspection fixtures beneath a downward-looking wrist."""
    mujoco.mj_forward(model, data)
    camera = model.camera(camera_name).id
    eye = data.cam_xpos[camera].copy()
    # MuJoCo's optical forward is camera-local -Z.
    direction = -data.cam_xmat[camera].reshape(3, 3)[:, 2]
    if direction[2] >= -.1:
        return
    center = eye + direction * ((.05 - eye[2]) / direction[2])
    for name, offset in (("gauge_block", [.08, 0., 0.]), ("inspection_boss", [-.09, -.03, 0.])):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if index >= 0 and model.geom_bodyid[index] == 0:
            delta = center + offset - model.geom_pos[index]
            for geom_id in range(model.ngeom):
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                if geom_name == name or geom_name.startswith(name + "_"):
                    model.geom_pos[geom_id] += delta
    mujoco.mj_forward(model, data)


def capture_command(args):
    """CLI workflow; physics owns the state and camera samples its current step."""
    from .robots import RobotSpec, fetch_menagerie, load_scene
    from .control import CartesianTeleoperator
    from .virtual_device import VirtualPhonePublisher
    from .cli import _camera_configuration, _viewer
    from .phone_lab import update_rgbd_preview
    if args.fast and (args.ip or args.input_episode):
        raise ValueError("--fast is for offline demo/hold motion; live/replayed input requires wall-clock pacing")
    configuration = _camera_configuration(args)
    if configuration is None and not args.profile:
        raise ValueError("capture-wrist requires --device or --profile")
    calibration = configuration.calibration if configuration is not None else CameraCalibration.load(args.profile)
    reference_mount = WristCameraMount.load(args.mount) if args.mount else default_wrist_mount(args.robot)
    mount = configured_wrist_mount(reference_mount, configuration)
    if args.robot in ("panda", "ur5e"):
        fetch_menagerie(args.robot, cache_dir=args.model_cache)
    selected = RobotSpec.from_json(args.robot) if args.robot.endswith(".json") else args.robot
    model, data, spec = load_scene(
        selected,
        phone_xml=wrist_camera_mjcf(calibration, reference_mount,
                                    show_phone=getattr(args, "show_phone", False),
                                    configuration=configuration, payload=getattr(args, "payload", False)),
        phone_parent_body=mount.parent_body, model_cache=args.model_cache)
    if args.fps > 1 / model.opt.timestep:
        raise ValueError("Capture FPS cannot exceed the simulation physics rate")
    if spec.source != "custom":
        _place_demo_targets(model, data, mount.camera_name)
    joint_ids = np.array([model.joint(name).qposadr[0] for name in spec.joint_names], dtype=int)
    actuator_ids = np.array([model.actuator(name).id for name in spec.actuator_names], dtype=int)
    home = data.qpos[joint_ids].copy()
    controller = CartesianTeleoperator(model, data, phone_to_world=np.eye(3),
                                      translation_scale=args.translation_scale, **spec.controller_kwargs)
    recorder = None
    started = time.monotonic()
    physics_steps = input_count = 0
    try:
        with ExitStack() as stack:
            viewer = _viewer(stack, model, data, args.headless)
            recorder = stack.enter_context(WristRGBDRecorder(model, data, spec, mount, calibration,
                                                            args.output, depth_scale=args.depth_scale,
                                                            configuration=configuration))
            publisher = stack.enter_context(VirtualPhonePublisher(args.publish, depth_scale=args.depth_scale)) if args.publish else None
            source = None
            if args.ip:
                from .streams import AsMagicSource
                source = stack.enter_context(AsMagicSource(args.ip, mode="teleop"))
                print("Connected input. Release Square once, then hold to move the wrist camera.", flush=True)
            elif args.input_episode:
                from .recording import ReplaySource
                source = stack.enter_context(ReplaySource(args.input_episode, realtime=True))
            # Keep the wire source timestamp positive; zero means absent in the
            # existing phone SDK. Sample only after integrating one whole step.
            next_capture = float(model.opt.timestep)
            pace_started = time.monotonic() - float(data.time)
            while (args.duration <= 0 or data.time < args.duration - 1e-9) and (viewer is None or viewer.is_running()):
                now = time.monotonic()
                if source is not None:
                    sample = source.poll(timeout_s=0)
                    if sample is not None:
                        input_count += 1
                        if sample.pose is not None:
                            controller.update(sample.pose, receive_time=sample.received_at,
                                              session_id=sample.session_id, clutch=sample.clutch,
                                              reset=sample.buttons.calibrate, gripper=sample.gripper)
                    controller.step(now=now, dt=model.opt.timestep)
                    if sample is None and args.input_episode and source.exhausted:
                        controller.hold()
                        break
                elif args.motion == "demo":
                    phase = np.arange(len(home)) * .4
                    desired = home + .08 * (np.sin(phase + data.time * .8) - np.sin(phase))
                    data.ctrl[actuator_ids] = np.clip(desired, controller.lower, controller.upper)
                mujoco.mj_step(model, data)
                physics_steps += 1
                if data.time + 1e-9 >= next_capture:
                    frame = recorder.capture()
                    if publisher:
                        publisher.publish(frame)
                    next_capture += 1 / args.fps
                    if viewer:
                        update_rgbd_preview(viewer, frame)
                        viewer.sync()
                if not args.fast:
                    delay = pace_started + data.time - time.monotonic()
                    if delay > 0:
                        time.sleep(min(delay, .02))
    finally:
        if recorder is not None:
            result = {"robot": spec.name, "camera_frames": recorder.count, "physics_steps": physics_steps,
                      "simulation_seconds": float(data.time), "wall_seconds": time.monotonic() - started,
                      "requested_simulation_fps": args.fps, "input_samples": input_count,
                      "motion": "phone" if args.ip else "recorded-input" if args.input_episode else args.motion,
                      "mount_measured": mount.measured, "output": str(recorder.directory.resolve())}
            (recorder.directory / "run.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
