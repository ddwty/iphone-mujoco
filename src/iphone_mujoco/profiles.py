"""Portable calibration profiles and phone camera modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .camera import CameraCalibration, RGBDFrame, camera_mjcf
from .geometry import validate_transform


@dataclass
class DeviceProfile:
    name: str
    calibration: CameraCalibration
    source: str = "runtime_asmagic_frame"
    device_identifier: str | None = None
    body_dimensions_m: tuple[float, float, float] | None = None
    camera_mount_measured: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, frame: RGBDFrame, *, name: str = "my-iphone",
                   device_identifier: str | None = None):
        """Snapshot K and image/depth sizes, retaining frame/session provenance."""
        return cls(name, frame.calibration,
                   source=frame.metadata.get("calibration_source", frame.source),
                   device_identifier=device_identifier,
                   metadata={"capture_timestamp": frame.timestamp,
                             "frame_id": frame.frame_id, "frame_source": frame.source,
                             "capture_metadata": frame.metadata,
                             "limitations": ["Single runtime calibration snapshot; ARKit intrinsics can vary with capture settings.",
                                             "Lens distortion, RGB-depth extrinsics and physical body parameters are not transmitted.",
                                             "Depth intrinsics assume ARKit sceneDepth registered to the full RGB raster."]})

    def to_dict(self):
        result = asdict(self)
        result["calibration"] = self.calibration.to_dict()
        result["schema_version"] = 1
        return result

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if values.pop("schema_version", None) != 1:
            raise ValueError("Unsupported device profile schema")
        values["calibration"] = CameraCalibration(**values["calibration"])
        if values.get("body_dimensions_m") is not None:
            values["body_dimensions_m"] = tuple(values["body_dimensions_m"])
        return cls(**values)


def synthetic_profile(width: int = 640, height: int = 480) -> DeviceProfile:
    """An explicitly synthetic example, never a factory iPhone calibration."""
    calibration = CameraCalibration(width, height, .85*width, .85*width,
                                    (width-1)/2, (height-1)/2,
                                    max(1, width//4), max(1, height//4))
    return DeviceProfile("synthetic-pinhole-demo", calibration, source="synthetic_example",
                         metadata={"calibration_status": "synthetic",
                                   "body_geometry_status": "approximate_visual_only",
                                   "mass_inertia_status": "unknown",
                                   "camera_mount_status": "unknown"})


def phone_mjcf(calibration: CameraCalibration, *, body_name: str = "iphone",
               camera_name: str = "iphone_camera", mocap: bool = False,
               visual_group: int = 2) -> str:
    """Return one MJCF body with approximate cosmetic geometry and calibrated K.

    Appearance is a generic 76 x 150 x 8 mm phone and a guessed rear lens mount;
    contact is disabled. The camera mount remains configurable. Use
    ``MujocoRGBDCamera.set_world_pose(..., move_body=True)`` to place the optical
    center from live poses while preserving the approximate appearance.
    """
    if not isinstance(visual_group, int) or not 0 <= visual_group <= 5:
        raise ValueError("visual_group must be an integer from 0 through 5")
    camera = camera_mjcf(calibration, name=camera_name).replace(
        'mode="fixed"', 'mode="fixed" pos="0.020 0.052 -0.005"')
    mocap_attr = ' mocap="true"' if mocap else ""
    return f'''<body name={quoteattr(body_name)}{mocap_attr}>
  <!-- APPROXIMATE VISUAL ONLY: physical dimensions, mass/inertia and lens mount are unmeasured. -->
  <geom name={quoteattr(body_name+'_shell')} type="box" size="0.038 0.075 0.004" rgba="0.12 0.13 0.15 1" density="0" mass="0" contype="0" conaffinity="0" group="{visual_group}"/>
  <geom name={quoteattr(body_name+'_screen')} type="box" pos="0 0 0.0041" size="0.035 0.071 0.0002" rgba="0.025 0.04 0.065 1" density="0" mass="0" contype="0" conaffinity="0" group="{visual_group}"/>
  <geom name={quoteattr(body_name+'_lens')} type="cylinder" pos="0.020 0.052 -0.0048" size="0.006 0.001" rgba="0.05 0.13 0.23 1" density="0" mass="0" contype="0" conaffinity="0" group="{visual_group}"/>
  {camera}
  <site name={quoteattr(body_name+'_optical_center')} pos="0.020 0.052 -0.005" size="0.003" rgba="0.1 0.8 1 0.4" group="3"/>
</body>'''


def configured_phone_mjcf(configuration, *, body_name: str = "iphone",
                          camera_name: str = "iphone_camera", mocap: bool = False,
                          visual_group: int = 2, payload: bool = False) -> str:
    """Return an MJCF module with catalog geometry, assets and selected camera.

    The body frame is screen-right, top, screen-out. Lens coordinates use
    Apple's mechanical cover centers; optical entrance-pupil offsets can be
    supplied as ``metadata['phone_from_camera']`` in a measured configuration.
    """
    from .devices import get_device, phone_geometry
    device = get_device(configuration.device_id)
    assets, body = phone_geometry(device, body_name=body_name,
                                  region=configuration.region, payload=payload,
                                  visual_group=visual_group)
    if mocap:
        body.set("mocap", "true")
    optical = configuration.metadata.get("phone_from_camera")
    transform = (validate_transform(np.asarray(optical, dtype=float)) if optical is not None
                 else device.phone_from_camera(configuration.lens))
    from .geometry import CV_TO_MUJOCO_CAMERA
    camera_pose = transform @ CV_TO_MUJOCO_CAMERA
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(camera_pose[:3, :3]).ravel())
    camera = ET.fromstring(camera_mjcf(configuration.calibration, name=camera_name))
    camera.set("pos", " ".join(map(str, camera_pose[:3, 3])))
    camera.set("quat", " ".join(map(str, quaternion)))
    body.append(camera)
    ET.SubElement(body, "site", name=body_name + "_optical_center",
                  pos=" ".join(map(str, transform[:3, 3])), size=".003",
                  rgba=".1 .8 1 .4", group="3")
    root = ET.Element("mujoco")
    root.append(assets)
    ET.SubElement(root, "worldbody").append(body)
    return ET.tostring(root, encoding="unicode")


def write_phone_module(path: str | Path, calibration: CameraCalibration, *,
                       body_name: str = "iphone", camera_name: str = "iphone_camera",
                       mocap: bool = False, visual_group: int = 2,
                       parent_from_body: np.ndarray | None = None,
                       overwrite: bool = False, configuration=None,
                       payload: bool = False) -> Path:
    """Export a portable MJCF include and its RGB/depth calibration sidecar.

    Include the XML inside worldbody, or under a robot body for a fixed mount.
    parent_from_body positions the approximate phone body, not its optical center.
    A mocap phone must be a direct child of worldbody. Geometry is visual only.
    """
    path = Path(path)
    sidecar = path.with_suffix(".calibration.json")
    asset_path = path.with_suffix(".assets.xml")
    device_path = path.with_suffix(".device.json")
    outputs = [path, sidecar] + ([asset_path, device_path] if configuration else [])
    if not overwrite and any(output.exists() for output in outputs):
        raise FileExistsError(f"Module or calibration already exists: {path}")
    if configuration is not None:
        if calibration != configuration.calibration:
            raise ValueError("Configuration and camera calibration must match")
        module = ET.fromstring(configured_phone_mjcf(
            configuration, body_name=body_name, camera_name=camera_name,
            mocap=mocap, visual_group=visual_group, payload=payload))
        body = module.find("worldbody/body")
    else:
        if payload:
            raise ValueError("A device configuration is required for payload mass")
        body = ET.fromstring(phone_mjcf(calibration, body_name=body_name, camera_name=camera_name,
                                        mocap=mocap, visual_group=visual_group))
    if parent_from_body is not None:
        transform = validate_transform(parent_from_body)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(transform[:3, :3]).ravel())
        body.set("pos", " ".join(map(str, transform[:3, 3])))
        body.set("quat", " ".join(map(str, quat)))
    root = ET.Element("mujocoinclude")
    if configuration:
        root.append(ET.Comment(f"Include {asset_path.name} inside the scene's asset element."))
    root.append(body)
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    calibration.save(sidecar)
    if configuration:
        asset_include = ET.Element("mujocoinclude")
        asset_include.extend(module.find("asset"))
        ET.indent(asset_include)
        asset_path.write_text(ET.tostring(asset_include, encoding="unicode") + "\n", encoding="utf-8")
        configuration.save(device_path)
    return path
