"""Calibrated pinhole RGB-D cameras shared by real iPhones and MuJoCo.

Calibration uses the unrotated RGB raster: integer pixel centers, origin at its
top left. Depth is optical-axis Z in meters, not distance along each ray.
The renderer models ideal registered RGB-D; it does not emulate LiDAR noise,
TrueDepth optics, rolling shutter, exposure, lens distortion or ARKit tracking.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr
import json

import numpy as np

from .geometry import (
    CV_TO_MUJOCO_CAMERA, convert_iphone_pose, pose_from_xyzw,
    transform_points, validate_transform,
)


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_width: int | None = None
    depth_height: int | None = None

    def __post_init__(self):
        for key in ("width", "height"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all() or min(self.fx, self.fy) <= 0:
            raise ValueError("Focal lengths must be positive and all intrinsics finite")
        if (self.depth_width is None) != (self.depth_height is None):
            raise ValueError("Depth width and height must be specified together")
        if self.depth_width is not None:
            for key in ("depth_width", "depth_height"):
                value = getattr(self, key)
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                    raise ValueError(f"{key} must be a positive integer")

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0., self.cx], [0., self.fy, self.cy], [0., 0., 1.]])

    @classmethod
    def from_column_major(cls, values, *, width: int, height: int, **kwargs):
        """Decode asMagic's nine-value column-major K, rejecting unsupported skew."""
        values = np.asarray(values, dtype=float)
        if values.size != 9:
            raise ValueError("Camera intrinsics must contain exactly nine column-major values")
        matrix = values.reshape((3, 3), order="F")
        if not np.isfinite(matrix).all() or not np.allclose(matrix[2], [0, 0, 1], atol=1e-6):
            raise ValueError("Camera intrinsics must be a finite pinhole matrix with K[2] = [0,0,1]")
        if not np.allclose([matrix[0, 1], matrix[1, 0]], 0, atol=1e-6):
            raise ValueError("MuJoCo pinhole camera cannot represent nonzero intrinsic skew")
        return cls(width, height, float(matrix[0, 0]), float(matrix[1, 1]),
                   float(matrix[0, 2]), float(matrix[1, 2]), **kwargs)

    def scaled(self, width: int, height: int) -> "CameraCalibration":
        """Resize with pixel-center alignment: u'=(u+.5)*scale-.5.

        This is also how registered ARKit depth pixels sample the raw RGB
        raster. This operation assumes resampling with no crop or rotation.
        """
        sx, sy = width / self.width, height / self.height
        return CameraCalibration(width, height, self.fx*sx, self.fy*sy,
                                 (self.cx+.5)*sx-.5, (self.cy+.5)*sy-.5)

    def depth_calibration(self) -> "CameraCalibration":
        return self.scaled(self.depth_width or self.width, self.depth_height or self.height)

    def to_dict(self) -> dict:
        return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in asdict(self).items()}

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if values.get("schema") == "iphone_mujoco.camera_configuration" or values.get("schema_version") == 1 and "calibration" in values:
            if values.get("schema_version") != 1:
                raise ValueError("Unsupported camera profile schema")
            values = values["calibration"]
        return cls(**values)


@dataclass
class PointCloud:
    """Compact valid points; colors are uint8 RGB and pixels refer to depth."""
    points: np.ndarray
    colors: np.ndarray | None
    pixels: np.ndarray
    frame: str

    def save_ply(self, path: str | Path):
        """Write portable ASCII PLY, preserving meter coordinates and RGB."""
        header = ["ply", "format ascii 1.0", "comment meters; "+self.frame,
                  f"element vertex {len(self.points)}", "property float x",
                  "property float y", "property float z"]
        if self.colors is not None:
            header += ["property uchar red", "property uchar green", "property uchar blue"]
        header += ["end_header"]
        with Path(path).open("w", encoding="ascii") as output:
            output.write("\n".join(header)+"\n")
            for i, point in enumerate(self.points):
                row = " ".join(f"{v:.9g}" for v in point)
                if self.colors is not None:
                    row += " "+" ".join(str(int(v)) for v in self.colors[i])
                output.write(row+"\n")


@dataclass
class RGBDFrame:
    timestamp: float
    rgb: np.ndarray | None
    depth_m: np.ndarray | None
    calibration: CameraCalibration
    world_from_camera: np.ndarray | None = None
    source: str = "simulation"
    frame_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: np.ndarray | None = None

    def __post_init__(self):
        if not np.isfinite(self.timestamp):
            raise ValueError("Frame timestamp must be finite")
        if self.rgb is not None:
            self.rgb = np.asarray(self.rgb)
            if self.rgb.shape != (self.calibration.height, self.calibration.width, 3) or self.rgb.dtype != np.uint8:
                raise ValueError("RGB must be uint8 (calibration.height, calibration.width, 3)")
        depth_shape = (self.calibration.depth_height or self.calibration.height,
                       self.calibration.depth_width or self.calibration.width)
        if self.depth_m is not None:
            self.depth_m = np.asarray(self.depth_m, dtype=np.float32)
            if self.depth_m.shape != depth_shape:
                raise ValueError("Depth shape does not match calibration depth dimensions")
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence)
            if self.depth_m is None or self.confidence.shape != depth_shape:
                raise ValueError("Confidence requires depth and matching depth dimensions")
        if self.world_from_camera is not None:
            self.world_from_camera = validate_transform(self.world_from_camera).copy()

    def point_cloud(self, *, world: bool = False, stride: int = 1,
                    min_depth: float = 0., max_depth: float = np.inf,
                    min_confidence: int = 0) -> PointCloud:
        """Unproject registered RGB-D; zero/nonfinite/nonpositive depth is absent.

        World output requires an explicitly registered pose. Confidence is an
        optional measurement; requesting a positive threshold with no confidence
        raises an error instead of inventing confidence values.
        """
        if self.depth_m is None:
            raise ValueError("This frame has no depth image")
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")
        if min_confidence > 0 and self.confidence is None:
            raise ValueError("Frame does not contain depth confidence")
        v, u = np.mgrid[:self.depth_m.shape[0]:stride, :self.depth_m.shape[1]:stride]
        z = self.depth_m[::stride, ::stride]
        valid = np.isfinite(z) & (z > max(0., min_depth)) & (z <= max_depth)
        if self.confidence is not None:
            valid &= self.confidence[::stride, ::stride] >= min_confidence
        u, v, z = u[valid], v[valid], z[valid]
        k = self.calibration.depth_calibration()
        points = np.column_stack(((u-k.cx)*z/k.fx, (v-k.cy)*z/k.fy, z))
        colors = None
        if self.rgb is not None:
            rgb_u = np.minimum(self.calibration.width-1,
                               np.floor((u+.5)*self.calibration.width/k.width).astype(int))
            rgb_v = np.minimum(self.calibration.height-1,
                               np.floor((v+.5)*self.calibration.height/k.height).astype(int))
            colors = self.rgb[rgb_v, rgb_u].copy()
        if world:
            if self.world_from_camera is None:
                raise ValueError("World point cloud requires a known world_from_camera transform")
            points = transform_points(points, self.world_from_camera)
        return PointCloud(points, colors, np.column_stack((u, v)), "world_z_up" if world else "camera_cv")


def frame_from_asmagic(frame, *, pose_convention: str | None = None,
                       pose_field: str = "local", world_from_sender=None,
                       legacy_depth_scale: float | None = None) -> RGBDFrame:
    """Adapt one asMagic ARFrame with no cross-frame fusion or guessed metadata.

    Intrinsics correspond to the encoded RGB dimensions, even if RGB bytes are
    disabled. Legacy senders with missing dimensions are rejected. A scale of
    zero is unknown: historical 0.0001 m/unit is accepted only if explicitly
    passed as ``legacy_depth_scale``. The saturated 65535 code and zero are
    unavailable depth, represented by NaN.
    """
    width, height = int(frame.rgb_width), int(frame.rgb_height)
    if width <= 0 or height <= 0:
        raise ValueError("Stream lacks RGB reference dimensions; cannot assign intrinsics")
    raw_depth = frame.depth_array
    depth_width = int(frame.depth_width) if raw_depth is not None else None
    depth_height = int(frame.depth_height) if raw_depth is not None else None
    calibration = CameraCalibration.from_column_major(
        frame.camera_intrinsics, width=width, height=height,
        depth_width=depth_width, depth_height=depth_height,
    )
    depth, scale = None, None
    scale_source = None
    saturated_count = 0
    if raw_depth is not None:
        scale = float(frame.depth_meters_per_unit)
        scale_source = "runtime"
        if scale == 0 and legacy_depth_scale is not None:
            scale = float(legacy_depth_scale)
            scale_source = "explicit_legacy_override"
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Stream lacks a valid depth_meters_per_unit; opt in to a known legacy scale")
        depth = raw_depth.astype(np.float32)*scale
        saturated_count = int(np.count_nonzero(raw_depth == 65535))
        depth[(raw_depth == 0) | (raw_depth == 65535)] = np.nan
    pose = None
    if pose_field not in {"local", "global"}:
        raise ValueError("pose_field must be local or global")
    if pose_convention is not None:
        if pose_field == "global" and not (frame.global_pose_is_shared and frame.shared_world_identifier):
            raise ValueError("Global pose is not marked as a named shared-world pose")
        raw_pose = getattr(frame, pose_field+"_pose")
        if np.asarray(raw_pose).size:
            pose = convert_iphone_pose(pose_from_xyzw(raw_pose), pose_convention,
                                       world_from_sender=world_from_sender)
    source_timestamp = float(frame.source_timestamp)
    metadata = {
        "timestamp_clock": "sender_monotonic" if source_timestamp > 0 else "sender_wall",
        "source_timestamp": source_timestamp, "sender_timestamp": float(frame.timestamp),
        "session_identifier": frame.session_identifier,
        "shared_world_identifier": frame.shared_world_identifier,
        "global_pose_is_shared": bool(frame.global_pose_is_shared),
        "pose_field": pose_field, "pose_convention": pose_convention,
        "world_from_sender": None if world_from_sender is None else validate_transform(world_from_sender).tolist(),
        "depth_meters_per_unit": scale, "depth_saturated_pixel_count": saturated_count,
        "depth_scale_source": scale_source,
        "image_rotation_quarter_turns_clockwise": int(frame.image_rotation_quarter_turns_clockwise),
        "calibration_source": "runtime_asmagic_frame", "depth_model": "registered_optical_axis_meters",
    }
    return RGBDFrame(source_timestamp if source_timestamp > 0 else float(frame.timestamp),
                     frame.color_array, depth, calibration, pose, "iphone", int(frame.frame_id),
                     metadata, frame.depth_confidence_array if raw_depth is not None else None)


def _principal_offset(calibration: CameraCalibration) -> tuple[float, float]:
    # MuJoCo principalpixel is a signed offset, not CV's absolute top-left
    # principal point. GL fragment centers lie .5 pixels from viewport edges.
    return ((calibration.width-1)/2-calibration.cx,
            calibration.cy-(calibration.height-1)/2)


def camera_mjcf(calibration: CameraCalibration, *, name: str = "iphone_camera") -> str:
    """An MJCF camera supporting fx != fy and a displaced principal point."""
    px, py = _principal_offset(calibration)
    return (f'<camera name={quoteattr(name)} mode="fixed" '
            f'sensorsize="{calibration.width} {calibration.height}" '
            f'resolution="{calibration.width} {calibration.height}" '
            f'focalpixel="{calibration.fx:.17g} {calibration.fy:.17g}" '
            f'principalpixel="{px:.17g} {py:.17g}"/>')


def apply_calibration(model, camera: str | int, calibration: CameraCalibration):
    """Update native MuJoCo intrinsics. Lens distortion remains unsupported."""
    import mujoco
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) if isinstance(camera, str) else int(camera)
    if camera_id < 0 or camera_id >= model.ncam:
        raise ValueError(f"Unknown camera: {camera}")
    if int(model.cam_mode[camera_id]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
        raise ValueError("Calibrated camera must use MuJoCo fixed mode")
    model.cam_sensorsize[camera_id] = [calibration.width, calibration.height]
    model.cam_resolution[camera_id] = [calibration.width, calibration.height]
    model.cam_intrinsic[camera_id] = [calibration.fx, calibration.fy, *_principal_offset(calibration)]
    model.cam_fovy[camera_id] = np.degrees(2*np.arctan(calibration.height/(2*calibration.fy)))
    return camera_id


class MujocoRGBDCamera:
    """Reusable offscreen renderer returning the same RGBDFrame as a real phone.

    MuJoCo rendering requires an available OpenGL context. On macOS use the
    desktop's graphical session. One renderer instance is owned by one thread.
    """
    def __init__(self, model, calibration: CameraCalibration, camera: str | int = "iphone_camera",
                 *, max_depth_m: float | None = None, hidden_geom_groups: tuple[int, ...] = ()):
        import mujoco
        self.model, self.calibration = model, calibration
        self.hidden_geom_groups = tuple(hidden_geom_groups)
        self._scene_option = mujoco.MjvOption()
        for group in self.hidden_geom_groups:
            if not isinstance(group, int) or not 0 <= group < len(self._scene_option.geomgroup):
                raise ValueError("hidden_geom_groups must contain indices 0 through 5")
            self._scene_option.geomgroup[group] = 0
        self.camera_id = apply_calibration(model, camera, calibration)
        if max_depth_m is not None and (not np.isfinite(max_depth_m) or max_depth_m <= 0):
            raise ValueError("max_depth_m must be positive")
        self.max_depth_m = max_depth_m
        sizes = [(calibration.width, calibration.height)]
        if calibration.depth_width:
            sizes.append((calibration.depth_width, calibration.depth_height))
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, *(x for x, _ in sizes))
        model.vis.global_.offheight = max(model.vis.global_.offheight, *(y for _, y in sizes))
        self._rgb_renderer = mujoco.Renderer(model, height=calibration.height, width=calibration.width)
        self._depth_renderer = None
        if sizes[-1] != sizes[0]:
            self._depth_renderer = mujoco.Renderer(model, height=sizes[-1][1], width=sizes[-1][0])

    def update_calibration(self, calibration: CameraCalibration):
        """Apply the current live frame's K, reallocating only for size changes."""
        if calibration == self.calibration:
            return
        old_sizes = (self.calibration.width, self.calibration.height,
                     self.calibration.depth_width, self.calibration.depth_height)
        new_sizes = (calibration.width, calibration.height,
                     calibration.depth_width, calibration.depth_height)
        if old_sizes == new_sizes:
            apply_calibration(self.model, self.camera_id, calibration)
            self.calibration = calibration
        else:
            model, camera_id, maximum, hidden = self.model, self.camera_id, self.max_depth_m, self.hidden_geom_groups
            self.close()
            self.__init__(model, calibration, camera_id, max_depth_m=maximum, hidden_geom_groups=hidden)

    def set_world_pose(self, data, world_from_camera: np.ndarray, *, move_body: bool = False):
        """Set CV optical pose; optionally move a static/mocap phone body too.

        The default sets the camera's local mount and does not move its parent
        robot. ``move_body=True`` moves a dedicated static or mocap phone body,
        preserving the camera mount. Jointed robot bodies cannot be teleported.
        """
        import mujoco
        desired = validate_transform(world_from_camera) @ CV_TO_MUJOCO_CAMERA
        mujoco.mj_forward(self.model, data)
        body_id = int(self.model.cam_bodyid[self.camera_id])
        if move_body and body_id != 0:
            mount = np.eye(4)
            mount[:3, 3] = self.model.cam_pos[self.camera_id]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, self.model.cam_quat[self.camera_id])
            mount[:3, :3] = rotation.reshape(3, 3)
            desired_body = desired @ np.linalg.inv(mount)
            mocap_id = int(self.model.body_mocapid[body_id])
            quat = np.empty(4)
            if mocap_id >= 0:
                data.mocap_pos[mocap_id] = desired_body[:3, 3]
                mujoco.mju_mat2Quat(quat, desired_body[:3, :3].flatten())
                data.mocap_quat[mocap_id] = quat
            else:
                if self.model.body_jntnum[body_id] != 0:
                    raise ValueError("move_body requires a dedicated static or mocap phone body")
                parent = int(self.model.body_parentid[body_id])
                parent_world = np.eye(4)
                parent_world[:3, :3] = data.xmat[parent].reshape(3, 3)
                parent_world[:3, 3] = data.xpos[parent]
                local = np.linalg.inv(parent_world) @ desired_body
                self.model.body_pos[body_id] = local[:3, 3]
                mujoco.mju_mat2Quat(quat, local[:3, :3].flatten())
                self.model.body_quat[body_id] = quat
        else:
            body_world = np.eye(4)
            body_world[:3, :3] = data.xmat[body_id].reshape(3, 3)
            body_world[:3, 3] = data.xpos[body_id]
            local = np.linalg.inv(body_world) @ desired
            self.model.cam_pos[self.camera_id] = local[:3, 3]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, local[:3, :3].flatten())
            self.model.cam_quat[self.camera_id] = quat
        mujoco.mj_forward(self.model, data)

    def render(self, data, *, timestamp: float | None = None, timestamp_clock: str | None = None) -> RGBDFrame:
        """Render same-state RGB and optical Z; clipped/no-return pixels are NaN."""
        import mujoco
        mujoco.mj_forward(self.model, data)
        self._rgb_renderer.disable_depth_rendering()
        self._rgb_renderer.update_scene(data, camera=self.camera_id, scene_option=self._scene_option)
        rgb = self._rgb_renderer.render().copy()
        renderer = self._depth_renderer or self._rgb_renderer
        # Native sensor width/height describes RGB; resizing the render viewport
        # preserves its full frustum. The scaled depth K uses pixel centers.
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=self.camera_id, scene_option=self._scene_option)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        far = float(self.model.vis.map.zfar * self.model.stat.extent)
        maximum = min(far*(1-1e-5), self.max_depth_m or np.inf)
        depth[(~np.isfinite(depth)) | (depth <= 0) | (depth >= maximum)] = np.nan
        pose = np.eye(4)
        pose[:3, :3] = data.cam_xmat[self.camera_id].reshape(3, 3)
        pose[:3, 3] = data.cam_xpos[self.camera_id]
        pose = pose @ CV_TO_MUJOCO_CAMERA
        return RGBDFrame(float(data.time if timestamp is None else timestamp), rgb, depth,
                         self.calibration, pose, "simulation", metadata={
                             "timestamp_clock": timestamp_clock or ("simulation" if timestamp is None else "caller_supplied"),
                             "camera_id": self.camera_id,
                             "depth_model": "ideal_optical_axis_meters",
                             "calibration_source": "configured_pinhole",
                             "physical_sensor_effects": "not_simulated",
                         })

    def close(self):
        self._rgb_renderer.close()
        if self._depth_renderer is not None:
            self._depth_renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
