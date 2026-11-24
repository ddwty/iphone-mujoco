"""asDevice placement and asTriad decorations for MuJoCo camera experiments.

Install the spatial extra to use this module. Model geometry, image calibration
and tracking remain separate: asDevice resolves nominal local reference frames;
MuJoCo (or a caller with a tracked pose) supplies the world placement.
"""
from __future__ import annotations

import numpy as np

from .devices import get_device
from .geometry import validate_transform


_MODELS = {
    "iphone-15-pro": ("iPhone15Pro", 1),
    "iphone-15-pro-max": ("iPhone15ProMax", 1),
    "iphone-16-pro": ("iPhone16Pro", 1),
    "iphone-16-pro-max": ("iPhone16ProMax", 1),
    "iphone-17-pro": ("iPhone17Pro", 5),
    "iphone-17-pro-max": ("iPhone17ProMax", 1),
}


class DevicePlacement:
    """Bridge the package's centred phone body to asDevice's screen-plane 01.

    Phone geometry uses portrait screen-right/up/out axes at the body midplane.
    asDevice 01 has the same axes at the front glass, half a thickness away.
    The selected optical mount is an explicit input from the camera configuration:
    a nominal cover-centre estimate, or a caller-supplied measured transform.
    Mechanical asDevice component references are never treated as optical K or
    a measured entrance pupil.
    """

    def __init__(self, configuration):
        try:
            from asDevice import DeviceFrames, Transform
        except ImportError as error:
            raise ImportError("Install iphone-mujoco[spatial] to use asDevice placement") from error
        self._transform = Transform
        model, revision = _MODELS[configuration.device_id]
        self.frames = DeviceFrames(model, orientation="portrait", revision=revision)
        self.device_id = configuration.device_id
        self.lens = configuration.lens
        hardware = get_device(configuration.device_id)
        self.body_from_01 = Transform.from_translation(0, 0, hardware.thickness_m / 2)
        measured = configuration.metadata.get("phone_from_camera")
        mount = hardware.phone_from_camera(self.lens) if measured is None else measured
        self._01_from_camera = self.body_from_01.inverse() @ self._pose(mount)
        self.mount_source = "catalog_cover_center_estimate" if measured is None else "configuration_phone_from_camera"

    def _pose(self, matrix):
        return self._transform(validate_transform(np.asarray(matrix, dtype=float)))

    def world_from_device(self, world_from_camera) -> np.ndarray:
        """Recover the current screen-plane 01 pose from an optical CV pose."""
        pose = self._pose(world_from_camera) @ self._01_from_camera.inverse()
        return np.asarray(pose.matrix)

    def world_from_camera(self, world_from_device) -> np.ndarray:
        """Place the selected optical camera from a supplied asDevice 01 pose."""
        return np.asarray((self._pose(world_from_device) @ self._01_from_camera).matrix)

    def world_from_frame(self, world_from_camera, name: str) -> np.ndarray:
        """Locate a registered asDevice frame using the current optical pose."""
        pose = self._pose(self.world_from_device(world_from_camera))
        return np.asarray((pose @ self.frames.transform(target="01", source=name)).matrix)

    def world_from_body(self, world_from_device) -> np.ndarray:
        """Place the existing centred MJCF shell without changing its geometry."""
        return np.asarray((self._pose(world_from_device) @ self.body_from_01.inverse()).matrix)

    def place_camera(self, camera, data, world_from_device):
        """Move a dedicated static/mocap phone via asDevice, preserving lens K."""
        camera.set_world_pose(data, self.world_from_camera(world_from_device), move_body=True)

    def describe(self, world_from_camera, *, pose_source: str) -> dict:
        """JSON-safe supplemental provenance; existing episode fields stay intact."""
        world_from_01 = self._pose(self.world_from_device(world_from_camera))
        world_from_03 = world_from_01 @ self.frames.transform(target="01", source="03")
        return {
            "provider": "asDevice", "model_key": self.frames.model_key,
            "definition_revision": self.frames.definition_revision,
            "orientation": "portrait", "unit": "m", "pose_source": pose_source,
            "world_axes": "MuJoCo Z up", "optical_axes": "X right, Y down, Z forward",
            "camera_mount_source": self.mount_source, "lens": self.lens,
            "world_from_01": world_from_01.to_list(),
            "world_from_03": world_from_03.to_list(),
            "world_from_optical": self._pose(world_from_camera).to_list(),
        }


class SpatialOverlay:
    """Append actual asTriad world/device/optical frames to overview scenes.

    No geoms are added to the physical model or sensor scenes. Call after scene
    assembly/frustum generation; the overlay preserves existing visual geoms.
    """

    def __init__(self, configuration):
        try:
            from asTriad import asTriad
        except ImportError as error:
            raise ImportError("Install iphone-mujoco[spatial] to render asTriad frames") from error
        self.placement = DevicePlacement(configuration)
        self.triads = (
            asTriad(size=35, name="W"),
            asTriad(size=24, name="D01"),
            asTriad(size=12, name="C"),
        )

    @property
    def geom_count(self):
        return sum(triad.geom_count for triad in self.triads)

    def append(self, scene, frame):
        if scene.ngeom + self.geom_count > scene.maxgeom:
            raise ValueError(f"Spatial overlay needs {self.geom_count} free visual geometry slots")
        if frame.world_from_camera is None:
            raise ValueError("Spatial overlay requires a known optical world pose")
        poses = (np.eye(4), self.placement.world_from_device(frame.world_from_camera), frame.world_from_camera)
        for triad, pose in zip(self.triads, poses):
            triad.set_pose(pose[:3, 3], pose[:3, :3])
            triad.append_to_scene(scene)
        return self.geom_count

    def annotate(self, frame):
        frame.metadata["spatial_frames"] = self.placement.describe(
            frame.world_from_camera, pose_source="simulation" if frame.source == "simulation" else frame.source)
