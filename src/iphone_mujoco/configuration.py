"""Select a documented iPhone camera or bind its captured intrinsics.

Apple's equivalent focal lengths describe a field of view, not factory pixel
intrinsics. The public-specification projection is an ideal pinhole model.
Captured intrinsics keep their lens and zoom identity when resized.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .camera import CameraCalibration
from .devices import get_device


_SCHEMA = "iphone_mujoco.camera_configuration"
_SCHEMA_VERSION = 1
_FULL_FRAME_DIAGONAL_MM = math.hypot(36, 24)


def _dimensions(value, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain width and height")
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in value):
        raise ValueError(f"{name} dimensions must be positive integers")
    return int(value[0]), int(value[1])


def _zoom(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("zoom must be a positive finite number")
    return float(value)


def _selection(device: str, lens: str | None, zoom: float | None, region: str):
    details = get_device(device).to_dict()
    if not isinstance(region, str) or region not in details["mass_variants"]:
        regions = ", ".join(details["mass_variants"])
        raise ValueError(f"Unsupported region {region!r}; choose {regions}")
    mass = details["mass_variants"][region]["grams"]
    if isinstance(mass, bool) or not isinstance(mass, Real) or not math.isfinite(mass) or mass <= 0:
        raise ValueError(f"No published mass for {details['id']} in region {region}")
    modes = details["zoom_modes"]
    if zoom is not None:
        zoom = _zoom(zoom)
        candidates = [mode for mode in modes if math.isclose(float(mode["display_zoom"]), zoom,
                                                            rel_tol=0., abs_tol=1e-9)]
        if lens is not None:
            candidates = [mode for mode in candidates if mode["lens"] == lens]
    else:
        lens = "wide" if lens is None else lens
        candidates = [mode for mode in modes if mode["lens"] == lens and not mode["crop"]]
    if len(candidates) != 1:
        raise ValueError(f"Unsupported lens/zoom combination for {details['id']}: lens={lens!r}, zoom={zoom!r}")
    return details, deepcopy(candidates[0])


@dataclass
class CameraConfiguration:
    device_id: str
    calibration: CameraCalibration
    calibration_source: str
    lens: str = "wide"
    zoom: float = 1.
    region: str = "us"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        details, mode = _selection(self.device_id, self.lens, self.zoom, self.region)
        self.device_id = details["id"]
        self.zoom = float(mode["display_zoom"])
        if not isinstance(self.calibration, CameraCalibration):
            raise ValueError("calibration must be CameraCalibration")
        if not isinstance(self.calibration_source, str) or not self.calibration_source.strip():
            raise ValueError("calibration_source must be a nonempty string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        self.metadata = deepcopy(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "device_id": self.device_id,
            "lens": self.lens,
            "zoom": self.zoom,
            "region": self.region,
            "calibration": self.calibration.to_dict(),
            "calibration_source": self.calibration_source,
            "metadata": deepcopy(self.metadata),
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def _from_dict(cls, values: dict[str, Any]) -> "CameraConfiguration":
        values = deepcopy(values)
        if values.pop("schema", None) != _SCHEMA or values.pop("schema_version", None) != _SCHEMA_VERSION:
            raise ValueError("Unsupported camera configuration schema")
        try:
            values["calibration"] = CameraCalibration(**values["calibration"])
            return cls(**values)
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid camera configuration") from error

    @classmethod
    def load(cls, path: str | Path) -> "CameraConfiguration":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Camera configuration must be a JSON object")
        return cls._from_dict(values)


def _profile(value):
    if isinstance(value, (CameraConfiguration, CameraCalibration)):
        return value, {}
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("profile must be a configuration, calibration or JSON path")
    if value.get("schema") == _SCHEMA or "device_id" in value:
        return CameraConfiguration._from_dict(value), {}
    try:
        # Legacy DeviceProfile files carry capture provenance around a bare K.
        if "calibration" in value:
            if value.get("schema_version") != 1:
                raise ValueError("Unsupported device profile schema")
            return CameraCalibration(**value["calibration"]), {
                "input_profile_source": value.get("source"),
                "input_profile_metadata": deepcopy(value.get("metadata", {})),
            }
        return CameraCalibration(**value), {}
    except TypeError as error:
        raise ValueError("Invalid calibration profile") from error


def _resize(calibration: CameraCalibration, resolution, depth_resolution) -> CameraCalibration:
    width, height = _dimensions(resolution, "resolution")
    if width * calibration.height != height * calibration.width:
        raise ValueError("A captured profile permits only same-aspect full-raster resizing; crop calibration must be captured separately")
    resized = calibration.scaled(width, height)
    depth = (None, None) if depth_resolution is None else _dimensions(depth_resolution, "depth_resolution")
    return CameraCalibration(width, height, resized.fx, resized.fy, resized.cx, resized.cy, *depth)


def configure_camera(device: str = "iphone-17-pro", *, lens: str | None = None,
                     zoom: float | None = None, resolution=(640, 480),
                     depth_resolution=(256, 192), region: str = "us",
                     profile=None) -> CameraConfiguration:
    """Choose a published lens preset, or use K bound to the same identity.

    A saved configuration retains its lens/zoom when those arguments are omitted.
    A bare K is explicitly user-bound to the selected device and camera; its
    origin is not promoted to a measured asMagic calibration. Depth dimensions
    describe a registered simulation raster, not hardware depth availability.
    """
    width, height = _dimensions(resolution, "resolution")
    depth = (None, None) if depth_resolution is None else _dimensions(depth_resolution, "depth_resolution")
    supplied, extra = _profile(profile) if profile is not None else (None, {})
    if isinstance(supplied, CameraConfiguration):
        lens = supplied.lens if lens is None else lens
        zoom = supplied.zoom if zoom is None else zoom
    details, mode = _selection(device, lens, zoom, region)
    selected_lens, selected_zoom = mode["lens"], float(mode["display_zoom"])
    if isinstance(supplied, CameraConfiguration):
        if (details["id"], selected_lens, selected_zoom) != (supplied.device_id, supplied.lens, supplied.zoom):
            raise ValueError("A camera profile must keep its exact device, lens and zoom identity")
        calibration = _resize(supplied.calibration, (width, height), depth_resolution)
        metadata = deepcopy(supplied.metadata)
        if calibration != supplied.calibration:
            metadata["resampling"] = {
                "operation": "full_raster_pixel_center_resize",
                "source_resolution": [supplied.calibration.width, supplied.calibration.height],
                "target_resolution": [width, height],
                "target_depth_resolution": list(depth) if depth_resolution is not None else None,
            }
        return CameraConfiguration(details["id"], calibration, supplied.calibration_source,
                                   selected_lens, selected_zoom, region, metadata)
    if isinstance(supplied, CameraCalibration):
        calibration = _resize(supplied, (width, height), depth_resolution)
        metadata = {**extra, "identity_source": "user_supplied", "depth_model": "registered_optical_axis_meters"}
        return CameraConfiguration(details["id"], calibration, "user_supplied_intrinsics",
                                   selected_lens, selected_zoom, region, metadata)

    focal = float(mode["equivalent_focal_mm"]) * math.hypot(width, height) / _FULL_FRAME_DIAGONAL_MM
    calibration = CameraCalibration(width, height, focal, focal, (width - 1) / 2, (height - 1) / 2, *depth)
    metadata = {
        "projection_model": "ideal_pinhole",
        "projection_reference": "35mm_full_frame_diagonal",
        "reference_diagonal_mm": _FULL_FRAME_DIAGONAL_MM,
        "camera_mode": mode,
        "specification_sources": {
            "mechanical_drawing": details["drawing_url"],
            "optics": details["optics_source_url"],
            "camera_mode": mode.get("source_url", details["optics_source_url"]),
            "mass": details["mass_variants"][region]["source_url"],
        },
        "depth_model": "ideal_registered_optical_axis_meters",
        "depth_source": "simulation" if depth_resolution is not None else "none",
        "hardware_depth_verified": False,
        "limitations": [
            "Equivalent focal length defines an ideal field of view, not measured pixel intrinsics.",
            "Distortion, stabilization crop and focus-dependent intrinsics are not modeled.",
            "Simulated depth does not establish LiDAR support for the selected lens or zoom.",
        ],
    }
    return CameraConfiguration(details["id"], calibration, "public_specs_ideal_pinhole",
                               selected_lens, selected_zoom, region, metadata)


def bind_runtime_calibration(calibration: CameraCalibration, *, device_id: str,
                             lens: str = "wide", zoom: float = 1.,
                             region: str = "us", metadata=None) -> CameraConfiguration:
    """Bind AR-stream K to a user-identified phone, keeping capture metadata.

    Current asMagic AR streams do not publish model/lens/zoom identity. This
    helper accepts the AR main-camera path only; it never derives other zooms
    or lenses from those intrinsics.
    """
    if lens != "wide" or _zoom(zoom) != 1.:
        raise ValueError("asMagic AR runtime calibration can only bind the wide lens at 1x")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    capture = deepcopy(metadata or {})
    capture["identity_source"] = "user_supplied_ar_main_camera"
    capture.setdefault("depth_model", "registered_optical_axis_meters")
    return CameraConfiguration(device_id, calibration, "runtime_asmagic_frame",
                               lens, zoom, region, capture)
