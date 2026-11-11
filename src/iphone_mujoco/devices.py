"""Sourced phone dimensions, nominal optics and reusable MuJoCo appearance.

The phone body frame has +X toward screen-right, +Y toward the top and +Z out
of the screen. Camera transforms use CV axes. Camera XY follows Apple's lens
cover centers; camera Z uses the cover plane until measured extrinsics replace
it. The catalogue records which geometry and dynamics quantities are estimates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any
import json
import xml.etree.ElementTree as ET

import numpy as np

from .geometry import CV_TO_ARKIT


@lru_cache(maxsize=1)
def _catalog() -> dict[str, dict[str, Any]]:
    path = files("iphone_mujoco").joinpath("assets/devices/iphone_catalog.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported iPhone hardware catalog schema")
    return {device["id"]: device for device in document["devices"]}


@dataclass(frozen=True)
class PhoneHardware:
    """One catalogue device; :meth:`to_dict` returns an independent data copy."""

    _values: dict[str, Any]

    @property
    def id(self) -> str:
        return self._values["id"]

    @property
    def name(self) -> str:
        return self._values["name"]

    @property
    def width_m(self) -> float:
        return self._values["body_whd_mm"][0] / 1000

    @property
    def height_m(self) -> float:
        return self._values["body_whd_mm"][1] / 1000

    @property
    def thickness_m(self) -> float:
        return self._values["body_whd_mm"][2] / 1000

    def mass_kg(self, region: str = "us") -> float:
        """Return Apple's nominal bare-phone mass for the selected region."""
        try:
            return self._values["mass_variants"][region]["grams"] / 1000
        except (KeyError, TypeError):
            choices = ", ".join(self._values["mass_variants"])
            raise ValueError(f"Unsupported mass region {region!r}; choose {choices}") from None

    def phone_from_camera(self, lens: str) -> np.ndarray:
        """Return body <- CV camera, using the selected rear cover center.

        The glass-plane position is a geometric mount estimate, not a measured
        optical entrance pupil. Cropped zoom modes use their physical lens.
        """
        try:
            x_mm, y_mm = self._values["rear_lens_xy_mm"][lens]
        except (KeyError, TypeError):
            raise ValueError("lens must be wide, ultra_wide or telephoto") from None
        result = CV_TO_ARKIT.copy()
        result[:3, 3] = [self.width_m / 2 - x_mm / 1000,
                         self.height_m / 2 - y_mm / 1000,
                         -self.thickness_m / 2 - self._values["lens_cover_surface_rise_mm"] / 1000]
        return result

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._values)


def device_ids() -> tuple[str, ...]:
    """Return supported canonical model identifiers in catalogue order."""
    return tuple(_catalog())


def get_device(identifier: str) -> PhoneHardware:
    try:
        values = _catalog()[identifier]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown iPhone model {identifier!r}; choose {', '.join(device_ids())}") from None
    return PhoneHardware(deepcopy(values))


def _numbers(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in np.asarray(values).ravel())


def _outline(width: float, height: float, corner: np.ndarray,
             bottom_corner: np.ndarray | None = None) -> np.ndarray:
    """Mirror supplied top-left ordinates into a clockwise convex outline."""
    bottom = corner if bottom_corner is None else bottom_corner
    return np.concatenate([
        corner * [1, -1] + [-width / 2, height / 2],
        corner[::-1] * [-1, -1] + [width / 2, height / 2],
        bottom * [-1, 1] + [width / 2, -height / 2],
        bottom[::-1] + [-width / 2, -height / 2],
    ])


def _circle_corner(radius: float) -> np.ndarray:
    angle = np.linspace(0, np.pi / 2, 13)
    return radius * np.column_stack((1 - np.cos(angle), 1 - np.sin(angle)))


def _mesh(asset: ET.Element, name: str, outline: np.ndarray, half_depth: float):
    # The outline is convex; MuJoCo builds its hull from these original vertices.
    vertices = np.concatenate([np.column_stack((outline, np.full(len(outline), z)))
                               for z in (-half_depth, half_depth)])
    ET.SubElement(asset, "mesh", name=name, vertex=_numbers(vertices))


def phone_geometry(device: str | PhoneHardware, *, body_name: str = "iphone",
                   region: str = "us", payload: bool = False,
                   visual_group: int = 2) -> tuple[ET.Element, ET.Element]:
    """Return ``(asset, body)`` for a selected phone, without adding a camera.

    Append ``asset`` at the MJCF root and ``body`` at its desired mount. Body
    outlines interpolate Apple's corner ordinates; edge bevels and the rear
    plateau contour are approximations. Geoms have zero mass and no contact.
    ``payload=True`` adds nominal phone mass with a body-centered uniform-box
    inertia estimate. It does not include a case, mount or measured mass tensor.
    """
    if isinstance(device, str):
        device = get_device(device)
    if not isinstance(device, PhoneHardware):
        raise TypeError("device must be a model identifier or PhoneHardware")
    if not isinstance(body_name, str) or not body_name.strip():
        raise ValueError("body_name must be nonempty")
    if type(visual_group) is not int or not 0 <= visual_group <= 5:
        raise ValueError("visual_group must be an integer from 0 through 5")
    if type(payload) is not bool:
        raise ValueError("payload must be a boolean")
    mass = device.mass_kg(region)
    values = device.to_dict()
    width, height, depth = device.width_m, device.height_m, device.thickness_m
    corner = np.asarray(values["corner_profile_xy_mm"], dtype=float) / 1000
    outline = _outline(width, height, corner)
    asset = ET.Element("asset")
    body = ET.Element("body", name=body_name)
    body.append(ET.Comment("Body outline: sourced corner ordinates with piecewise-linear interpolation; edge bevel and plateau contour approximate."))
    if payload:
        body.append(ET.Comment(f"Nominal bare-phone mass ({region}); body-centered uniform-box inertia estimate."))
        inertia = mass / 12 * np.array([height**2 + depth**2, width**2 + depth**2,
                                        width**2 + height**2])
        ET.SubElement(body, "inertial", pos="0 0 0", mass=_numbers([mass]),
                      diaginertia=_numbers(inertia))

    common = {"mass": "0", "density": "0", "contype": "0", "conaffinity": "0",
              "group": str(visual_group)}

    def geom(suffix: str, **attributes):
        return ET.SubElement(body, "geom", name=body_name + "_" + suffix,
                             **common, **attributes)

    _mesh(asset, body_name + "_shell_mesh", outline, depth / 2)
    geom("shell", type="mesh", mesh=body_name + "_shell_mesh", rgba="0.42 0.44 0.47 1")
    # Keep explicit box extents available to generic mount builders.
    geom("shell_bounds", type="box", size=_numbers([width / 2, height / 2, depth / 2]),
         rgba="0 0 0 0")

    screen_width, screen_height = np.asarray(values["screen_wh_mm"]) / 1000
    screen_outline = outline * [screen_width / width, screen_height / height]
    _mesh(asset, body_name + "_screen_mesh", screen_outline, .00005)
    geom("screen", type="mesh", mesh=body_name + "_screen_mesh",
         pos=_numbers([0, 0, depth / 2 + .00005]), rgba="0.018 0.026 0.04 1")

    left, top, right, bottom = np.asarray(values["plateau_rear_bounds_mm"]) / 1000
    plateau_width, plateau_height = right - left, bottom - top
    radius = values["plateau_bottom_corner_radius_mm"] / 1000
    lower_corner = _circle_corner(radius)
    upper_corner = corner if left == 0 and top == 0 else lower_corner
    plateau_outline = _outline(plateau_width, plateau_height, upper_corner, lower_corner)
    rise = values["camera_plateau_rise_mm"] / 1000
    _mesh(asset, body_name + "_plateau_mesh", plateau_outline, rise / 2)
    geom("plateau", type="mesh", mesh=body_name + "_plateau_mesh",
         pos=_numbers([width / 2 - (left + right) / 2, height / 2 - (top + bottom) / 2,
                       -depth / 2 - rise / 2]), rgba="0.36 0.38 0.41 1")

    cover_rise = values["lens_cover_surface_rise_mm"] / 1000
    lens_height = cover_rise - rise
    lens_radius = values["rear_lens_outer_diameter_mm"] / 2000
    for lens in values["rear_lens_xy_mm"]:
        position = device.phone_from_camera(lens)[:3, 3]
        rim_position = position.copy()
        rim_position[2] += lens_height / 2
        geom(lens + "_rim", type="cylinder", pos=_numbers(rim_position),
             size=_numbers([lens_radius, lens_height / 2]), rgba="0.16 0.17 0.19 1")
        cover_position = position.copy()
        cover_position[2] += .00005
        geom(lens + "_lens", type="cylinder", pos=_numbers(cover_position),
             size=_numbers([lens_radius * .90, .00005]), rgba="0.018 0.045 0.072 1")

    for feature, color in (("flash", "0.88 0.88 0.78 1"), ("lidar", "0.025 0.025 0.03 1")):
        if f"rear_{feature}_xy_mm" in values:
            x_mm, y_mm = values[f"rear_{feature}_xy_mm"]
            radius = values[f"rear_{feature}_outer_diameter_mm"] / 2000
            geom(feature, type="cylinder", pos=_numbers([width / 2 - x_mm / 1000,
                 height / 2 - y_mm / 1000, -depth / 2 - rise - .00001]),
                 size=_numbers([radius, .00001]), rgba=color)
    return asset, body
