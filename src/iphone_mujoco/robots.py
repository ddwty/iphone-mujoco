"""Robot descriptions, pinned Menagerie assets, and extensible MJCF loading.

UR5e and Panda use the unmodified Google DeepMind MuJoCo Menagerie asset files.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ASSETS = Path(__file__).parent / "assets" / "robots"
MENAGERIE_COMMIT = "cda5236f98383f6fbaa4820c6b18dba892cd087f"
MENAGERIE_REPOSITORY = "google-deepmind/mujoco_menagerie"


def default_model_cache() -> Path:
    """A writable local cache; preserve previous installs without redownloading."""
    override = os.environ.get("IPHONE_MUJOCO_MODEL_CACHE") or os.environ.get("ASMAGIC_MUJOCO_MODEL_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    canonical = Path(".cache/iphone_mujoco/menagerie")
    previous = Path(".cache") / ("asmagic" + "_mujoco") / "menagerie"
    return (previous if previous.is_dir() and not canonical.exists() else canonical).resolve()


@dataclass(frozen=True)
class RobotSpec:
    name: str
    model_path: Path | str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    site_name: str
    home_qpos: tuple[float, ...] = ()
    gripper_actuator_names: tuple[str, ...] = ()
    gripper_open: tuple[float, ...] = ()
    gripper_closed: tuple[float, ...] = ()
    description: str = "Custom MJCF robot"
    # For an upstream model lacking a TCP site, inject this explicit local frame.
    add_site_body: str | None = None
    add_site_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source: str = "custom"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path).expanduser().resolve())
        for key in ("joint_names", "actuator_names", "home_qpos", "gripper_actuator_names", "gripper_open", "gripper_closed", "add_site_pos"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if not self.joint_names or len(self.joint_names) != len(self.actuator_names):
            raise ValueError("Specify one position actuator for every arm joint")
        if len(set(self.joint_names)) != len(self.joint_names) or len(set(self.actuator_names)) != len(self.actuator_names):
            raise ValueError("Arm joint and actuator names must be unique")
        if self.home_qpos and len(self.home_qpos) != len(self.joint_names):
            raise ValueError("home_qpos must contain one angle per arm joint")
        if not (len(self.gripper_actuator_names) == len(self.gripper_open) == len(self.gripper_closed)):
            raise ValueError("Gripper actuator names and open/closed controls must match")

    @property
    def controller_kwargs(self) -> dict[str, Any]:
        result = {key: getattr(self, key) for key in ("joint_names", "actuator_names", "site_name", "gripper_actuator_names", "gripper_open", "gripper_closed")}
        result["home_qpos"] = self.home_qpos or None
        return result

    @classmethod
    def from_json(cls, path: Path | str) -> "RobotSpec":
        """Load a custom adapter; model_path resolves relative to this JSON file.

        Position servos must command joint coordinates with unit gear. Torque,
        velocity, coupled arm tendons, ball/free joints need a separate adapter.
        """
        path = Path(path).expanduser().resolve()
        payload = json.loads(path.read_text())
        payload["model_path"] = str(path.parent / payload["model_path"])
        return cls(**payload)


def list_robots() -> tuple[str, ...]:
    return ("panda", "ur5e")


def get_robot(name: str, model_cache: Path | str | None = None) -> RobotSpec:
    cache = Path(model_cache).expanduser().resolve() if model_cache is not None else default_model_cache()
    if name == "ur5e":
        return RobotSpec(name, cache / "universal_robots_ur5e/ur5e.xml",
                         ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
                         ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"), "attachment_site",
                         (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0),
                         description="MuJoCo Menagerie Universal Robots UR5e, bare flange (no gripper).",
                         source=f"menagerie@{MENAGERIE_COMMIT}")
    if name == "panda":
        return RobotSpec(name, cache / "franka_emika_panda/panda.xml",
                         tuple(f"joint{i}" for i in range(1, 8)), tuple(f"actuator{i}" for i in range(1, 8)),
                         "asmagic_tcp", (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853),
                         ("actuator8",), (255.0,), (0.0,),
                         "MuJoCo Menagerie Franka Emika Panda with parallel gripper; adapter TCP is 103.4 mm along hand +Z.",
                         add_site_body="hand", add_site_pos=(0.0, 0.0, 0.1034), source=f"menagerie@{MENAGERIE_COMMIT}")
    raise ValueError(f"Unknown robot {name!r}; choose from {', '.join(list_robots())}, or load RobotSpec.from_json()")


def _blob_sha(payload: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def _safe_destination(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if not relative or "\\" in relative or "\0" in relative or rel.is_absolute() or any(part in ("..", ".") for part in rel.parts):
        raise ValueError(f"Unsafe model asset path: {relative!r}")
    destination = root.joinpath(*rel.parts)
    if root.is_symlink() or any(parent.is_symlink() for parent in (destination, *destination.parents) if parent == root or root in parent.parents):
        raise ValueError(f"Model cache contains a symlink: {destination}")
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Model asset leaves cache directory")
    return destination


def fetch_menagerie(name: str, cache_dir: Path | str | None = None, *, workers: int = 4) -> Path:
    """Download a pinned, size/hash verified model, including its license.

    Resumes only checksum-valid local files. No archive extraction, executable
    code, unpinned branches, or writes through cache symlinks are used.
    Git object hashes come from the pinned upstream Git tree; they establish
    asset identity, not the physical accuracy of the original robot model.
    """
    if name not in ("ur5e", "panda"):
        raise ValueError("Menagerie models: ur5e, panda")
    manifest = json.loads((ASSETS / f"{name}_manifest.json").read_text())
    if manifest["commit"] != MENAGERIE_COMMIT or manifest["repository"] != MENAGERIE_REPOSITORY:
        raise ValueError("Unexpected Menagerie manifest provenance")
    cache = Path(cache_dir).expanduser().absolute() if cache_dir is not None else default_model_cache()
    root = _safe_destination(cache, manifest["directory"])
    root.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{MENAGERIE_REPOSITORY}/{MENAGERIE_COMMIT}/{manifest['directory']}/"

    def fetch(item: dict[str, Any]) -> None:
        destination = _safe_destination(root, item["path"])
        expected_size = item["size"]
        if not isinstance(expected_size, int) or not 0 <= expected_size <= 50_000_000:
            raise ValueError("Invalid model file size")
        if destination.is_file():
            payload = destination.read_bytes()
            if len(payload) == expected_size and _blob_sha(payload) == item["git_blob_sha1"]:
                return
        request = Request(base + item["path"], headers={"User-Agent": "iphone-mujoco/0.1"})
        with urlopen(request, timeout=60) as response:
            payload = response.read(expected_size + 1)
        if len(payload) != expected_size or _blob_sha(payload) != item["git_blob_sha1"]:
            raise RuntimeError(f"Upstream model checksum mismatch: {item['path']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                temporary.write(payload)
                temporary_path = Path(temporary.name)
            temporary_path.replace(destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        list(pool.map(fetch, manifest["files"]))
    return root


def _resolve_xml_paths(root: ET.Element, parent: Path) -> None:
    """Keep relative include, mesh and texture paths valid when loading a string."""
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    assetdir = compiler.get("assetdir", "")
    for key in ("meshdir", "texturedir", "assetdir"):
        if compiler.get(key):
            compiler.set(key, str((parent / compiler.get(key, "")).resolve()))
    if not compiler.get("meshdir"):
        compiler.set("meshdir", str((parent / assetdir).resolve()))
    if not compiler.get("texturedir"):
        compiler.set("texturedir", str((parent / assetdir).resolve()))
    for include in root.iter("include"):
        include.set("file", str((parent / include.attrib["file"]).resolve()))


def load_scene(robot: str | RobotSpec = "panda", phone_xml: str | None = None, *, model_cache: Path | str | None = None,
               phone_parent_body: str | None = None) -> tuple[mujoco.MjModel, mujoco.MjData, RobotSpec]:
    """Compile robot and optional phone body; initialize vendor home keyframe.

    Custom MJCF keeps its own scene. Built-in models receive a floor, lighting,
    and visible reference objects for RGB/depth experiments. Download Menagerie
    models explicitly with fetch_menagerie() before loading them. The default
    is the Menagerie Franka Panda. Set
    phone_parent_body to rigidly attach a camera body to a named robot link;
    its local transform then follows the link through normal forward kinematics.
    """
    spec = get_robot(robot, model_cache) if isinstance(robot, str) else robot
    if not spec.model_path.is_file():
        raise FileNotFoundError(f"Robot MJCF not found: {spec.model_path}. For official robots run: iphone-mujoco models --fetch {spec.name}")
    root = ET.parse(spec.model_path).getroot()
    _resolve_xml_paths(root, spec.model_path.parent)
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    if spec.add_site_body is not None:
        body = root.find(f".//body[@name='{spec.add_site_body}']")
        if body is None:
            raise ValueError(f"Missing TCP parent body: {spec.add_site_body}")
        ET.SubElement(body, "site", name=spec.site_name, pos=" ".join(map(str, spec.add_site_pos)), size="0.008", rgba="0.2 0.9 0.4 1")
    if spec.source != "custom":
        ET.SubElement(world, "geom", name="asmagic_floor", type="plane", size="3 3 .1", rgba=".35 .39 .42 1")
        ET.SubElement(world, "light", name="asmagic_light", pos="0 -2 3", dir="0 .5 -1", diffuse=".8 .8 .8")
        ET.SubElement(world, "geom", name="reference_cube", type="box", pos=".5 .3 .05", size=".05 .05 .05", rgba=".8 .25 .12 1")
        ET.SubElement(world, "geom", name="reference_sphere", type="sphere", pos=".4 -.25 .05", size=".05", rgba=".12 .45 .8 1")
    if phone_xml:
        phone = ET.fromstring(phone_xml)
        if phone.tag == "body":
            parent = world
            if phone_parent_body is not None:
                parent = next((body for body in world.iter("body") if body.get("name") == phone_parent_body), None)
                if parent is None:
                    raise ValueError(f"Missing camera parent body: {phone_parent_body}")
            parent.append(phone)
        elif phone.tag == "mujoco":
            parent = world
            if phone_parent_body is not None:
                parent = next((body for body in world.iter("body") if body.get("name") == phone_parent_body), None)
                if parent is None:
                    raise ValueError(f"Missing camera parent body: {phone_parent_body}")
            for child in phone:
                if child.tag == "worldbody":
                    if phone_parent_body is not None and any(item.tag != "body" for item in child):
                        raise ValueError("Mounted phone module worldbody must contain only rigid bodies")
                    parent.extend(child)
                else:
                    existing = root.find(child.tag)
                    if existing is not None:
                        existing.extend(child)
                    else:
                        root.append(child)
        else:
            raise ValueError("phone_xml must contain a body or mujoco root")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home)
    if spec.home_qpos:
        for joint_name, position in zip(spec.joint_names, spec.home_qpos):
            joint = model.joint(joint_name)
            if model.jnt_type[joint.id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError("Only scalar hinge/slide arm joints are supported")
            data.qpos[joint.qposadr[0]] = position
        for actuator_name, position in zip(spec.actuator_names, spec.home_qpos):
            data.ctrl[model.actuator(actuator_name).id] = position
    for actuator_name, position in zip(spec.gripper_actuator_names, spec.gripper_open):
        data.ctrl[model.actuator(actuator_name).id] = position
    mujoco.mj_forward(model, data)
    # Validate all adapter names and control assumptions before any experiment.
    from .control import CartesianTeleoperator
    CartesianTeleoperator(model, data, **spec.controller_kwargs)
    return model, data, spec
