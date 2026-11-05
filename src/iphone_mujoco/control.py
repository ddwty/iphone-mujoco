"""Bounded Cartesian servo for simulation, driven by locally received phone poses.

This is Jacobian damped least-squares IK with position servos. Joint and target
limits are enforced; collision avoidance and hardware control are not provided.
Call update() and step() from the same simulation thread, then mujoco.mj_step().
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import mujoco
import numpy as np

from .geometry import ARKIT_WORLD_TO_MUJOCO


@dataclass(frozen=True)
class ControlStatus:
    mode: str
    input_age: float | None = None
    position_error: float = 0.0
    rotation_error: float = 0.0
    joint_limit_active: bool = False

    @property
    def state(self) -> str:
        return self.mode


def _pose(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Phone pose must be a finite 4x4 rigid transform")
    rotation = value[:3, :3]
    if not np.allclose(value[3], (0, 0, 0, 1), atol=1e-5) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError("Phone pose must contain an orthonormal proper rotation")
    return value.copy()


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Shortest SO(3) logarithm, including rotations close to pi."""
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=float).reshape(9))
    if quaternion[0] < 0:
        quaternion *= -1
    sine = np.linalg.norm(quaternion[1:])
    if sine < 1e-10:
        return 2 * quaternion[1:]
    return quaternion[1:] * (2 * np.arctan2(sine, max(0.0, quaternion[0])) / sine)


def _bounded(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector * min(1.0, maximum / max(norm, 1e-12))


class CartesianTeleoperator:
    """Map phone displacement to TCP displacement with clutch/reconnect anchors.

    phone_to_world maps phone-world vectors to robot-world vectors. By default
    ARKit becomes the app's robotics basis: (x, y, z) -> (-z, -x, y).
    Pass np.eye(3) when the transport already supplies MuJoCo-world poses.
    The first sample after release, stale input, session change, or reset only
    establishes an anchor. receive_time/now use the local monotonic clock, never
    a phone wall clock or AR timestamp. gripper=0 opens, gripper=1 closes.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 joint_names: Sequence[str], actuator_names: Sequence[str], site_name: str,
                 *, home_qpos: Sequence[float] | None = None,
                 gripper_actuator_names: Sequence[str] = (),
                 gripper_open: Sequence[float] = (), gripper_closed: Sequence[float] = (),
                 phone_to_world: np.ndarray | None = None, translation_scale: float = 1.0,
                 stale_timeout: float = 0.35, joint_velocity_limit: float = 1.0,
                 max_linear_velocity: float = 0.25, max_angular_velocity: float = 1.0,
                 workspace_radius: float = 0.5, damping: float = 0.06,
                 position_gain: float = 5.0, rotation_gain: float = 4.0,
                 gripper_rate: float = 2.0, max_input_jump: float = 0.5,
                 max_rotation_jump: float = 1.5):
        self.model, self.data = model, data
        if not joint_names or len(joint_names) != len(actuator_names):
            raise ValueError("Specify one position actuator per arm joint")
        self.joint_ids = np.array([model.joint(name).id for name in joint_names], dtype=int)
        self.actuator_ids = np.array([model.actuator(name).id for name in actuator_names], dtype=int)
        self.site_id = model.site(site_name).id
        if len(set(self.joint_ids)) != len(self.joint_ids) or len(set(self.actuator_ids)) != len(self.actuator_ids):
            raise ValueError("Duplicate arm joints or actuators")
        self.qpos_ids = model.jnt_qposadr[self.joint_ids].copy()
        self.dof_ids = model.jnt_dofadr[self.joint_ids].copy()
        if any(model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for j in self.joint_ids):
            raise ValueError("Only scalar hinge/slide arm joints are supported")
        for joint, actuator in zip(self.joint_ids, self.actuator_ids):
            if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT or model.actuator_trnid[actuator, 0] != joint:
                raise ValueError("Each arm actuator must directly drive its configured joint")
            if not np.allclose(model.actuator_gear[actuator], (1, 0, 0, 0, 0, 0)):
                raise ValueError("Arm position actuators must have unit gear")
            gain = model.actuator_gainprm[actuator, 0]
            bias = model.actuator_biasprm[actuator]
            if (model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE or gain <= 0
                    or not np.isclose(bias[1], -gain) or bias[2] > 0):
                raise ValueError("Arm actuators must be affine position servos, not torque/velocity motors")
        self.gripper_ids = np.array([model.actuator(name).id for name in gripper_actuator_names], dtype=int)
        self.gripper_open = np.asarray(gripper_open, dtype=float)
        self.gripper_closed = np.asarray(gripper_closed, dtype=float)
        if len(self.gripper_ids) != len(self.gripper_open) or self.gripper_open.shape != self.gripper_closed.shape:
            raise ValueError("Gripper actuator names and endpoint controls must match")
        if not np.isfinite(self.gripper_open).all() or not np.isfinite(self.gripper_closed).all():
            raise ValueError("Gripper controls must be finite")
        if len(set(self.gripper_ids)) != len(self.gripper_ids) or set(self.gripper_ids) & set(self.actuator_ids):
            raise ValueError("Gripper actuator names must be unique and separate from arm actuators")
        self.lower = np.where(model.jnt_limited[self.joint_ids], model.jnt_range[self.joint_ids, 0], -np.inf)
        self.upper = np.where(model.jnt_limited[self.joint_ids], model.jnt_range[self.joint_ids, 1], np.inf)
        limited = model.actuator_ctrllimited[self.actuator_ids].astype(bool)
        self.lower = np.maximum(self.lower, np.where(limited, model.actuator_ctrlrange[self.actuator_ids, 0], -np.inf))
        self.upper = np.minimum(self.upper, np.where(limited, model.actuator_ctrlrange[self.actuator_ids, 1], np.inf))
        if np.any(self.lower >= self.upper):
            raise ValueError("Joint and actuator control limits do not overlap")
        self.phone_to_world = np.asarray(phone_to_world if phone_to_world is not None else ARKIT_WORLD_TO_MUJOCO[:3, :3], dtype=float)
        if self.phone_to_world.shape != (3, 3) or not np.isfinite(self.phone_to_world).all() or not np.allclose(self.phone_to_world.T @ self.phone_to_world, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(self.phone_to_world), 1):
            raise ValueError("phone_to_world must be a proper rotation matrix")
        parameters = dict(translation_scale=translation_scale, stale_timeout=stale_timeout, joint_velocity_limit=joint_velocity_limit,
                          max_linear_velocity=max_linear_velocity, max_angular_velocity=max_angular_velocity,
                          workspace_radius=workspace_radius, damping=damping, position_gain=position_gain,
                          rotation_gain=rotation_gain, gripper_rate=gripper_rate, max_input_jump=max_input_jump,
                          max_rotation_jump=max_rotation_jump)
        for name, value in parameters.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, name, float(value))
        self.speed_scale = 1.0
        self.home_qpos = np.asarray(home_qpos if home_qpos is not None else data.qpos[self.qpos_ids], dtype=float).copy()
        if self.home_qpos.shape != self.qpos_ids.shape or not np.isfinite(self.home_qpos).all() or np.any(self.home_qpos < self.lower) or np.any(self.home_qpos > self.upper):
            raise ValueError("home_qpos must be within arm joint limits")
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._session_id: str | None = None
        self._last_receive: float | None = None
        self._last_step: float | None = None
        self._last_pose: np.ndarray | None = None
        self._phone_anchor: np.ndarray | None = None
        self._tcp_anchor: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self._command = np.clip(data.qpos[self.qpos_ids].copy(), self.lower, self.upper)
        self._gripper_target = 0.0
        self._gripper_command = 0.0
        self._holding = False
        self.status = ControlStatus("waiting")

    def tcp_pose(self) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        result[:3, 3] = self.data.site_xpos[self.site_id]
        return result

    def hold(self, reanchor: bool = True, *, reason: str = "held") -> None:
        # Freeze once on entry. Repeated clutch packets or stale simulation
        # steps must not chase gravity-induced servo error by recapturing qpos.
        if not self._holding:
            self._command = np.clip(self.data.qpos[self.qpos_ids].copy(), self.lower, self.upper)
            self._holding = True
        self.data.ctrl[self.actuator_ids] = self._command
        if reanchor:
            self._phone_anchor = self._tcp_anchor = self.target_pose = None
            self._last_pose = None
        self.status = ControlStatus(reason)

    def reset(self) -> None:
        """Re-anchor the next input without resetting simulation state or time."""
        self.hold(reason="waiting")
        self._last_receive = None
        self._last_step = None

    def change_speed(self, factor: float) -> float:
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Speed factor must be positive and finite")
        self.speed_scale = float(np.clip(factor, 0.05, 3.0))
        return self.speed_scale

    def update(self, pose4x4: np.ndarray, receive_time: float | None = None,
               gripper: float | None = None, session_id: str | None = None,
               reset: bool = False, clutch: bool = False) -> bool:
        pose = _pose(pose4x4)
        received = time.monotonic() if receive_time is None else float(receive_time)
        if not np.isfinite(received):
            raise ValueError("receive_time must be finite local monotonic time")
        if gripper is not None and not np.isfinite(gripper):
            raise ValueError("Gripper must be finite")
        if self._last_receive is not None and received < self._last_receive:
            return False
        session_changed = session_id != self._session_id
        gap = self._last_receive is not None and received - self._last_receive > self.stale_timeout
        if reset or session_changed or gap:
            self.hold(reason="waiting")
        self._session_id = session_id
        self._last_receive = received
        if clutch:
            self.hold(reason="clutched")
            return False
        if self._last_pose is not None:
            jump = np.linalg.norm(pose[:3, 3] - self._last_pose[:3, 3])
            angle = np.linalg.norm(rotation_vector(pose[:3, :3] @ self._last_pose[:3, :3].T))
            if jump > self.max_input_jump or angle > self.max_rotation_jump:
                self.hold(reason="discontinuity")
                return False
        self._last_pose = pose
        if self._phone_anchor is None:
            self._phone_anchor = pose.copy()
            self._tcp_anchor = self.tcp_pose()
            self.target_pose = self._tcp_anchor.copy()
            self._command = np.clip(self.data.qpos[self.qpos_ids].copy(), self.lower, self.upper)
        else:
            assert self._tcp_anchor is not None
            self.target_pose = self._tcp_anchor.copy()
            displacement = self.phone_to_world @ (pose[:3, 3] - self._phone_anchor[:3, 3]) * self.translation_scale
            self.target_pose[:3, 3] += _bounded(displacement, self.workspace_radius)
            delta_rotation = pose[:3, :3] @ self._phone_anchor[:3, :3].T
            self.target_pose[:3, :3] = self.phone_to_world @ delta_rotation @ self.phone_to_world.T @ self._tcp_anchor[:3, :3]
        if gripper is not None:
            self._gripper_target = float(np.clip(gripper, 0, 1))
        self._holding = False
        self.status = ControlStatus("active", 0.0)
        return True

    def _step_time(self, now: float | None, dt: float | None) -> tuple[float, float]:
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now):
            raise ValueError("now must be finite local monotonic time")
        if dt is None:
            dt = self.model.opt.timestep if self._last_step is None else now - self._last_step
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("Control dt must be finite and nonnegative")
        self._last_step = now
        # A delayed event loop cannot issue a large catch-up joint command.
        return now, min(float(dt), 0.05)

    def _write_gripper(self, dt: float) -> None:
        delta = np.clip(self._gripper_target - self._gripper_command, -self.gripper_rate * dt, self.gripper_rate * dt)
        self._gripper_command += float(delta)
        values = self.gripper_open + self._gripper_command * (self.gripper_closed - self.gripper_open)
        if len(self.gripper_ids):
            limited = self.model.actuator_ctrllimited[self.gripper_ids].astype(bool)
            lower = np.where(limited, self.model.actuator_ctrlrange[self.gripper_ids, 0], -np.inf)
            upper = np.where(limited, self.model.actuator_ctrlrange[self.gripper_ids, 1], np.inf)
            self.data.ctrl[self.gripper_ids] = np.clip(values, lower, upper)

    def step(self, now: float | None = None, dt: float | None = None) -> ControlStatus:
        now, dt = self._step_time(now, dt)
        age = None if self._last_receive is None else now - self._last_receive
        if age is not None and (age < -0.001 or age > self.stale_timeout):
            self.hold(reason="stale")
            self.status = ControlStatus("stale", age)
            return self.status
        if self.target_pose is None or self.status.mode != "active":
            return self.status
        current = self.tcp_pose()
        error_p = self.target_pose[:3, 3] - current[:3, 3]
        error_r = rotation_vector(self.target_pose[:3, :3] @ current[:3, :3].T)
        velocity = np.concatenate((_bounded(self.position_gain * error_p, self.max_linear_velocity * self.speed_scale),
                                   _bounded(self.rotation_gain * error_r, self.max_angular_velocity * self.speed_scale)))
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        jacobian = np.vstack((self._jacp[:, self.dof_ids], self._jacr[:, self.dof_ids]))
        # A length scale balances metres and radians in the least-squares task.
        weights = np.array((1, 1, 1, .3, .3, .3))
        weighted_jacobian = weights[:, None] * jacobian
        qdot = weighted_jacobian.T @ np.linalg.solve(weighted_jacobian @ weighted_jacobian.T + self.damping ** 2 * np.eye(6), weights * velocity)
        limit = self.joint_velocity_limit * self.speed_scale
        qdot = np.clip(qdot, -limit, limit)
        proposed = self._command + qdot * dt
        bounded = np.clip(proposed, self.lower, self.upper)
        self._command = bounded
        self.data.ctrl[self.actuator_ids] = self._command
        self._write_gripper(dt)
        self.status = ControlStatus("active", age, float(np.linalg.norm(error_p)), float(np.linalg.norm(error_r)), bool(np.any(bounded != proposed)))
        return self.status

    def go_home(self, now: float | None = None, dt: float | None = None) -> ControlStatus:
        """Move servo targets toward home with joint-rate limits; do not teleport."""
        _, dt = self._step_time(now, dt)
        self._phone_anchor = self._tcp_anchor = self.target_pose = self._last_pose = None
        self._holding = False
        maximum = self.joint_velocity_limit * self.speed_scale * dt
        self._command = np.clip(self._command + np.clip(self.home_qpos - self._command, -maximum, maximum), self.lower, self.upper)
        self.data.ctrl[self.actuator_ids] = self._command
        self.status = ControlStatus("home")
        return self.status
