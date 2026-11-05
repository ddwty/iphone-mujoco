"""Bounded, latest-sample adapters around the published asMagic SDK.

Sockets and video decoding belong to ``asmagic``. This module owns a receiving
thread per SDK subscription and exposes snapshots without blocking simulation.
Local monotonic receive time drives freshness; sender timestamps remain data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import threading
import time
from typing import Any, Callable
import uuid

import numpy as np

from .geometry import convert_iphone_pose, pose_from_xyzw


@dataclass(frozen=True)
class ControlButtons:
    """asMagic Robot Control actions, preserving the App's button meanings."""

    move: bool = False
    calibrate: bool = False
    home: bool = False
    record: bool = False
    speed_up: bool = False
    speed_down: bool = False
    r1: bool = False
    trigger: float = 0.0

    @classmethod
    def from_asmagic(cls, frame: Any) -> "ControlButtons":
        named = frame.button
        trigger = float(named.r2)
        if not math.isfinite(trigger):
            raise ValueError("Received non-finite right trigger")
        return cls(
            move=bool(named.square), calibrate=bool(named.cross),
            home=bool(named.circle), record=bool(named.triangle),
            speed_up=bool(named.up), speed_down=bool(named.down),
            r1=bool(named.r1), trigger=float(np.clip(trigger, 0.0, 1.0)),
        )

    @property
    def gripper(self) -> float:
        """App's claw command: 0 released/open, 1 R1 or full R2 travel."""
        return max(self.trigger, float(self.r1))


@dataclass(frozen=True)
class LiveSample:
    """One latest snapshot; independently received controls are never frame-paired.

    ``received_at`` is the pose's *local monotonic* receipt time. ``timestamp``
    and ``source_timestamp`` preserve sender wall/source clocks respectively.
    ``pose`` maps CV camera coordinates to the explicitly selected Z-up world.
    Check ``clutch`` before passing the pose to a robot controller.
    """

    timestamp: float
    received_at: float
    rgbd: Any | None = None
    pose: np.ndarray | None = None
    buttons: ControlButtons = field(default_factory=ControlButtons)
    gripper: float | None = None
    session_id: str = ""
    frame_id: int | None = None
    source_timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pose_stale: bool = False
    control_stale: bool = False

    @property
    def clutch(self) -> bool:
        return self.pose_stale or self.control_stale or not self.buttons.move


class AsMagicSource:
    """Receive real iPhone packets using ``asmagic>=1.0.1``.

    ``mode='camera'`` reads AR RGB-D on TCP 8000. ``mode='teleop'`` reads the
    Robot Control page's pose on 8000 plus buttons on 8020. That page does not
    publish camera images. Use Through Mac in the App and the iPhone's IP.

    Camera coordinate settings are absent from the wire. The default requires
    App Y up with no custom adjustment. Choose ``robotics`` for App Z up, or
    supply ``world_from_sender`` for a measured world registration. Robot
    Control always publishes raw ARKit poses and therefore requires ``arkit``.
    """

    def __init__(
        self, ip: str, *, mode: str = "teleop", pose_field: str = "local",
        pose_convention: str = "arkit", world_from_sender: np.ndarray | None = None,
        stale_after_s: float = 0.3, ar_port: int = 8000, joystick_port: int = 8020,
        receiver_factories: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        if mode not in {"camera", "teleop"}:
            raise ValueError("mode must be camera or teleop")
        if pose_field not in {"local", "global"}:
            raise ValueError("pose_field must be local or global")
        if not math.isfinite(stale_after_s) or stale_after_s <= 0:
            raise ValueError("stale_after_s must be finite and positive")
        if mode == "teleop" and pose_convention != "arkit":
            raise ValueError("Robot Control publishes raw ARKit poses; use pose_convention='arkit'")
        # Validate coordinate registration before starting any receivers.
        convert_iphone_pose(np.eye(4), pose_convention, world_from_sender=world_from_sender)
        if receiver_factories is None:
            try:
                from asmagic import ARDataSubscriber, JoystickDataSubscriber
            except ImportError as error:
                raise ImportError("Install live streaming with: pip install 'iphone-mujoco[live]'") from error
            receiver_factories = {"ar": ARDataSubscriber, "joystick": JoystickDataSubscriber}
        self.ip, self.mode, self.pose_field = ip, mode, pose_field
        self.pose_convention, self.world_from_sender = pose_convention, world_from_sender
        self.stale_after_s = float(stale_after_s)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._sample: LiveSample | None = None
        self._buttons = ControlButtons()
        self._button_received_at: float | None = None
        self._button_timestamp: float | None = None
        self._updated_at: float | None = None
        self._generation = self._consumed_generation = 0
        self._local_session = "receiver-" + str(uuid.uuid4())
        self._last_source_timestamp: float | None = None
        self._needs_release = mode == "teleop"
        self._stats: dict[str, Any] = {
            "ar_frames": 0, "joystick_frames": 0, "replaced_samples": 0,
            "decode_errors": 0, "session_resets": 0, "last_error": None,
        }
        channels = [("ar", ar_port)] + ([("joystick", joystick_port)] if mode == "teleop" else [])
        self._threads = [threading.Thread(
            target=self._run, args=(name, port, receiver_factories[name]),
            name="iphone-mujoco-" + name, daemon=True,
        ) for name, port in channels]
        for thread in self._threads:
            thread.start()

    def _run(self, channel: str, port: int, factory: Callable[..., Any]) -> None:
        receiver = None
        try:
            # SDK AR conflation occurs after HEVC decode. Never conflate encoded
            # video independently or share a ZeroMQ socket across threads.
            receiver = factory(self.ip, port=port, hwm=4 if channel == "ar" else 1, conflate=True)
            while not self._stop.is_set():
                try:
                    frame = receiver.receive(timeout=50)
                    if frame is None:
                        continue
                    received_at = time.monotonic()
                    if channel == "ar":
                        sample = self._convert_ar(frame, received_at)
                        with self._condition:
                            self._publish_ar(sample)
                    else:
                        buttons = ControlButtons.from_asmagic(frame)
                        with self._condition:
                            if self._button_received_at is not None and (
                                received_at - self._button_received_at > self.stale_after_s
                            ):
                                self._needs_release = True
                            self._buttons = buttons
                            self._button_received_at = received_at
                            self._button_timestamp = float(frame.timestamp)
                            self._updated_at = time.monotonic()
                            if not buttons.move:
                                self._needs_release = False
                            self._stats["joystick_frames"] += 1
                            self._generation += 1
                            self._condition.notify_all()
                except (RuntimeError, ValueError, OSError) as error:
                    with self._condition:
                        self._stats["decode_errors"] += 1
                        self._stats["last_error"] = str(error)
                        self._condition.notify_all()
        except StopIteration:
            pass
        except Exception as error:
            with self._condition:
                self._stats["last_error"] = str(error)
                self._condition.notify_all()
        finally:
            if receiver is not None:
                receiver.close()
            with self._condition:
                self._condition.notify_all()

    def _convert_ar(self, frame: Any, received_at: float) -> LiveSample:
        if not math.isfinite(float(frame.timestamp)) or not math.isfinite(float(frame.source_timestamp)):
            raise ValueError("Received non-finite sender timestamp")
        raw_local = np.asarray(frame.local_pose).tolist()
        raw_global = np.asarray(frame.global_pose).tolist()
        raw_pose = raw_local if self.pose_field == "local" else raw_global
        if self.mode == "camera" and self.pose_field == "global" and not (
            frame.global_pose_is_shared and frame.shared_world_identifier
        ):
            raise ValueError("Global camera pose requires a valid shared-world identifier")
        pose = None if not raw_pose else convert_iphone_pose(
            pose_from_xyzw(raw_pose), self.pose_convention,
            world_from_sender=self.world_from_sender,
        )
        rgbd = None
        if self.mode == "camera" and (
            frame.has_color_image or getattr(frame, "has_depth_image", False)
            or (frame.rgb_width > 0 and frame.rgb_height > 0 and np.asarray(frame.camera_intrinsics).size == 9)
        ):
            from .camera import frame_from_asmagic
            rgbd = frame_from_asmagic(
                frame, pose_convention=self.pose_convention, pose_field=self.pose_field,
                world_from_sender=self.world_from_sender,
            )
        metadata = {
            "source": "asmagic", "mode": self.mode, "ip": self.ip,
            "pose_field": self.pose_field, "pose_convention": self.pose_convention,
            "raw_local_pose": raw_local, "raw_global_pose": raw_global,
            "raw_camera_intrinsics": np.asarray(frame.camera_intrinsics).tolist(),
            "sender_session_id": frame.session_identifier,
            "shared_world_identifier": frame.shared_world_identifier,
            "global_pose_is_shared": bool(frame.global_pose_is_shared),
            "rgb_codec": frame.rgb_codec_name, "rgb_width": frame.rgb_width,
            "rgb_height": frame.rgb_height, "depth_width": frame.depth_width,
            "depth_height": frame.depth_height, "depth_meters_per_unit": frame.depth_meters_per_unit,
            "image_rotation_quarter_turns_clockwise": frame.image_rotation_quarter_turns_clockwise,
            "controls_are_independently_received": self.mode == "teleop",
            "wire_frame_id": int(frame.frame_id),
            "wire_source_timestamp": float(frame.source_timestamp),
        }
        if self.world_from_sender is not None:
            metadata["world_from_sender"] = np.asarray(self.world_from_sender).tolist()
        return LiveSample(
            timestamp=float(frame.timestamp), received_at=received_at, rgbd=rgbd, pose=pose,
            session_id=frame.session_identifier, frame_id=int(frame.frame_id) if frame.frame_id else None,
            source_timestamp=float(frame.source_timestamp) if frame.source_timestamp else None,
            metadata=metadata,
        )

    def _publish_ar(self, sample: LiveSample) -> None:
        previous = self._sample
        discontinuity = previous is not None and (
            sample.received_at - previous.received_at > self.stale_after_s
            or (sample.session_id and sample.session_id != previous.metadata.get("sender_session_id"))
            or (sample.source_timestamp is not None and self._last_source_timestamp is not None
                and sample.source_timestamp < self._last_source_timestamp)
            or sample.timestamp < previous.timestamp
        )
        if discontinuity:
            self._local_session = "receiver-" + str(uuid.uuid4())
            self._needs_release = self.mode == "teleop"
            self._stats["session_resets"] += 1
        self._last_source_timestamp = sample.source_timestamp
        # A sender session can remain unchanged across a long network outage;
        # append the local receipt epoch so controllers still reanchor.
        session_id = (sample.session_id + "/" if sample.session_id else "") + self._local_session
        if self._generation != self._consumed_generation:
            self._stats["replaced_samples"] += 1
        self._sample = replace(sample, session_id=session_id)
        self._updated_at = time.monotonic()
        self._stats["ar_frames"] += 1
        self._stats["last_error"] = None
        self._generation += 1
        self._condition.notify_all()

    def _snapshot(self) -> LiveSample | None:
        if self._sample is None:
            return None
        now = time.monotonic()
        pose_stale = self._stop.is_set() or now - self._sample.received_at > self.stale_after_s
        control_stale = self._stop.is_set() or self._button_received_at is None or (
            now - self._button_received_at > self.stale_after_s
        )
        buttons = ControlButtons() if control_stale else self._buttons
        if self._needs_release:
            buttons = replace(buttons, move=False)
        metadata = dict(self._sample.metadata, joystick_timestamp=self._button_timestamp,
                        joystick_received_at=self._button_received_at,
                        move_release_required=self._needs_release,
                        raw_buttons=asdict(self._buttons),
                        update_received_at=self._updated_at)
        return replace(
            self._sample, pose=None if pose_stale else self._sample.pose,
            buttons=buttons, gripper=None if control_stale else buttons.gripper,
            pose_stale=pose_stale, control_stale=control_stale, metadata=metadata,
        )

    @property
    def latest(self) -> LiveSample | None:
        with self._condition:
            return self._snapshot()

    @property
    def stats(self) -> dict[str, Any]:
        with self._condition:
            return dict(self._stats, has_received_data=self._sample is not None,
                        threads_alive=sum(t.is_alive() for t in self._threads),
                        last_pose_received_at=None if self._sample is None else self._sample.received_at)

    def poll(self, timeout_s: float = 0.0) -> LiveSample | None:
        """Wait at most timeout_s for an update, returning only the latest one."""
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("timeout_s must be finite and nonnegative")
        with self._condition:
            self._condition.wait_for(
                lambda: self._stop.is_set() or self._generation != self._consumed_generation,
                timeout_s,
            )
            if self._stop.is_set() or self._generation == self._consumed_generation:
                return None
            self._consumed_generation = self._generation
            return self._snapshot()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2.0)
        if any(t.is_alive() for t in self._threads):
            raise RuntimeError("asMagic receiver did not close within two seconds")

    def __enter__(self) -> "AsMagicSource":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
