"""Publish a simulated camera through the existing asMagic Phone wire schema.

World axes are robotics/MuJoCo (X forward, Y left, Z up). Wire camera-local
axes remain ARKit (X right, Y up, -Z forward), exactly as in the app.
"""
from __future__ import annotations

from io import BytesIO
import time
import uuid

import mujoco
import numpy as np
from PIL import Image

from .camera import RGBDFrame


def quantize_depth(depth_m: np.ndarray, depth_scale: float = 0.0001) -> np.ndarray:
    """asMagic-compatible UInt16 depth: zero invalid, 65535 reserved saturation."""
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be positive and finite")
    depth = np.asarray(depth_m, dtype=float)
    valid = np.isfinite(depth) & (depth > 0) & (depth < 65534.5 * depth_scale)
    encoded = np.zeros(depth.shape, dtype="<u2")
    encoded[valid] = np.clip(np.rint(depth[valid] / depth_scale), 1, 65534).astype("<u2")
    return encoded


def encode_phone_frame(frame: RGBDFrame, *, sequence: int = 0,
                       session_id: str = "simulation", depth_scale: float = 0.0001) -> bytes:
    """Encode JPEG + optical-axis metric depth. Overflow/invalid depth becomes zero."""
    try:
        from asmagic.proto.phone_msg_pb2 import Phone
    except ImportError as error:
        raise ImportError("Install iphone-mujoco[live] to publish SDK-compatible frames") from error
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be positive and finite")
    wall_timestamp = time.time() if frame.metadata.get("timestamp_clock") == "simulation" else float(frame.timestamp)
    message = Phone(timestamp=wall_timestamp, frame_id=sequence,
                    source_timestamp=float(frame.metadata.get("simulation_time", frame.timestamp)),
                    rgb_width=frame.calibration.width, rgb_height=frame.calibration.height,
                    depth_meters_per_unit=depth_scale, session_identifier=session_id,
                    rgb_codec=1, rgb_is_key_frame=True)
    message.camersIntrinsics.extend(frame.calibration.matrix.flatten(order="F"))
    if frame.rgb is not None:
        buffer = BytesIO()
        Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).save(buffer, format="JPEG", quality=95)
        message.color_img = buffer.getvalue()
    if frame.depth_m is not None:
        quantized = quantize_depth(frame.depth_m, depth_scale)
        valid = quantized > 0
        message.depth_height, message.depth_width = quantized.shape
        message.depth_img = quantized.tobytes()
        confidence = np.where(valid, 2, 0).astype(np.uint8)
        if frame.confidence is not None:
            confidence = np.where(valid, frame.confidence, 0).astype(np.uint8)
        message.depth_confidence = confidence.tobytes()
    if frame.world_from_camera is not None:
        # CV optical camera -> ARKit local camera, while retaining robotics world.
        pose = np.asarray(frame.world_from_camera) @ np.diag([1., -1., -1., 1.])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(pose[:3, :3]).ravel())
        wire = [*pose[:3, 3], *quaternion[1:], quaternion[0]]
        message.local_pose.extend(wire)
        message.global_pose.extend(wire)
    return message.SerializeToString()


class VirtualPhonePublisher:
    """Single-thread-owned PUB socket. A subscriber connects to this computer's IP."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:18000", *, depth_scale: float = 0.0001):
        try:
            import zmq
        except ImportError as error:
            raise ImportError("Install iphone-mujoco[live] to publish a virtual phone") from error
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.setsockopt(zmq.LINGER, 0)
        try:
            self._socket.bind(endpoint)
        except Exception:
            self.close()
            raise
        self.endpoint = endpoint
        self.depth_scale = depth_scale
        self.session_id = "mujoco-" + uuid.uuid4().hex
        self.sent_frames = 0

    def publish(self, frame: RGBDFrame):
        self._socket.send(encode_phone_frame(frame, sequence=self.sent_frames,
                                            session_id=self.session_id, depth_scale=self.depth_scale))
        self.sent_frames += 1

    def close(self):
        if getattr(self, "_socket", None) is not None:
            self._socket.close(linger=0)
            self._socket = None
        if getattr(self, "_context", None) is not None:
            self._context.term()
            self._context = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
