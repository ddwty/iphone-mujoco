"""Round-trip virtual output through the independently installed PyPI SDK."""
import time

import numpy as np
import pytest

pytest.importorskip("asmagic", reason="Virtual wire round-trips require the optional live SDK")
from asmagic.decoders import ar_frame
from asmagic.proto.phone_msg_pb2 import Phone
from iphone_mujoco.camera import CameraCalibration, RGBDFrame, frame_from_asmagic
from iphone_mujoco.geometry import pose_from_xyzw
from iphone_mujoco.virtual_device import encode_phone_frame


def _decode(frame, **kwargs):
    message = Phone()
    message.ParseFromString(encode_phone_frame(frame, **kwargs))
    return message, ar_frame(message)


def test_sdk_roundtrip_rgb_depth_confidence_K_pose_and_two_clocks():
    calibration = CameraCalibration(6, 4, 8.2, 6.3, 2.1, 1.7, 3, 2)
    rgb = np.full((4, 6, 3), [90, 140, 180], dtype=np.uint8)
    depth = np.array([[1.00003, 2.00002, np.nan], [0., 3., np.inf]], dtype=np.float32)
    confidence = np.array([[2, 1, 2], [0, 2, 1]], dtype=np.uint8)
    pose = pose_from_xyzw([.2, -.3, .8, .1, .2, .3, .9])
    frame = RGBDFrame(1_800_000_000., rgb, depth, calibration, pose,
                      metadata={"simulation_time": 2.5, "timestamp_clock": "host_wall"},
                      confidence=confidence)
    message, raw = _decode(frame, sequence=23, session_id="sim-A")
    assert message.timestamp == frame.timestamp
    assert message.source_timestamp == 2.5
    assert message.frame_id == 23 and message.session_identifier == "sim-A"
    assert not message.global_pose_is_shared and not message.shared_world_identifier
    np.testing.assert_allclose(raw.camera_intrinsics_matrix, calibration.matrix, atol=1e-6)
    result = frame_from_asmagic(raw, pose_convention="robotics")
    assert result.timestamp == 2.5
    assert result.metadata["sender_timestamp"] == frame.timestamp
    assert result.calibration.depth_width == 3 and result.calibration.depth_height == 2
    np.testing.assert_allclose(result.world_from_camera, pose, atol=1e-6)
    np.testing.assert_allclose(result.rgb, rgb, atol=3)
    valid = np.isfinite(depth) & (depth > 0)
    np.testing.assert_allclose(result.depth_m[valid], depth[valid], atol=.000051)
    assert np.isnan(result.depth_m[~valid]).all()
    np.testing.assert_equal(result.confidence, [[2, 1, 0], [0, 2, 0]])
    # Same pixel-center RGB-to-depth mapping survives the wire encoding.
    cloud = result.point_cloud(min_confidence=1)
    assert cloud.points.shape == (3, 3)


def test_sdk_roundtrip_optional_images_and_pose_stay_absent():
    calibration = CameraCalibration(6, 4, 8., 6., 2.1, 1.7)
    _, raw = _decode(RGBDFrame(1000., None, None, calibration))
    assert raw.color_array is None and raw.depth_array is None
    assert raw.local_pose.size == 0 and raw.global_pose.size == 0
    frame = frame_from_asmagic(raw, pose_convention="robotics")
    assert frame.rgb is None and frame.depth_m is None and frame.world_from_camera is None


def test_virtual_depth_reserves_saturated_code_for_missing_data():
    calibration = CameraCalibration(3, 2, 3., 3., 1., .5)
    depth = np.array([[6.5534, 6.5535, 6.5536], [0., np.nan, np.inf]], dtype=np.float32)
    message, raw = _decode(RGBDFrame(1000., None, depth, calibration))
    quantized = raw.depth_array
    assert not np.any(quantized == 65535)
    assert quantized[0, 0] == 65534
    np.testing.assert_equal(quantized[0, 1:], [0, 0])
    result = frame_from_asmagic(raw)
    assert np.isfinite(result.depth_m[0, 0])
    assert np.isnan(result.depth_m[0, 1:]).all()
    np.testing.assert_equal(np.frombuffer(message.depth_confidence, dtype=np.uint8), [2, 0, 0, 0, 0, 0])


def test_simulation_clock_is_not_encoded_as_unix_timestamp():
    calibration = CameraCalibration(2, 2, 2., 2., .5, .5)
    frame = RGBDFrame(2.5, None, None, calibration, metadata={"timestamp_clock": "simulation"})
    before = time.time()
    message, _ = _decode(frame)
    assert before <= message.timestamp <= time.time()
    assert message.source_timestamp == 2.5


@pytest.mark.parametrize("scale", [0., -1., np.nan, np.inf])
def test_invalid_depth_scale_is_rejected(scale):
    calibration = CameraCalibration(2, 2, 2., 2., .5, .5)
    with pytest.raises(ValueError, match="depth_scale"):
        encode_phone_frame(RGBDFrame(1., None, None, calibration), depth_scale=scale)
