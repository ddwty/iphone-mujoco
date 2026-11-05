"""Focused protocol/freshness/portable-episode smoke checks."""
from dataclasses import replace
import json
import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest

from iphone_mujoco.recording import RecordingWriter, ReplaySource
from iphone_mujoco.streams import AsMagicSource, ControlButtons, LiveSample


class FakeReceiver:
    def __init__(self):
        self.queue = queue.Queue()
        self.closed = False

    def receive(self, timeout):
        try:
            return self.queue.get(timeout=timeout / 1000)
        except queue.Empty:
            return None

    def close(self):
        self.closed = True


def ar_frame(timestamp=10.0, x=0.0, session=""):
    return SimpleNamespace(
        timestamp=timestamp, source_timestamp=timestamp,
        local_pose=np.array([x, 0, 0, 0, 0, 0, 1]),
        global_pose=np.array([x, 0, 0, 0, 0, 0, 1]),
        camera_intrinsics=np.array([]), session_identifier=session,
        shared_world_identifier="", global_pose_is_shared=False,
        has_color_image=False, frame_id=int(timestamp), rgb_codec_name="jpeg",
        rgb_width=0, rgb_height=0, depth_width=0, depth_height=0,
        depth_meters_per_unit=0.0, image_rotation_quarter_turns_clockwise=0,
    )


def joystick(move=False, **kwargs):
    fields = dict(square=move, cross=False, circle=False, triangle=False,
                  up=False, down=False, r1=False, r2=0.0)
    fields.update(kwargs)
    return SimpleNamespace(button=SimpleNamespace(**fields), timestamp=10.0)


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Receiver did not publish within timeout")
        time.sleep(0.005)


def make_source(**kwargs):
    ar, stick = FakeReceiver(), FakeReceiver()
    source = AsMagicSource("127.0.0.1", receiver_factories={
        "ar": lambda *a, **k: ar, "joystick": lambda *a, **k: stick,
    }, **kwargs)
    return source, ar, stick


def test_current_button_mapping_and_claw():
    controls = ControlButtons.from_asmagic(joystick(
        True, cross=True, circle=True, triangle=True, up=True, down=True,
        r1=True, r2=0.4,
    ))
    assert controls == ControlButtons(True, True, True, True, True, True, True, 0.4)
    assert controls.gripper == 1.0
    assert ControlButtons.from_asmagic(joystick(r2=0.7)).gripper == 0.7
    with pytest.raises(ValueError, match="non-finite"):
        ControlButtons.from_asmagic(joystick(r2=float("nan")))


def test_latest_bounded_pose_and_context_cleanup():
    source, ar, stick = make_source()
    with source:
        for i in range(20):
            ar.queue.put(ar_frame(timestamp=10.0 + i, x=float(i)))
        wait_until(lambda: source.stats["ar_frames"] == 20)
        sample = source.poll(0)
        assert sample.frame_id == 29
        assert sample.metadata["raw_local_pose"][0] == 19
        np.testing.assert_allclose(sample.pose[:3, 3], [0, -19, 0])
        assert sample.clutch
        assert source.poll(0) is None
        assert source.stats["replaced_samples"] >= 18
    assert ar.closed and stick.closed
    assert source.stats["threads_alive"] == 0


def test_move_requires_initial_release_and_independent_controls_expire():
    source, ar, stick = make_source(stale_after_s=0.08)
    with source:
        ar.queue.put(ar_frame())
        stick.queue.put(joystick(True, r2=0.5))
        wait_until(lambda: source.stats["ar_frames"] and source.stats["joystick_frames"])
        assert source.latest.clutch  # Already-held Move cannot arm a new connection.
        stick.queue.put(joystick(False))
        wait_until(lambda: source.stats["joystick_frames"] == 2)
        stick.queue.put(joystick(True, r2=0.5))
        wait_until(lambda: source.stats["joystick_frames"] == 3)
        assert not source.latest.clutch
        assert source.latest.gripper == 0.5
        original_receive_time = source.latest.received_at
        time.sleep(0.09)
        assert source.latest.pose_stale and source.latest.control_stale
        assert source.latest.pose is None and source.latest.gripper is None
        assert source.latest.received_at == original_receive_time


def test_session_epoch_changes_after_sender_restart():
    source, ar, stick = make_source()
    with source:
        ar.queue.put(ar_frame(timestamp=10, session="sender"))
        wait_until(lambda: source.stats["ar_frames"] == 1)
        old_session = source.latest.session_id
        ar.queue.put(ar_frame(timestamp=1, session="sender"))
        wait_until(lambda: source.stats["ar_frames"] == 2)
        assert source.latest.session_id != old_session
        assert source.latest.metadata["sender_session_id"] == "sender"
        assert source.stats["session_resets"] == 1


def test_robot_control_coordinate_convention_is_fixed():
    with pytest.raises(ValueError, match="raw ARKit"):
        make_source(pose_convention="robotics")


def test_ur_six_value_rotation_vector_is_rejected():
    source, ar, stick = make_source()
    with source:
        frame = ar_frame()
        frame.local_pose = np.array([0, 0, 0, 0, 0, 0])
        ar.queue.put(frame)
        wait_until(lambda: source.stats["decode_errors"] == 1)
        assert source.latest is None
        assert "seven" in source.stats["last_error"]


def test_episode_roundtrip_preserves_provenance_without_pickle(tmp_path):
    directory = tmp_path / "episode"
    sample = LiveSample(
        timestamp=1720000000.125, received_at=1234.5, pose=np.eye(4),
        buttons=ControlButtons(move=True, trigger=0.7), gripper=0.7,
        session_id="sender/receiver", frame_id=42, source_timestamp=123.25,
        metadata={"source": "asmagic", "raw_local_pose": [0, 0, 0, 0, 0, 0, 1]},
    )
    with RecordingWriter(directory, metadata={"experiment": "smoke"}) as writer:
        assert writer.write(sample) == 0
        writer.write(replace(sample, timestamp=sample.timestamp + 0.01, received_at=1234.51))
    with np.load(directory / "frames/00000000.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["pose"], np.eye(4))
    with ReplaySource(directory) as replay:
        actual = replay.poll()
        assert actual.timestamp == sample.timestamp
        assert actual.source_timestamp == 123.25
        assert actual.session_id == sample.session_id and actual.frame_id == 42
        assert actual.buttons == sample.buttons and actual.gripper == sample.gripper
        assert actual.received_at != sample.received_at
        assert actual.metadata["replay"]["original_received_at"] == 1234.5
        assert actual.metadata["raw_local_pose"] == sample.metadata["raw_local_pose"]
        assert replay.poll() is not None
        assert replay.exhausted and replay.poll() is None
    assert json.loads((directory / "manifest.json").read_text())["frame_count"] == 2
    with pytest.raises(FileExistsError):
        RecordingWriter(directory)


def test_episode_rejects_nonmonotonic_receipts_and_objects(tmp_path):
    with RecordingWriter(tmp_path / "episode") as writer:
        writer.write(LiveSample(10.0, 100.0))
        with pytest.raises(ValueError, match="nondecreasing"):
            writer.write(LiveSample(11.0, 99.0))
        with pytest.raises(ValueError, match="Object arrays"):
            writer.write(LiveSample(11.0, 101.0, pose=np.array([object()])))


def test_rgbd_episode_preserves_calibration_confidence_and_pixel_data(tmp_path):
    from iphone_mujoco.camera import CameraCalibration, RGBDFrame
    calibration = CameraCalibration(4, 3, 4.2, 4.4, 1.2, 1.1, 2, 2)
    frame = RGBDFrame(
        timestamp=123.25, rgb=np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
        depth_m=np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32),
        calibration=calibration, world_from_camera=np.eye(4), source="iphone", frame_id=4,
        metadata={"depth_meters_per_unit": 0.0001, "session_identifier": "iphone-session"},
        confidence=np.array([[2, 0], [1, 2]], dtype=np.uint8),
    )
    with RecordingWriter(tmp_path / "rgbd") as writer:
        writer.write(LiveSample(10000.0, 100.0, rgbd=frame, pose=np.eye(4)))
    with ReplaySource(tmp_path / "rgbd") as replay:
        actual = replay.poll().rgbd
        assert actual.calibration == calibration
        assert actual.timestamp == frame.timestamp and actual.metadata == frame.metadata
        np.testing.assert_array_equal(actual.rgb, frame.rgb)
        np.testing.assert_array_equal(actual.depth_m, frame.depth_m)
        np.testing.assert_array_equal(actual.confidence, frame.confidence)


def test_replay_preserves_old_pose_age_on_independent_button_update(tmp_path):
    sample = LiveSample(10.0, 100.0, pose=np.eye(4),
                        metadata={"update_received_at": 100.2})
    with RecordingWriter(tmp_path / "independent-controls") as writer:
        writer.write(sample)
    with ReplaySource(tmp_path / "independent-controls") as replay:
        actual = replay.poll()
        assert 0.19 < time.monotonic() - actual.received_at < 0.3
