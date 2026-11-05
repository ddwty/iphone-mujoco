"""Wrist RGB-D contracts; image rendering is opt-in for OpenGL sessions."""
import json
import os

import mujoco
import numpy as np
from PIL import Image
import pytest

from iphone_mujoco.camera import CameraCalibration
from iphone_mujoco.geometry import CV_TO_MUJOCO_CAMERA, pose_from_xyzw
from iphone_mujoco.recording import ReplaySource
from iphone_mujoco.robots import load_scene
from iphone_mujoco.wrist import (
    WristCameraMount, WristRGBDRecorder, wrist_camera_mjcf,
)


def _world_from_body(data, body_id):
    pose = np.eye(4)
    pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
    pose[:3, 3] = data.xpos[body_id]
    return pose


def _world_from_cv_camera(data, camera_id):
    pose = np.eye(4)
    pose[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3)
    pose[:3, 3] = data.cam_xpos[camera_id]
    return pose @ CV_TO_MUJOCO_CAMERA


def test_wrist_camera_follows_joint_chain_without_teleporting_robot():
    calibration = CameraCalibration(64, 48, 59., 57., 29.3, 21.8, 32, 24)
    mount = WristCameraMount(
        "tool", pose_from_xyzw([.02, -.07, .04, .1, .2, -.3, .9]),
        camera_name="wrist_camera",
    )
    baseline, _, _ = load_scene("reference6")
    model, data, _ = load_scene(
        "reference6", phone_xml=wrist_camera_mjcf(calibration, mount),
        phone_parent_body=mount.parent_body,
    )
    assert model.nq == baseline.nq and model.nv == baseline.nv
    assert model.nmocap == 0
    parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mount.parent_body)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, mount.camera_name)
    local_camera_position = model.cam_pos[camera_id].copy()
    local_body_positions = model.body_pos.copy()
    poses = []
    for angles in ([0., -.65, 1.25, 0., -.6, 0.], [.4, -.4, .7, -.3, -.2, .5]):
        data.qpos[:6] = angles
        mujoco.mj_forward(model, data)
        expected = _world_from_body(data, parent_id) @ mount.parent_from_camera
        actual = _world_from_cv_camera(data, camera_id)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        poses.append(actual.copy())
    assert not np.allclose(poses[0], poses[1])
    np.testing.assert_equal(model.cam_pos[camera_id], local_camera_position)
    np.testing.assert_equal(model.body_pos, local_body_positions)


def test_wrist_mount_rejects_nonrigid_transform_and_missing_parent():
    invalid = np.eye(4)
    invalid[0, 0] = 2.
    with pytest.raises(ValueError):
        WristCameraMount("tool", invalid)
    with pytest.raises(ValueError):
        WristCameraMount("tool", np.full((4, 4), np.nan))
    with pytest.raises(ValueError):
        WristCameraMount("", np.eye(4))
    calibration = CameraCalibration(64, 48, 59., 57., 29.3, 21.8, 32, 24)
    mount = WristCameraMount("body_that_does_not_exist", np.eye(4))
    with pytest.raises(ValueError, match="parent|body"):
        load_scene(
            "reference6", phone_xml=wrist_camera_mjcf(calibration, mount),
            phone_parent_body=mount.parent_body,
        )


@pytest.mark.skipif((os.getenv("IPHONE_MUJOCO_TEST_RENDER") or os.getenv("ASMAGIC_TEST_RENDER")) != "1", reason="Set IPHONE_MUJOCO_TEST_RENDER=1 in an OpenGL session")
def test_wrist_rgbd_recording_preserves_same_frame_state_and_iphone_depth(tmp_path):
    calibration = CameraCalibration(64, 48, 59., 57., 29.3, 21.8, 32, 24)
    # At the reference arm's home pose, this camera faces down at the floor.
    mount = WristCameraMount("tool", pose_from_xyzw([.03, -.1, .08, 1., 0., 0., 0.]))
    model, data, spec = load_scene(
        "reference6", phone_xml=wrist_camera_mjcf(calibration, mount),
        phone_parent_body=mount.parent_body,
    )
    directory = tmp_path / "wrist"
    scale = .0001
    frames, snapshots = [], []
    with WristRGBDRecorder(model, data, spec, mount, calibration, directory, depth_scale=scale) as recorder:
        for index in range(2):
            data.qpos[0] = index * .15
            data.time = 1.25 + index * .1
            mujoco.mj_forward(model, data)
            snapshots.append({name: getattr(data, name).copy() for name in ("qpos", "qvel", "ctrl")})
            frames.append(recorder.capture())
            assert data.time == frames[-1].timestamp
            for name, expected in snapshots[-1].items():
                np.testing.assert_equal(getattr(data, name), expected)

    assert CameraCalibration.load(directory / "calibration.json") == calibration
    info = json.loads((directory / "camera_info.json").read_text())
    np.testing.assert_allclose(info["rgb"]["K"], calibration.matrix)
    np.testing.assert_allclose(info["depth"]["K"], calibration.depth_calibration().matrix)
    assert info["depth"]["meters_per_unit"] == scale
    assert info["mount"]["measured"] is False
    states = [json.loads(line) for line in (directory / "states.jsonl").read_text().splitlines()]
    assert len(states) == 2
    with ReplaySource(directory / "episode") as replay:
        samples = list(replay)
    assert len(samples) == 2
    for index, (frame, state, sample, snapshot) in enumerate(zip(frames, states, samples, snapshots)):
        assert frame.rgb.shape == (48, 64, 3) and frame.depth_m.shape == (24, 32)
        assert frame.calibration == calibration
        assert frame.frame_id == index and state["index"] == index
        assert state["simulation_time"] == frame.timestamp == sample.timestamp
        for name, expected in snapshot.items():
            np.testing.assert_equal(state[name], expected)
        np.testing.assert_allclose(state["world_from_camera"], frame.world_from_camera)
        np.testing.assert_allclose(sample.rgbd.world_from_camera, frame.world_from_camera)
        np.testing.assert_equal(sample.rgbd.rgb, frame.rgb)
        np.testing.assert_equal(sample.rgbd.depth_m, frame.depth_m)
        assert sample.rgbd.calibration == calibration
        np.testing.assert_equal(np.asarray(Image.open(directory / "rgb" / f"{index:08d}.png")), frame.rgb)
        path = directory / "depth" / f"{index:08d}.png"
        # PNG IHDR records 16-bit grayscale, independent of Pillow's decode dtype.
        assert path.read_bytes()[24:26] == bytes([16, 0])
        encoded = np.asarray(Image.open(path))
        assert encoded.shape == (24, 32) and not np.any(encoded == 65535)
        valid = np.isfinite(frame.depth_m)
        assert np.count_nonzero(valid) > 100
        np.testing.assert_allclose(encoded[valid] * scale, frame.depth_m[valid], atol=5e-7)
        np.testing.assert_equal(encoded[~valid], 0)
