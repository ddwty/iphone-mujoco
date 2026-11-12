"""Portable camera attachment and same-state mobile episode contracts."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from iphone_mujoco import CameraCalibration
from iphone_mujoco.cli import _camera_configuration
from iphone_mujoco.experiments import load_camera_scene, collect_camera_episode
from iphone_mujoco.geometry import CV_TO_MUJOCO_CAMERA
from iphone_mujoco.recording import ReplaySource
from iphone_mujoco.wrist import WristCameraMount


def test_bare_calibration_is_preserved_with_default_catalog_appearance(tmp_path):
    calibration = CameraCalibration(160, 120, 177., 169., 73.1, 62.4, 80, 60)
    path = tmp_path / "calibration.json"
    calibration.save(path)
    config = _camera_configuration(SimpleNamespace(profile=str(path)))
    assert config.calibration == calibration
    assert config.calibration_source == "user_supplied_intrinsics"
    assert config.device_id == "iphone-17-pro"


def test_custom_scene_keeps_relative_include_home_and_camera_kinematics(tmp_path):
    (tmp_path / "axis.xml").write_text('<mujocoinclude><body name="sensor_axis" pos=".2 .1 .5">'
        '<joint name="axis" type="slide" axis="1 0 0"/><geom type="box" size=".05 .05 .05" mass="1"/>'
        '</body></mujocoinclude>')
    path = tmp_path / "scene.xml"
    path.write_text('<mujoco><worldbody><include file="axis.xml"/></worldbody>'
                    '<keyframe><key name="home" qpos=".07"/></keyframe></mujoco>')
    calibration = CameraCalibration(64, 48, 60., 61., 31., 24.)
    mount = WristCameraMount("sensor_axis", np.eye(4), camera_name="inspection_camera")
    model, data, name = load_camera_scene("custom", calibration, scene=path, mount=mount)
    assert name == "inspection_camera" and model.nq == model.nkey == 1
    assert data.qpos[0] == .07
    camera = model.camera(name).id
    before = data.cam_xpos[camera].copy()
    data.qpos[0] += .04
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.cam_xpos[camera] - before, [.04, 0., 0.], atol=1e-10)
    np.testing.assert_allclose(data.cam_xmat[camera].reshape(3, 3) @ CV_TO_MUJOCO_CAMERA[:3, :3], np.eye(3), atol=1e-10)


@pytest.mark.skipif(os.getenv("IPHONE_MUJOCO_TEST_RENDER") != "1", reason="Requires an OpenGL context")
def test_mobile_episode_keeps_warmup_clocks_encoders_and_same_frame_state(tmp_path):
    from iphone_mujoco.configuration import configure_camera
    config = configure_camera(resolution=(160, 120), depth_resolution=(80, 60))
    result = collect_camera_episode("mobile", tmp_path / "mobile", configuration=config,
                                    duration=.2, fps=10., headless=True, motion="straight")
    assert result["frames"] == 2
    assert result["simulation_start_time"] >= .5 - 1e-9
    assert result["simulation_seconds"] == pytest.approx(.2)
    assert result["motion_model"] == "differential_drive_wheel_contact_dynamics"
    states = [json.loads(line) for line in (tmp_path / "mobile/states.jsonl").read_text().splitlines()]
    samples = list(ReplaySource(tmp_path / "mobile/episode"))
    assert len(samples) == len(states) == 2
    for sample, state in zip(samples, states):
        frame = sample.rgbd
        assert frame.timestamp == state["simulation_time"]
        assert frame.timestamp > result["simulation_start_time"]
        assert len(state["qpos"]) > 7 and len(state["ctrl"]) > 2
        assert frame.rgb.shape == (120, 160, 3) and frame.depth_m.shape == (60, 80)
        assert np.isfinite(frame.depth_m).any()
        np.testing.assert_array_equal(frame.metadata["scene_state"]["qpos"], state["qpos"])
        np.testing.assert_allclose(frame.world_from_camera, state["world_from_camera"])
        assert len(state["wheel_position_rad"]) == len(state["wheel_velocity_rad_s"]) == 2
        assert state["wheel_odometry"]["x_m"] >= 0
    assert states[-1]["wheel_position_rad"] != states[0]["wheel_position_rad"]
