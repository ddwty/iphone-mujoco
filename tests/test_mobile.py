"""Pinned mobile assets and real differential-wheel integration."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from iphone_mujoco.configuration import configure_camera
from iphone_mujoco.mobile import DifferentialDriveController, MobileRobotSpec, WheelOdometry, load_mobile_scene
from iphone_mujoco.robots import ASSETS, default_model_cache


@pytest.fixture
def mobile():
    cache = default_model_cache()
    if not (cache / "pal_tiago/tiago.xml").is_file():
        pytest.skip("Fetch the pinned TIAGo model with iphone-mujoco models --fetch tiago")
    return load_mobile_scene(configuration=configure_camera(resolution=(64, 48), depth_resolution=(32, 24)),
                             model_cache=cache)


def _relative_base(initial, final):
    return (np.linalg.inv(np.asarray(initial)) @ np.asarray(final))


def test_tiago_manifest_source_and_local_identity():
    manifest = json.loads((ASSETS / "tiago_manifest.json").read_text())
    assert manifest["repository"] == "google-deepmind/mujoco_menagerie"
    assert manifest["commit"] == "370c43965323c0944b433b202aac5058b5f907a9"
    assert manifest["directory"] == "pal_tiago"
    paths = {item["path"] for item in manifest["files"]}
    assert {"LICENSE", "README.md", "tiago.xml", "tiago_position.xml", "assets/wheels/wheel.stl"} <= paths
    assert len(paths) == len(manifest["files"])
    assert all(not Path(path).is_absolute() and ".." not in Path(path).parts for path in paths)
    cache = default_model_cache() / "pal_tiago"
    if cache.is_dir():
        for item in manifest["files"]:
            path = cache / item["path"]
            if path.is_file():
                raw = path.read_bytes()
                digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                assert len(raw) == item["size"] and digest == item["git_blob_sha1"]


def test_wheel_odometry_integrates_encoder_arc_and_rejects_invalid_input():
    odometry = WheelOdometry(.1, .4)
    odometry.update(0, 0)
    odometry.update(2, 4)
    expected_angle, expected_radius = .5, .6
    np.testing.assert_allclose(list(odometry.state().values()),
                               [expected_radius * np.sin(expected_angle),
                                expected_radius * (1 - np.cos(expected_angle)), expected_angle])
    with pytest.raises(ValueError):
        odometry.update(np.nan, 0)
    with pytest.raises(ValueError):
        WheelOdometry(0, .4)


def test_mobile_uses_real_free_base_wheels_and_supported_head_camera(mobile):
    model, data, spec = mobile
    assert spec.name == "tiago" and spec.radius_m == .0985 and spec.track_m == .4044
    assert model.nmocap == 0 and model.jnt_type[model.joint("reference").id] == mujoco.mjtJoint.mjJNT_FREE
    assert np.isclose(data.time, spec.metadata["warmup_seconds"]) and data.time >= .5
    assert spec.metadata["license"] == "Apache-2.0"
    assert Path(spec.metadata["license_path"]).is_file()
    assert model.body_parentid[model.body("iphone").id] == model.body("head_2_link").id
    assert model.body_mass[model.body("iphone").id] == 0
    for name in ("iphone_mobile_backplate", "iphone_mobile_clamp_left", "iphone_mobile_clamp_right", "iphone_mobile_support"):
        geom = model.geom(name).id
        assert model.geom_contype[geom] == model.geom_conaffinity[geom] == 0
    controller = DifferentialDriveController(model, data, spec)
    initial = controller.state()
    pose = np.array(initial["world_from_camera"])
    assert pose[0, 2] > .99 and pose[2, 3] > 1.1
    before = data.qpos.copy()
    controller.command(.1, .2)
    np.testing.assert_array_equal(data.qpos, before)
    controller.step(.2)
    assert data.time > spec.metadata["warmup_seconds"]
    assert np.any(np.abs(controller.state()["wheel_position_rad"]) > .01)
    assert not np.any(data.warning.number)


@pytest.mark.parametrize("linear,angular", [(.2, 0), (.12, .4), (0, .4)])
def test_wheel_driven_straight_arc_and_rotation_match_encoder_odometry(mobile, linear, angular):
    model, data, spec = mobile
    controller = DifferentialDriveController(model, data, spec)
    initial = controller.state()
    controller.command(linear, angular)
    controller.step(2)
    final = controller.state()
    physical = _relative_base(initial["world_from_base"], final["world_from_base"])
    yaw = np.arctan2(physical[1, 0], physical[0, 0])
    odometry = final["wheel_odometry"]
    np.testing.assert_allclose(physical[:2, 3], [odometry["x_m"], odometry["y_m"]], atol=.025)
    np.testing.assert_allclose(yaw, odometry["yaw_rad"], atol=.045)
    assert np.linalg.norm(np.asarray(final["wheel_position_rad"]) - initial["wheel_position_rad"]) > .5
    if angular == 0:
        assert physical[0, 3] > .3 and abs(physical[1, 3]) < .01
    elif linear == 0:
        assert abs(yaw) > .6 and np.linalg.norm(physical[:2, 3]) < .015
    else:
        assert physical[0, 3] > .15 and physical[1, 3] > .04 and yaw > .6
    assert not np.any(data.warning.number)


def test_wheel_commands_are_acceleration_bounded_and_stop(mobile):
    model, data, spec = mobile
    controller = DifferentialDriveController(model, data, spec)
    controller.command(10, -10)
    assert controller.state()["command_twist"] == {"linear_mps": .25, "angular_rps": -.6}
    controller.step()
    state = controller.state()
    assert state["applied_twist"]["linear_mps"] <= .5 * model.opt.timestep
    assert abs(state["applied_twist"]["angular_rps"]) <= 1.05 * model.opt.timestep
    controller.step(1)
    controller.command(0, 0)
    controller.step(1)
    np.testing.assert_allclose(controller.state()["wheel_target_rad_s"], 0, atol=1e-12)
    np.testing.assert_allclose(controller.state()["wheel_velocity_rad_s"], 0, atol=.025)
    with pytest.raises(ValueError):
        controller.command(np.inf, 0)
    with pytest.raises(ValueError):
        controller.step(model.opt.timestep * 1.5)
