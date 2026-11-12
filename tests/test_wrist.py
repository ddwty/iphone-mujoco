"""Wrist RGB-D contracts; image rendering is opt-in for OpenGL sessions."""
import json
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
import pytest

from iphone_mujoco.camera import CameraCalibration
from iphone_mujoco.geometry import CV_TO_MUJOCO_CAMERA, pose_from_xyzw, project_points, transform_points
from iphone_mujoco.phone_scene import demo_calibration
from iphone_mujoco.recording import ReplaySource
from iphone_mujoco.robots import get_robot, load_scene
from iphone_mujoco.wrist import (
    WristCameraMount, WristRGBDRecorder, configured_wrist_mount, default_wrist_mount,
    wrist_camera_mjcf, _place_demo_targets,
)


@pytest.fixture(params=("panda", "ur5e"))
def robot_name(request):
    name = request.param
    if not get_robot(name).model_path.is_file():
        pytest.skip(f"Download the pinned {name} model with iphone-mujoco models --fetch {name}")
    return name


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


def test_wrist_camera_follows_joint_chain_without_teleporting_robot(robot_name):
    calibration = CameraCalibration(64, 48, 59., 57., 29.3, 21.8, 32, 24)
    mount = WristCameraMount(
        default_wrist_mount(robot_name).parent_body,
        pose_from_xyzw([.02, -.07, .04, .1, .2, -.3, .9]),
        camera_name="wrist_camera",
    )
    baseline, _, _ = load_scene(robot_name)
    model, data, spec = load_scene(
        robot_name, phone_xml=wrist_camera_mjcf(calibration, mount),
        phone_parent_body=mount.parent_body,
    )
    assert model.nq == baseline.nq and model.nv == baseline.nv
    assert model.nmocap == 0
    parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mount.parent_body)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, mount.camera_name)
    local_camera_position = model.cam_pos[camera_id].copy()
    local_body_positions = model.body_pos.copy()
    joint_positions = [model.joint(name).qposadr[0] for name in spec.joint_names]
    home = data.qpos[joint_positions].copy()
    poses = []
    for angles in (home, home + np.linspace(-.15, .15, len(home))):
        data.qpos[joint_positions] = angles
        mujoco.mj_forward(model, data)
        expected = _world_from_body(data, parent_id) @ mount.parent_from_camera
        actual = _world_from_cv_camera(data, camera_id)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        poses.append(actual.copy())
    assert not np.allclose(poses[0], poses[1])
    np.testing.assert_equal(model.cam_pos[camera_id], local_camera_position)
    np.testing.assert_equal(model.body_pos, local_body_positions)


def test_wrist_mount_rejects_nonrigid_transform():
    invalid = np.eye(4)
    invalid[0, 0] = 2.
    with pytest.raises(ValueError):
        WristCameraMount("hand", invalid)
    with pytest.raises(ValueError):
        WristCameraMount("hand", np.full((4, 4), np.nan))
    with pytest.raises(ValueError):
        WristCameraMount("", np.eye(4))
    with pytest.raises(ValueError):
        WristCameraMount("hand", np.eye(4), parent_from_support=invalid)


def test_support_mount_roundtrip_preserves_legacy_json(tmp_path):
    legacy = WristCameraMount("hand", np.eye(4))
    path = tmp_path / "mount.json"
    legacy.save(path)
    assert "parent_from_support" not in json.loads(path.read_text())
    assert WristCameraMount.load(path).parent_from_support is None
    mounted = default_wrist_mount("ur5e")
    mounted.save(path)
    loaded = WristCameraMount.load(path)
    np.testing.assert_array_equal(loaded.parent_from_support, mounted.parent_from_support)
    assert not loaded.parent_from_support.flags.writeable


@pytest.mark.parametrize("robot_name", ["ur5e"], indirect=True)
def test_ur_phone_is_in_tcp_plane_with_connected_massless_support(robot_name):
    calibration = CameraCalibration(64, 48, 59., 57., 29.3, 21.8)
    mount = default_wrist_mount(robot_name)
    baseline, baseline_data, baseline_spec = load_scene(
        robot_name, phone_xml=wrist_camera_mjcf(calibration, mount),
        phone_parent_body=mount.parent_body,
    )
    model, data, spec = load_scene(
        robot_name, phone_xml=wrist_camera_mjcf(calibration, mount, show_phone=True),
        phone_parent_body=mount.parent_body,
    )
    assert (model.nq, model.nv, model.nu) == (baseline.nq, baseline.nv, baseline.nu)
    parent_id = model.body(mount.parent_body).id
    camera_id = model.camera(mount.camera_name).id
    tcp_id = model.site("attachment_site").id
    support_prefix = mount.camera_name + "_support"
    support_geoms = [index for index in range(model.ngeom)
                     if model.geom(index).name.startswith(support_prefix)]
    assert len(support_geoms) == 5
    np.testing.assert_equal(model.geom_contype[support_geoms], 0)
    np.testing.assert_equal(model.geom_conaffinity[support_geoms], 0)
    for body_id in np.unique(model.geom_bodyid[support_geoms]):
        assert model.body_mass[body_id] == 0
    joint_positions = [model.joint(name).qposadr[0] for name in spec.joint_names]
    baseline_positions = [baseline.joint(name).qposadr[0] for name in baseline_spec.joint_names]
    home = data.qpos[joint_positions].copy()
    for angles in (home, home + np.linspace(-.15, .15, len(home))):
        data.qpos[joint_positions] = angles
        baseline_data.qpos[baseline_positions] = angles
        mujoco.mj_forward(model, data)
        mujoco.mj_forward(baseline, baseline_data)
        optical = _world_from_cv_camera(data, camera_id)
        expected = _world_from_body(data, parent_id) @ mount.parent_from_camera
        np.testing.assert_allclose(optical, expected, atol=1e-10)
        outward = data.xmat[parent_id].reshape(3, 3)[:, 1]
        np.testing.assert_allclose(np.dot(optical[:3, 3] - data.site_xpos[tcp_id], outward), 0., atol=1e-10)
        np.testing.assert_allclose(optical[:3, 2], outward, atol=1e-10)
        np.testing.assert_allclose(data.qM, baseline_data.qM, atol=1e-12)

        brace = model.geom(support_prefix + "_brace").id
        axis = data.geom_xmat[brace].reshape(3, 3)[:, 2]
        endpoints = data.geom_xpos[brace] + np.outer([-1, 1], axis * model.geom_size[brace, 1])
        collar = model.geom(support_prefix + "_collar").id
        collar_points = (endpoints - data.geom_xpos[collar]) @ data.geom_xmat[collar].reshape(3, 3)
        in_collar = ((np.linalg.norm(collar_points[:, :2], axis=1) <= model.geom_size[collar, 0] + 1e-10)
                     & (np.abs(collar_points[:, 2]) <= model.geom_size[collar, 1] + 1e-10))
        plate = model.geom(support_prefix + "_plate").id
        plate_points = (endpoints - data.geom_xpos[plate]) @ data.geom_xmat[plate].reshape(3, 3)
        in_plate = np.all(np.abs(plate_points) <= model.geom_size[plate] + 1e-10, axis=1)
        assert np.any(in_collar) and np.any(in_plate)

        shell = model.geom(mount.camera_name + "_phone_shell").id
        shell_from_world = data.geom_xmat[shell].reshape(3, 3)
        plate_in_shell = (data.geom_xpos[plate] - data.geom_xpos[shell]) @ shell_from_world
        np.testing.assert_allclose(plate_in_shell[2] - model.geom_size[plate, 2], model.geom_size[shell, 2], atol=1e-10)
        for side, sign in (("left", -1), ("right", 1)):
            clamp = model.geom(support_prefix + "_clamp_" + side).id
            clamp_in_shell = (data.geom_xpos[clamp] - data.geom_xpos[shell]) @ shell_from_world
            np.testing.assert_allclose(sign * clamp_in_shell[0] - model.geom_size[clamp, 0], model.geom_size[shell, 0], atol=1e-10)


def test_wrist_mount_rejects_missing_parent(robot_name):
    calibration = CameraCalibration(64, 48, 59., 57., 29.3, 21.8, 32, 24)
    mount = WristCameraMount("body_that_does_not_exist", np.eye(4))
    with pytest.raises(ValueError, match="parent|body"):
        load_scene(
            robot_name, phone_xml=wrist_camera_mjcf(calibration, mount),
            phone_parent_body=mount.parent_body,
        )


def test_visible_phone_preserves_optical_mount_and_robot_dynamics(robot_name):
    calibration = CameraCalibration(64, 48, 59., 57., 29.3, 21.8, 32, 24)
    mount = WristCameraMount(default_wrist_mount(robot_name).parent_body,
                             pose_from_xyzw([.03, -.1, .08, .1, .2, -.3, .9]))
    baseline, baseline_data, baseline_spec = load_scene(
        robot_name, phone_xml=wrist_camera_mjcf(calibration, mount),
        phone_parent_body=mount.parent_body,
    )
    model, data, spec = load_scene(
        robot_name, phone_xml=wrist_camera_mjcf(calibration, mount, show_phone=True),
        phone_parent_body=mount.parent_body,
    )
    assert (model.nq, model.nv, model.nu, model.nmocap) == (
        baseline.nq, baseline.nv, baseline.nu, baseline.nmocap)
    assert model.ncam == baseline.ncam == 1
    for body_id in range(baseline.nbody):
        current = model.body(baseline.body(body_id).name).id
        np.testing.assert_equal(model.body_mass[current], baseline.body_mass[body_id])
        np.testing.assert_equal(model.body_inertia[current], baseline.body_inertia[body_id])
    phone_id = model.body(mount.camera_name + "_phone").id
    phone_geoms = np.flatnonzero(model.geom_bodyid == phone_id)
    assert len(phone_geoms) == 3
    assert model.body_mass[phone_id] == 0
    np.testing.assert_equal(model.geom_contype[phone_geoms], 0)
    np.testing.assert_equal(model.geom_conaffinity[phone_geoms], 0)
    camera_id = model.camera(mount.camera_name).id
    parent_id = model.body(mount.parent_body).id
    rig_id = int(model.cam_bodyid[camera_id])
    assert model.body_parentid[rig_id] == parent_id
    optical_site = model.site(mount.camera_name + "_phone_optical_center").id
    joint_positions = [model.joint(name).qposadr[0] for name in spec.joint_names]
    baseline_positions = [baseline.joint(name).qposadr[0] for name in baseline_spec.joint_names]
    home = data.qpos[joint_positions].copy()
    for angles in (home, home + np.linspace(-.15, .15, len(home))):
        baseline_data.qpos[baseline_positions] = angles
        data.qpos[joint_positions] = angles
        mujoco.mj_forward(baseline, baseline_data)
        mujoco.mj_forward(model, data)
        expected = _world_from_body(data, parent_id) @ mount.parent_from_camera
        np.testing.assert_allclose(_world_from_cv_camera(data, camera_id), expected, atol=1e-10)
        np.testing.assert_allclose(_world_from_cv_camera(baseline_data, camera_id), expected, atol=1e-10)
        np.testing.assert_allclose(data.site_xpos[optical_site], expected[:3, 3], atol=1e-10)
        np.testing.assert_allclose(data.qM, baseline_data.qM, atol=1e-12)


def test_visible_phone_overrides_inherited_robot_geom_mass():
    calibration = CameraCalibration(64, 48, 59., 57., 29.3, 21.8)
    mount = WristCameraMount("parent", np.eye(4))
    scene = ('<mujoco><default><geom mass=".5"/></default><worldbody>'
             '<body name="parent"><joint name="joint"/><geom type="box" size=".1 .1 .1"/>'
             '{rig}</body></worldbody></mujoco>')
    baseline = mujoco.MjModel.from_xml_string(scene.format(rig=wrist_camera_mjcf(calibration, mount)))
    model = mujoco.MjModel.from_xml_string(scene.format(rig=wrist_camera_mjcf(calibration, mount, show_phone=True)))
    baseline_data, data = mujoco.MjData(baseline), mujoco.MjData(model)
    mujoco.mj_forward(baseline, baseline_data)
    mujoco.mj_forward(model, data)
    assert model.body(mount.camera_name + "_phone").mass[0] == 0
    np.testing.assert_allclose(data.qM, baseline_data.qM, atol=1e-12)


def _configured_parent_scene(rig):
    document = ET.fromstring(rig)
    root = ET.fromstring('<mujoco><default><geom mass=".5"/></default><worldbody>'
                         '<body name="parent"><joint name="joint"/>'
                         '<geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>')
    assets = document.find("asset")
    if assets is not None:
        root.append(assets)
    root.find("worldbody/body").append(document.find("worldbody/body"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qpos[0] = .4
    mujoco.mj_forward(model, data)
    return model, data


def test_configured_lens_changes_optical_center_without_moving_phone():
    from iphone_mujoco.configuration import configure_camera
    from iphone_mujoco.devices import get_device
    mount = WristCameraMount("parent", pose_from_xyzw([.03, -.1, .08, .1, .2, -.3, .9]),
                             parent_from_support=pose_from_xyzw([.01, -.1, .08, .1, .2, -.3, .9]))
    expected_phone = mount.parent_from_camera @ np.linalg.inv(get_device("iphone-17-pro").phone_from_camera("wide"))
    optical_poses = []
    support_positions = None
    for lens in ("wide", "ultra_wide", "telephoto"):
        configuration = configure_camera("iphone-17-pro", lens=lens, resolution=(64, 48), depth_resolution=(32, 24))
        model, data = _configured_parent_scene(wrist_camera_mjcf(
            configuration.calibration, mount, show_phone=True, configuration=configuration))
        parent = model.body(mount.parent_body).id
        phone = model.body(mount.camera_name + "_phone").id
        camera = model.camera(mount.camera_name).id
        effective = configured_wrist_mount(mount, configuration)
        np.testing.assert_allclose(_world_from_cv_camera(data, camera),
                                   _world_from_body(data, parent) @ effective.parent_from_camera, atol=1e-10)
        np.testing.assert_allclose(_world_from_body(data, phone),
                                   _world_from_body(data, parent) @ expected_phone, atol=1e-10)
        assert model.body_parentid[model.cam_bodyid[camera]] == parent
        assert model.body_mass[phone] == 0
        support_ids = [index for index in range(model.ngeom)
                       if model.geom(index).name.startswith(mount.camera_name + "_support")]
        if support_positions is None:
            support_positions = data.geom_xpos[support_ids].copy()
        else:
            np.testing.assert_allclose(data.geom_xpos[support_ids], support_positions, atol=1e-10)
        optical_poses.append(_world_from_cv_camera(data, camera))
    assert not np.allclose(optical_poses[0][:3, 3], optical_poses[1][:3, 3])
    assert not np.allclose(optical_poses[0][:3, 3], optical_poses[2][:3, 3])


def test_configured_mount_uses_measured_selected_lens_transform():
    from iphone_mujoco.configuration import configure_camera
    from iphone_mujoco.devices import get_device
    configuration = configure_camera("iphone-17-pro", lens="telephoto", resolution=(64, 48))
    measured = pose_from_xyzw([.021, .049, -.006, 0., .01, 0., 1.])
    configuration.metadata["phone_from_camera"] = measured.tolist()
    mount = WristCameraMount("parent", np.eye(4))
    device = get_device(configuration.device_id)
    expected = np.linalg.inv(device.phone_from_camera("wide")) @ measured
    np.testing.assert_allclose(configured_wrist_mount(mount, configuration).parent_from_camera, expected, atol=1e-10)
    model, data = _configured_parent_scene(wrist_camera_mjcf(
        configuration.calibration, mount, show_phone=True, configuration=configuration))
    parent = model.body(mount.parent_body).id
    np.testing.assert_allclose(_world_from_cv_camera(data, model.camera(mount.camera_name).id),
                               _world_from_body(data, parent) @ expected, atol=1e-10)


def test_configured_payload_mass_is_opt_in_and_independent_of_visibility():
    from iphone_mujoco.configuration import configure_camera
    from iphone_mujoco.devices import get_device
    configuration = configure_camera("iphone-17-pro", region="cn", resolution=(64, 48))
    mount = WristCameraMount("parent", np.eye(4))
    baseline, baseline_data = _configured_parent_scene(wrist_camera_mjcf(
        configuration.calibration, mount, configuration=configuration, show_phone=True))
    assert baseline.body(mount.camera_name + "_phone").mass[0] == 0
    loaded_inertia = None
    for show_phone in (True, False):
        model, data = _configured_parent_scene(wrist_camera_mjcf(
            configuration.calibration, mount, configuration=configuration, show_phone=show_phone, payload=True))
        phone = model.body(mount.camera_name + "_phone").id
        np.testing.assert_allclose(model.body_mass[phone], get_device(configuration.device_id).mass_kg("cn"))
        assert (model.nq, model.nv) == (baseline.nq, baseline.nv)
        assert not np.allclose(data.qM, baseline_data.qM)
        if loaded_inertia is None:
            loaded_inertia = data.qM.copy()
        else:
            np.testing.assert_allclose(data.qM, loaded_inertia, atol=1e-12)
        if not show_phone:
            assert not np.any(model.geom_bodyid == phone)


def test_payload_requires_device_configuration():
    with pytest.raises(ValueError, match="payload requires"):
        wrist_camera_mjcf(demo_calibration(), WristCameraMount("parent", np.eye(4)), payload=True)


def test_mainstream_default_wrist_observes_demo_targets(robot_name):
    calibration = demo_calibration()
    mount = default_wrist_mount(robot_name)
    model, data, _ = load_scene(
        robot_name, phone_xml=wrist_camera_mjcf(calibration, mount, show_phone=True),
        phone_parent_body=mount.parent_body,
    )
    _place_demo_targets(model, data, mount.camera_name)
    pose = _world_from_cv_camera(data, model.camera(mount.camera_name).id)
    assert pose[2, 2] < -.8
    targets = np.array([data.geom_xpos[model.geom(name).id] for name in ("gauge_block", "inspection_boss")])
    optical_points = transform_points(targets, np.linalg.inv(pose))
    assert np.all(optical_points[:, 2] > 0)
    pixels = project_points(optical_points, calibration)
    assert np.all((pixels[:, 0] >= 0) & (pixels[:, 0] < calibration.width))
    assert np.all((pixels[:, 1] >= 0) & (pixels[:, 1] < calibration.height))


@pytest.mark.skipif((os.getenv("IPHONE_MUJOCO_TEST_RENDER") or os.getenv("ASMAGIC_TEST_RENDER")) != "1", reason="Set IPHONE_MUJOCO_TEST_RENDER=1 in an OpenGL session")
def test_wrist_rgbd_recording_preserves_same_frame_state_and_iphone_depth(tmp_path, robot_name):
    calibration = CameraCalibration(64, 48, 59., 57., 29.3, 21.8, 32, 24)
    mount = default_wrist_mount(robot_name)
    model, data, spec = load_scene(
        robot_name, phone_xml=wrist_camera_mjcf(calibration, mount),
        phone_parent_body=mount.parent_body,
    )
    directory = tmp_path / "wrist"
    scale = .0001
    frames, snapshots = [], []
    joint_position = int(model.joint(spec.joint_names[0]).qposadr[0])
    home_position = data.qpos[joint_position]
    with WristRGBDRecorder(model, data, spec, mount, calibration, directory, depth_scale=scale) as recorder:
        for index in range(2):
            data.qpos[joint_position] = home_position + index * .15
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
