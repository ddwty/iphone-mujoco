"""Portable iPhone modules and laboratory scenes; no OpenGL is required."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from iphone_mujoco.camera import CameraCalibration
from iphone_mujoco.geometry import pose_from_xyzw
from iphone_mujoco.phone_scene import LAB_METRICS, SCENE_PATH, export_phone_lab, load_phone_lab


def _world_from_body(data, body_id):
    pose = np.eye(4)
    pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
    pose[:3, 3] = data.xpos[body_id]
    return pose


def _world_from_mujoco_camera(data, camera_id):
    pose = np.eye(4)
    pose[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3)
    pose[:3, 3] = data.cam_xpos[camera_id]
    return pose


def test_phone_module_export_works_under_another_body_and_preserves_existing_files(tmp_path):
    from iphone_mujoco.profiles import write_phone_module

    calibration = CameraCalibration(160, 120, 149., 137., 76.2, 63.1, 80, 60)
    local_pose = pose_from_xyzw([.12, -.04, .08, .1, -.2, .15, .95])
    path = tmp_path / "phone_a.xml"
    write_phone_module(path, calibration, body_name="phone_a", camera_name="camera_a",
                       parent_from_body=local_pose)
    sidecar = path.with_suffix(".calibration.json")
    assert CameraCalibration.load(sidecar) == calibration
    include = ET.parse(path).getroot()
    assert include.tag == "mujocoinclude"
    phone = include.find("body[@name='phone_a']")
    assert phone is not None
    camera = phone.find("camera[@name='camera_a']")
    assert camera is not None
    local_mount = np.eye(4)
    local_mount[:3, 3] = np.fromstring(camera.get("pos", "0 0 0"), sep=" ")
    quaternion = np.fromstring(camera.get("quat", "1 0 0 0"), sep=" ")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    local_mount[:3, :3] = rotation.reshape(3, 3)
    scene = tmp_path / "host.xml"
    scene.write_text('<mujoco><worldbody><body name="host" pos=".7 -.3 .4" '
                     'quat=".9238795325 0 0 .3826834324"><include file="phone_a.xml"/>'
                     '</body></worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera_id, host_id = model.camera("camera_a").id, model.body("host").id
    expected = _world_from_body(data, host_id) @ local_pose @ local_mount
    np.testing.assert_allclose(_world_from_mujoco_camera(data, camera_id), expected, atol=1e-9)
    assert model.nmocap == 0
    original_xml, original_json = path.read_bytes(), sidecar.read_bytes()
    with pytest.raises(FileExistsError):
        write_phone_module(path, calibration, body_name="replacement", camera_name="other")
    assert path.read_bytes() == original_xml and sidecar.read_bytes() == original_json


def test_phone_lab_has_visible_mocap_phone_and_requested_calibration():
    calibration = CameraCalibration(320, 240, 298., 271., 145.2, 127.4, 128, 96)
    model, data = load_phone_lab(calibration)
    phone_id, camera_id = model.body("iphone").id, model.camera("iphone_camera").id
    assert model.body_mocapid[phone_id] >= 0
    assert model.cam_bodyid[camera_id] == phone_id
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index) for index in range(model.ncam)]
    assert len(names) == len(set(names))
    np.testing.assert_equal(model.cam_resolution[camera_id], [calibration.width, calibration.height])
    np.testing.assert_allclose(model.cam_intrinsic[camera_id],
                               [calibration.fx, calibration.fy,
                                (calibration.width-1)/2-calibration.cx,
                                calibration.cy-(calibration.height-1)/2], atol=1e-5)
    options = mujoco.MjvOption()
    phone_geoms = np.flatnonzero(model.geom_bodyid == phone_id)
    assert len(phone_geoms) >= 3
    assert np.all(options.geomgroup[model.geom_group[phone_geoms]] != 0)
    invisible = phone_geoms[model.geom_rgba[phone_geoms, 3] == 0]
    assert [model.geom(index).name for index in invisible] == ["iphone_shell_bounds"]
    assert np.all(model.geom_contype[phone_geoms] == 0)
    np.testing.assert_allclose(data.cam_xpos[camera_id], [0., -.8, 1.05], atol=1e-8)


def test_inspection_bench_preserves_metric_fixtures_and_has_a_connected_phone_stand():
    from iphone_mujoco.devices import get_device
    model, data = load_phone_lab()
    table = model.geom("tabletop").id
    top = data.geom_xpos[table, 2] + model.geom_size[table, 2]
    np.testing.assert_allclose(top, LAB_METRICS["table_top_height_m"], atol=1e-12)
    for height in (40, 80, 120):
        step = model.geom(f"step_{height}mm").id
        np.testing.assert_allclose(data.geom_xpos[step, 2] - model.geom_size[step, 2], top, atol=1e-12)
        np.testing.assert_allclose(data.geom_xpos[step, 2] + model.geom_size[step, 2] - top, height / 1000, atol=1e-12)
    checker = [model.geom(f"checker_0_{col}").id for col in range(8)]
    np.testing.assert_allclose(np.diff(data.geom_xpos[checker, 0]), LAB_METRICS["checker_square_m"], atol=1e-12)
    ruler = [model.geom(f"ruler_tick_{index}").id for index in range(11)]
    np.testing.assert_allclose(np.diff(data.geom_xpos[ruler, 0]), LAB_METRICS["ruler_major_spacing_m"], atol=1e-12)
    np.testing.assert_allclose(model.geom_size[model.geom("ruler_minor_tick_1").id, 0], .00035)

    assert model.nmesh >= 3
    device = get_device("iphone-17-pro")
    phone = model.body("iphone").id
    plate = model.geom("iphone_mount_backplate").id
    local_plate = (data.geom_xpos[plate] - data.xpos[phone]) @ data.xmat[phone].reshape(3, 3)
    np.testing.assert_allclose(local_plate[2] - model.geom_size[plate, 2], device.thickness_m / 2, atol=1e-12)
    for side, sign in (("left", -1), ("right", 1)):
        jaw = model.geom("iphone_mount_jaw_" + side).id
        local_jaw = (data.geom_xpos[jaw] - data.xpos[phone]) @ data.xmat[phone].reshape(3, 3)
        np.testing.assert_allclose(sign * local_jaw[0] - model.geom_size[jaw, 0], device.width_m / 2, atol=1e-12)
    socket = model.geom("stand_ball_head_socket").id
    axis = data.geom_xmat[socket].reshape(3, 3)[:, 2]
    endpoints = data.geom_xpos[socket] + np.outer([-1, 1], axis * model.geom_size[socket, 1])
    ball = data.geom_xpos[model.geom("iphone_mount_ball").id]
    np.testing.assert_allclose(np.min(np.linalg.norm(endpoints - ball, axis=1)), 0., atol=1e-12)


def test_standalone_inspection_stage_moves_through_its_servo():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    joint = model.joint("inspection_slide").id
    actuator = model.actuator("inspection_slide_drive").id
    address = int(model.jnt_qposadr[joint])
    np.testing.assert_allclose(model.jnt_range[joint], [-.08, .08])
    data.ctrl[actuator] = .06
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert .04 < data.qpos[address] < .08
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()


def test_exported_phone_lab_compiles_after_relocation(tmp_path):
    calibration = CameraCalibration(160, 120, 148., 152., 79.1, 58.7, 64, 48)
    original = tmp_path / "original"
    exported_scene = export_phone_lab(original, calibration)
    assert Path(exported_scene) == original / "scene.xml"
    assert CameraCalibration.load(original / "calibration.json") == calibration
    moved = tmp_path / "elsewhere" / "downloaded_lab"
    shutil.copytree(original, moved)
    original.rename(tmp_path / "original_is_no_longer_at_export_path")
    for path in moved.glob("*.xml"):
        root = ET.parse(path).getroot()
        for element in root.iter():
            if "file" in element.attrib:
                assert not Path(element.attrib["file"]).is_absolute()
                assert (moved / element.attrib["file"]).is_file()
            for value in element.attrib.values():
                assert str(original) not in value
    model = mujoco.MjModel.from_xml_path(str(moved / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.body_mocapid[model.body("iphone").id] >= 0
    camera_id = model.camera("iphone_camera").id
    np.testing.assert_equal(model.cam_resolution[camera_id], [160, 120])
    assert np.isfinite(data.cam_xpos[camera_id]).all()


def test_selected_lens_changes_optical_pose_without_moving_lab_phone():
    from iphone_mujoco.configuration import configure_camera
    from iphone_mujoco.devices import get_device
    poses = []
    for zoom in (.5, 1., 2., 4., 8.):
        config = configure_camera(zoom=zoom, region="cn")
        model, data = load_phone_lab(configuration=config)
        phone = _world_from_body(data, model.body("iphone").id)
        optical = _world_from_mujoco_camera(data, model.camera("iphone_camera").id)
        from iphone_mujoco.geometry import CV_TO_MUJOCO_CAMERA
        expected = phone @ get_device(config.device_id).phone_from_camera(config.lens)
        np.testing.assert_allclose(optical @ CV_TO_MUJOCO_CAMERA, expected, atol=1e-10)
        poses.append(phone)
    for pose in poses[1:]:
        np.testing.assert_allclose(pose, poses[0], atol=1e-10)


def test_catalog_export_is_portable_for_lab_and_robot_body(tmp_path):
    from iphone_mujoco.configuration import CameraConfiguration, configure_camera
    from iphone_mujoco.profiles import write_phone_module
    config = configure_camera("iphone-17-pro", zoom=.5, region="cn", resolution=(320, 240))
    original = tmp_path / "original"
    export_phone_lab(original, configuration=config)
    moved = tmp_path / "moved"
    shutil.move(str(original), moved)
    model = mujoco.MjModel.from_xml_path(str(moved / "scene.xml"))
    assert model.nmesh > 0 and model.ncam >= 1
    assert CameraConfiguration.load(moved / "device.json").lens == "ultra_wide"
    write_phone_module(moved / "payload.xml", config.calibration,
                       configuration=config, payload=True, body_name="payload_phone")
    host = moved / "host.xml"
    host.write_text('<mujoco><asset><include file="payload.assets.xml"/></asset>'
                    '<worldbody><body name="host" pos="0 0 1"><joint/>'
                    '<include file="payload.xml"/></body></worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_path(str(host))
    np.testing.assert_allclose(model.body_mass.sum(), .204, atol=1e-10)
    assert CameraCalibration.load(moved / "payload.device.json") == config.calibration
