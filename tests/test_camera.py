"""Focused camera geometry tests; rendering is opt-in for graphical sessions."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from iphone_mujoco.camera import (
    CameraCalibration, RGBDFrame, MujocoRGBDCamera, apply_calibration,
    camera_mjcf, frame_from_asmagic,
)
from iphone_mujoco.geometry import (
    ARKIT_WORLD_TO_MUJOCO, CV_TO_ARKIT, convert_iphone_pose,
    pose_from_xyzw, project_points,
)
from iphone_mujoco.profiles import DeviceProfile, phone_mjcf, synthetic_profile


def test_column_major_K_and_pixel_center_depth_scaling(tmp_path):
    calibration = CameraCalibration.from_column_major(
        [4, 0, 0, 0, 6, 0, 1.2, 1.8, 1], width=4, height=4,
        depth_width=2, depth_height=2,
    )
    np.testing.assert_allclose(calibration.matrix, [[4, 0, 1.2], [0, 6, 1.8], [0, 0, 1]])
    np.testing.assert_allclose(calibration.depth_calibration().matrix,
                               [[2, 0, .35], [0, 3, .65], [0, 0, 1]])
    calibration.save(tmp_path / "K.json")
    assert CameraCalibration.load(tmp_path / "K.json") == calibration
    with pytest.raises(ValueError, match="skew"):
        CameraCalibration.from_column_major([4, 1, 0, 0, 6, 0, 1, 1, 1], width=4, height=4)
    with pytest.raises(ValueError):
        CameraCalibration(0, 4, 1, 1, 0, 0)


def test_point_cloud_round_trip_metric_optical_depth_and_invalid_samples():
    calibration = CameraCalibration(4, 4, 4, 6, 1.2, 1.8, 2, 2)
    rgb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    frame = RGBDFrame(1., rgb, [[2., 0.], [np.nan, 3.]], calibration,
                      CV_TO_ARKIT, confidence=np.array([[2, 0], [1, 2]], dtype=np.uint8))
    cloud = frame.point_cloud(min_confidence=2)
    assert cloud.points.shape == (2, 3)
    np.testing.assert_equal(cloud.pixels, [[0, 0], [1, 1]])
    np.testing.assert_allclose(cloud.points[:, 2], [2, 3])
    # Z depth differs from Euclidean range away from the principal ray.
    assert np.linalg.norm(cloud.points[0]) > 2
    np.testing.assert_allclose(project_points(cloud.points, calibration.depth_calibration()), cloud.pixels)
    np.testing.assert_equal(cloud.colors, rgb[[1, 3], [1, 3]])
    np.testing.assert_allclose(frame.point_cloud(world=True).points, cloud.points*[1, -1, -1])
    with pytest.raises(ValueError, match="no depth"):
        RGBDFrame(1., None, None, calibration).point_cloud()
    with pytest.raises(ValueError, match="known world"):
        RGBDFrame(1., None, np.ones((2, 2)), calibration).point_cloud(world=True)


def test_explicit_phone_basis_and_quaternion_layout():
    pose = pose_from_xyzw([1, 2, 3, 0, 0, 0, 1])
    converted = convert_iphone_pose(pose, "arkit")
    np.testing.assert_allclose(converted[:3, 3], [-3, -1, 2])
    # CV forward -> ARKit -Z -> robotics +X. CV right -> robotics -Y.
    np.testing.assert_allclose(converted[:3, :3] @ [0, 0, 1], [1, 0, 0])
    np.testing.assert_allclose(converted[:3, :3] @ [1, 0, 0], [0, -1, 0])
    np.testing.assert_allclose(convert_iphone_pose(pose, "robotics"), pose @ CV_TO_ARKIT)
    np.testing.assert_allclose(ARKIT_WORLD_TO_MUJOCO[:3, :3].T @ ARKIT_WORLD_TO_MUJOCO[:3, :3], np.eye(3))
    with pytest.raises(ValueError, match="custom"):
        convert_iphone_pose(pose, "custom")
    with pytest.raises(ValueError, match="zero"):
        pose_from_xyzw([0]*7)


def _wire_frame(**kwargs):
    values = dict(rgb_width=4, rgb_height=4,
                  camera_intrinsics=np.array([4, 0, 0, 0, 6, 0, 1.2, 1.8, 1]),
                  depth_array=np.array([[10000, 0], [65535, 20000]], dtype=np.uint16),
                  depth_width=2, depth_height=2, depth_meters_per_unit=.0001,
                  local_pose=np.array([1, 2, 3, 0, 0, 0, 1]), global_pose=np.zeros(7),
                  global_pose_is_shared=False, shared_world_identifier="", source_timestamp=12.5,
                  timestamp=1000., frame_id=7, session_identifier="capture-A", color_array=None,
                  image_rotation_quarter_turns_clockwise=1, depth_confidence_array=None)
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_live_frame_uses_same_frame_calibration_and_preserves_missing_information(tmp_path):
    frame = frame_from_asmagic(_wire_frame(), pose_convention="arkit")
    np.testing.assert_allclose(frame.depth_m[[0, 1], [0, 1]], [1, 2])
    assert np.isnan(frame.depth_m[0, 1]) and np.isnan(frame.depth_m[1, 0])
    assert frame.timestamp == 12.5
    assert frame.rgb is None and frame.confidence is None
    assert frame.metadata["depth_saturated_pixel_count"] == 1
    assert frame.metadata["image_rotation_quarter_turns_clockwise"] == 1
    # Display orientation is retained separately; raw K and depth are unrotated.
    assert frame.calibration.cx == 1.2
    assert frame_from_asmagic(_wire_frame()).world_from_camera is None
    with pytest.raises(ValueError, match="reference dimensions"):
        frame_from_asmagic(_wire_frame(rgb_width=0))
    with pytest.raises(ValueError, match="depth_meters_per_unit"):
        frame_from_asmagic(_wire_frame(depth_meters_per_unit=0))
    legacy = frame_from_asmagic(_wire_frame(depth_meters_per_unit=0), legacy_depth_scale=.0001)
    assert legacy.depth_m[0, 0] == 1
    with pytest.raises(ValueError, match="shared-world"):
        frame_from_asmagic(_wire_frame(), pose_convention="arkit", pose_field="global")
    profile = DeviceProfile.from_frame(frame)
    profile.save(tmp_path / "phone.json")
    restored = DeviceProfile.load(tmp_path / "phone.json")
    assert restored.calibration == frame.calibration
    assert restored.body_dimensions_m is None and not restored.camera_mount_measured
    assert restored.metadata["capture_metadata"]["session_identifier"] == "capture-A"


def test_native_mujoco_frustum_matches_offcenter_K():
    """No GL needed: verify native frustum signs and both focal lengths."""
    import mujoco
    calibration = CameraCalibration(160, 120, 170., 95., 51., 80.)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody>'+camera_mjcf(calibration)+'</worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=10)
    camera = mujoco.MjvCamera()
    camera.type, camera.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, 0
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    view = scene.camera[0]
    near = view.frustum_near
    # MuJoCo's frustum_width stores the half-width at the near plane.
    left = view.frustum_center-view.frustum_width
    right = view.frustum_center+view.frustum_width
    # The raster uses pixel centers .5 from each viewport edge. GL Y is up.
    principal = [(-left)/(right-left)*calibration.width-.5,
                 view.frustum_top/(view.frustum_top-view.frustum_bottom)*calibration.height-.5]
    focal = [near/(right-left)*calibration.width,
             near/(view.frustum_top-view.frustum_bottom)*calibration.height]
    np.testing.assert_allclose(principal, [calibration.cx, calibration.cy], atol=2e-5)
    np.testing.assert_allclose(focal, [calibration.fx, calibration.fy], atol=2e-5)
    rig = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+phone_mjcf(calibration, mocap=True)+'</worldbody></mujoco>')
    assert rig.ncam == 1 and rig.nmocap == 1
    assert synthetic_profile().source == "synthetic_example"


@pytest.mark.skipif((os.getenv("IPHONE_MUJOCO_TEST_RENDER") or os.getenv("ASMAGIC_TEST_RENDER")) != "1", reason="Set IPHONE_MUJOCO_TEST_RENDER=1 in an OpenGL session")
def test_render_metric_depth_principal_point_and_missing_returns():
    import mujoco
    calibration = CameraCalibration(160, 120, 170., 95., 51., 80., 80, 60)
    xml = ('<mujoco><visual><headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>'
           '<worldbody>'+camera_mjcf(calibration)+
           '<geom type="plane" pos="0 0 -3" size="5 5 .1" rgba=".2 .2 .2 1"/>'
           '<geom type="sphere" pos=".12 -.18 -2" size=".045" rgba="1 0 0 1"/>'
           '</worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with MujocoRGBDCamera(model, calibration) as camera:
        frame = camera.render(data)
        assert frame.rgb.shape == (120, 160, 3) and frame.depth_m.shape == (60, 80)
        red = (frame.rgb[..., 0] > 100) & (frame.rgb[..., 1] < 50)
        v, u = np.nonzero(red)
        assert len(u) > 5
        np.testing.assert_allclose([u.mean(), v.mean()], [61.2, 88.55], atol=.8)
        np.testing.assert_allclose(np.median(frame.depth_m), 3., atol=2e-4)
        np.testing.assert_allclose(np.median(frame.point_cloud().points[:, 2]), 3., atol=2e-4)
        camera.set_world_pose(data, CV_TO_ARKIT)
        np.testing.assert_allclose(camera.render(data).world_from_camera, CV_TO_ARKIT, atol=1e-6)
        changed_K = CameraCalibration(160, 120, 155., 105., 62., 57., 80, 60)
        camera.update_calibration(changed_K)
        assert camera.render(data).calibration == changed_K
    empty = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+camera_mjcf(calibration)+'</worldbody></mujoco>')
    with MujocoRGBDCamera(empty, calibration) as camera:
        assert np.isnan(camera.render(mujoco.MjData(empty)).depth_m).all()
    rig = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+phone_mjcf(calibration, mocap=True)+'</worldbody></mujoco>')
    rig_data = mujoco.MjData(rig)
    with MujocoRGBDCamera(rig, calibration) as camera:
        desired = CV_TO_ARKIT.copy()
        desired[:3, 3] = [.2, -.1, .8]
        camera.set_world_pose(rig_data, desired, move_body=True)
        np.testing.assert_allclose(camera.render(rig_data).world_from_camera, desired, atol=1e-6)
