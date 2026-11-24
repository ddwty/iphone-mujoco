"""Independent frame/placement checks; no physical tracking is implied."""
import json
import os

import mujoco
import numpy as np
import pytest

pytest.importorskip("asDevice")
pytest.importorskip("asTriad")

from asDevice import Transform
from iphone_mujoco.camera import MujocoRGBDCamera, RGBDFrame
from iphone_mujoco.configuration import configure_camera
from iphone_mujoco.devices import device_ids, get_device
from iphone_mujoco.geometry import CV_TO_MUJOCO_CAMERA, pose_from_xyzw
from iphone_mujoco.phone_scene import load_phone_lab
from iphone_mujoco.recording import RecordingWriter, ReplaySource
from iphone_mujoco.spatial import DevicePlacement, SpatialOverlay


@pytest.mark.parametrize("device", device_ids())
@pytest.mark.parametrize("lens", ("wide", "ultra_wide", "telephoto"))
def test_device_01_is_front_glass_and_lens_selection_preserves_body(device, lens):
    config = configure_camera(device, lens=lens)
    placement = DevicePlacement(config)
    body = pose_from_xyzw([.2, -.4, .9, .12, -.2, .05, .96])
    optical = body @ get_device(device).phone_from_camera(lens)
    # Independent body datum: 01 is on the front surface, not at the shell centre.
    expected = body.copy()
    expected[:3, 3] += body[:3, 2] * get_device(device).thickness_m / 2
    actual = placement.world_from_device(optical)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(placement.world_from_camera(actual), optical, atol=1e-12)
    np.testing.assert_allclose(placement.world_from_body(actual), body, atol=1e-12)
    rear = np.diag([-1., 1., -1., 1.])
    rear[2, 3] = -get_device(device).thickness_m
    np.testing.assert_allclose(placement.world_from_frame(optical, "03"), expected @ rear, atol=1e-12)
    assert np.linalg.det(placement.world_from_frame(optical, "03")[:3, :3]) == pytest.approx(1)


def test_nominal_cover_reference_and_optical_axes_remain_distinct():
    placement = DevicePlacement(configure_camera())
    optical = np.eye(4)
    cover = placement.world_from_frame(optical, "11")
    np.testing.assert_allclose(cover[:3, 3], optical[:3, 3], atol=1e-12)
    # 11 has rear-view XYZ. CV optical has a different proper rotation.
    np.testing.assert_allclose(cover[:3, :3], np.diag([-1, -1, 1]), atol=1e-12)
    assert placement.frames.definition_revision == 5


def test_measured_mount_overrides_cover_estimate_and_metadata_replays(tmp_path):
    config = configure_camera()
    config.metadata["phone_from_camera"] = pose_from_xyzw([.03, .04, -.012, .2, .1, .3, .9]).tolist()
    placement = DevicePlacement(config)
    device_pose = np.asarray(Transform.from_rotation_translation(
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)), (.4, -.2, .8)).matrix)
    optical = placement.world_from_camera(device_pose)
    np.testing.assert_allclose(placement.world_from_device(optical), device_pose, atol=1e-12)
    frame = RGBDFrame(2., np.zeros((480, 640, 3), np.uint8), np.ones((192, 256), np.float32),
                      config.calibration, optical, "simulation", metadata={"timestamp_clock": "simulation"})
    SpatialOverlay(config).annotate(frame)
    assert frame.metadata["spatial_frames"]["camera_mount_source"] == "configuration_phone_from_camera"
    with RecordingWriter(tmp_path / "episode") as writer:
        writer.write(frame)
    with ReplaySource(tmp_path / "episode") as replay:
        restored = replay.poll().rgbd
    assert restored.metadata == frame.metadata
    json.dumps(restored.metadata, allow_nan=False)
    np.testing.assert_allclose(restored.world_from_camera, optical)
    np.testing.assert_array_equal(restored.depth_m, frame.depth_m)


def test_real_astriad_appends_without_changing_model_or_sensor_arrays():
    config = configure_camera()
    model, data = load_phone_lab(configuration=config)
    optical = np.eye(4)
    optical[:3, :3] = data.cam_xmat[model.camera("iphone_camera").id].reshape(3, 3)
    optical[:3, 3] = data.cam_xpos[model.camera("iphone_camera").id]
    optical = optical @ CV_TO_MUJOCO_CAMERA
    frame = RGBDFrame(0., np.zeros((480, 640, 3), np.uint8), np.ones((192, 256), np.float32),
                      config.calibration, optical, "simulation")
    overlay = SpatialOverlay(config)
    scene = mujoco.MjvScene(model, maxgeom=overlay.geom_count + 1)
    scene.ngeom = 1
    model_geoms = model.ngeom
    rgb, depth = frame.rgb.copy(), frame.depth_m.copy()
    assert all(type(triad).__module__ == "asTriad.core" for triad in overlay.triads)
    overlay.append(scene, frame)
    assert scene.ngeom == overlay.geom_count + 1
    assert model.ngeom == model_geoms
    np.testing.assert_array_equal(frame.rgb, rgb)
    np.testing.assert_array_equal(frame.depth_m, depth)
    with pytest.raises(ValueError, match="free visual geometry"):
        overlay.append(scene, frame)
    assert scene.ngeom == overlay.geom_count + 1


@pytest.mark.skipif(os.environ.get("IPHONE_MUJOCO_TEST_RENDER") != "1", reason="requires OpenGL")
def test_asdevice_drives_phone_body_and_preserves_optical_capture():
    config = configure_camera(zoom=.5, resolution=(160, 120), depth_resolution=(80, 60))
    model, data = load_phone_lab(configuration=config)
    placement = DevicePlacement(config)
    target = pose_from_xyzw([.12, -.75, 1.1, .1, .2, .3, .9])
    with MujocoRGBDCamera(model, config.calibration) as camera:
        placement.place_camera(camera, data, target)
        frame = camera.render(data)
    body = model.body("iphone").id
    expected_body = placement.world_from_body(target)
    np.testing.assert_allclose(data.xpos[body], expected_body[:3, 3], atol=1e-10)
    np.testing.assert_allclose(data.xmat[body].reshape(3, 3), expected_body[:3, :3], atol=1e-10)
    np.testing.assert_allclose(frame.world_from_camera, placement.world_from_camera(target), atol=1e-10)
    assert frame.calibration == config.calibration
