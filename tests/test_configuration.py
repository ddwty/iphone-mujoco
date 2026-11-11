"""Camera selection, provenance and measured-profile reuse contracts."""
import math

import pytest

from iphone_mujoco.camera import CameraCalibration
from iphone_mujoco.configuration import (
    CameraConfiguration, bind_runtime_calibration, configure_camera,
)


def test_public_specs_use_equivalent_diagonal_and_registered_simulation_depth():
    camera = configure_camera("iphone-17-pro")
    expected_focal = 24 * math.hypot(640, 480) / math.hypot(36, 24)
    assert camera.calibration.fx == pytest.approx(expected_focal)
    assert camera.calibration.fy == camera.calibration.fx
    assert camera.calibration.cx == 319.5 and camera.calibration.cy == 239.5
    assert (camera.calibration.depth_width, camera.calibration.depth_height) == (256, 192)
    assert camera.calibration_source == "public_specs_ideal_pinhole"
    assert camera.metadata["depth_source"] == "simulation"
    assert camera.metadata["hardware_depth_verified"] is False
    assert camera.metadata["specification_sources"]["optics"].startswith("https://support.apple.com/")


def test_zoom_presets_select_the_physical_lens_and_crop_keeps_its_identity():
    wide = configure_camera("iphone-17-pro-max")
    crop = configure_camera("iphone-17-pro-max", zoom=2)
    crop_28mm = configure_camera("iphone-17-pro-max", zoom=1.2)
    tele = configure_camera("iphone-17-pro-max", lens="telephoto")
    tele_crop = configure_camera("iphone-17-pro-max", zoom=8)
    ultra = configure_camera("iphone-17-pro-max", lens="ultra_wide")
    assert wide.lens == crop.lens == "wide"
    assert crop.calibration.fx == pytest.approx(wide.calibration.fx * 2)
    assert crop_28mm.lens == "wide"
    assert crop_28mm.calibration.fx == pytest.approx(wide.calibration.fx * 28 / 24)
    assert "iph72395b28f" in crop_28mm.metadata["specification_sources"]["camera_mode"]
    assert tele.lens == tele_crop.lens == "telephoto"
    assert tele.zoom == 4 and tele_crop.zoom == 8 and ultra.zoom == .5
    assert tele_crop.metadata["camera_mode"]["crop"] is True
    assert tele_crop.calibration.fx == pytest.approx(tele.calibration.fx * 2)
    with pytest.raises(ValueError, match="lens/zoom"):
        configure_camera(lens="wide", zoom=8)
    with pytest.raises(ValueError, match="lens/zoom"):
        configure_camera(zoom=3)


def test_runtime_binding_round_trip_preserves_intrinsics_and_distinct_clocks(tmp_path):
    calibration = CameraCalibration(1280, 960, 1000., 1001., 601.2, 476.3, 256, 192)
    clocks = {"sender_timestamp": 1000., "source_timestamp": 12.5,
              "local_received_timestamp": 1400., "session_identifier": "session-A"}
    camera = bind_runtime_calibration(calibration, device_id="iphone-17-pro", metadata=clocks)
    clocks["source_timestamp"] = 99
    path = tmp_path / "nested" / "iphone.json"
    camera.save(path)
    restored = CameraConfiguration.load(path)
    assert restored.to_dict() == camera.to_dict()
    assert restored.calibration == calibration
    assert restored.calibration_source == "runtime_asmagic_frame"
    assert restored.metadata["source_timestamp"] == 12.5
    assert restored.metadata["sender_timestamp"] == 1000.
    assert restored.metadata["local_received_timestamp"] == 1400.
    assert restored.metadata["identity_source"] == "user_supplied_ar_main_camera"
    with pytest.raises(ValueError, match="wide lens at 1x"):
        bind_runtime_calibration(calibration, device_id="iphone-17-pro", zoom=2)


def test_bound_profile_only_allows_same_identity_and_pixel_center_resize(tmp_path):
    calibration = CameraCalibration(1280, 960, 1000., 1001., 601.2, 476.3, 256, 192)
    original = bind_runtime_calibration(calibration, device_id="iphone-17-pro")
    path = tmp_path / "measured.json"
    original.save(path)
    resized = configure_camera(profile=path, resolution=(640, 480), depth_resolution=(320, 240))
    assert resized.calibration.fx == 500
    assert resized.calibration.fy == 500.5
    assert resized.calibration.cx == pytest.approx((601.2 + .5) / 2 - .5)
    assert (resized.calibration.depth_width, resized.calibration.depth_height) == (320, 240)
    assert resized.calibration_source == original.calibration_source
    assert resized.metadata["resampling"]["operation"] == "full_raster_pixel_center_resize"
    with pytest.raises(ValueError, match="same-aspect"):
        configure_camera(profile=path, resolution=(640, 360))
    with pytest.raises(ValueError, match="exact device"):
        configure_camera(profile=path, device="iphone-17-pro-max")
    with pytest.raises(ValueError, match="exact device"):
        configure_camera(profile=path, zoom=2)


def test_bound_crop_profile_inherits_lens_and_zoom_without_deriving_other_zoom():
    original = configure_camera(zoom=8, resolution=(1280, 960))
    restored = configure_camera(profile=original)
    assert restored.lens == "telephoto" and restored.zoom == 8
    assert restored.calibration.fx == pytest.approx(original.calibration.fx / 2)
    with pytest.raises(ValueError, match="exact device"):
        configure_camera(profile=original, zoom=4)


def test_bare_calibration_is_user_bound_and_retains_legacy_capture_provenance():
    calibration = CameraCalibration(640, 480, 530., 531., 318.3, 240.1, 256, 192)
    bare = configure_camera(profile=calibration.to_dict(), zoom=2)
    assert bare.calibration == calibration
    assert bare.calibration_source == "user_supplied_intrinsics"
    assert bare.metadata["identity_source"] == "user_supplied"
    legacy = {"schema_version": 1, "calibration": calibration.to_dict(),
              "source": "runtime_asmagic_frame", "metadata": {"source_timestamp": 12.5}}
    bound = configure_camera(profile=legacy)
    assert bound.calibration_source == "user_supplied_intrinsics"
    assert bound.metadata["input_profile_metadata"]["source_timestamp"] == 12.5


@pytest.mark.parametrize("kwargs", [
    {"zoom": float("nan")}, {"zoom": float("inf")}, {"zoom": True},
    {"resolution": (0, 480)}, {"resolution": (640., 480)},
    {"depth_resolution": (256, False)}, {"region": "unknown"},
])
def test_invalid_selection_is_rejected(kwargs):
    with pytest.raises(ValueError):
        configure_camera(**kwargs)


def test_configuration_rejects_unknown_schema_and_copies_metadata(tmp_path):
    camera = configure_camera()
    exported = camera.to_dict()
    exported["metadata"]["camera_mode"]["lens"] = "changed"
    assert camera.metadata["camera_mode"]["lens"] == "wide"
    path = tmp_path / "unknown.json"
    path.write_text('{"schema":"iphone_mujoco.camera_configuration","schema_version":2}')
    with pytest.raises(ValueError, match="schema"):
        CameraConfiguration.load(path)
    with pytest.raises(ValueError, match="calibration profile"):
        configure_camera(profile={"focal_length": 24})
