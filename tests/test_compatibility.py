"""Public migration and NumPy calibration serialization contracts."""
import importlib
import warnings

import numpy as np

from iphone_mujoco.camera import CameraCalibration
from iphone_mujoco.recording import RecordingWriter, ReplaySource
from iphone_mujoco.streams import LiveSample
from iphone_mujoco.robots import default_model_cache


def test_legacy_alias_uses_canonical_sample_identity_and_recording(tmp_path):
    legacy_name = "asmagic" + "_mujoco"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = importlib.import_module(legacy_name + ".streams")
        old_recording = importlib.import_module(legacy_name + ".recording")
        root = importlib.import_module(legacy_name)
    assert legacy.LiveSample is LiveSample
    assert root.streams is legacy
    assert old_recording.RecordingWriter is RecordingWriter
    episode = tmp_path / "episode"
    with old_recording.RecordingWriter(episode) as writer:
        writer.write(legacy.LiveSample(timestamp=1., received_at=2., session_id="migration"))
    with ReplaySource(episode) as source:
        assert source.poll().session_id == "migration"


def test_numpy_calibration_exports_nested_json_and_loads(tmp_path):
    calibration = CameraCalibration(np.int64(64), np.int32(48), np.float32(59.25),
                                    np.float32(60.5), np.float32(31.5), np.float32(23.5))
    path = tmp_path / "profiles" / "camera.json"
    calibration.save(path)
    assert CameraCalibration.load(path) == calibration


def test_model_cache_reuses_previous_install_and_prefers_explicit_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("IPHONE_MUJOCO_MODEL_CACHE", "ASMAGIC_MUJOCO_MODEL_CACHE"):
        monkeypatch.delenv(name, raising=False)
    previous = tmp_path / ".cache" / ("asmagic" + "_mujoco") / "menagerie"
    previous.mkdir(parents=True)
    assert default_model_cache() == previous
    canonical = tmp_path / ".cache" / "iphone_mujoco" / "menagerie"
    canonical.mkdir(parents=True)
    assert default_model_cache() == canonical
    override = tmp_path / "models"
    monkeypatch.setenv("IPHONE_MUJOCO_MODEL_CACHE", str(override))
    assert default_model_cache() == override
