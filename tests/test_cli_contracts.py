"""Public CLI checks that need neither graphics nor a physical phone."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from iphone_mujoco.camera import CameraCalibration, RGBDFrame
from iphone_mujoco.cli import main
from iphone_mujoco.recording import RecordingWriter
from iphone_mujoco.streams import LiveSample


@pytest.mark.parametrize("argv", [
    ["phone-lab", "--duration", "nan"],
    ["phone-lab", "--fps", "inf"],
    ["calibrate", "--ip", "fixture", "--output", "unused.json", "--timeout", "-1"],
    ["capture-wrist", "--profile", "unused.json", "--output", "unused", "--depth-scale", "nan"],
    ["replay", "unused", "--speed", "0"],
    ["simulate", "--translation-scale=-inf"],
])
def test_invalid_numbers_are_rejected_before_starting_runtime(argv, capsys):
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    assert "must be" in capsys.readouterr().err


def test_doctor_reports_graphics_as_unprobed(monkeypatch, capsys):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    assert main(["doctor"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["graphics"]["status"] == "not_probed"
    assert result["graphics"]["requested_backend"] == "egl"
    assert result["graphics"]["context_required_for_headless_rgbd"] is True
    assert "iphone_mujoco" in result


def _episode(directory):
    calibration = CameraCalibration(8, 6, 7.5, 7.2, 3.3, 2.7, 4, 3)
    depth = np.full((3, 4), 1.25, dtype=np.float32)
    depth[0, 0] = np.nan
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    frame = RGBDFrame(1., rgb, depth, calibration, np.eye(4),
                      metadata={"timestamp_clock": "simulation", "depth_meters_per_unit": .0001})
    empty = RGBDFrame(2., None, None, calibration, np.eye(4))
    with RecordingWriter(directory, metadata={"fixture": "same recording contract as users"}) as writer:
        writer.write(LiveSample(timestamp=10., received_at=100., source_timestamp=4., rgbd=frame,
                                pose=np.eye(4), session_id="session-a", metadata={"source": "fixture"}))
        writer.write(LiveSample(timestamp=11., received_at=100.1, source_timestamp=3., rgbd=empty,
                                pose=np.eye(4), session_id="session-a", metadata={"source": "fixture"}))
        writer.write(LiveSample(timestamp=1., received_at=100.2, source_timestamp=.2,
                                session_id="new-origin", metadata={"source": "pose-only"}))
    return calibration


def _fingerprint(directory):
    return {str(p.relative_to(directory)): sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


def test_inspection_reports_shapes_calibration_missing_data_and_independent_clocks(tmp_path, capsys):
    directory = tmp_path / "episode"
    calibration = _episode(directory)
    before = _fingerprint(directory)
    assert main(["inspect-episode", str(directory), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert _fingerprint(directory) == before
    assert result["frame_count"] == 3 and result["session_count"] == 2
    assert result["sources"] == {"fixture": 2, "pose-only": 1}
    assert result["rgb_shapes"] == [{"shape": [6, 8, 3], "frames": 1}]
    assert result["depth_shapes"] == [{"shape": [3, 4], "frames": 1}]
    np.testing.assert_equal(result["calibrations"][0]["K_rgb"], calibration.matrix)
    assert result["missing"] == {"rgbd": 1, "rgb": 2, "depth": 2, "pose": 1, "depth_confidence": 3}
    assert result["depth"]["valid_pixels"] == 11
    assert result["depth"]["meters_per_unit_values"] == [.0001]
    assert result["timestamps"]["source_timestamp"]["regressions"] == 1
    # The sender clock restarted in a new session; receipt remained monotonic.
    assert result["timestamps"]["sender_timestamp"]["regressions"] == 0
    assert result["timestamps"]["relative_received_at"]["nondecreasing"] is True
    assert "No cross-device latency" in result["clock_note"]


@pytest.mark.parametrize("count", [True, 1.5])
def test_inspection_rejects_invalid_manifest_count_with_path(tmp_path, capsys, count):
    directory = tmp_path / "episode"
    _episode(directory)
    path = directory / "manifest.json"
    document = json.loads(path.read_text())
    document["frame_count"] = count
    path.write_text(json.dumps(document))
    assert main(["inspect-episode", str(directory), "--json"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert str(path) in captured.err and "frame_count" in captured.err


def test_inspection_names_corrupt_frame_array_without_traceback(tmp_path, capsys):
    directory = tmp_path / "episode"
    _episode(directory)
    path = directory / "frames" / "00000000.npz"
    path.write_bytes(b"corrupt archive")
    assert main(["inspect-episode", str(directory), "--json"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert str(path) in captured.err and "Traceback" not in captured.err


def test_calibration_creates_nested_output_parent(tmp_path, monkeypatch, capsys):
    from iphone_mujoco import streams
    calibration = CameraCalibration(8, 6, 7.5, 7.2, 3.3, 2.7)
    frame = RGBDFrame(1., np.zeros((6, 8, 3), dtype=np.uint8), None, calibration)
    sample = LiveSample(timestamp=2., received_at=3., rgbd=frame, session_id="fixture", frame_id=1)

    @contextmanager
    def fake_source(*args, **kwargs):
        yield SimpleNamespace(poll=lambda **unused: sample)

    # Only isolate the network for an output-path unit check; no decoder is replaced.
    monkeypatch.setattr(streams, "AsMagicSource", fake_source)
    output = tmp_path / "new" / "nested" / "camera.json"
    assert main(["calibrate", "--ip", "fixture", "--output", str(output)]) == 0
    capsys.readouterr()
    assert CameraCalibration.load(output) == calibration
    assert output.with_suffix(".json.source.json").is_file()
