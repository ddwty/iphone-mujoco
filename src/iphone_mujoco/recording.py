"""Portable JSON + numeric NPZ episodes; reading never enables pickle.

Every sample keeps its sender clocks, spatial identity, provenance and optional
RGB-D calibration. Replay schedules by local receipt intervals and rebases only
the controller's monotonic receipt time, preserving the original in metadata.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .streams import ControlButtons, LiveSample


FORMAT = "asmagic-mujoco-episode"
FORMAT_VERSION = 1


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class RecordingWriter:
    """Write a new episode directory without replacing an existing recording."""

    def __init__(self, directory: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError("Recording directory must be new or empty: " + str(self.directory))
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "frames").mkdir()
        self.count = 0
        self._first_received_at: float | None = None
        self._last_received_at: float | None = None
        self._closed = False
        self._manifest = {
            "format": FORMAT, "version": FORMAT_VERSION, "frame_count": 0,
            "created_at_unix": time.time(), "metadata": metadata or {},
            "array_format": "numpy-npz-no-pickle",
            "pose_convention": "simulation-world-from-cv-camera",
            "depth_units": "meters", "rgb_order": "RGB",
        }
        self._flush()

    def _flush(self) -> None:
        _write_json(self.directory / "manifest.json", self._manifest)

    def write(self, sample: LiveSample | Any) -> int:
        """Record a LiveSample or a simulation RGBDFrame; return its index."""
        if self._closed:
            raise RuntimeError("Recording is closed")
        if not isinstance(sample, LiveSample):
            # Simulation frames share the same artifact format as real frames.
            sample = LiveSample(
                timestamp=float(sample.timestamp), received_at=time.monotonic(),
                rgbd=sample, pose=sample.world_from_camera, frame_id=sample.frame_id,
                session_id=str(sample.metadata.get("session_id", "simulation")),
                metadata={"source": sample.source},
            )
        event_received_at = float(sample.metadata.get("update_received_at") or sample.received_at)
        if not all(math.isfinite(t) for t in (sample.received_at, sample.timestamp, event_received_at)):
            raise ValueError("Recording timestamps must be finite")
        if self._last_received_at is not None and event_received_at < self._last_received_at:
            raise ValueError("Recording receipt timestamps must be nondecreasing")
        arrays: dict[str, np.ndarray] = {}
        if sample.pose is not None:
            arrays["pose"] = np.asarray(sample.pose)
        frame_metadata = None
        if sample.rgbd is not None:
            frame = sample.rgbd
            for name in ("rgb", "depth_m", "world_from_camera", "confidence"):
                value = getattr(frame, name)
                if value is not None:
                    arrays[name] = np.asarray(value)
            frame_metadata = {
                "timestamp": frame.timestamp, "calibration": frame.calibration.to_dict(),
                "source": frame.source, "frame_id": frame.frame_id, "metadata": frame.metadata,
            }
        if any(value.dtype.hasobject for value in arrays.values()):
            raise ValueError("Object arrays are not supported in portable recordings")
        if self._first_received_at is None:
            self._first_received_at = event_received_at
        index = self.count
        stem = self.directory / "frames" / f"{index:08d}"
        sample_metadata = {
            "index": index, "timestamp": sample.timestamp,
            "received_at": sample.received_at,
            "event_received_at": event_received_at,
            "relative_received_at": event_received_at - self._first_received_at,
            "source_timestamp": sample.source_timestamp, "session_id": sample.session_id,
            "frame_id": sample.frame_id, "buttons": asdict(sample.buttons),
            "gripper": sample.gripper, "metadata": sample.metadata,
            "pose_stale": sample.pose_stale, "control_stale": sample.control_stale,
            "rgbd": frame_metadata, "array_keys": list(arrays),
        }
        # Serialize first, so invalid metadata cannot leave an indexed NPZ.
        json.dumps(_json_value(sample_metadata), allow_nan=False)
        temporary = stem.with_suffix(".npz.tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(output, **arrays)
        temporary.replace(stem.with_suffix(".npz"))
        _write_json(stem.with_suffix(".json"), sample_metadata)
        self.count += 1
        self._last_received_at = event_received_at
        self._manifest["frame_count"] = self.count
        self._flush()
        return index

    add = write

    def close(self) -> None:
        if not self._closed:
            self._manifest["closed_at_unix"] = time.time()
            self._flush()
            self._closed = True

    def __enter__(self) -> "RecordingWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class ReplaySource:
    """Read recorded samples, optionally using the original receipt intervals."""

    def __init__(self, directory: str | Path, *, realtime: bool = False, speed: float = 1.0) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Replay speed must be finite and positive")
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != FORMAT or self.manifest.get("version") != FORMAT_VERSION:
            raise ValueError("Unsupported episode format or version")
        self.count = int(self.manifest["frame_count"])
        if self.count < 0:
            raise ValueError("Invalid episode frame count")
        self.realtime, self.speed = bool(realtime), float(speed)
        self._index = 0
        self._started_at: float | None = None
        self._closed = False
        self._pending: dict[str, Any] | None = None
        self.latest: LiveSample | None = None

    @property
    def exhausted(self) -> bool:
        return self._closed or self._index >= self.count

    def _read(self, document: dict[str, Any]) -> LiveSample:
        stem = self.directory / "frames" / f"{self._index:08d}"
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in document["array_keys"]}
        frame = None
        if document["rgbd"] is not None:
            from .camera import CameraCalibration, RGBDFrame
            info = document["rgbd"]
            calibration = CameraCalibration(**info["calibration"])
            frame = RGBDFrame(
                timestamp=info["timestamp"], rgb=arrays.get("rgb"),
                depth_m=arrays.get("depth_m"), calibration=calibration,
                world_from_camera=arrays.get("world_from_camera"), source=info["source"],
                frame_id=info["frame_id"], metadata=info["metadata"],
                confidence=arrays.get("confidence"),
            )
        replay_received_at = time.monotonic()
        metadata = dict(document["metadata"], update_received_at=replay_received_at, replay={
            "directory": str(self.directory.resolve()), "index": self._index,
            "original_received_at": document["received_at"],
            "original_update_received_at": document.get("event_received_at", document["received_at"]),
            "relative_received_at": document["relative_received_at"],
        })
        # A button update can reuse an older pose. Preserve that pose age rather
        # than making old motion fresh each time an independent control arrives.
        pose_age = max(0.0, document.get("event_received_at", document["received_at"])
                       - document["received_at"])
        return LiveSample(
            timestamp=document["timestamp"], received_at=replay_received_at - pose_age / self.speed,
            rgbd=frame, pose=arrays.get("pose"), buttons=ControlButtons(**document["buttons"]),
            gripper=document["gripper"], session_id=document["session_id"],
            frame_id=document["frame_id"], source_timestamp=document["source_timestamp"],
            metadata=metadata, pose_stale=document["pose_stale"], control_stale=document["control_stale"],
        )

    def poll(self, timeout_s: float = 0.0) -> LiveSample | None:
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("timeout_s must be finite and nonnegative")
        if self.exhausted:
            return None
        if self._pending is None:
            path = self.directory / "frames" / f"{self._index:08d}.json"
            self._pending = json.loads(path.read_text(encoding="utf-8"))
        if self._started_at is None:
            self._started_at = time.monotonic()
        if self.realtime:
            due = self._started_at + self._pending["relative_received_at"] / self.speed
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, timeout_s))
                if time.monotonic() < due:
                    return None
        self.latest = self._read(self._pending)
        self._pending = None
        self._index += 1
        return self.latest

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "ReplaySource":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __iter__(self):
        while not self.exhausted:
            sample = self.poll(0.05)
            if sample is not None:
                yield sample


EpisodeWriter = RecordingWriter
