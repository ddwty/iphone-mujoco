"""Read-only validation and summaries for recorded episodes; no graphics or SDK."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .geometry import validate_transform
from .recording import FORMAT, FORMAT_VERSION, ReplaySource


def _document(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"Cannot read episode JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Episode JSON must contain an object: {path}")
    return value


def _number(document, name, path, *, optional=False, nonnegative=False):
    value = document.get(name)
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{path}: {name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{path}: {name} must be nonnegative")
    return float(value)


def inspect_episode(directory: str | Path) -> dict:
    """Validate every indexed frame and summarize clocks without comparing clocks.

    Structural errors name the offending manifest/frame file. Sender clock
    regressions are reported per session rather than being called network
    latency; sender wall clocks may legitimately be adjusted.
    """
    directory = Path(directory).expanduser()
    manifest_path = directory / "manifest.json"
    manifest = _document(manifest_path)
    if manifest.get("format") != FORMAT or type(manifest.get("version")) is not int or manifest["version"] != FORMAT_VERSION:
        raise ValueError(f"{manifest_path}: unsupported episode format or version")
    count = manifest.get("frame_count")
    if type(count) is not int or count < 0:
        raise ValueError(f"{manifest_path}: frame_count must be a nonnegative integer")
    if not isinstance(manifest.get("metadata", {}), dict):
        raise ValueError(f"{manifest_path}: metadata must be an object")

    sources, sessions, metadata_keys = Counter(), set(), set()
    rgb_shapes, depth_shapes, calibration_counts = Counter(), Counter(), Counter()
    calibration_details = {}
    missing = Counter({name: 0 for name in ("rgbd", "rgb", "depth", "pose", "depth_confidence")})
    clock_names = ("sender_timestamp", "source_timestamp", "relative_received_at")
    clocks = {name: {"available": 0, "missing": 0, "regressions": 0, "first": None, "last": None} for name in clock_names}
    previous = {name: {} for name in clock_names}
    depth_pixels = valid_depth_pixels = 0
    depth_min = depth_max = None
    scales = set()

    with ReplaySource(directory, realtime=False) as source:
        for index in range(count):
            stem = directory / "frames" / f"{index:08d}"
            document_path = stem.with_suffix(".json")
            array_path = stem.with_suffix(".npz")
            document = _document(document_path)
            if type(document.get("index")) is not int or document["index"] != index:
                raise ValueError(f"{document_path}: index must match its frame filename")
            timestamp = _number(document, "timestamp", document_path)
            _number(document, "received_at", document_path)
            source_time = _number(document, "source_timestamp", document_path, optional=True)
            relative = _number(document, "relative_received_at", document_path, nonnegative=True)
            if not isinstance(document.get("metadata"), dict):
                raise ValueError(f"{document_path}: metadata must be an object")
            if not isinstance(document.get("session_id"), str):
                raise ValueError(f"{document_path}: session_id must be a string")
            keys = document.get("array_keys")
            if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
                raise ValueError(f"{document_path}: array_keys must be a list of unique strings")
            try:
                with np.load(array_path, allow_pickle=False) as archive:
                    if set(archive.files) != set(keys):
                        raise ValueError("array_keys does not match the NPZ contents")
                sample = source.poll()
                if sample is None:
                    raise ValueError("indexed frame produced no sample")
                if sample.pose is not None:
                    validate_transform(sample.pose)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, BadZipFile, EOFError) as error:
                raise ValueError(f"Cannot decode episode frame {document_path} / {array_path}: {error}") from error

            session = sample.session_id
            sessions.add(session)
            metadata_keys.update(sample.metadata.keys() - {"replay", "update_received_at"})
            sources[str(sample.metadata.get("source") or (sample.rgbd.source if sample.rgbd else "unknown"))] += 1
            # Receipt offsets have one recording timeline. Sender clocks are
            # independently checked per session, never subtracted from receipt.
            values = (timestamp, source_time, relative)
            for name, value in zip(clock_names, values):
                stat = clocks[name]
                if value is None:
                    stat["missing"] += 1
                    continue
                group = "episode" if name == "relative_received_at" else session
                if group in previous[name] and value < previous[name][group]:
                    stat["regressions"] += 1
                previous[name][group] = value
                stat["available"] += 1
                if stat["first"] is None:
                    stat["first"] = value
                stat["last"] = value

            missing["pose"] += sample.pose is None
            frame = sample.rgbd
            missing["rgbd"] += frame is None
            missing["rgb"] += frame is None or frame.rgb is None
            missing["depth"] += frame is None or frame.depth_m is None
            missing["depth_confidence"] += frame is None or frame.confidence is None
            if frame is None:
                continue
            key = json.dumps(frame.calibration.to_dict(), sort_keys=True)
            calibration_counts[key] += 1
            calibration_details[key] = {
                **frame.calibration.to_dict(), "K_rgb": frame.calibration.matrix.tolist(),
                "K_depth": frame.calibration.depth_calibration().matrix.tolist(),
            }
            if frame.rgb is not None:
                rgb_shapes[frame.rgb.shape] += 1
            if frame.depth_m is not None:
                depth_shapes[frame.depth_m.shape] += 1
                valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
                depth_pixels += frame.depth_m.size
                valid_depth_pixels += int(valid.sum())
                if valid.any():
                    low, high = float(frame.depth_m[valid].min()), float(frame.depth_m[valid].max())
                    depth_min = low if depth_min is None else min(depth_min, low)
                    depth_max = high if depth_max is None else max(depth_max, high)
            scale = frame.metadata.get("depth_meters_per_unit")
            if scale is not None:
                if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                    raise ValueError(f"{document_path}: depth_meters_per_unit must be positive and finite when present")
                scales.add(float(scale))

    warnings = []
    for name, stat in clocks.items():
        stat["nondecreasing"] = stat["regressions"] == 0 if stat["available"] else None
        if stat["regressions"]:
            warnings.append(f"{name}: {stat['regressions']} backwards transition(s)")
    receipt = clocks["relative_received_at"]
    duration = None if receipt["first"] is None else receipt["last"] - receipt["first"]
    return {
        "directory": str(directory.resolve()), "format": FORMAT, "version": FORMAT_VERSION,
        "frame_count": count, "sources": dict(sorted(sources.items())), "session_count": len(sessions),
        "metadata": manifest.get("metadata", {}), "frame_metadata_keys": sorted(metadata_keys),
        "missing": dict(missing),
        "rgb_shapes": [{"shape": list(shape), "frames": number} for shape, number in sorted(rgb_shapes.items())],
        "depth_shapes": [{"shape": list(shape), "frames": number} for shape, number in sorted(depth_shapes.items())],
        "calibrations": [{**calibration_details[key], "frames": number} for key, number in calibration_counts.items()],
        "depth": {"units": "meters", "total_pixels": depth_pixels, "valid_pixels": valid_depth_pixels,
                  "valid_fraction": valid_depth_pixels / depth_pixels if depth_pixels else None,
                  "minimum_m": depth_min, "maximum_m": depth_max, "meters_per_unit_values": sorted(scales)},
        "timestamps": clocks, "recorded_receipt_duration_s": duration,
        "clock_note": "Sender clocks are checked per session; receipt offsets are one local timeline. No cross-device latency is inferred.",
        "validation": {"structure": "ok", "warnings": warnings},
    }
