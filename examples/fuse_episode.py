"""Fuse static-scene RGB-D using recorded world poses and voxel averaging.

Run: python examples/fuse_episode.py recordings/scan/episode --output recordings/map
The input poses provide registration. Scene motion produces ghosted geometry.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from iphone_mujoco.camera import PointCloud
from iphone_mujoco.recording import ReplaySource


def fuse_episode(episode, output, *, voxel_size=.01, stride=4, max_voxels=200_000):
    """Stream registered frames into a bounded voxel map and preserve clocks.

    Session/registration changes, explicit reset flags and regressing source
    times reject the run. The source episode supplies camera poses and remains
    the authoritative record of each observation.
    """
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    if not isinstance(stride, int) or stride < 1 or not isinstance(max_voxels, int) or max_voxels < 1:
        raise ValueError("stride and max_voxels must be positive integers")
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    voxels = {}
    frames = points_seen = 0
    identity = None
    previous_time = previous_source_time = previous_frame_id = None
    with tempfile.TemporaryDirectory(prefix=".rgbd-fusion-", dir=output.parent) as temporary:
        staging = Path(temporary)
        with (staging / "frames.jsonl").open("w", encoding="utf-8") as timing:
            with ReplaySource(episode) as source:
                for sample in source:
                    frame = sample.rgbd
                    if frame is None or frame.rgb is None or frame.depth_m is None or frame.world_from_camera is None:
                        raise ValueError(f"Frame {frames} requires RGB, depth and a known world_from_camera")
                    metadata = dict(sample.metadata, **frame.metadata)
                    current_identity = {
                        "session_id": sample.session_id, "frame_session_id": frame.metadata.get("session_id"),
                        "source": frame.source, "timestamp_clock": metadata.get("timestamp_clock", "unspecified"),
                        "sender_session_id": metadata.get("sender_session_id"),
                        "shared_world_identifier": metadata.get("shared_world_identifier"),
                        "world_from_sender": metadata.get("world_from_sender"),
                        "pose_field": metadata.get("pose_field"), "pose_convention": metadata.get("pose_convention"),
                    }
                    if not sample.session_id or (identity is not None and current_identity != identity):
                        raise ValueError(f"Frame {frames} has a missing or changed session/world registration")
                    reset_fields = ("world_reset", "origin_reset", "tracking_reset", "session_reset", "discontinuity")
                    if any(metadata.get(name) for name in reset_fields):
                        raise ValueError(f"Frame {frames} reports a reset/discontinuity; split the episode")
                    if previous_time is not None and frame.timestamp < previous_time:
                        raise ValueError(f"Frame {frames} has regressing frame time; split the episode")
                    if sample.source_timestamp is not None:
                        if previous_source_time is not None and sample.source_timestamp < previous_source_time:
                            raise ValueError(f"Frame {frames} has regressing source time; split the episode")
                        previous_source_time = sample.source_timestamp
                    if frame.frame_id is not None:
                        if previous_frame_id is not None and frame.frame_id < previous_frame_id:
                            raise ValueError(f"Frame {frames} has a regressing frame ID; split the episode")
                        previous_frame_id = frame.frame_id
                    identity, previous_time = current_identity, frame.timestamp
                    cloud = frame.point_cloud(world=True, stride=stride)
                    with np.errstate(over="ignore", invalid="ignore"):
                        grid = np.floor(cloud.points / voxel_size)
                    if not np.isfinite(grid).all():
                        raise ValueError("World coordinates exceed the finite voxel grid; increase --voxel-size")
                    for point, color, cell_index in zip(cloud.points, cloud.colors, grid):
                        key = tuple(cell_index)
                        cell = voxels.get(key)
                        if cell is None:
                            if len(voxels) >= max_voxels:
                                raise ValueError("Voxel limit reached; increase --voxel-size or --max-voxels")
                            voxels[key] = [point.copy(), color.astype(np.float64), 1]
                        else:
                            cell[0] += point
                            cell[1] += color
                            cell[2] += 1
                    points_seen += len(cloud.points)
                    replay = sample.metadata["replay"]
                    timing.write(json.dumps({
                        "index": frames, "frame_id": frame.frame_id, "frame_timestamp": frame.timestamp,
                        "timestamp_clock": current_identity["timestamp_clock"],
                        "recorded_sample_timestamp": sample.timestamp,
                        "sender_timestamp": metadata.get("sender_timestamp"),
                        "source_timestamp": sample.source_timestamp,
                        "recorded_received_at": replay["original_received_at"],
                        "relative_received_at": replay["relative_received_at"],
                        "world_from_camera": frame.world_from_camera.tolist(),
                    }, allow_nan=False) + "\n")
                    frames += 1
        if not voxels:
            raise ValueError("Episode contains no valid RGB-D points")
        values = list(voxels.values())
        points = np.asarray([value[0] / value[2] for value in values])
        colors = np.rint([value[1] / value[2] for value in values]).astype(np.uint8)
        PointCloud(points, colors, np.empty((len(points), 2), dtype=int), "world_z_up").save_ply(staging / "points.ply")
        summary = {
            "method": "known-pose static-scene voxel averaging", "units": "meters", "frame": "world_z_up",
            "source_episode": str(Path(episode).expanduser().resolve()), "source_identity": identity,
            "frames": frames, "input_points": points_seen, "output_voxels": len(voxels),
            "voxel_size_m": voxel_size, "stride": stride, "max_voxels": max_voxels,
            "bounds_m": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
            "clocks": "Per-frame values retain separate source, sender and recorded receipt clocks in frames.jsonl",
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        # Reserve a new destination after all input validation. Publish the
        # summary last so an interrupted file transfer has no completion marker.
        output.mkdir()
        for name in ("points.ply", "frames.jsonl", "summary.json"):
            (staging / name).replace(output / name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New result directory")
    parser.add_argument("--voxel-size", type=float, default=.01, help="Voxel edge length in meters")
    parser.add_argument("--stride", type=int, default=4, help="Depth raster sampling stride")
    parser.add_argument("--max-voxels", type=int, default=200_000, help="Maximum accumulated voxels")
    args = parser.parse_args()
    try:
        result = fuse_episode(args.episode, args.output, voxel_size=args.voxel_size,
                              stride=args.stride, max_voxels=args.max_voxels)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
