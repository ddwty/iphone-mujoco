"""Native-viewer acceptance smoke. Run with mjpython, not ordinary macOS Python.

Injects application commands without synthesizing OS keyboard/mouse input.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from iphone_mujoco.phone_lab import run_phone_lab
from iphone_mujoco.phone_scene import default_world_from_camera
from iphone_mujoco.recording import ReplaySource

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--profile")
args = parser.parse_args()
run_phone_lab(SimpleNamespace(profile=args.profile, headless=False, duration=2., fps=8.,
                              output=args.output, scan=False),
              initial_keys=(262, ord("E"), ord("L"), ord("P")))
capture = Path(args.output) / "capture-0000"
with ReplaySource(capture / "episode") as source:
    frame = source.poll().rgbd
expected_position = default_world_from_camera()[:3, 3] + [.025, 0, .025]
np.testing.assert_allclose(frame.world_from_camera[:3, 3], expected_position, atol=1e-9)
assert not np.allclose(frame.world_from_camera[:3, :3], default_world_from_camera()[:3, :3])
raw = np.asarray(Image.open(capture / "depth_uint16.png"))
assert raw.dtype == np.uint16 and not np.any(raw == 65535)
valid = raw > 0
np.testing.assert_allclose(frame.depth_m[valid], raw[valid] * .0001, atol=5e-7)
assert np.array_equal(np.isnan(frame.depth_m), ~valid)
np.testing.assert_array_equal(np.load(capture / "depth_m.npy"), frame.depth_m)
assert (capture / "points_world.ply").stat().st_size > 100
result = {"native_viewer": "completed", "command_route": "application queue, not OS key synthesis",
          "translated_and_rotated": True, "saved_snapshot": str(capture.resolve()),
          "rgb_shape": list(frame.rgb.shape), "depth_shape": list(frame.depth_m.shape),
          "valid_depth_ratio": float(valid.mean()), "metric_and_uint16_consistent": True}
(Path(args.output) / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
