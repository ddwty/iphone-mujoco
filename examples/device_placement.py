"""Place device 01 through asDevice and save clean RGB-D plus an asTriad overview.

Requires the spatial extra and OpenGL. This supplies a simulation pose, not a
physical tracking measurement. Use a fresh --output directory.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from asDevice import Transform
from PIL import Image

from iphone_mujoco import MujocoRGBDCamera, configure_camera
from iphone_mujoco.phone_scene import load_phone_lab
from iphone_mujoco.recording import RecordingWriter
from iphone_mujoco.spatial import SpatialOverlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="iphone-17-pro")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = configure_camera(args.device, region="cn", zoom=.5)
    model, data = load_phone_lab(configuration=config)
    spatial = SpatialOverlay(config)
    world_from_01 = Transform.from_rotation_translation(
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)), (0, -.8, 1.0))
    with MujocoRGBDCamera(model, config.calibration) as camera:
        spatial.placement.place_camera(camera, data, np.asarray(world_from_01.matrix))
        frame = camera.render(data)
        spatial.annotate(frame)
        with RecordingWriter(args.output / "episode") as writer:
            writer.write(frame)
        with mujoco.Renderer(model, height=480, width=640) as overview:
            overview.update_scene(data, camera="lab_overview")
            spatial.append(overview.scene, frame)
            Image.fromarray(overview.render()).save(args.output / "scene.png")
    config.save(args.output / "device.json")
    print(f"Saved asDevice placement and asTriad overview: {args.output}")


if __name__ == "__main__":
    main()
