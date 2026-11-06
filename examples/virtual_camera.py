"""Render RGB-D around the Menagerie Franka Panda with a virtual iPhone.

The first run downloads and verifies the pinned Panda assets.
"""
from pathlib import Path

from iphone_mujoco import MujocoRGBDCamera, phone_mjcf
from iphone_mujoco.cli import demo_calibration, look_at
from iphone_mujoco.robots import fetch_menagerie, load_scene

if __name__ == "__main__":
    fetch_menagerie("panda")
    k = demo_calibration()  # Explicitly synthetic; load a device JSON for real K.
    model, data, spec = load_scene("panda", phone_xml=phone_mjcf(k))
    with MujocoRGBDCamera(model, k) as camera:
        camera.set_world_pose(data, look_at(), move_body=True)
        frame = camera.render(data)
        output = Path("recordings/virtual-camera")
        output.mkdir(parents=True, exist_ok=True)
        frame.point_cloud(stride=4).save_ply(output / "camera_points.ply")
        print(frame.rgb.shape, frame.depth_m.shape)
