"""Create the same RGBDFrame API without a physical phone."""
from iphone_mujoco import MujocoRGBDCamera, phone_mjcf
from iphone_mujoco.cli import demo_calibration, look_at
from iphone_mujoco.robots import load_scene

k = demo_calibration()  # Explicitly synthetic; load a device JSON for real K.
model, data, spec = load_scene("reference6", phone_xml=phone_mjcf(k))
with MujocoRGBDCamera(model, k) as camera:
    camera.set_world_pose(data, look_at(), move_body=True)
    frame = camera.render(data)
    frame.point_cloud(stride=4).save_ply("camera_points.ply")
    print(frame.rgb.shape, frame.depth_m.shape)
