# Camera lab and scene mounting

Collect RGB-D, intrinsics and optical poses from the virtual iPhone's named
camera. Simulation and real-phone inputs share the `RGBDFrame` interface.

[![Camera lab with RGB and depth](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.png)](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.gif)

[GIF](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.gif)

## Open the lab

Run from the project root.

```bash
mjpython -m iphone_mujoco phone-lab
```

Use `mjpython` on macOS and `python` on Linux/Windows. On macOS, you can also
open **Open Phone Lab.command** or choose option 1 in **run.command**.
The window shows the phone, stand, test objects and camera frustum, with RGB
and metric-depth previews alongside the scene.

| Key | Action |
|---|---|
| Arrow keys | Move the phone horizontally |
| Q / E | Lower / raise the phone |
| J / L | Turn left / right |
| I / K | Tilt the camera |
| 1 / 2 | Overview / phone camera view |
| P | Save RGB-D, calibration, pose and point clouds |
| R | Reset the phone pose |
| B | Toggle scanning motion |
| V / F | Toggle previews / frustum |

Scroll to zoom and drag to change the overview. The phone pose determines the
sensor view. Add `--profile profiles/my-iphone.json` to load camera intrinsics.

## Lab geometry

The table is 1.10 m wide and 0.80 m deep. It holds a red cube, blue sphere,
green cylinder, 40/80/120 mm steps, a checkerboard with 35 mm squares, a ruler
with 100 mm spacing and an occluder.

| Experiment | Procedure | Output to check |
|---|---|---|
| Projection and calibration | Detect checkerboard corners and calculate rays with K | Reprojection error, field of view and principal-point handling |
| Depth geometry | Back-project the steps and objects into world coordinates | Scale, planes, distances and point coordinates |
| Occlusion and fusion | Press B to scan and save multiple views | Depth edges, visible surfaces and registration input |
| Object detection | Capture objects with known colors and shapes | Detector input, viewpoint coverage and visualization |

Press P to create a capture in the local output directory.

```text
rgb.png                 Lossless RGB
depth_m.npy             float32 optical Z in meters, NaN invalid
depth_uint16.png        UInt16 depth, 0.0001 m/code, zero invalid
depth_preview.png       Color preview
calibration.json        RGB K and depth dimensions
capture.json            Camera world pose, time, source and units
points.ply              Optical-camera point cloud
points_world.ply        World-frame point cloud
episode/                Replayable recording
```

The depth preview spans 0.15–2 m. Read numeric depth from the metric array or
UInt16 file. Values outside the UInt16 range use the invalid code. PNG and
metric arrays use the same quantization.

## Collect a camera sequence

Collect images, poses and scene state over time.

```bash
python examples/scene_camera.py --scenario tabletop \
  --headless --duration 4 --fps 10 --output recordings/tabletop
python examples/scene_camera.py --scenario mobile \
  --headless --duration 6 --fps 10 --output recordings/mobile
```

Omit `--headless` to open the viewer and use `mjpython` on macOS. Set `--profile`
to load your calibration. Omitting it uses the example intrinsics.

![Mobile platform with RGB-D](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/mobile-camera.gif)

[PNG](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/mobile-camera.png) · [GIF](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/mobile-camera.gif)

The tabletop scene keeps the camera fixed while the target moves. The mobile
scene moves the camera with the base along a preset trajectory.
`states.jsonl` stores each optical pose and scene state. Mobile captures also
include `world_from_base`.

```text
rgb/00000000.png        Lossless RGB
depth/00000000.png      UInt16 depth
calibration.json        Camera calibration
camera_info.json        RGB/depth K, units and source
states.jsonl            Poses, simulation time and state
episode/                Replayable recording
preview_points.ply      First-frame world point cloud
run.json                Frame count, timing and run information
```

## Export the complete scene

```bash
iphone-mujoco export-lab --output recordings/exported-lab
python -m mujoco.viewer --mjcf recordings/exported-lab/scene.xml
```

Use `mjpython -m mujoco.viewer` on macOS. The export contains `scene.xml`,
`iphone.xml`, `calibration.json`, `scene_info.json`, and a fixed-mount
`iphone_fixed.xml` with its calibration. Relative references let you copy the
whole directory. Select `iphone_camera` in the standard viewer's camera list.
Use the package's `phone-lab` entry point for RGB-D panels and P-key captures.

## Attach a camera to an existing MJCF

This example adds a second camera to the exported lab's `camera_stand`.
Set `--scene` and the mount's `parent_body` to use a robot base, head, flange
or fixed support in your own scene.

```bash
iphone-mujoco export-lab --output recordings/custom-lab
python - <<'PY'
from iphone_mujoco.phone_scene import default_world_from_camera
from iphone_mujoco.wrist import WristCameraMount

pose = default_world_from_camera()
pose[0, 3] -= 0.25
WristCameraMount(
    "camera_stand", pose,
    camera_name="scene_iphone_camera", measured=False,
).save("recordings/custom-lab/mount.json")
PY
python examples/scene_camera.py --scenario custom \
  --scene recordings/custom-lab/scene.xml --parent-body camera_stand \
  --mount recordings/custom-lab/mount.json --headless \
  --duration 4 --fps 10 --output recordings/custom-data
```

`parent_from_camera` maps CV optical-camera coordinates into parent-body
coordinates. It is a 4×4 matrix with translation in meters. This stand is at
the world origin, so the example uses the world pose directly. For a robot
link, supply the local mounting transform. Set `measured` after measuring or
calibrating that transform. Custom captures save `mount.json`, and the camera
follows the parent body's motion.

## Export the camera module

```bash
iphone-mujoco export-phone --output recordings/iphone.xml
```

Add `--profile my-iphone.json` to export your calibration. The generated
`iphone.calibration.json` stores depth dimensions. Include the module in your MJCF.

```xml
<mujoco>
  <worldbody>
    <body name="camera_mount" pos="0 -0.8 1.0" euler="1.3 0 0">
      <include file="iphone.xml"/>
    </body>
  </worldbody>
</mujoco>
```

Load the complete `scene.xml` in the viewer. For a robot mount, place the
module include under the selected body. Assign unique names to multiple
cameras with options such as `--body-name phone_left --camera-name camera_left`.

```python
from iphone_mujoco import CameraCalibration, MujocoRGBDCamera, write_phone_module

k = CameraCalibration.load("my-iphone.json")
write_phone_module("iphone.xml", k)
# Include the module in your MJCF and create model/data before rendering.
with MujocoRGBDCamera(model, k, camera="iphone_camera") as camera:
    frame = camera.render(data)
    rgb, depth_m = frame.rgb, frame.depth_m
    world_points = frame.point_cloud(world=True)
```

Install the `[live]` extra to connect through the PyPI `asmagic` SDK.
