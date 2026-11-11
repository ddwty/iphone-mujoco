# Wrist RGB-D and robot-state collection

Attach a camera rigidly to the robot's end link. Each RGB image, depth image,
joint position, velocity and control sample comes from the same `mjData` state.
Forward kinematics and the mount transform determine the camera world pose.

[![iPhone wrist camera with RGB-D](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/robot-wrist.png)](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/robot-wrist.gif)

[GIF](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/robot-wrist.gif)

## Start with Franka Panda

Download the Panda model and select the mounted phone.

```bash
iphone-mujoco models --fetch panda
mjpython -m iphone_mujoco capture-wrist \
  --robot panda --device iphone-17-pro --region cn --zoom 1 \
  --resolution 640x480 --depth-resolution 256x192 \
  --show-phone --duration 5 --fps 10 \
  --output recordings/wrist-panda
```

Open the viewer with `mjpython` on macOS or `python` on Linux/Windows.
`--show-phone` displays the selected phone and its wrist support.
`--motion hold` keeps a fixed joint target. The default motion makes small joint movements.
Choose a new or empty output directory for each run.

| Arm | `--robot` | Camera parent body | End effector | Model |
|---|---|---|---|---|
| Franka Emika Panda, seven axes | `panda` | `hand` | Parallel gripper, upstream 0–255 control range | Menagerie MJCF and meshes |
| Universal Robots UR5e, six axes | `ur5e` | `wrist_3_link` | Bare flange | Menagerie MJCF and meshes |

Models use Google DeepMind's MuJoCo Menagerie assets at a pinned commit,
with per-file checks and upstream licenses. The first download needs a network
connection. Later runs reuse the local cache.

Download UR5e and select it with `--robot`.

```bash
iphone-mujoco models --fetch ur5e
python -m iphone_mujoco capture-wrist \
  --robot ur5e --device iphone-17-pro-max --region cn --zoom 0.5 \
  --show-phone --headless --fast --duration 5 --fps 10 \
  --output recordings/wrist-ur5e
```

The UR5e bracket connects the flange to the phone clamp, with the rear
lens-cover plane flush with the TCP plane. Selecting another lens keeps the
phone body and bracket in place. A crop preset, such as 2× or 8×, keeps the
same physical lens center.

## Camera configuration and batch collection

Save a configuration to use the same model, optics and output dimensions across
runs. Inspect the available models and source specifications with
`iphone-mujoco devices --device iphone-17-pro`.

```bash
iphone-mujoco configure-camera --device iphone-17-pro --region cn --zoom 4 \
  --resolution 1280x960 --depth-resolution 256x192 \
  --output profiles/iphone-17-pro-telephoto.json
```

Use `--headless --fast` to generate data without a window or wall-clock pacing.

```bash
python -m iphone_mujoco capture-wrist \
  --robot panda --profile profiles/iphone-17-pro-telephoto.json \
  --show-phone --headless --fast --duration 5 --fps 30 \
  --output recordings/wrist-offline
```

`--fps` counts frames per simulation second. Each state records its sampling
time. Headless rendering uses OpenGL. Linux servers can use EGL or Xvfb/GLFW.
Add `--payload` to include the phone's published regional mass and estimated
uniform-box inertia. Phone and support geometry have no contact collisions;
the case and mount mass must be modeled separately for a physical payload.

Public presets use the published equivalent focal length to define an ideal
pinhole field of view. For the K from your own iPhone, enable RGB/depth in AR
mode on the asMagic Stream page and capture it:

```bash
python -m pip install -e '.[live]'
iphone-mujoco calibrate --ip YOUR_IP --device iphone-17-pro --region cn \
  --output profiles/my-iphone.json
```

Pass that file with `--profile`. It binds the received AR main-camera K at 1×
to the chosen phone. Same-aspect full-raster resizing is supported; requesting
another device, lens or zoom is rejected. Capture a new calibration after
changing phones or capture settings. The app's camera-only RGB stream supports
lens selection but currently provides neither K nor depth.

The [camera guide](phone-lab.md#camera-configuration) covers lens presets,
output resolution, calibration sources and module export.

## Control the arm with an iPhone

Open **Robot control → Through Mac** on the iPhone and start sending.
Release Square once after connecting, then hold it to move. Release it to
hold the current target.

```bash
mjpython -m iphone_mujoco capture-wrist \
  --robot panda --profile profiles/my-iphone.json \
  --show-phone --ip YOUR_IP --duration 0 \
  --output recordings/wrist-phone-control
```

Press Ctrl-C to finish recording. The phone supplies control input and MuJoCo
renders wrist RGB-D. The app's Robot control page sends poses and controls,
while the Stream page sends camera observations.

Replay recorded control input to collect wrist images from the same sequence.

```bash
mjpython -m iphone_mujoco capture-wrist \
  --robot panda --profile profiles/my-iphone.json \
  --show-phone --input-episode recordings/my-teleoperation --duration 0 \
  --output recordings/wrist-from-input
```

Live and replayed input use the host clock for freshness and apply rate,
joint and stale-input limits. This entry point maps Square, Cross and the
gripper. Use `teleop` for the other robot-button workflows.

## Read the output

```text
wrist-panda/
  rgb/00000000.png          # Lossless RGB uint8
  depth/00000000.png        # Raw UInt16 depth
  calibration.json         # RGB K and depth dimensions
  camera_info.json         # RGB/depth K, units, source and clocks
  mount.json               # Optical-camera-to-parent transform and measurement flag
  states.jsonl             # Time, joints, velocities, controls and optical pose
  episode/                 # NPZ/JSON recording
  preview_points.ply       # First-frame optical-camera point cloud
  run.json                 # Frame count, simulation time and wall time
```

Matching indices identify the image and state from one frame. Read them with
`ReplaySource`. For a selected device, `camera_info.json` and each frame's
metadata also retain `camera_configuration`: model, lens, zoom, region,
calibration source and specification references.

```python
from iphone_mujoco.recording import ReplaySource

with ReplaySource("recordings/wrist-panda/episode") as source:
    frame = source.poll().rgbd
    rgb, depth_m = frame.rgb, frame.depth_m
    K_rgb = frame.calibration.matrix
    K_depth = frame.calibration.depth_calibration().matrix
    world_from_camera = frame.world_from_camera
    qpos = frame.metadata["robot_state"]["qpos"]
    world_points = frame.point_cloud(world=True)
```

| Data | Convention |
|---|---|
| RGB | `uint8 (H,W,3)`, RGB channels, raw camera orientation |
| Depth PNG | `uint16`, meters/code stored in `camera_info.json` |
| Depth definition | Optical-axis Z in meters, NaN for invalid metric samples |
| Invalid depth | Zero in PNG, 65535 reserved |
| Camera axes | X right, Y down, Z forward |
| `world_from_camera` | 4×4 transform from optical camera to MuJoCo Z-up world |
| Time | Simulation seconds, matching the state's `simulation_time` |

Preserve the PNG's 16-bit values and apply its saved scale.

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image

root = Path("recordings/wrist-panda")
info = json.loads((root / "camera_info.json").read_text())
raw = np.asarray(Image.open(root / "depth/00000000.png"))
depth_m = raw.astype(np.float32) * info["depth"]["meters_per_unit"]
depth_m[(raw == 0) | (raw == 65535)] = np.nan
```

`ReplaySource` reads frames immediately by default. `realtime=True` follows
recorded host receipt/write intervals. For simulation-time playback, schedule
using each frame's simulation timestamp.

Add `--publish tcp://127.0.0.1:18000` to let the PyPI SDK subscribe to the
virtual camera. Receive with `asmagic.ARDataSubscriber("127.0.0.1", port=18000)`.
The wire stream uses JPEG. Dataset RGB PNGs are lossless.

## Measured mounts and custom robots

`parent_from_camera` maps CV optical-camera coordinates into parent-link
coordinates.

```text
T_world_camera = T_world_parent × T_parent_camera
```

With a device configuration, the mount describes the **wide-camera reference**.
The phone pose follows that reference and stays fixed while another lens is
selected. Without a device selection, it describes the generic optical camera
directly. Panda uses the following reference mount. `measured` records whether
the extrinsics have been measured.

```json
{
  "parent_body": "hand",
  "parent_from_camera": [
    [1, 0, 0, 0.09],
    [0, 1, 0, 0],
    [0, 0, 1, 0.06],
    [0, 0, 0, 1]
  ],
  "camera_name": "wrist_camera",
  "measured": false
}
```

Measure the physical mounting transform or run hand-eye calibration, then load
it with `--mount measured-mount.json`. The catalog places cameras at mechanical
lens-cover centers. A measured phone-body ← CV-camera transform can override
that estimate through `configuration.metadata["phone_from_camera"]`.
A custom arm also needs an adapter JSON.

```bash
mjpython -m iphone_mujoco capture-wrist \
  --robot path/to/adapter.json --mount path/to/mount.json \
  --profile profiles/my-iphone.json --show-phone \
  --duration 5 --fps 10 --output recordings/custom-wrist
```

See the developer guide for actuator types, TCP and gripper configuration.
[Robot adapters](development.md#adding-a-robot)
