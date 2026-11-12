# iPhone MuJoCo

[![CI](https://github.com/ddwty/iphone-mujoco/actions/workflows/ci.yml/badge.svg)](https://github.com/ddwty/iphone-mujoco/actions/workflows/ci.yml)

Mount an iPhone camera module in MuJoCo scenes, on robot wrists and on mobile
platforms. Select a phone, rear lens and output size, then record RGB-D,
calibration and camera poses in replayable episodes.
Real iPhone streams connect through the PyPI `asmagic` SDK.
[Python SDK](https://pypi.org/project/asmagic/)

[![MuJoCo phone laboratory with RGB and depth](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.png)](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.gif)

## Installation

Use Python 3.10 or newer to install from source.

```bash
git clone https://github.com/ddwty/iphone-mujoco.git
cd iphone-mujoco
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
iphone-mujoco doctor
```

On Windows, create the environment with `py -m venv .venv` and install with
`.venv\Scripts\python.exe -m pip install -e .`. Use the environment's
`python` and `iphone-mujoco` for the remaining commands, or their full paths
`.venv\Scripts\python.exe` and `.venv\Scripts\iphone-mujoco.exe`.

| Platform | Open the interactive laboratory |
|---|---|
| macOS | `mjpython -m iphone_mujoco phone-lab --device iphone-17-pro --region cn` |
| Linux | `.venv/bin/python -m iphone_mujoco phone-lab --device iphone-17-pro --region cn` |
| Windows | `.venv\Scripts\python.exe -m iphone_mujoco phone-lab --device iphone-17-pro --region cn` |

Move the virtual camera with arrow keys and Q/E, rotate with J/L and I/K,
press B to sweep the inspection stage, and P to save RGB-D and point clouds.
On macOS you can also open
`Open Phone Lab.command` after running `setup.command`.

Rendering uses OpenGL. A headless Linux host needs a supported EGL driver or
Xvfb/GLFW configuration. `--headless` still renders sensor images.

## Choose the iPhone camera

![iPhone rear camera fields of view](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/camera-lenses.gif)

[PNG](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/camera-lenses.png) · [GIF](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/camera-lenses.gif)

| Model IDs | Rear camera presets |
|---|---|
| `iphone-15-pro` | 0.5×, 1×, 1.2×, 1.5×, 2×, 3× |
| `iphone-15-pro-max` | 0.5×, 1×, 1.2×, 1.5×, 2×, 5× |
| `iphone-16-pro`, `iphone-16-pro-max` | 0.5×, 1×, 1.2×, 1.5×, 2×, 5× |
| `iphone-17-pro`, `iphone-17-pro-max` | 0.5×, 1×, 1.2×, 1.5×, 2×, 4×, 8× |

```bash
iphone-mujoco devices
iphone-mujoco devices --device iphone-17-pro
iphone-mujoco configure-camera --device iphone-17-pro --region cn \
  --zoom 0.5 --resolution 1280x960 --depth-resolution 256x192 \
  --output profiles/iphone-17-pro-ultrawide.json
mjpython -m iphone_mujoco phone-lab \
  --profile profiles/iphone-17-pro-ultrawide.json
```

Use `--lens ultra_wide`, `--lens wide` or `--lens telephoto` to select the
physical lens directly. The 1.2×, 1.5× and 2× presets crop the main camera;
8× on iPhone 17 Pro crops the telephoto camera. Switching lenses moves the optical origin while
the mounted phone stays in place. RGB and depth output sizes are independent.

The catalog uses Apple's dimensional drawings, technical specifications and
regional masses. Public camera presets derive an ideal field of view from the
published equivalent focal length. Use `calibrate --device` for your phone's
actual AR RGB-D intrinsics. See [camera configuration and mounting](docs/phone-lab.md#camera-configuration)
for the distinction and [catalog sources](src/iphone_mujoco/assets/devices/iphone_catalog.json)
for individual references.

## Camera experiments

| Experiment | Camera placement | Output | Use |
|---|---|---|---|
| Inspection bench | External camera on a stand | RGB-D, K, optical pose, stage state and point clouds | Part inspection, depth geometry and calibration |
| Robot wrist | Rigidly attached to an arm link | RGB-D with same-state joints and controls | Eye-in-hand observations, trajectory datasets, visual-servo input |
| TIAGo mobile base | iPhone fixed to the robot | RGB-D, wheel state/odometry, base and camera poses | Visual odometry, encoder comparison and mapping datasets |
| Custom MJCF | Camera attached to a named body | Same frame/episode API | Existing robot, room or task scenes |

### Inspection camera

![Camera laboratory with RGB and depth](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.gif)

[PNG](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.png) · [GIF](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.gif)

```bash
python examples/scene_camera.py --scenario tabletop \
  --device iphone-17-pro --region cn --zoom 0.5 \
  --duration 4 --fps 10 --headless --output recordings/tabletop
iphone-mujoco inspect-episode recordings/tabletop/episode --json
```

The bench contains an actuated inspection stage, machined-part fixtures, depth
gauges and a calibration board. The part moves through a position-controlled
slide joint. The sequence examples accept the same camera options, including a
saved configuration with `--profile profiles/my-iphone.json`.

### TIAGo mobile RGB-D

![PAL Robotics TIAGo with a mounted iPhone RGB-D camera](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/mobile-camera.gif)

[PNG](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/mobile-camera.png) · [GIF](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/mobile-camera.gif)

```bash
iphone-mujoco models --fetch tiago
mjpython -m iphone_mujoco capture-mobile --robot tiago \
  --device iphone-17-pro --region cn --zoom 1 \
  --motion route --linear-speed .18 --angular-speed .35 \
  --duration 8 --fps 15 --output recordings/mobile
```

Use `python -m iphone_mujoco` outside macOS; add `--headless` for offline
collection. The PAL Robotics TIAGo uses Menagerie meshes, joints and inertias.
Wheel velocity servos drive the free base through floor contact. The route
selects velocity commands; MuJoCo determines the motion. `--motion straight`
and `--motion hold` provide simpler collection runs.

Control it directly from Python:

```python
from iphone_mujoco import MujocoRGBDCamera, configure_camera
from iphone_mujoco.mobile import DifferentialDriveController, load_mobile_scene

config = configure_camera("iphone-17-pro", region="cn")
model, data, robot = load_mobile_scene(configuration=config)
drive = DifferentialDriveController(model, data, robot)
drive.command(0.15, 0.2)  # forward m/s, yaw rad/s
with MujocoRGBDCamera(model, config.calibration) as camera:
    for _ in range(100):
        drive.step(0.02)
        frame = camera.render(data)
        wheel_state = drive.state()
```

Record wheel encoder odometry separately from the simulator's base pose, then
compare either against RGB-D odometry. The [camera guide](docs/phone-lab.md#mobile-data-and-control)
describes the state fields, package API and coordinate frames.

### Wrist cameras

![Mounted iPhone and wrist RGB-D on robot arms](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/robot-wrist.gif)

[PNG](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/robot-wrist.png) · [GIF](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/robot-wrist.gif)

| `--robot` | Laboratory arm | Camera parent | End effector |
|---|---|---|---|
| `panda` | Franka Emika Panda, 7 arm axes | `hand` | Parallel gripper |
| `ur5e` | Universal Robots UR5e, 6 arm axes | `wrist_3_link` | Bare flange |

```bash
iphone-mujoco models --fetch panda
mjpython -m iphone_mujoco capture-wrist --robot panda \
  --device iphone-17-pro --region cn --zoom 1 --show-phone \
  --duration 5 --fps 10 --output recordings/wrist-panda
```

Use `python -m iphone_mujoco` outside macOS. Models are cached after the first
download.

```bash
iphone-mujoco models --fetch ur5e
mjpython -m iphone_mujoco capture-wrist --robot ur5e \
  --device iphone-17-pro-max --region cn --zoom 0.5 --show-phone \
  --duration 5 --fps 10 --output recordings/wrist-ur5e
```

Use `--show-phone` to display the phone and its wrist support. The UR5e mount
places the rear lens-cover plane level with the TCP plane. Use
`--mount measured-mount.json` for your mounting transform. RGB, depth and robot
state are captured together. Add `--payload` to include the selected region's
published phone mass and an estimated uniform-box inertia.
Use `--motion hold` for a stationary joint target. The viewer displays RGB and
depth alongside the robot.

Both arms use Google DeepMind's MuJoCo Menagerie models. Model sources and
robot adapter details are in the developer guide. [Menagerie](https://github.com/google-deepmind/mujoco_menagerie) · [Developer guide](docs/development.md)

### Custom MJCF scenes

Use an existing MJCF and a mount JSON describing the local optical-camera-to-parent
transform.

```bash
python examples/scene_camera.py --scenario custom \
  --scene path/to/scene.xml --parent-body flange \
  --mount path/to/mount.json --profile profiles/my-iphone.json \
  --duration 4 --fps 10 --output recordings/custom-scene
```

Attach the camera to a robot flange, mobile base, head or static support using
its body name. The camera guide includes a complete example and mount format. [Camera mounting guide](docs/phone-lab.md)

Export an MJCF include for a `worldbody` or robot body.

```bash
iphone-mujoco export-phone --device iphone-17-pro --region cn \
  --zoom 1 --output recordings/iphone/iphone.xml
```

The export provides body and asset includes, calibration and a saved device
configuration. The [mounting guide](docs/phone-lab.md#export-the-camera-module)
shows where each include belongs in an MJCF scene.

## Read recordings

Collection writes RGB PNG, UInt16 depth PNG, calibration, `states.jsonl`, a
point-cloud preview and an `episode/` directory. Wrist runs add mounting
extrinsics and joint controls; mobile runs add wheel state and odometry.
Read the episode to access images, calibration, poses and frame metadata.

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

Fuse a mobile-camera sequence into a voxelized world-frame point cloud using
its recorded optical poses.

```bash
python examples/fuse_episode.py recordings/mobile/episode \
  --output recordings/mobile-map --voxel-size .01 --stride 4
```

The fusion script averages static-scene points and colors in a voxel grid and
writes `points.ply`, frame provenance and a summary. See the wrist guide for
dataset fields and depth PNG reading.
[Wrist data guide](docs/wrist-rgbd.md)

## iPhone connection

In asMagic, open **Stream** in AR mode, enable RGB/depth, choose **Y up** and
disable custom coordinate adjustments. Capture the device's camera intrinsics.

```bash
python -m pip install -e '.[live]'
iphone-mujoco calibrate --ip YOUR_IP --device iphone-17-pro --region cn \
  --output profiles/my-iphone.json
```

The saved configuration binds the received K to the AR main camera at 1×.
It can be resized without cropping at the same aspect ratio. Changing its
device, lens or zoom requires a separate calibration. The app's Z-up preset
uses `--pose-convention robotics`. Reacquire K after changing capture settings.
The app's camera-only mode supports other physical lenses, but its RGB stream
currently carries neither K nor depth; it cannot supply a calibrated all-lens
RGB-D profile.

Use live phone poses and intrinsics to move the virtual camera and record paired
real/simulated episodes:

```bash
mjpython -m iphone_mujoco camera --ip YOUR_IP --device iphone-17-pro \
  --region cn --duration 10 --output recordings/paired-camera
```

For wrist-camera control, switch to **Robot control → Through Mac** and start
sending. Release Square once after connecting, then hold it to move.

```bash
mjpython -m iphone_mujoco capture-wrist --robot panda \
  --profile profiles/my-iphone.json --show-phone --ip YOUR_IP \
  --duration 0 --output recordings/phone-controlled-wrist
```

This workflow uses phone poses/buttons for control and MuJoCo RGB-D for the
wrist observations. Stream and Robot control share port 8000 and operate as
separate app workflows. To record real RGB-D, use the Stream page and run
`iphone-mujoco record --ip YOUR_IP --mode camera --duration 10 --output recordings/phone`.

## Data conventions

| Quantity | Convention |
|---|---|
| RGB | `uint8`, H×W×3, RGB channels, raw raster orientation |
| Depth | Optical-axis Z in meters, invalid metric samples are NaN |
| UInt16 export | Explicit meters/code, normally `0.0001`, zero invalid, 65535 reserved |
| Intrinsics | Full fx/fy/cx/cy with separate RGB/depth dimensions |
| Optical pose | MuJoCo Z-up world ← CV camera, X right, Y down, Z forward |
| Time | Simulation, sender and local receipt clocks remain distinct |

Use device calibration and measured mounting transforms for a physical rig.
Sensor models and controller behavior are described in the developer guide.

## Documentation

- [Camera laboratory and mounting](docs/phone-lab.md) · [Wrist RGB-D datasets](docs/wrist-rgbd.md)
- [Developer guide](docs/development.md) · [Contributing](.github/CONTRIBUTING.md) · [Security](.github/SECURITY.md)
- [Releases](https://github.com/ddwty/iphone-mujoco/releases) · [MIT license](LICENSE)

Keep recordings, profiles, model caches and validation output in ignored local
directories. Use a new output directory for each experiment.
