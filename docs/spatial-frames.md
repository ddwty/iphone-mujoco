# Device coordinates and rendering

The camera-lab, rear-lens, wrist and mobile PNG/GIF sequences use the actual
asTriad library for coordinate arrows. asDevice resolves the phone's device
reference and composes its placement with the selected optical camera mount.

## Install and open

```bash
python -m pip install -e '.[spatial]'
mjpython -m iphone_mujoco phone-lab --device iphone-17-pro --region cn --spatial
```

Use `python` in place of `mjpython` on Linux/Windows. The spatial extra installs
fixed Git commits of [asTriad](https://github.com/asMagicApp/asTraid) and
[asDevice](https://github.com/ddwty/asDevice), requiring Git and access to those
repositories. They are not published on PyPI. The live transport continues to
use the PyPI `asmagic` SDK through the independent `live` extra.

With `--spatial`, the lab's arrows/Q/E/J/L/I/K edit the device 01 pose. The
selected camera moves rigidly with the phone through asDevice transforms.
Press P to save RGB-D plus device/optical pose provenance. Add `--headless
--output recordings/spatial-first` for a single offline capture.

Wrist and mobile captures accept `--spatial` too:

```bash
iphone-mujoco capture-wrist --robot panda --device iphone-17-pro --show-phone \
  --spatial --headless --fast --duration 1 --output recordings/spatial-wrist
iphone-mujoco capture-mobile --device iphone-17-pro --spatial --headless \
  --duration 1 --output recordings/spatial-mobile
python examples/device_placement.py --output recordings/device-placement
```

The scene example also accepts `--spatial` for tabletop, mobile and custom MJCF
episodes. All these captures retain the same asDevice pose metadata when
headless; interactive viewers add the asTriad overview decorations.

## Read the frames

| Render label | Origin and axes |
|---|---|
| W | Actual MuJoCo world origin, Z up |
| D01 | asDevice 01: front-glass centre, portrait screen-right / up / out |
| C | Selected CV optical camera: X right / Y down / Z forward |

asTriad uses red X, green Y and blue Z. The inset magnifies the same physical
phone and frame origins. The device and camera origins are not interchangeable:
the existing MJCF shell is centred at the body midplane, while 01 is half a
body thickness toward the front glass. 03 is the rear-body datum with rear-view
axes; it is available through the API and metadata, without another overlapping
triad in the overview.

The iPhone17Pro definition is pinned to revision 5; iPhone15Pro/Max,
iPhone16Pro/Max and iPhone17ProMax use revision 1. The integration explicitly
selects portrait orientation to match the package's phone body axes.

## Place a phone with asDevice

```python
import numpy as np
from asDevice import Transform
from iphone_mujoco import MujocoRGBDCamera, configure_camera
from iphone_mujoco.phone_scene import load_phone_lab
from iphone_mujoco.spatial import DevicePlacement

config = configure_camera("iphone-17-pro", region="cn", zoom=0.5)
model, data = load_phone_lab(configuration=config)
placement = DevicePlacement(config)
# Supply the simulation-world pose of screen-plane 01, not the optical centre.
world_from_01 = Transform.from_rotation_translation(
    ((1, 0, 0), (0, 0, -1), (0, 1, 0)), (0, -0.8, 1.0))
with MujocoRGBDCamera(model, config.calibration) as camera:
    placement.place_camera(camera, data, np.asarray(world_from_01.matrix))
    frame = camera.render(data)
    device_pose = placement.world_from_device(frame.world_from_camera)
    rear_pose = placement.world_from_frame(frame.world_from_camera, "03")
```

For a rigid wrist/mobile mount, MuJoCo supplies the current optical world pose;
`world_from_device` resolves the same-state device 01 pose without moving robot
links. To decorate a separate overview, call
`SpatialOverlay(config).append(renderer.scene, frame)` after `update_scene`;
reserve at least 876 free geometry slots. `annotate(frame)` adds JSON-safe
`frame.metadata['spatial_frames']` before recording.

The camera configuration still owns K and the camera-to-body transform.
Catalog mounts estimate the optical origin at the physical lens-cover centre.
`metadata['phone_from_camera']` supplies a measured replacement. asDevice
mechanical camera/window references have their own axes and do not provide a
tracked world pose or camera sensor calibration. Physical use needs tracking,
world registration and measured extrinsics from the host application.

## Reproduce the documentation media

```bash
iphone-mujoco models --fetch panda
iphone-mujoco models --fetch ur5e
iphone-mujoco models --fetch tiago
python tools/render_readme_media.py --output recordings/spatial-media
```

OpenGL is required, including for offline rendering. Choose a fresh output
directory. All four PNG/GIF pairs are rendered from the package's scenes, using
asDevice poses and actual asTriad geometry on every overview frame. Only the
overview and inset receive decorations; sensor RGB-D stays clean. JSON sidecars
record package versions, per-frame poses and media SHA-256 hashes.

These are simulation renders. They do not validate physical tracking accuracy,
latency, mounting tolerances or real robot control.

The [validation note](spatial-validation.md) records the local tests, wheel
installation smoke and regenerated media counts.
