# Development and data contracts

The MuJoCo renderer and real-iPhone adapter share the `RGBDFrame` data contract.
The public package/import names are `iphone-mujoco` / `iphone_mujoco`;
legacy import aliases reference the same canonical classes.

## Module ownership

| Modules | Responsibility |
|---|---|
| `camera`, `geometry` | K, frame validation, projection, transforms and rendering |
| `profiles` | Calibration/provenance and cosmetic phone MJCF |
| `phone_scene`, `phone_lab` | Offline laboratory, interaction and snapshots |
| `streams` | SDK adapters, local receipt state and independent freshness |
| `virtual_device` | Existing SDK-schema output and depth quantization |
| `robots`, `control` | Robot adapters, pinned models and bounded simulated targets |
| `wrist` | Rigid mounting and same-state camera/robot capture |
| `recording`, `inspection` | Versioned episodes, replay and offline validation |
| `cli` | User workflows |

The PyPI `asmagic` SDK provides ZeroMQ, Protobuf and JPEG/HEVC decoding.
Additional sensor adapters use its public subscribers and declare their
frames, units and clocks.

## Live SDK workflows

The iPhone publishes and the computer subscribes to its LAN address.
Select Stream or Robot control in the app to match the adapter mode.
Both workflows use port 8000.

| App workflow | TCP | Payload | Adapter |
|---|---:|---|---|
| Stream | 8000 | RGB/depth/confidence, K, frame-paired pose, source/session identity | `AsMagicSource(ip, mode="camera")` |
| Robot control → Through Mac | 8000 | Pose/velocity and sender wall time | `AsMagicSource(ip, mode="teleop")` |
| Robot control → Through Mac | 8020 | Buttons, sticks and triggers | Same teleop adapter |

Monitor sample counts after connecting. Robot control supplies pose and buttons.
Its receiver-created epoch identifies connection continuity. A second phone can
provide camera observations while the first provides control.

Camera input defaults to frame-paired `localPose`. A guest's `globalPose` may
be an older participant estimate: global selection requires both
`global_pose_is_shared` and a nonempty `shared_world_identifier`. World/robot
registration remains a separate measured transform.

Robot control copies the raw ARKit camera transform into both wire pose fields.
Those are absolute AR-session poses. The receiving controller establishes the
clutch anchor and relative motion.
Stopping the app closes publishers without an explicit emergency-stop packet.
The app's robot TCP commands use a separate interface.

| Control | App meaning | Package field |
|---|---|---|
| Square | Hold Move; release resets motion anchor | `buttons.move` |
| Cross | Capture alignment pose | `buttons.calibrate` |
| Circle | Hold Home; release stops homing | `buttons.home` |
| Triangle rising edge | Toggle remote motion recording | `buttons.record` |
| D-pad up/down rising edge | Sensitivity change by 0.05 | `buttons.speed_up/speed_down` |
| R2 / R1 | Claw fraction `max(R2, R1)` | `gripper` in [0,1] |

The legacy face-button order is Triangle, Square, Cross, Circle. Use SDK named
controls. Joystick `is_connected` describes attached controller hardware.
On-screen controls remain usable. CLI workflows expose the actions listed in
their usage guides.

## RGB-D and calibration

`RGBDFrame` carries timestamp, optional RGB/depth/confidence, calibration,
optional optical world pose, source identity and metadata. Missing observations
remain `None`; invalid metric depth samples are NaN.

- RGB: `uint8 (H,W,3)`, RGB channels and raw raster orientation. Display rotation
  hints do not rotate stored pixels or K.
- Depth: optical-axis Z in meters, not range along a ray. Wire bytes are tightly
  packed little-endian UInt16 with an explicit meters/code scale, normally
  `0.0001`. Zero is invalid; generated exports reserve 65535 for saturation.
- The UInt16 encoding range ends near 6.5535 m. Legacy scale-less files require
  an explicit scale override in the conversion API.
- K preserves fx/fy/cx/cy, unequal focal lengths and off-center principal points;
  nonzero skew is rejected. The wire field `camersIntrinsics` is column-major.
  The sender scales K when resizing RGB.
- Depth K assumes full-frame RGB registration and center-aligned resampling:
  `u' = (u + 0.5) * scale - 0.5`. Unknown crop/registration cannot be inferred.
- Confidence is optional per-depth-pixel ARConfidenceLevel UInt8.

`calibrate` writes the `CameraCalibration` JSON accepted by `--profile` for the
current device and capture settings. `DeviceProfile.from_frame(frame)` provides
a separate, richer record with provenance and physical-property fields.

The SDK owns HEVC decoder state and receives encoded packets without ZeroMQ
conflation, then retains decoded output in a bounded queue. This package keeps
one latest sample. Physics advances independently of receive/decode threads.

## Spatial frames

`RGBDFrame.world_from_camera` maps **CV optical camera → MuJoCo Z-up world**.
Transforms multiply column vectors; translation occupies the last column;
distances are meters. The wire pose is seven values
`[tx,ty,tz,qx,qy,qz,qw]`; six-value robot rotation-vector poses are rejected.

| Frame | Axes |
|---|---|
| CV optical camera | +X right, +Y down, +Z forward |
| ARKit/MuJoCo local camera | +X right, +Y up, −Z forward |
| Package world | Robotics Z-up |

For camera streams, `pose_convention="arkit"` expects app Y-up without custom
adjustments; `"robotics"` expects the app Z-up/X-forward preset. Custom sender
offsets/rotations require `world_from_sender`. The wire does not serialize those
settings, so the adapter cannot infer them. Robot control bypasses camera
settings and always uses raw ARKit input.

ARKit world conversion is `simulation_x=-arkit_z`, `simulation_y=-arkit_x`,
`simulation_z=arkit_y`. `convert_iphone_pose` applies world registration on the
left and the local optical conversion `diag(1,-1,-1,1)` on the right.

Poses returned by `AsMagicSource` are already converted. When passing them to
`CartesianTeleoperator`, use `phone_to_world=np.eye(3)`. The controller's default
ARKit conversion is for raw callers; never apply both conversions.

Wrist mounts store `parent_from_camera`, a fixed parent-link ← CV-camera
transform: `world_from_camera = world_from_parent @ parent_from_camera`.
The camera follows robot forward kinematics. `move_body=True` is for a dedicated
static/mocap phone body; do not teleport a jointed robot to reproduce phone
motion. Attached cameras can simply render from their moving parent body.

`wrist_camera_mjcf(..., show_phone=True)` adds the phone shell at the optical
mount. Its cosmetic geoms use zero mass and disabled contacts, including when
a parent MJCF provides default geom mass.
Use an explicit saved mount JSON to reproduce a specific experiment rig.

## Clocks, sessions and freshness

| Clock | Meaning |
|---|---|
| Simulation | `mjData.time`, advanced by physics |
| Sender wall | App Unix timestamp, not synchronized receive time |
| Sender source | ARKit monotonic source time, scoped to source/session |
| Local receipt | Host monotonic arrival/event time used for freshness |

Use local receipt time for freshness checks. Cross-device timing requires clock
synchronization. Camera frame IDs and source/session identities identify
discontinuities. Robot control supplies wall time. Re-establish world
registration after an origin reset.

Pose and buttons have independent receipt times/freshness: a button update
cannot refresh an old pose. First connect/reconnect and receipt gaps must observe
Move released before accepting held Move. Invalid, stale or discontinuous input
holds the target and reanchors subsequent movement. Limits still apply during
replay.

Wrist RGB, depth and robot state come from one `mjData` state after a complete
physics step, with no step between samples. Sampling starts at positive
simulation time because zero source-time means missing in the current SDK path.
Requested wrist FPS counts frames per simulation second.

## Episodes and replay

An episode contains a versioned manifest, numbered numeric NPZ files and JSON
sidecars. Writers reject existing nonempty directories. Readers disable pickle,
reject object arrays and unknown versions, and preserve calibration, confidence,
raw poses/K, sender/source/receipt clocks, source/session identity and buttons.
Image/PLY exports accompany the episode metadata.

Episodes use the format identifier `asmagic-mujoco-episode`.

`ReplaySource` defaults to immediate iteration. With `realtime=True`, it schedules
by recorded local event intervals and replay speed, rebases controller receipt
time while preserving original pose age, and retains original timing under
`metadata["replay"]`. Fast-generated simulation receipt intervals may differ
from simulation time; resampling must explicitly select the intended clock.

`inspect-episode` reports coverage, shape/K, source/session and clock consistency.

## Adding a robot

Place an adapter JSON beside the MJCF:

```json
{
  "name": "my-arm",
  "model_path": "scene.xml",
  "joint_names": ["shoulder", "elbow", "wrist"],
  "actuator_names": ["shoulder_servo", "elbow_servo", "wrist_servo"],
  "site_name": "tcp",
  "home_qpos": [0, 0, 0],
  "description": "My arm model"
}
```

Run `iphone-mujoco teleop --ip YOUR_IP --robot path/to/adapter.json`.
The Cartesian controller requires named scalar hinge/slide joints and one
direct unit-gear **position servo** per joint, a TCP site, and home values within
joint/control bounds. Torque/velocity motors, ball/free joints and tendon
couplings need a different controller adapter and are rejected. Three axes
cannot reproduce an arbitrary six-dimensional target.

Optional `gripper_actuator_names`, `gripper_open`, `gripper_closed` arrays specify
actuator control values. Panda uses its upstream
0–255 range; UR5e uses a bare flange.
See [the custom robot example](../examples/custom_robot.py) for Python loading.
Keep model/data and controller stepping on one simulation thread.

## Models, rigs and scope

Public robot experiments use the seven-axis Franka Emika Panda with parallel
gripper and the six-axis Universal Robots UR5e with bare flange. Both use MJCF
and mesh assets maintained in Google DeepMind's MuJoCo Menagerie, pinned to
commit `cda5236f98383f6fbaa4820c6b18dba892cd087f`, with file hash checks and
original licenses. Use `IPHONE_MUJOCO_MODEL_CACHE` for a shared
local cache; legacy `ASMAGIC_MUJOCO_MODEL_CACHE` and previous caches remain
compatible. Caches are excluded from source and release distributions.

The phone shell and default mounts provide visual placement. For a physical rig,
supply measured hand-eye/extrinsic calibration and load parameters for mass,
inertia and lens mounting. Rendering uses an ideal pinhole model. Experiments
with LiDAR confidence/noise, distortion, rolling shutter, exposure or sensor
latency require corresponding sensor models.

The controller uses damped Jacobian IK and position servos with joint,
workspace and rate bounds. Hardware drivers, motion planning, collision
avoidance and additional SDK signal adapters require separate integrations.
