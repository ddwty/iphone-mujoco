# Development and data contracts

The MuJoCo renderer and optional real-iPhone adapter share an explicit RGB-D
contract. Keep simulated geometry, device measurements and unavailable values
distinct. The public package/import names are `iphone-mujoco` / `iphone_mujoco`;
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
| `cli` | User workflows, not another protocol implementation |

The existing PyPI `asmagic` SDK owns ZeroMQ, Protobuf and JPEG/HEVC decoding.
Do not vendor its schemas, receivers or decoder. Installation must not depend
on a Swift or editable SDK checkout. Additional sensor adapters should consume
the SDK's public subscriber objects and declare their frames, units and clocks.

## Live SDK workflows

The iPhone publishes; the computer subscribes to its LAN address. Camera and
Robot control are separate workflows using the same port, so a page switch
does not establish simultaneous camera/control delivery from one phone.

| App workflow | TCP | Payload | Adapter |
|---|---:|---|---|
| Stream | 8000 | RGB/depth/confidence, K, frame-paired pose, source/session identity | `AsMagicSource(ip, mode="camera")` |
| Robot control → Through Mac | 8000 | Pose/velocity and sender wall time | `AsMagicSource(ip, mode="teleop")` |
| Robot control → Through Mac | 8020 | Buttons, sticks and triggers | Same teleop adapter |

A socket connection is not frame delivery; check actual samples. Robot control
does not send RGB, depth, K, session IDs or frame IDs. A receiver-created epoch
identifies receipt continuity, not a measured AR session. A second phone can
provide camera observations while the first provides control.

Camera input defaults to frame-paired `localPose`. A guest's `globalPose` may
be an older participant estimate: global selection requires both
`global_pose_is_shared` and a nonempty `shared_world_identifier`. World/robot
registration remains a separate measured transform.

Robot control copies the raw ARKit camera transform into both wire pose fields.
Those are absolute AR-session poses, without a robot-base registration or
clutch anchor; relative motion is established by the receiving controller.
Stopping the app closes publishers without an explicit emergency-stop packet.
The separate robot TCP command interface is not the 8000/8020 protobuf stream.

| Control | App meaning | Package field |
|---|---|---|
| Square | Hold Move; release resets motion anchor | `buttons.move` |
| Cross | Capture alignment pose | `buttons.calibrate` |
| Circle | Hold Home; release stops homing | `buttons.home` |
| Triangle rising edge | Toggle remote motion recording | `buttons.record` |
| D-pad up/down rising edge | Sensitivity change by 0.05 | `buttons.speed_up/speed_down` |
| R2 / R1 | Claw fraction `max(R2, R1)` | `gripper` in [0,1] |

The legacy face-button order is Triangle, Square, Cross, Circle; use SDK named
controls. Joystick `is_connected` describes controller hardware, not permission
to move: on-screen controls remain usable. Different CLI workflows may expose
only a subset of the app actions; check their usage guide.

## RGB-D and calibration

`RGBDFrame` carries timestamp, optional RGB/depth/confidence, calibration,
optional optical world pose, source identity and metadata. Missing observations
remain `None`; invalid metric depth samples are NaN.

- RGB: `uint8 (H,W,3)`, RGB channels and raw raster orientation. Display rotation
  hints do not rotate stored pixels or K.
- Depth: optical-axis Z in meters, not range along a ray. Wire bytes are tightly
  packed little-endian UInt16 with an explicit meters/code scale, normally
  `0.0001`. Zero is invalid; generated exports reserve 65535 for saturation.
- The wire representation saturates near 6.5535 m; this is not a claim about
  reliable iPhone sensing range. Quantization/range gating are separate from
  ideal rendering. No missing scale or sensor observation is invented; legacy
  scale-less files require an explicit override in the conversion API.
- K preserves fx/fy/cx/cy, unequal focal lengths and off-center principal points;
  nonzero skew is rejected. The wire field `camersIntrinsics` is column-major.
  The sender scales K when resizing RGB.
- Depth K assumes full-frame RGB registration and center-aligned resampling:
  `u' = (u + 0.5) * scale - 0.5`. Unknown crop/registration cannot be inferred.
- Confidence is optional per-depth-pixel ARConfidenceLevel UInt8. Renderer
  validity and ideal wire confidence are not measured LiDAR confidence.

`calibrate` writes the bare `CameraCalibration` JSON accepted by `--profile`.
Its K is a runtime snapshot for that device/settings, not a universal factory
specification. `DeviceProfile.from_frame(frame)` adds provenance and unknown
physical properties; its richer JSON is not a direct CLI profile replacement.

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

`wrist_camera_mjcf(..., show_phone=True)` adds the approximate phone shell at
the optical mount. Its cosmetic geoms explicitly use zero mass and disabled
contacts, including when a parent MJCF provides default geom mass. The
Reference6/7 demonstration mounts face the initial floor targets from 0.4.0.
Use an explicit saved mount JSON to reproduce a specific experiment rig.

## Clocks, sessions and freshness

| Clock | Meaning |
|---|---|
| Simulation | `mjData.time`, advanced by physics |
| Sender wall | App Unix timestamp, not synchronized receive time |
| Sender source | ARKit monotonic source time, scoped to source/session |
| Local receipt | Host monotonic arrival/event time used for freshness |

Do not subtract clocks on different machines to report one-way latency. Camera
frame IDs and source/session identities preserve discontinuities; Robot control
has only wall time. Origin resets cannot silently reuse an unrelated measured
world registration. A packet timestamp does not prove cross-device simultaneity.

Pose and buttons have independent receipt times/freshness: a button update
cannot refresh an old pose. First connect/reconnect and receipt gaps must observe
Move released before accepting held Move. Invalid, stale or discontinuous input
holds the target and reanchors subsequent movement. Limits still apply during
replay. These are simulator safeguards, not physical robot safety guarantees.

Wrist RGB, depth and robot state come from one `mjData` state after a complete
physics step, with no step between samples. Sampling starts at positive
simulation time because zero source-time means missing in the current SDK path.
Requested wrist FPS is per simulation second, not guaranteed wall-clock rate.

## Episodes and replay

An episode contains a versioned manifest, numbered numeric NPZ files and JSON
sidecars. Writers reject existing nonempty directories. Readers disable pickle,
reject object arrays and unknown versions, and preserve calibration, confidence,
raw poses/K, sender/source/receipt clocks, source/session identity and buttons.
Image/PLY exports are convenience views, not replacements for that metadata.

The existing format identifier `asmagic-mujoco-episode` survives the package
rename. Branding does not justify rewriting recordings or losing compatibility.

`ReplaySource` defaults to immediate iteration. With `realtime=True`, it schedules
by recorded local event intervals and replay speed, rebases controller receipt
time while preserving original pose age, and retains original timing under
`metadata["replay"]`. Fast-generated simulation receipt intervals may differ
from simulation time; resampling must explicitly select the intended clock.

`inspect-episode` is read-only and reports coverage, shape/K, source/session and
separate clock consistency. Structural validity does not establish physical
calibration quality, experiment correctness or a live phone connection.

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

Optional `gripper_actuator_names`, `gripper_open`, `gripper_closed` arrays use
actuator control values, not guessed joint angles. Panda uses its upstream
0–255 range; reference arms use finger position servos; UR5e has no gripper.
See [the custom robot example](../examples/custom_robot.py) for Python loading.
Keep model/data and controller stepping on one simulation thread.

## Models, rigs and scope

Official robot assets are downloaded separately at a pinned commit with file
hash checks and original licenses. Use `IPHONE_MUJOCO_MODEL_CACHE` for a shared
local cache; legacy `ASMAGIC_MUJOCO_MODEL_CACHE` and previous caches remain
compatible. Caches are excluded from source and release distributions.

Default phone geometry and wrist mounts are demonstrations. Real rigs require
measured hand-eye/extrinsic calibration; mass, inertia and lens mounting remain
unknown until measured. Ideal rendering does not model LiDAR noise/confidence,
distortion, rolling shutter, exposure or hardware latency.

The `phone-lab` scene contains static camera-test geometry. Damped Jacobian IK and position servos
provide bounded simulation targets, with potentially substantial moving-target
residuals. There is no hardware driver, motion planner or collision avoidance.
IMU, hand/body/face retargeting and other SDK signals need explicit simulator
adapters before becoming supported capabilities. Track proposals in Issues;
use Actions/releases for checks actually run and their precise scope.
