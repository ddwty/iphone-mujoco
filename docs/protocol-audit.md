# asMagic protocol audit

Audited against the local App and Python SDK sources on 2026-10-23. The SDK
package declares version **1.0.1**, and that version was available in this
workspace's Python environment. Live reception uses its public
`ARDataSubscriber` and `JoystickDataSubscriber`; this package does not fork
protobuf definitions, implement its own video decoder, or replace transport.

## Two workflows on the same port

| App workflow | TCP port | Data | Adapter |
|---|---:|---|---|
| Camera / Live Data AR stream | 8000 | `Phone.Phone`: optional RGB, optional depth/confidence, intrinsics, frame-paired local pose, spatial/session identity | `AsMagicSource(ip, mode="camera")` |
| Robot Control → Through Mac | 8000 | `Phone.Phone`: pose and velocity, sender wall timestamp | `AsMagicSource(ip, mode="teleop")` |
| Robot Control → Through Mac | 8020 | `Joystick.Joystick`: buttons, sticks, triggers | Same teleop adapter |

The iPhone **publishes** and the computer subscribes to its IP. A successful
local socket/subscription is not proof that frames are arriving. Check actual
sample counts. The Robot Control sender does **not** include RGB, depth,
intrinsics, session ID or sequence/frame ID. Camera and Robot Control use the
same 8000 publisher port: switching pages is a workflow switch, not evidence
that both publishers can run simultaneously on one iPhone. A second iPhone can
provide camera observations while the first controls a robot.

## Poses and reference frames

Both current senders put **seven** floats in pose fields:
`[tx, ty, tz, qx, qy, qz, qw]`, in meters and quaternion XYZW order. UR's separate
six-value robot TCP pose uses a rotation vector; it is not this quaternion
message. The adapter rejects six-value input instead of guessing.

Robot Control's `URZMQRemoteSender.makePhoneMessage` copies the same raw ARKit
camera transform into **both** `localPose` and `globalPose`. These are absolute
poses in that AR session; no clutch anchor, robot-base registration, shared
world, session identifier or origin setting is transmitted. Motion becomes
relative in the receiving controller: hold Move to establish an anchor, apply
subsequent deltas, release to hold and reanchor. A numeric `globalPose` field
here does not mean a globally calibrated robot/world coordinate system.

The Camera sender takes `frame.camera.transform` for `localPose` from the same
ARFrame as the images, depth and intrinsics. `globalPose` is the host camera
pose or the latest received participant pose on a guest; the latter can be an
older estimate. Global selection therefore requires both
`global_pose_is_shared` and a nonempty `shared_world_identifier`. Local pose is
the default for RGB-D fusion. World/base registration is a separate measured
transform supplied by the experiment.

Camera output settings can rotate/translate the sender world. The current App
default is **Y up**, with raw ARKit world axes. Choose `pose_convention="arkit"`
for Y up without custom adjustments, or `"robotics"` for the App's
**Z up · X forward** preset without extra adjustments. Custom offsets/rotations
require `world_from_sender`. Settings are not serialized in `Phone.Phone`, so
the package cannot infer them from a packet. Robot Control bypasses these
camera settings and always requires `"arkit"`.

Geometry uses column vectors and a right-handed Z-up simulation world. The
ARKit-to-world rotation is the App's `CoordinateFrameTransform.roboticsRotation`:

```text
simulation_x = -arkit_z
simulation_y = -arkit_x
simulation_z =  arkit_y
```

Camera-local conversion is separate: ARKit/MuJoCo camera axes look along -Z
with +Y up; the RGB-D optical convention is +Z forward, +X right, +Y down.
`convert_iphone_pose` applies the world registration on the left and
`diag(1,-1,-1,1)` on the right. Poses exposed by `LiveSample.pose` and
`RGBDFrame.world_from_camera` therefore mean **simulation world ← CV optical
camera**. A robot controller receiving these poses must not repeat the ARKit
world conversion.

## Buttons, clutch and stop behavior

| Wire control | App meaning | Exposed field |
|---|---|---|
| Square | Hold Move; release resets motion anchor | `buttons.move` |
| Cross | Capture alignment pose | `buttons.calibrate` |
| Circle | Hold Home; release stops homing | `buttons.home` |
| Triangle rising edge | Toggle remote motion recording | `buttons.record` |
| D-pad up/down rising edge | Sensitivity changes by 0.05 | `buttons.speed_up/speed_down` |
| R2 / R1 | Claw fraction `max(R2, R1)` | `gripper` in [0,1] |

The legacy face-button array order is **Triangle, Square, Cross, Circle**.
Named SDK controls handle this order. `is_connected` in a joystick message can
describe attached controller hardware; on-screen controls remain usable, so it
is not the teleop permission flag.

`URZMQRemoteSender.stop()` closes publishers and clears sender history; it sends
**no explicit emergency-stop packet**. Robot-control idle command
`[3,0,0,0,0,0,0]` belongs to the App's downstream robot TCP interface, not the
8000/8020 protobuf stream. This package holds on expired local receipt time.
Pose and control channels expire independently. Initial connection and receipt
gaps require Move to be released before arming again. A receiver-created UUID
epoch distinguishes UR restarts; metadata labels this as receiver identity,
never as a measured AR session ID.

## RGB, depth and clocks

RGB is JPEG (unspecified/1) or HEVC (2). The SDK owns HEVC decoder state. Its AR
pump receives encoded packets without ZeroMQ conflation, then replaces decoded
output in a bounded queue. This package likewise retains one latest sample.
Simulation advances independently of receive/decode threads.

`camersIntrinsics` is the sender's original misspelled wire field. It holds a
column-major 3×3 pinhole K at the transmitted RGB resolution. The sender scales
fx, fy, cx and cy when resizing RGB. RGB/depth arrays stay in sensor pixel
orientation; `image_rotation_quarter_turns_clockwise` is a display hint. Rotating
only the image would invalidate K and pixel-to-ray correspondence.

Depth is tightly packed little-endian UInt16, with
`depth_meters_per_unit=0.0001` in the current sender. Invalid/nonpositive depth
becomes zero; finite depth is clipped to UInt16, so this representation
saturates around **6.5535 m**. This is a wire encoding limit, not a claim about
the iPhone sensor's reliable range. Confidence is one ARConfidenceLevel UInt8
per depth pixel. Resolution and confidence are validated. The package requires
an explicit positive scale and does not invent missing calibration or sensor
depth. Legacy files with no scale require an explicit legacy scale override in
the camera conversion API.

`timestamp` is sender Unix wall time. Camera `source_timestamp` is the paired
ARKit source clock, with `frame_id` and `session_identifier` for provenance.
Robot Control supplies only wall time. No clock synchronization, network delay,
or cross-device capture simultaneity can be inferred from these values.
Freshness/watchdog calculations use computer `time.monotonic()` receipt time.

## Recording and replay contract

`RecordingWriter(directory).write(sample)` writes `manifest.json` and numbered
numeric `.npz` plus JSON sidecars. It preserves sender wall/source timestamps,
frame ID, sender/session/receiver epoch, original raw poses and K, depth units,
RGB-D calibration, confidence, source metadata, independent button state, and
local receipt intervals. NPZ reading always uses `allow_pickle=False` and
object arrays are rejected. Existing recordings are never overwritten.

`ReplaySource(directory, realtime=True, speed=1).poll(timeout_s)` schedules by
recorded local receipt intervals. It preserves sender timestamps and frame data,
but rebases controller receipt time to the replay machine's monotonic clock;
the original receipt time remains in `metadata["replay"]`. Replay is a source of
recorded observations, not evidence of a live iPhone connection.

## Source evidence

Paths below are repository relative; the App repository is
`MagiClaw-ddwty 2`, and the SDK repository is `zmqTEST/asmagic`.

- SDK `pyproject.toml`, `asmagic/arDataSubscriber.py`,
  `asmagic/joystickDataSubscriber.py`, `asmagic/_subscriber.py`,
  `asmagic/_ar_receiver.py`: published dependency/API, thread ownership,
  bounded decode queue and deterministic cleanup.
- SDK `asmagic/data_types.py`, `asmagic/joystick_controls.py`,
  `asmagic/proto/schemas/phone_msg.proto`: field semantics, intrinsics layout,
  named button mapping and byte formats.
- App `RoboticDataCollection/ReadURArm/Remote/URZMQRemoteSender.swift:9`,
  `:40`, `:53`, `:87`, `:103`: port binding, close-only stop, packets and poses.
- App `RoboticDataCollection/ReadURArm/URBodyControlARViewController.swift:185`
  and `URBodyControlOverlay.swift:650`: raw AR transform and page forwarding.
- App `RoboticDataCollection/ReadURArm/Remote/URRemoteProtobufBridge.swift:157`
  and `URTerminalEngine.swift:229`: pose selection and exact App action mapping.
- App `RoboticDataCollection/View/RemoteView/RemoteARViewController+ARSession.swift:324`,
  `:431`: same-frame camera pose and capture/session snapshot.
- App `RoboticDataCollection/View/RemoteView/RemoteARViewController+Protobuf.swift:94`,
  `:106`, `:256`, `:458`: source clocks, pose frame change, 0.0001 depth scale,
  resized RGB intrinsics.
- App `RoboticDataCollection/View/RemoteView/RemoteARViewController+SceneProcessing.swift:72`,
  `:224`: XYZW extraction and shared-world participant estimate.
- App `RoboticDataCollection/ViewModel/RemoteARViewModel/CooridinateDataConvert.swift:46`
  and `Model/Data/CoordinateFrameTransform.swift:8`: default coordinate mode,
  world rotation and user origin adjustments.

Source audit plus synthetic/loopback checks establish software behavior.
Physical RGB/depth quality, camera extrinsics, thermal/timing behavior, a real
robot, and simultaneous real-device workflows need their own device evidence.
