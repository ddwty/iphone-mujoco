# Adding robots and sensors

The package keeps the existing PyPI `asmagic` SDK as the transport and codec
provider. New adapters consume SDK objects; they must not introduce another
copy of the app's Protobuf schema or another incompatible camera protocol.

## A custom robot

Create a JSON adapter beside your MJCF:

```json
{
  "name": "my-arm",
  "model_path": "scene.xml",
  "joint_names": ["shoulder", "elbow", "wrist"],
  "actuator_names": ["shoulder_servo", "elbow_servo", "wrist_servo"],
  "site_name": "tcp",
  "home_qpos": [0, 0, 0],
  "description": "My measured arm model"
}
```

The MJCF must contain those named scalar hinge/slide joints, one direct
unit-gear **position servo** per joint, and the TCP site. If you configure a
home, put it within joint/control limits. A three-axis example cannot reproduce
an arbitrary six-dimensional target; use an arm with enough degrees of freedom.
Arm torque/velocity motors, ball/free joints and tendon couplings require their
own controller adapter. They are rejected by the Cartesian controller.

```bash
iphone-mujoco teleop --ip YOUR_IP --robot path/to/adapter.json
```

Optional gripper configuration adds `gripper_actuator_names`, `gripper_open`,
and `gripper_closed` arrays. These are actuator control values, not guessed
finger joint angles. The official Panda adapter uses its upstream 0–255 control
range; the reference arms use their finger position servos. UR5e has no gripper.

```python
import numpy as np
import mujoco
from iphone_mujoco.robots import RobotSpec, load_scene
from iphone_mujoco.control import CartesianTeleoperator
from iphone_mujoco.streams import AsMagicSource

spec = RobotSpec.from_json("adapter.json")
model, data, spec = load_scene(spec)
control = CartesianTeleoperator(model, data, phone_to_world=np.eye(3),
                               **spec.controller_kwargs)
with AsMagicSource("YOUR_IP", mode="teleop") as phone:
    # Run in a paced simulation loop, on one thread with model/data.
    sample = phone.poll(timeout_s=.1)
    if sample and sample.pose is not None:
        control.update(sample.pose, receive_time=sample.received_at,
                       session_id=sample.session_id, clutch=sample.clutch,
                       gripper=sample.gripper)
    control.step(dt=model.opt.timestep)
    mujoco.mj_step(model, data)
```

`AsMagicSource` already returns Z-up poses. Pass identity `phone_to_world`
as above. The controller's default ARKit basis conversion is only for callers
passing **raw** ARKit-world input. Never apply the world conversion twice.

## A measured iPhone rig

`DeviceProfile.from_frame(rgbd)` preserves measured runtime intrinsics and their
provenance. Physical dimensions, mass/inertia and camera-to-body mounting
extrinsics remain unknown unless measured. The included shell is cosmetic.
For a robot-mounted camera, add `phone_mjcf(K)` under the robot's flange and set
its mount transform. Update the camera using `set_world_pose` with
`move_body=False`, or simply render the camera attached to that moving body.
`move_body=True` is for a dedicated static/mocap phone body, not a jointed arm.

All public `RGBDFrame.world_from_camera` poses map CV optical coordinates
(X right, Y down, Z forward) into the MuJoCo Z-up world. `camera_mjcf` and the
renderer handle MuJoCo's local camera basis. Depth is optical-axis Z in meters.
Projection/unprojection use integer pixel centers and center-aligned resampling.

## Other asMagic streams

Use the SDK's `IMUDataSubscriber`, `HandPoseDataSubscriber`,
`BodyTrackingDataSubscriber` or `FaceDataSubscriber` for their respective
streams. New sim adapters can use the same pattern as `AsMagicSource`:

1. Keep sender time, local monotonic receipt time, frame identity and coordinate
   provenance separate.
2. Declare which quantities are measured, simulated or unavailable.
3. Keep independent streams independent until timestamps and registration permit
   a justified association.
4. Add an explicit capability entry before claiming a feature exists.

The current virtual publisher implements camera/pose only. It binds to
127.0.0.1 by default in examples; bind to `tcp://0.0.0.0:18000` only when LAN
clients should subscribe. There is no discovery or authentication layer in the
existing PUB/SUB wire protocol.

## Releasing

`python -m build` generates an sdist and wheel. Check the wheel contains the
phone XML, both reference models and both pinned model manifests. The large
official assets are downloaded on demand and keep their individual licenses.
Package installation is independent of any editable SDK or Swift app checkout.
Publishing to PyPI and creating a public repository are separate release steps.
