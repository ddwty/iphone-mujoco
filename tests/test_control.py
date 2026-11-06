"""Headless dynamics and controller contracts, without camera or hardware claims."""
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from iphone_mujoco.control import CartesianTeleoperator, rotation_vector
from iphone_mujoco.robots import _safe_destination, get_robot, load_scene


@pytest.fixture(params=("panda", "ur5e"))
def robot_name(request):
    name = request.param
    if not get_robot(name).model_path.is_file():
        pytest.skip(f"Download the pinned {name} model with iphone-mujoco models --fetch {name}")
    return name


def controller(name, **kwargs):
    model, data, spec = load_scene(name)
    return model, data, CartesianTeleoperator(model, data, **spec.controller_kwargs, **kwargs)


def test_vendor_dynamics_tracks_bounded_cartesian_displacement(robot_name):
    model, data, control = controller(robot_name, phone_to_world=np.eye(3))
    pose = np.eye(4)
    initial = control.tcp_pose().copy()
    assert control.update(pose, receive_time=0)
    assert np.allclose(control.target_pose, initial)  # First frame is a pure anchor.
    pose[0, 3] = .025
    for index in range(500):
        now = index * model.opt.timestep
        control.update(pose, receive_time=now, gripper=.5)
        control.step(now=now, dt=model.opt.timestep)
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert control.tcp_pose()[0, 3] - initial[0, 3] > .01
    assert np.linalg.norm(control.tcp_pose()[:3, 3] - control.target_pose[:3, 3]) < .02


def test_clutch_stale_session_and_rezero_reanchor_without_jump(robot_name):
    _, data, control = controller(robot_name)
    pose = np.eye(4)
    control.update(pose, receive_time=0, session_id="a")
    pose[0, 3] = .1
    control.update(pose, receive_time=.1, session_id="a")
    control.step(now=.1, dt=.01)
    assert not control.update(pose, receive_time=.11, session_id="a", clutch=True)
    assert control.status.mode == "clutched"
    pose[0, 3] = 4  # Reposition phone while clutch is released.
    control.update(pose, receive_time=.12, session_id="a")
    assert np.allclose(control.target_pose, control.tcp_pose())
    assert control.step(now=1).mode == "stale"
    assert np.allclose(data.ctrl[control.actuator_ids], data.qpos[control.qpos_ids])
    pose[0, 3] = 10
    control.update(pose, receive_time=1.01, session_id="b")
    assert np.allclose(control.target_pose, control.tcp_pose())
    pose[0, 3] += .2
    control.update(pose, receive_time=1.02, session_id="b", reset=True)
    assert np.allclose(control.target_pose, control.tcp_pose())


@pytest.mark.parametrize("held_mode", ["clutched", "stale"])
def test_repeated_hold_preserves_servo_target_under_gravity(held_mode):
    """A loaded uncompensated link must sag once, not ratchet on each hold.

    This small pendulum intentionally has no gravcomp. It exercises hundreds
    of physics steps between received packets and does not require downloads.
    """
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <compiler angle="radian"/>
      <option timestep=".002" gravity="0 0 -9.81" integrator="implicitfast"/>
      <worldbody><body pos="0 0 1">
        <joint name="joint" axis="0 1 0" range="-2 2" damping="1"/>
        <geom type="capsule" fromto="0 0 0 .6 0 0" size=".04" mass="2"/>
        <site name="tcp" pos=".6 0 0"/>
      </body></worldbody>
      <actuator><position name="servo" joint="joint" kp="200" kv="20"
        ctrllimited="true" ctrlrange="-2 2"/></actuator>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[0] = data.ctrl[0] = .5
    mujoco.mj_forward(model, data)
    control = CartesianTeleoperator(model, data, ["joint"], ["servo"], "tcp")
    pose = np.eye(4)
    control.update(pose, receive_time=0)
    if held_mode == "stale":
        control.step(now=.36, dt=model.opt.timestep)
    else:
        control.update(pose, receive_time=.001, clutch=True)
    frozen_target = data.ctrl.copy()
    for index in range(2400):  # 4.8 s; 150 clutch packets or 2400 stale steps.
        now = .36 + index * model.opt.timestep
        if held_mode == "clutched" and index % 16 == 0:
            control.update(pose, receive_time=now, clutch=True)
        control.step(now=now, dt=model.opt.timestep)
        mujoco.mj_step(model, data)
        assert np.array_equal(data.ctrl, frozen_target)
    assert control.status.mode == held_mode
    assert .005 < abs(data.qpos[0] - frozen_target[0]) < .06


def test_pose_discontinuity_out_of_order_and_invalid_input(robot_name):
    _, _, control = controller(robot_name)
    pose = np.eye(4)
    control.update(pose, receive_time=2)
    pose[0, 3] = 5
    assert not control.update(pose, receive_time=2.1)
    assert control.status.mode == "discontinuity"
    assert not control.update(np.eye(4), receive_time=1)
    assert control.update(pose, receive_time=2.2)
    assert np.allclose(control.target_pose, control.tcp_pose())
    with pytest.raises(ValueError):
        control.update(np.full((4, 4), np.nan))
    invalid = np.eye(4)
    invalid[0, 0] = 2
    with pytest.raises(ValueError):
        control.update(invalid)


def test_rate_joint_workspace_gripper_and_home_limits(robot_name):
    model, data, control = controller(robot_name, joint_velocity_limit=.3, workspace_radius=.1)
    pose = np.eye(4)
    control.update(pose, receive_time=0)
    pose[0, 3] = .3
    control.update(pose, receive_time=.1, gripper=1)
    assert np.linalg.norm(control.target_pose[:3, 3] - control._tcp_anchor[:3, 3]) <= .1000001
    before = data.ctrl[control.actuator_ids].copy()
    control.step(now=.1, dt=.02)
    assert np.max(np.abs(data.ctrl[control.actuator_ids] - before)) <= .3 * .02 + 1e-12
    assert np.all(data.ctrl[control.actuator_ids] >= control.lower)
    assert np.all(data.ctrl[control.actuator_ids] <= control.upper)
    expected_gripper = control.gripper_open + 2 * .02 * (control.gripper_closed - control.gripper_open)
    np.testing.assert_allclose(data.ctrl[control.gripper_ids], expected_gripper)
    qpos_before = data.qpos.copy()
    before = data.ctrl[control.actuator_ids].copy()
    control.go_home(now=.12, dt=.02)
    assert np.array_equal(data.qpos, qpos_before)  # Home only sends bounded servo targets.
    assert np.max(np.abs(data.ctrl[control.actuator_ids] - before)) <= .006 + 1e-12
    assert control.status.mode == "home"


def test_rotation_log_handles_pi_and_teleop_maps_orientation(robot_name):
    rotation = np.diag((1., -1., -1.))
    assert np.isclose(np.linalg.norm(rotation_vector(rotation)), np.pi)
    _, _, control = controller(robot_name, phone_to_world=np.eye(3))
    pose = np.eye(4)
    control.update(pose, receive_time=0)
    initial = control.tcp_pose()
    angle = .2
    pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    control.update(pose, receive_time=.1)
    assert np.allclose(control.target_pose[:3, :3], pose[:3, :3] @ initial[:3, :3])
    assert np.allclose(control.target_pose[:3, 3], initial[:3, 3])


def test_torque_motors_are_rejected(robot_name):
    model, data, control = controller(robot_name)
    model.actuator_biasprm[control.actuator_ids[0]] = 0
    with pytest.raises(ValueError, match="position servos"):
        CartesianTeleoperator(model, data, **get_robot(robot_name).controller_kwargs)


def test_custom_mjcf_adapter_preserves_named_vendor_joints(robot_name):
    source = get_robot(robot_name)
    custom = replace(source, name="my-arm", source="custom")
    baseline, _, _ = load_scene(robot_name)
    model, data, spec = load_scene(custom)
    assert model.nq == baseline.nq and spec.name == "my-arm"
    joint_positions = [model.joint(name).qposadr[0] for name in spec.joint_names]
    np.testing.assert_allclose(data.qpos[joint_positions], custom.home_qpos)
    _, _, no_home = load_scene(replace(custom, name="no-home", home_qpos=()))
    assert no_home.controller_kwargs["home_qpos"] is None


def test_safe_download_paths(tmp_path):
    for invalid in ("../escape", "/absolute", "assets/../../escape", "assets\\escape"):
        with pytest.raises(ValueError):
            _safe_destination(tmp_path, invalid)
    (tmp_path / "assets").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="symlink"):
        _safe_destination(tmp_path, "assets/escape.obj")
