"""Render RGB-D around the Menagerie Franka Panda with a virtual iPhone.

The first run downloads and verifies the pinned Panda assets.
"""
import argparse
import json
from pathlib import Path

from iphone_mujoco import CameraCalibration, MujocoRGBDCamera, configured_phone_mjcf, phone_mjcf
from iphone_mujoco.cli import _camera_configuration, _camera_options, _save_frame, demo_calibration, look_at
from iphone_mujoco.robots import fetch_menagerie, load_scene
from iphone_mujoco.wrist import WristCameraMount, configured_wrist_mount


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Captured camera calibration or a saved device configuration")
    _camera_options(parser)
    parser.add_argument("--output", type=Path, default=Path("recordings/virtual-camera"))
    args = parser.parse_args(argv)
    configuration = _camera_configuration(args)
    k = (configuration.calibration if configuration else
         CameraCalibration.load(args.profile) if args.profile else demo_calibration())
    fetch_menagerie("panda")
    phone = configured_phone_mjcf(configuration) if configuration else phone_mjcf(k)
    model, data, spec = load_scene("panda", phone_xml=phone)
    # The wide lens anchors the body; switching lenses keeps the phone in place.
    mount = configured_wrist_mount(WristCameraMount("world", look_at()), configuration)
    metadata = {"source": "mujoco-simulation", "timestamp_clock": "simulation",
                "calibration_source": "caller_profile" if args.profile else "synthetic_example"}
    if configuration:
        metadata.update(camera_configuration=configuration.to_dict(),
                        calibration_source=configuration.calibration_source)
    with MujocoRGBDCamera(model, k) as camera:
        camera.set_world_pose(data, mount.parent_from_camera, move_body=True)
        frame = camera.render(data)
        frame.metadata.update(metadata)
        args.output.mkdir(parents=True, exist_ok=True)
        _save_frame(frame, args.output)
        (args.output / "camera_info.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if configuration:
            configuration.save(args.output / "device.json")
        print(frame.rgb.shape, frame.depth_m.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
