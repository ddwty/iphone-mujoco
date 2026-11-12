"""Collect camera episodes using the installed package's scene APIs.

Use mjpython for interactive viewers on macOS. Headless capture needs OpenGL.
"""
import argparse
import json

from iphone_mujoco.cli import _camera_configuration, _camera_options
from iphone_mujoco.configuration import configure_camera
from iphone_mujoco.experiments import (
    load_camera_scene as build_scene, advance_scene, collect_camera_episode,
)
from iphone_mujoco.wrist import WristCameraMount


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("tabletop", "mobile", "custom"), default="tabletop")
    parser.add_argument("--profile", help="Runtime calibration or bound device configuration")
    _camera_options(parser)
    parser.add_argument("--scene", help="Original MJCF path for a custom scene")
    parser.add_argument("--parent-body")
    parser.add_argument("--mount", help="WristCameraMount JSON for a custom scene")
    parser.add_argument("--model-cache")
    parser.add_argument("--motion", choices=("route", "straight", "hold"), default="route")
    parser.add_argument("--linear-speed", type=float, default=.18)
    parser.add_argument("--angular-speed", type=float, default=.35)
    parser.add_argument("--duration", type=float, default=8.)
    parser.add_argument("--fps", type=float, default=15.)
    parser.add_argument("--output", required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    if args.scenario == "custom" and (not args.scene or not args.mount):
        parser.error("Custom scenes require --scene and --mount")
    if args.scenario != "custom" and (args.scene or args.mount or args.parent_body):
        parser.error("--scene, --mount and --parent-body are for custom scenes")
    try:
        configuration = _camera_configuration(args) or configure_camera()
        result = collect_camera_episode(
            args.scenario, args.output, configuration=configuration,
            duration=args.duration, fps=args.fps, headless=args.headless,
            scene=args.scene, parent_body=args.parent_body,
            mount=WristCameraMount.load(args.mount) if args.mount else None,
            model_cache=args.model_cache, motion=args.motion,
            linear_speed=args.linear_speed, angular_speed=args.angular_speed)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps({key: result[key] for key in ("frames", "simulation_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
