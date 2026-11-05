"""Load an adapter for your own position-actuated MJCF; see docs/extending.md."""
import argparse
from iphone_mujoco.control import CartesianTeleoperator
from iphone_mujoco.robots import RobotSpec, load_scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_json")
    args = parser.parse_args()
    spec = RobotSpec.from_json(args.adapter_json)
    model, data, spec = load_scene(spec)
    controller = CartesianTeleoperator(model, data, **spec.controller_kwargs)
    print(spec.name, model.nq, model.nu)
