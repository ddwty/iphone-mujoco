"""Receive with the published SDK: python examples/live_camera.py --ip YOUR_IP."""
import argparse
from iphone_mujoco.streams import AsMagicSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    args = parser.parse_args()
    with AsMagicSource(args.ip, mode="camera", pose_convention="arkit") as source:
        while True:
            sample = source.poll(timeout_s=.1)
            if sample and sample.rgbd:
                frame = sample.rgbd
                print(frame.calibration.matrix, frame.world_from_camera)
                if frame.depth_m is not None:
                    print(frame.point_cloud(stride=8).points.shape)


if __name__ == "__main__":
    main()
