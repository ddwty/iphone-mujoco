"""Run with mjpython on macOS; reuses the package's capture-wrist workflow.

Example: mjpython examples/collect_wrist_rgbd.py --robot panda
         --profile profiles/iphone-live.json --output recordings/wrist
"""
import sys
from iphone_mujoco.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["capture-wrist", *sys.argv[1:]]))
