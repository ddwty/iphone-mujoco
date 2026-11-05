"""mjpython run_iphone.py --ip YOUR_IP [--robot reference6]."""
import sys
from iphone_mujoco.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["teleop", "--duration", "0", *sys.argv[1:]]))
