"""macOS: mjpython examples/open_phone_lab.py [--profile my-iphone.json]."""
import sys
from iphone_mujoco.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["phone-lab", *sys.argv[1:]]))
