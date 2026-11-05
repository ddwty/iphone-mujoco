#!/bin/bash
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/mjpython ]; then
  echo 'Run setup.command first.'
  exit 1
fi
echo '1  Virtual iPhone RGB-D laboratory'
echo '2  iPhone controls teaching arm'
echo '3  iPhone controls UR5e'
echo '4  iPhone controls Panda'
echo '5  Live iPhone camera + paired simulation'
echo '6  Collect wrist RGB-D with iPhone calibration'
echo '7  Teaching arm + virtual iPhone'
read -r -p 'Choose [1]: ' choice
case "${choice:-1}" in
  1)
    if [ -f profiles/iphone-live.json ]; then
      exec .venv/bin/mjpython -m iphone_mujoco phone-lab --profile profiles/iphone-live.json
    fi
    exec .venv/bin/mjpython -m iphone_mujoco phone-lab ;;
  7)
    if [ -f profiles/iphone-live.json ]; then
      exec .venv/bin/mjpython -m iphone_mujoco simulate --profile profiles/iphone-live.json --duration 0
    fi
    exec .venv/bin/mjpython -m iphone_mujoco simulate --duration 0 ;;
  2|3|4|5)
    read -r -p 'iPhone IP: ' phone_ip
    if [ -z "$phone_ip" ]; then echo 'Enter the IP shown in the iPhone app.'; exit 1; fi
    case "$choice" in
      2) exec .venv/bin/mjpython -m iphone_mujoco teleop --ip "$phone_ip" --robot reference6 --duration 0 ;;
      3) exec .venv/bin/mjpython -m iphone_mujoco teleop --ip "$phone_ip" --robot ur5e --duration 0 ;;
      4) exec .venv/bin/mjpython -m iphone_mujoco teleop --ip "$phone_ip" --robot panda --duration 0 ;;
      5) exec .venv/bin/mjpython -m iphone_mujoco camera --ip "$phone_ip" --duration 0 ;;
    esac ;;
  6)
    read -r -p 'Calibration JSON path: ' wrist_profile
    if [ ! -f "$wrist_profile" ]; then echo 'Choose a calibration JSON exported by calibrate or export-lab.'; exit 1; fi
    wrist_output="recordings/wrist-$(date '+%Y%m%d-%H%M%S')"
    exec .venv/bin/mjpython -m iphone_mujoco capture-wrist --robot panda --profile "$wrist_profile" --duration 5 --fps 30 --output "$wrist_output" ;;
  *) echo 'Choose a number from 1 to 7.'; exit 1 ;;
esac
