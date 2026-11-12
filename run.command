#!/bin/bash
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/mjpython ]; then
  echo 'Run setup.command first.'
  exit 1
fi
echo '1  iPhone optical inspection bench'
echo '2  Franka Panda + virtual iPhone'
echo '3  Universal Robots UR5e + virtual iPhone'
echo '4  iPhone controls Franka Panda'
echo '5  iPhone controls Universal Robots UR5e'
echo '6  Live iPhone camera + paired Panda simulation'
echo '7  Collect Panda wrist RGB-D with iPhone calibration'
echo '8  PAL Robotics TIAGo mobile RGB-D experiment'
read -r -p 'Choose [1]: ' choice
case "${choice:-1}" in
  1)
    if [ -f profiles/iphone-live.json ]; then
      exec .venv/bin/mjpython -m iphone_mujoco phone-lab --profile profiles/iphone-live.json
    fi
    exec .venv/bin/mjpython -m iphone_mujoco phone-lab ;;
  2|3)
    robot=panda
    if [ "$choice" = 3 ]; then robot=ur5e; fi
    if [ -f profiles/iphone-live.json ]; then
      exec .venv/bin/mjpython -m iphone_mujoco simulate --robot "$robot" --profile profiles/iphone-live.json --duration 0
    fi
    exec .venv/bin/mjpython -m iphone_mujoco simulate --robot "$robot" --duration 0 ;;
  4|5|6)
    read -r -p 'iPhone IP: ' phone_ip
    if [ -z "$phone_ip" ]; then echo 'Enter the IP shown in the iPhone app.'; exit 1; fi
    case "$choice" in
      4) exec .venv/bin/mjpython -m iphone_mujoco teleop --ip "$phone_ip" --robot panda --duration 0 ;;
      5) exec .venv/bin/mjpython -m iphone_mujoco teleop --ip "$phone_ip" --robot ur5e --duration 0 ;;
      6) exec .venv/bin/mjpython -m iphone_mujoco camera --ip "$phone_ip" --robot panda --duration 0 ;;
    esac ;;
  7)
    read -r -p 'Calibration JSON path: ' wrist_profile
    if [ ! -f "$wrist_profile" ]; then echo 'Choose a calibration JSON exported by calibrate or export-lab.'; exit 1; fi
    wrist_output="recordings/wrist-$(date '+%Y%m%d-%H%M%S')"
    exec .venv/bin/mjpython -m iphone_mujoco capture-wrist --robot panda --profile "$wrist_profile" --show-phone --duration 5 --fps 30 --output "$wrist_output" ;;
  8)
    mobile_output="recordings/tiago-$(date '+%Y%m%d-%H%M%S')"
    exec .venv/bin/mjpython -m iphone_mujoco capture-mobile --robot tiago --device iphone-17-pro --duration 8 --fps 15 --output "$mobile_output" ;;
  *) echo 'Choose a number from 1 to 8.'; exit 1 ;;
esac
