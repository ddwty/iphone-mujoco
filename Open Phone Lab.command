#!/bin/bash
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/mjpython ]; then
  echo 'Run setup.command once, then open this file again.'
  read -r -p 'Press Return to close. ' unused
  exit 1
fi
lab_args=()
if [ -f profiles/iphone-live.json ]; then
  lab_args+=(--profile "$PWD/profiles/iphone-live.json")
fi
exec .venv/bin/mjpython -m iphone_mujoco phone-lab "${lab_args[@]}"
