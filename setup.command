#!/bin/bash
set -eu
cd "$(dirname "$0")"
install_target='.'
if [ "${1:-}" = '--live' ]; then install_target='.[live]'; fi
if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="$PWD/.cache/uv"
  if [ ! -x .venv/bin/python ]; then uv venv --python python3 .venv; fi
  uv pip install --python .venv/bin/python -e "$install_target"
else
  if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
  .venv/bin/python -m ensurepip --upgrade
  .venv/bin/python -m pip install -e "$install_target"
fi
.venv/bin/python -m iphone_mujoco doctor
echo 'Setup complete. Open run.command to start.'
echo 'For live iPhone input, run setup.command --live once.'
