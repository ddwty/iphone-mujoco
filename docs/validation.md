# Validation and evidence

Keep package, graphics, protocol and physical-device evidence distinct. This
page describes reproducible checks; private recordings and machine-specific
logs are intentionally not part of the repository.

## Candidate 0.3.0 local checks (2026-11-05)

- 12 new offline CLI/episode contracts passed, including nonfinite argument
  rejection, nested calibration output and useful errors for corrupt episodes.
- 3 source-boundary/export fixtures passed. They protect ignored datasets,
  private machine values and explicit public-source selection.
- 3 migration/calibration contracts passed. Old imports use the same classes as
  the canonical package; NumPy-valued calibration saves as portable JSON; model
  caches from previous installs remain usable without another download.
- An actual optional-SDK localhost TCP smoke passed using the installed PyPI
  SDK. This checks fixtures/encoding/recording, not wireless-device delivery.
- Wheel and sdist builds passed. An isolated clean environment installs the
  wheel without the optional live SDK, private profiles or model cache.

The graphical acceptance procedure renders a synthetic phone-lab observation
from that clean installation, checks RGB/depth dimensions, UInt16/metric depth
and invalid masks, exports point clouds, then runs offline episode inspection.
The macOS driver reports limited depth precision when ARB_clip_control is
unavailable; the toolkit does not conceal that diagnostic.

## GitHub CI

The repository defines focused installed-wheel contracts on Ubuntu, macOS and
Windows, with an additional Ubuntu Python 3.10 job. A separate Linux Xvfb/Mesa
job renders a synthetic RGB-D observation and checks the saved data.
The optional SDK wire checks run in one Ubuntu job.

Consult the actual Actions result for the commit you use. Workflow presence,
skipped rendering or a package build alone is not evidence of a rendered or
physical-device result. The CI pipeline never needs a real iPhone or robot.

## Reproduce a bounded check

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_cli_contracts.py tests/test_compatibility.py -q
python tools/check_repo_hygiene.py --mode tracked
```

With a working graphics context:

```bash
iphone-mujoco phone-lab --headless --output recordings/synthetic-smoke
iphone-mujoco inspect-episode recordings/synthetic-smoke/capture-0000/episode --json
```

Native-viewer acceptance can use `mjpython tools/smoke_phone_lab.py` on macOS
(provide a new `--output` directory). It injects application commands rather
than OS keyboard events, checks the camera moved/rotated and verifies the saved
RGB-D snapshot. Other platforms may use Python with an active graphics session.

## What has not been established

Default phone bodies and wrist mounts are approximate. Ideal depth rendering
does not establish physical LiDAR noise, RGB/depth extrinsics, rolling shutter,
exposure or confidence. Runtime K snapshots are not universal factory values.

Previous local phone-to-simulation experiments exercised pose/button delivery
and clutch/holding for two official arms. They produced substantial dynamic
target-following residuals; no precision-teleoperation criterion was met. They
do not establish hardware safety, absolute tracking accuracy, grasping success
or one-way network latency. Current releases require separate device evidence
before making such claims; see [acceptance gates](roadmap.md).
