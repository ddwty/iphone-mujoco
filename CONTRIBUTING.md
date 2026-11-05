# Contributing

Contributions should make the virtual/real RGB-D workflow easier to reproduce
without weakening its data or control contracts. Small issues, documentation
fixes, robot adapters and focused bug fixes are welcome.

## Development setup

Create a branch from the default branch and work from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
iphone-mujoco doctor
```

Install `'.[live,dev]'` when working on optional SDK integration. Offline work
does not require private calibration or an iPhone.

## Describe a bounded change

Give the observed problem, intended behavior and smallest reproducible case.
For robot changes state model source/license, joints/actuators, TCP frame, units
and controller type. For cameras state raster sizes, K, coordinate convention,
depth units and missing-data policy.

Transport, Protobuf schemas and codec support belong to the existing PyPI
`asmagic` SDK. Adapt that SDK rather than copying receivers or HEVC decoders.
Label simulated, measured and unavailable quantities explicitly.

## Verify the changed contract

Run checks relevant to the change before expanding validation, for example:

```bash
python -m pytest tests/test_phone_lab.py -q
python -m pytest tests/test_streams.py -q
```

Camera rendering checks require a real graphics context and opt in:

```bash
IPHONE_MUJOCO_TEST_RENDER=1 python -m pytest tests/test_camera.py tests/test_wrist.py -q
```

Native-viewer smoke on macOS:

```bash
mjpython tools/smoke_phone_lab.py --output recordings/gui-smoke
```

That smoke injects application commands, not physical OS keyboard input. Use a
fresh output directory. Other platforms can use `python` with a graphics session.
Optional-model tests skip when assets are absent; deliberately download assets
when changing that adapter.

The SDK loopback smoke requires live dependencies and local TCP access. It
checks protocol integration, not wireless delivery or a physical camera.
Device claims need separate device/configuration, inputs, timestamps, results
and failure evidence.

When changing packaging or bundled assets:

```bash
python -m build
```

Inspect both artifacts and test import from an isolated installation. Runtime
must not depend on neighboring checkouts or personal paths.

## Keep source and experiments separate

Do not commit virtual environments, caches, downloaded models, recordings,
private calibration/provenance, local validation logs, builds or credentials.
Use synthetic fixtures or small sanitized reproductions. Do not upload an
entire personal capture to report a bug. Preserve original user data while
fixing recorders or importers.

## Pull requests and compatibility

Explain concrete before/after behavior, checks actually run, skips and remaining
device/graphics limits. Update docs when CLI, formats or frames change. Add a
focused regression for consequential bugs; avoid tests that merely repeat the
implementation. Version data/schema changes and reject unsupported formats.

The public package/import names are `iphone-mujoco` / `iphone_mujoco`. Legacy
import aliases and the existing episode format identifier may remain for
compatibility; do not rename recorded formats without an explicit migration.
Record user-visible changes in the unreleased changelog and keep commits scoped.

## Licensing and security

Original contributions follow the [MIT license](LICENSE). Preserve model-specific
licenses and third-party notices; downloaded assets are not project-owned code.
By submitting a contribution you represent that you can contribute it under
these terms. See [SECURITY.md](SECURITY.md) and [the release guide](docs/releasing.md).
