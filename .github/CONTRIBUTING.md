# Contributing

Describe the observed problem, resulting behavior and checks run.
Review [the data contracts](../docs/development.md)
before changing formats, frames, timing, controllers or SDK integration.

## Develop and check

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
iphone-mujoco doctor
```

Use `'.[live,dev]'` for optional SDK work. Reuse the PyPI `asmagic` transport,
schemas and codecs. Offline work needs no phone or private calibration.
Run a focused check for the changed contract, then build when changing
packaging/assets:

```bash
python -m pytest tests/test_phone_lab.py -q
python -m build
python tools/check_repo_hygiene.py --mode tracked
```

Camera tests require a working graphics context and opt in:
`IPHONE_MUJOCO_TEST_RENDER=1 python -m pytest tests/test_camera.py tests/test_wrist.py -q`.
For a native-viewer smoke on macOS, use
`mjpython tools/smoke_phone_lab.py --output recordings/gui-smoke` with a new
output directory.
Use `python` with an active graphics session on other platforms.

Download model assets before running their tests. SDK loopback requires live
dependencies and local TCP access. For device integrations, record the device,
settings, inputs, timestamps and results. Installed-wheel contracts and a Linux
RGB-D smoke run in [CI](workflows/ci.yml).

Generate the README animations with a graphics context and the Panda/UR5e models.

```bash
iphone-mujoco models --fetch panda
iphone-mujoco models --fetch ur5e
python tools/render_readme_media.py --output recordings/readme-media
```

Use `--robots panda` or `--robots ur5e` for a single-arm recording. Review media
before copying it into `docs/images/`. Keep each public image below the source
export's 2 MB limit.

## Source and compatibility

Keep environments, model caches, recordings, private calibration/provenance,
validation logs, builds and credentials out of Git. Preserve original local
data; share only synthetic or deliberately sanitized reproductions. Before
publishing, run `python tools/check_repo_hygiene.py --mode staged` and review
the staged diff. `python tools/export_source.py` creates a reviewed-source ZIP
and per-file hash manifest without workbench data or dependency caches.

Add focused regressions for consequential bugs. Update usage docs when CLI or
data behavior changes. Preserve legacy aliases as the canonical classes and
version schema changes; reject unsupported formats. Robot changes must specify
model source/license, joints/actuators, TCP, units and controller semantics.

## Release

1. Align `pyproject.toml`, `iphone_mujoco.__version__` and capability metadata.
2. Build wheel/sdist; inspect bundled assets and test a clean installation.
   Check focused CI for the candidate commit and review its source selection.
3. Create the matching `vX.Y.Z` tag and a draft GitHub Release describing actual
   behavior, compatibility, checks and remaining limits; review before publishing.
4. Publishing triggers [release artifacts](workflows/release.yml): the tag must
   match the package version. The workflow attaches wheel, sdist, clean source
   ZIP, manifest and `SHA256SUMS`. Verify remote assets/checksums and the commit.

Use [GitHub Releases](https://github.com/ddwty/iphone-mujoco/releases) for release
notes. Never rewrite a published tag or main history as routine maintenance.
PyPI publication is separate and is not configured by this workflow; use
Trusted Publishing when ownership and exact artifacts have been verified.

Contributions follow the [MIT license](../LICENSE). Retain third-party licenses
and notices; downloaded model assets remain under their original terms. See
[the security policy](SECURITY.md) for private reports and network boundaries.
