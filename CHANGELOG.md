# Changelog

Public APIs use semantic versioning and datasets have explicit format versions.
Before 1.0, minor releases may change APIs; document migrations when they do.
Older entries are local development milestones, not claims of GitHub or PyPI
publication.

## 0.3.0 — 2026-11-05 (initial preview)

### Added

- Fresh-checkout English and Chinese onboarding, contribution guidance,
  architecture/data/time contracts and a bounded roadmap.
- Offline episode inspection and explicit environment/graphics diagnostics.
- Open-source repository hygiene and release preparation.

### Changed

- Public project/distribution/CLI name: **iPhone MuJoCo / `iphone-mujoco`**.
  Public Python import: `iphone_mujoco`. The optional `asmagic` SDK keeps its
  existing upstream name; recorded episode identifiers retain compatibility.
- Quick starts begin offline without private profiles, network addresses or
  neighboring repositories. Live connectivity remains optional.
- Private recordings, calibration and local validation stay outside source.

## 0.2.0 — 2026-11-04

- Added visible phone/tabletop laboratory, native RGB/depth previews, frustum,
  motion controls and snapshots.
- Added portable lab and fixed-camera export with calibration sidecars.
- Added module/relocation and graphics smoke checks.

## 0.1.1 — 2026-10-23

- Added rigid wrist cameras and synchronized RGB-D/robot-state datasets.
- Added RGB PNG, UInt16 depth PNG and replayable episodes.
- Preserved the final replay input and sampled after a physics step to avoid
  zero source-time ambiguity.

## 0.1.0 — 2026-10-23

- Added calibrated cameras, point clouds, SDK adapters and recording/replay.
- Added reference arms, pinned official UR5e/Panda adapters and bounded control.
- Fixed repeated hold updates that let servo targets follow gravity-induced
  joint displacement.
