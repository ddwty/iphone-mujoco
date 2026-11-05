# Roadmap and acceptance gates

This is an experimental simulation/data workbench. Follow-up should preserve
the optional SDK boundary and improve one observable workflow at a time. The
items below are proposals, not implemented capability claims.

## Current baseline

- Offline phone laboratory, portable MJCF exports and explicit RGB-D contracts.
- Optional real-device adapters, independent freshness, recording/replay and
  offline episode inspection.
- Wrist datasets, two reference arms and optional pinned UR5e/Panda models.

Update the capability manifest when implementations change, rather than only
renaming a capability label.

## 1. Reproducible installation and data

Make a new checkout usable without private files. Keep ordinary runtime/tests
offline except deliberate dependency/model installation.

**Acceptance:** wheel and sdist build/import; bundled scenes load after
relocation; numerical contracts pass; graphics checks identify backend and
skips. A graphics-skipped job is not rendered proof. Reports should include a
small synthetic reproduction, not a private capture directory.

## 2. Quantitative simulated following

Separate steady-state precision, dynamic lag, saturation and rejected input.
Start with slow bounded trajectories and a fixed model/TCP/controller profile.

**Acceptance:** publish trajectories, parameters, clock/session handling, error
distributions and saturation/failure counts. A proposed precision mode should
first meet a declared slow-motion gate, for example position p95 ≤5 mm and
rotation p95 ≤0.03 rad under the stated conditions. This is a proposed gate,
not current performance. Do not bypass rate/joint limits to improve the score.

## 3. Measured camera rigs

Add hand-eye/extrinsic calibration and optional empirical depth/noise models,
retaining source settings, missing returns and uncertainty.

**Acceptance:** preserve raw observations, device/configuration, held-out
reprojection/depth errors and failure cases. Ideal renderer checks cannot
replace physical calibration. Body/mass parameters remain unknown until their
own measurements exist.

## 4. Additional robot adapters

Prefer a documented adapter over a new control loop. Cover actuator semantics,
TCP orientation, gripper mapping and startup/hold behavior.

**Acceptance:** correct units, validated actuators, retained source/license and
bounded joint/Cartesian smoke. Declare illustrative, vendor-derived or measured
dynamics explicitly. A downloaded model does not establish hardware readiness.

## 5. Additional iPhone signals

IMU simulation, hand/body/face retargeting, ROS 2 and video return are future
adapters. A signal available in an SDK does not establish a simulator model.

**Acceptance:** each has explicit frame/time/units, missing-data behavior and
a useful reproducible experiment. Hardware control/safety needs a separate
scope and independent review, not a routine simulator adapter.

## Release decisions

Ship a smaller verified workflow when a feature's acceptance gate is unmet.
Document compatibility changes and limits in the changelog. CI scope, package
publication, physical-device results and precision claims are separate decisions;
evidence for one does not substitute for another.
