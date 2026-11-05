# Development instructions

- Public package/CLI: `iphone_mujoco` / `iphone-mujoco`.
- Reuse the PyPI `asmagic` SDK for live transport and codecs. Do not vendor its
  implementation or independently redefine the wire schemas.
- Preserve recorded format/version, pixel units, optical Z depth, coordinate
  transforms and the distinction between sender, source and local clocks.
- Keep legacy import aliases as references to canonical classes, not copies.
- Do not delete local recordings or validation outputs while cleaning source.
  Use the explicit source/index hygiene tools before staging or publishing.
- Run checks proportionate to the changed contract. A wheel build plus focused
  smoke is the default; GUI/physical-device checks require separate evidence.
- Update examples and architecture/validation notes when behavior changes.
  Do not claim hardware control, latency, accuracy or CI success from unrelated
  simulation/build results.
