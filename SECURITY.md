# Security policy

This is an early research/development toolkit. The simulated camera and robot
controllers are not hardware safety systems. See [the roadmap](docs/roadmap.md)
for capability and validation boundaries.

## Reporting a vulnerability

For credential exposure, unsafe file handling or another security issue, use
the repository's **Security → Report a vulnerability** private reporting flow
when available. If private reporting is unavailable, contact the repository
maintainer through their GitHub profile and ask for a private channel. Do not
attach tokens, private recordings, device identifiers or exploitation details
to a public issue. Ordinary reproducible bugs belong in the bug template.

## Network boundary

The optional live adapter uses the published `asmagic` SDK. Its ZeroMQ camera
and control channels do not provide application-level authentication or TLS.
Use a trusted network. A virtual publisher should bind to loopback unless LAN
clients are intentionally required; do not expose it directly to the Internet.
The package does not require an account or a network connection for offline
simulation and dataset inspection.

## Local files and releases

- Recordings may contain identifiable camera imagery, room geometry and motion.
  Keep them in ignored local directories and share only deliberately selected,
  redacted samples.
- Replay uses numeric NPZ arrays with `allow_pickle=False`. Treat downloaded
  datasets, MJCF and third-party assets as untrusted input; process trusted
  sources and avoid running downloaded Python scripts as part of an experiment.
- Official robot assets are fetched at a pinned commit with size/Git-blob
  checks, and their licenses are retained. Hash checks identify files; they do
  not establish the physical accuracy of a model.
- `tools/check_repo_hygiene.py` and the source export allowlist prevent common
  accidental inclusions. They supplement human review and GitHub secret
  scanning; they are not comprehensive secret detection.
- Use SHA256SUMS from the same reviewed GitHub Release to check artifact
  integrity. The project does not yet provide signed provenance attestations.

Before changing network behavior, external publishing, dependency sources or
recording schemas, include the security and compatibility effect in the PR.
