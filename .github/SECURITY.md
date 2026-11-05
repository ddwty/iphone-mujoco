# Security policy

This experimental simulation toolkit is not a hardware safety system. See
[development contracts](../docs/development.md) for supported behavior and limits.

## Report privately

Use **Security → Report a vulnerability** in the repository when available.
Otherwise contact the maintainer through their GitHub profile for a private
channel. Do not attach tokens, private recordings, device identifiers or exploit
details to public issues. Ordinary reproducible bugs belong in the bug template.

## Network and files

- The optional PyPI `asmagic` SDK uses ZeroMQ without application-level TLS or
  authentication. Use a trusted network; keep virtual publishers on loopback
  unless LAN clients are required. Do not expose them directly to the Internet.
- Recordings can identify people, rooms and motion. Keep them ignored locally
  and share only deliberately selected, redacted samples.
- Replay reads numeric NPZ with `allow_pickle=False`. Treat downloaded datasets,
  MJCF and assets as untrusted; use trusted sources and avoid running unknown
  Python scripts as part of an experiment.
- Robot assets are commit-pinned and hash checked, with retained licenses.
  Hashes identify files; they do not establish physical model accuracy.
- Source allowlisting and hygiene checks supplement human review and secret
  scanning; they are not comprehensive credential detection.
- Verify artifact integrity using `SHA256SUMS` from the same reviewed Release.
  Signed provenance attestations are not currently provided.

Changes to network behavior, publishing, dependency sources or recording
schemas should describe their security and compatibility effects in the PR.
