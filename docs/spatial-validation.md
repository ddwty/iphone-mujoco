# Spatial integration validation — 24 November 2026

This records local simulation and packaging evidence for the asDevice/asTriad
integration. Physical iPhone tracking, measured mounting accuracy and robot
hardware were not tested in this change. No remote CI result is inferred here.

| Check | Result |
|---|---|
| Complete local suite with `IPHONE_MUJOCO_TEST_RENDER=1` | 147 passed, including OpenGL checks |
| Spatial contracts | 22 passed: six models × three lenses, datum/axis checks, measured mount override, replay, actual asTriad geometry and pose-driven capture |
| Built wheel in a fresh virtual environment | Installed with the pinned `spatial` extra; imported from site-packages and rendered a headless spatial capture |
| CLI captures | Lab, Panda wrist and TIAGo mobile captures saved asDevice device/optical metadata |
| Runnable examples | Device placement and spatial tabletop episodes completed |
| Replay of all six smoke outputs | Device optical transform matches the recorded camera pose; simulation clock remains labelled |
| Source hygiene | Source and staged-index checks passed before publication; local outputs remain excluded |

The pinned integrations are asTriad 0.12.0 at
`0243a4a96e001bc47917ac9690058befb4f0d8fa` and asDevice 0.3.0a1 at
`8f9f94f5eaa1921c90dca5340980d41fa22bdd4d`. The local renderer used MuJoCo 3.5.0
on Apple Silicon macOS. It reports `ARB_clip_control` unavailable with
`mjDEPTH_ZEROFAR`, so the depth previews are illustrative rather than a
physical accuracy measurement.

## Regenerated media

All files are 1120 × 760. Each overview and phone inset appends actual asTriad
W/D01/C geometry; the device frame comes from the asDevice adapter. Sensor RGB-D
is captured before adding overview decorations. PNG posters are selected from
the same rendered sequence used by the GIF.

| Pair | Encoded GIF frames | Simulated duration / presentation |
|---|---:|---|
| `phone-lab.png` / `.gif` | 64 | 8.124 s; actuated inspection stage |
| `robot-wrist.png` / `.gif` | 56 | 8.292 s; Panda followed by UR5e, with wrist closeups |
| `mobile-camera.png` / `.gif` | 64 | 8.632 s; wheel-driven TIAGo route |
| `camera-lenses.png` / `.gif` | 5 | 5.0 s; five lens/crop presets, repeated frames merged by GIF encoding |

`tools/render_readme_media.py` writes JSON sidecars with package versions,
per-frame poses and media hashes into the chosen local output directory.
The documentation images were visually inspected, and all PNG/GIF pairs were
copied identically into the GitBook asset directory.

See [device coordinates and reproduction](spatial-frames.md) for commands.
