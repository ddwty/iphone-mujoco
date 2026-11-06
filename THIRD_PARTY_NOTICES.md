# Third-party notices

The toolkit source and original camera-test scene geometry use the repository's
MIT license. Existing copyright notices are retained. Installed dependencies and
downloaded robot assets keep their own licenses; this repository does not
relicense them.

| Component | Use | License/source |
|---|---|---|
| MuJoCo | Physics, cameras and native viewer | Apache-2.0; [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) |
| NumPy | Geometry and array operations | BSD-3-Clause; [numpy/numpy](https://github.com/numpy/numpy) |
| Pillow | PNG/JPEG and preview images | HPND/Pillow license; [python-pillow/Pillow](https://github.com/python-pillow/Pillow) |
| `asmagic` | Optional live transport, schemas and codecs | MIT; [published SDK](https://pypi.org/project/asmagic/) |
| MuJoCo Menagerie Panda assets | Franka Emika Panda MJCF and meshes | Apache-2.0; upstream model LICENSE |
| MuJoCo Menagerie UR5e assets | Universal Robots UR5e MJCF and meshes | BSD-3-Clause, ROS Industrial Consortium; upstream model LICENSE |

The optional Menagerie download is pinned to commit
`cda5236f98383f6fbaa4820c6b18dba892cd087f` in
[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie).
This source repository contains manifests, not the large vendor meshes. The
downloaded model directories include the upstream LICENSE files. If you
redistribute those assets, carry their notices and satisfy their licenses.

The phone geometry is original work. The project is an independent community
tool; use of product names
does not imply endorsement by Apple, Google DeepMind, Franka or Universal Robots.
