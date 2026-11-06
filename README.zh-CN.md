# iPhone MuJoCo

[English](README.md) · [![CI](https://github.com/ddwty/iphone-mujoco/actions/workflows/ci.yml/badge.svg)](https://github.com/ddwty/iphone-mujoco/actions/workflows/ci.yml)

把虚拟 iPhone 放在桌面支架、机械臂手腕或移动平台上，采集 RGB-D、内参、
相机位姿与可回放的数据。需要真机输入时，通过已有 PyPI `asmagic` SDK 连接 iPhone。
[Python SDK](https://pypi.org/project/asmagic/)

[![MuJoCo 相机实验台、RGB 与深度](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.png)](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.gif)

## 先运行一个离线实验

使用 Python 3.10 及以上版本获取源码并安装。

```bash
git clone https://github.com/ddwty/iphone-mujoco.git
cd iphone-mujoco
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
iphone-mujoco doctor
```

Windows 使用 `py -m venv .venv` 创建环境，再运行
`.venv\Scripts\python.exe -m pip install -e .`。后续命令使用该环境中的
`python` 与 `iphone-mujoco`，也可分别使用完整路径
`.venv\Scripts\python.exe` 和 `.venv\Scripts\iphone-mujoco.exe`。

| 系统 | 打开交互实验台 |
|---|---|
| macOS | `mjpython -m iphone_mujoco phone-lab` |
| Linux | `.venv/bin/python -m iphone_mujoco phone-lab` |
| Windows | `.venv\Scripts\python.exe -m iphone_mujoco phone-lab` |

方向键和 Q/E 移动手机，J/L 与 I/K 转动手机，B 开启扫描，P 保存 RGB-D 与点云。
macOS 也可先运行 `setup.command`，再双击 `Open Phone Lab.command`。

渲染使用 OpenGL。无桌面的 Linux 主机需要兼容 EGL 驱动或 Xvfb/GLFW 环境。
`--headless` 会继续渲染传感器图像。

## 选择一个具体实验

| 实验 | 相机装在哪里 | 采集什么 | 可以用来做什么 |
|---|---|---|---|
| 桌面观察 | 固定支架上的外部相机 | RGB-D、K、光学位姿、点云 | 目标检测、深度几何、遮挡与标定程序 |
| 机械臂腕部 | 刚性安装到末端 link | 同一状态的图像、关节和控制量 | 眼在手观察、轨迹数据、视觉伺服输入 |
| 移动平台 | 安装到沿预设路径运动的底盘 | RGB-D、底盘/相机位姿、仿真时间 | RGB-D 里程计输入、多视角融合、视点覆盖 |
| 自己的 MJCF | 安装到指定 body | 相同的数据帧和 episode 接口 | 复用现有机器人、房间或任务场景 |

### 固定相机与桌面采集

![相机实验台、RGB 与深度](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.gif)

[PNG](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.png) · [GIF](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/phone-lab.gif)

```bash
python examples/scene_camera.py --scenario tabletop \
  --duration 4 --fps 10 --headless --output recordings/tabletop
iphone-mujoco inspect-episode recordings/tabletop/episode --json
```

添加 `--profile profiles/my-iphone.json` 可使用自己的设备内参，默认使用包内相机配置。

### 移动机器人上的相机

![沿预设路径运动的移动平台与 RGB-D](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/mobile-camera.gif)

[PNG](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/mobile-camera.png) · [GIF](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/mobile-camera.gif)

```bash
python examples/scene_camera.py --scenario mobile \
  --duration 6 --fps 10 --output recordings/mobile
```

macOS 打开交互窗口时，将 `python` 换成 `mjpython`。底盘沿预设路径行驶，
窗口同时显示 RGB 与深度，每帧记录底盘和相机位姿。

### 不同机械臂上的腕部相机

![机械臂上安装的虚拟 iPhone 与腕部 RGB-D](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/robot-wrist.gif)

[PNG](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/robot-wrist.png) · [GIF](https://raw.githubusercontent.com/ddwty/iphone-mujoco/main/docs/images/robot-wrist.gif)

| `--robot` | 实验室机械臂 | 相机父 body | 末端配置 |
|---|---|---|---|
| `panda` | Franka Emika Panda，七轴 | `hand` | 平行夹爪 |
| `ur5e` | Universal Robots UR5e，六轴 | `wrist_3_link` | 裸法兰 |

下载 Panda 模型并采集一段腕部数据。

```bash
iphone-mujoco models --fetch panda
iphone-mujoco export-lab --output recordings/demo-rig
mjpython -m iphone_mujoco capture-wrist --robot panda \
  --profile recordings/demo-rig/calibration.json --show-phone \
  --duration 5 --fps 10 --output recordings/wrist-panda
```

Linux/Windows 使用 `python -m iphone_mujoco`。六轴机械臂实验可选择 UR5e。
每种模型首次下载后，后续运行会复用本地缓存。

```bash
iphone-mujoco models --fetch ur5e
mjpython -m iphone_mujoco capture-wrist --robot ur5e \
  --profile recordings/demo-rig/calibration.json --show-phone \
  --duration 5 --fps 10 --output recordings/wrist-ur5e
```

`--show-phone` 显示手机，`--mount measured-mount.json` 载入自己的安装外参。
RGB、深度和机器人状态同步采集。`--motion hold` 保持固定关节目标，
采集窗口同时显示机器人、RGB 与深度。

两种机械臂均采用 Google DeepMind 的 MuJoCo Menagerie 模型，模型来源与适配方法见开发文档。
[Menagerie](https://github.com/google-deepmind/mujoco_menagerie) · [开发文档](docs/development.md)

### 放入自己的实验场景

为已有 MJCF 指定安装 body，并通过 mount JSON 描述光学相机坐标到父 link 坐标的固定变换。

```bash
python examples/scene_camera.py --scenario custom \
  --scene path/to/scene.xml --parent-body flange \
  --mount path/to/mount.json --profile profiles/my-iphone.json \
  --duration 4 --fps 10 --output recordings/custom-scene
```

通过 body 名称将相机装到机械臂法兰、移动底盘、头部或固定支架。
完整例子与 mount 格式见相机指南。[相机安装指南](docs/phone-lab.md)

也可导出原生 MJCF 相机模组，include 到自己的 `worldbody` 或机器人 body 中。

```bash
iphone-mujoco export-phone --output recordings/iphone.xml
```

## 把采集结果用于算法

腕部采集会保存 RGB PNG、UInt16 深度 PNG、标定、安装外参、`states.jsonl`、
点云预览与 `episode/`。通过 episode 读取图像、内参、位姿和逐帧元数据。

```python
from iphone_mujoco.recording import ReplaySource

with ReplaySource("recordings/wrist-panda/episode") as source:
    frame = source.poll().rgbd
    rgb, depth_m = frame.rgb, frame.depth_m
    K_rgb = frame.calibration.matrix
    K_depth = frame.calibration.depth_calibration().matrix
    world_from_camera = frame.world_from_camera
    qpos = frame.metadata["robot_state"]["qpos"]
    world_points = frame.point_cloud(world=True)
```

使用录制的光学位姿，可将移动相机序列融合成世界坐标下的体素点云。

```bash
python examples/fuse_episode.py recordings/mobile/episode \
  --output recordings/mobile-map --voxel-size .01 --stride 4
```

融合脚本在体素网格中平均静态场景的点和颜色，输出 `points.ply`、逐帧来源及摘要。
数据字段与深度 PNG 读取方法见腕部指南。[腕部数据指南](docs/wrist-rgbd.md)

## 接入自己的真实 iPhone

安装可选的真机依赖，并采集当前设备的 K。

```bash
python -m pip install -e '.[live]'
iphone-mujoco calibrate --ip YOUR_IP --output profiles/my-iphone.json
```

在 asMagic 打开 **Stream**，开启 RGB/深度发送，选择 **Y up** 并关闭自定义坐标调整。
App 的 Z-up 预设使用 `--pose-convention robotics`。更改拍摄配置后重新采集标定。

要控制仿真中的腕部相机，切换到 **Robot control → Through Mac** 并开启发送。
连接后先松开 Square，再按住移动。

```bash
mjpython -m iphone_mujoco capture-wrist --robot panda \
  --profile profiles/my-iphone.json --show-phone --ip YOUR_IP \
  --duration 0 --output recordings/phone-controlled-wrist
```

此工作流使用手机位姿和按钮作为控制输入，腕部 RGB-D 由 MuJoCo 渲染。
Stream 与 Robot control 共享 8000 端口，作为独立 App 工作流使用。
录制真实 RGB-D 时，在 Stream 页面运行
`iphone-mujoco record --ip YOUR_IP --mode camera --duration 10 --output recordings/phone`。

## 数据约定

| 数据 | 约定 |
|---|---|
| RGB | `uint8`，H×W×3，RGB 通道，原始相机方向 |
| 深度 | 光轴 Z，单位米，无效米制深度为 NaN |
| UInt16 导出 | 明确每单位多少米，通常 `0.0001`，0 无效，65535 保留 |
| 内参 | 完整 fx/fy/cx/cy，RGB 与深度各自的尺寸 |
| 光学位姿 | MuJoCo Z-up 世界 ← CV 相机，X 右、Y 下、Z 前 |
| 时间 | 区分仿真、发送端和本机收到时间 |

真实安装使用设备标定与实测安装外参，传感器模型和控制器行为见开发文档。

## 文档

- [相机实验台与安装](docs/phone-lab.md) · [腕部 RGB-D 数据集](docs/wrist-rgbd.md)
- [开发文档](docs/development.md) · [贡献指南](.github/CONTRIBUTING.md) · [安全说明](.github/SECURITY.md)
- [发布版本](https://github.com/ddwty/iphone-mujoco/releases) · [MIT 许可证](LICENSE)

录制、标定、模型缓存和验证输出保存在本地忽略目录，每次实验使用新的输出目录。
