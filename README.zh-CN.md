# iPhone MuJoCo

[English](README.md) · [![CI](https://github.com/ddwty/iphone-mujoco/actions/workflows/ci.yml/badge.svg)](https://github.com/ddwty/iphone-mujoco/actions/workflows/ci.yml)

把虚拟 iPhone 放在桌面支架、机械臂手腕或移动平台上，采集 RGB-D、内参、
相机位姿与可回放的数据。需要真机输入时，通过已有 PyPI `asmagic` SDK 连接 iPhone。
[Python SDK](https://pypi.org/project/asmagic/)

![MuJoCo 相机实验台、RGB 与深度](docs/images/phone-lab.gif)

动图来自实际 MuJoCo 渲染，使用 synthetic 示例内参。实验台包括已知尺寸物体、
深度台阶、棋盘板和遮挡物，可用于检查相机投影、点云与数据采集程序。

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

```bash
python examples/scene_camera.py --scenario tabletop \
  --duration 4 --fps 10 --headless --output recordings/tabletop
iphone-mujoco inspect-episode recordings/tabletop/episode --json
```

使用已知尺寸检查重建尺度、深度边缘和像素反投影。添加
`--profile profiles/my-iphone.json` 可换成自己的实测内参，省略时使用明确标记的
synthetic 配置。

### 移动机器人上的相机

![沿预设路径运动的移动平台与 RGB-D](docs/images/mobile-camera.gif)

```bash
python examples/scene_camera.py --scenario mobile \
  --duration 6 --fps 10 --output recordings/mobile
```

macOS 打开交互窗口时，将 `python` 换成 `mjpython`。示例底盘由基础几何组成，
沿脚本指定的运动学路径行驶。相机提供变化的视角与位姿，可把这些观测接到自己的
里程计、点云配准或重建算法。
原生窗口在场景旁显示 RGB 与米制深度预览。

### 不同机械臂上的腕部相机

![机械臂上安装的虚拟 iPhone 与腕部 RGB-D](docs/images/robot-wrist.gif)

| `--robot` | 机械臂关节 | 相机父 body | 模型来源 |
|---|---|---|---|
| `reference6` | `joint1`–`joint6` | `tool` | 包内教学模型 |
| `reference7` | `joint1`–`joint7` | `tool` | 包内教学模型 |
| `panda` | `joint1`–`joint7` | `hand` | 按需下载 Menagerie Franka Panda |
| `ur5e` | 肩部旋转/俯仰、肘、三个腕部关节 | `wrist_3_link` | 按需下载 Menagerie UR5e |

先生成离线示例内参，再运行一段腕部采集。

```bash
iphone-mujoco export-lab --output recordings/demo-rig
mjpython -m iphone_mujoco capture-wrist --robot reference6 \
  --profile recordings/demo-rig/calibration.json --show-phone \
  --duration 5 --fps 10 --output recordings/wrist-reference6
```

Linux/Windows 使用 `python -m iphone_mujoco`。将 `--robot` 换成 `reference7`
即可使用另一台内置机械臂。官方机械臂先下载固定版本模型，再选择对应名称。

```bash
iphone-mujoco models --fetch panda
iphone-mujoco models --fetch ur5e
mjpython -m iphone_mujoco capture-wrist --robot panda \
  --profile recordings/demo-rig/calibration.json --show-phone \
  --duration 5 --fps 10 --output recordings/wrist-panda
```

下载器保留上游许可证并校验文件。`--show-phone` 显示近似手机外壳。默认安装外参
是演示配置，实际安装可用 `--mount measured-mount.json` 载入实测结果。
相机通过正向运动学跟随关节，RGB、深度和机器人状态取自同一物理状态。
`--motion hold` 可保持固定关节目标。
腕部采集窗口也会在机器人场景旁显示本帧 RGB 与深度。

官方模型来自 MuJoCo Menagerie，精确关节、执行器及适配约定见开发文档。
[Menagerie](https://github.com/google-deepmind/mujoco_menagerie) · [开发文档](docs/development.md)

### 放入自己的实验场景

为已有 MJCF 指定安装 body，并通过 mount JSON 描述光学相机坐标到父 link 坐标的固定变换。

```bash
python examples/scene_camera.py --scenario custom \
  --scene path/to/scene.xml --parent-body flange \
  --mount path/to/mount.json --profile profiles/my-iphone.json \
  --duration 4 --fps 10 --output recordings/custom-scene
```

父 body 可以是机械臂法兰、移动底盘、头部或固定支架，相机会随它的变换更新。
相机指南提供可以直接复现的自定义场景例子与 mount 格式。[相机安装指南](docs/phone-lab.md)

也可导出原生 MJCF 相机模组，include 到自己的 `worldbody` 或机器人 body 中。

```bash
iphone-mujoco export-phone --output recordings/iphone.xml
```

## 把采集结果用于算法

腕部采集会保存 RGB PNG、UInt16 深度 PNG、标定、安装外参、`states.jsonl`、
点云预览与 `episode/`。场景示例使用相同的回放 API，读取 episode 可以同时保留
内参、位姿、元数据和各自的时间基准。

```python
from iphone_mujoco.recording import ReplaySource

with ReplaySource("recordings/wrist-reference6/episode") as source:
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

示例面向静态场景，在体素网格中平均点和颜色，输出 `points.ply`、逐帧来源以及点数
和包围盒摘要。腕部图像与状态配对也可以接到自己的模仿学习数据转换程序。
完整文件结构及深度 PNG 读取方法见腕部指南。[腕部数据指南](docs/wrist-rgbd.md)

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

## 数据约定与当前范围

| 数据 | 约定 |
|---|---|
| RGB | `uint8`，H×W×3，RGB 通道，原始相机方向 |
| 深度 | 光轴 Z，单位米，无效米制深度为 NaN |
| UInt16 导出 | 明确每单位多少米，通常 `0.0001`，0 无效，65535 保留 |
| 内参 | 完整 fx/fy/cx/cy，RGB 与深度各自的尺寸 |
| 光学位姿 | MuJoCo Z-up 世界 ← CV 相机，X 右、Y 下、Z 前 |
| 时间 | 区分仿真、发送端和本机收到时间 |

手机外形和默认安装外参为近似示例。真实安装需要实测外参及设备标定。
RGB-D 使用理想针孔模型，LiDAR 噪声/置信度、曝光、滚动快门和畸变需要额外建模。
机械臂控制提供受限仿真目标，精密跟随和真实机器人实验需要各自的验证。

## 文档

- [相机实验台与安装](docs/phone-lab.md) · [腕部 RGB-D 数据集](docs/wrist-rgbd.md)
- [开发文档](docs/development.md) · [贡献指南](.github/CONTRIBUTING.md) · [安全说明](.github/SECURITY.md)
- [发布版本](https://github.com/ddwty/iphone-mujoco/releases) · [MIT 许可证](LICENSE)

录制、标定、模型缓存和验证输出保存在本地忽略目录，每次实验使用新的输出目录。
