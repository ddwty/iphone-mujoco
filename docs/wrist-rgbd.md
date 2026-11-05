# 采集与 iPhone 接口一致的腕部 RGB-D

`capture-wrist` 把相机刚性挂到机械臂末端。相机的世界位姿由机器人关节状态和固定安装外参计算，RGB、深度、关节状态都来自同一个 `mjData` 时刻。

先用 `calibrate --ip YOUR_IP --output profiles/my-iphone.json` 保存自己的真机内参。当前公开演示配置的 RGB 是 **640×480**，深度是 **256×192**；具体真机尺寸以收到的标定为准。采集器使用完整 `fx/fy/cx/cy`，深度单独使用对应分辨率的 K。更改手机拍摄模式后，先在 asMagic Stream 开启发送，再重新运行 `calibrate` 获取参数。

## 开始采集

在 PhoneMujuco 根目录运行：

```bash
.venv/bin/mjpython -m iphone_mujoco capture-wrist \
  --robot panda \
  --profile profiles/my-iphone.json \
  --duration 5 --fps 30 \
  --output recordings/wrist-panda
```

默认执行小幅度的关节演示运动，相机随腕部移动。使用 `--robot ur5e` 切换机械臂；`--motion hold` 保持模型起始目标。所有输出目录必须是新的或空的。

仅生成数据、关闭交互窗口时：

```bash
.venv/bin/python -m iphone_mujoco capture-wrist \
  --robot panda --profile profiles/my-iphone.json \
  --headless --fast --duration 5 --fps 30 \
  --output recordings/wrist-offline
```

`fps` 是每个**仿真秒**采多少帧。渲染、压缩、写盘可能比真实时间慢；采集器逐仿真步保存完整序列，记录实际仿真时间戳和运行耗时。`--headless` 仍需要可用的 OpenGL；Linux 服务器可配置 MuJoCo EGL。

## 用 iPhone 遥控时同步采集腕部图像

iPhone 打开 **Robot control → Through Mac** 并发送，然后运行：

```bash
.venv/bin/mjpython -m iphone_mujoco capture-wrist \
  --robot panda --profile profiles/my-iphone.json \
  --ip YOUR_IP --duration 0 \
  --output recordings/wrist-phone-control
```

连接后先松开 Square，再按住移动，松开保持；Ctrl-C 结束并关闭录制。
这里由 iPhone 提供控制输入，腕部 RGB-D 来自 MuJoCo 渲染。
asMagic 当前 Robot control 页本身只发送位姿和控制量。

也可以用已经录好的遥控输入生成仿真图像：

```bash
.venv/bin/mjpython -m iphone_mujoco capture-wrist \
  --robot panda --profile profiles/my-iphone.json \
  --input-episode recordings/my-teleoperation --duration 0 \
  --output recordings/wrist-from-input
```

实时/回放输入使用本机时钟检查新鲜度，并保留控制器的速度、关节和断流限制。
渲染计算量会影响运行速度，因此这类数据不应直接作为真实系统延迟测量。
该入口当前映射 Square、Cross 和夹爪；完整机械臂按钮实验可使用原有 `teleop` 命令。

## 输出数据

```text
wrist-panda/
  rgb/00000000.png          # RGB uint8，640×480
  depth/00000000.png        # 原始深度 uint16，256×192
  calibration.json         # 可由 CameraCalibration.load 直接读取
  camera_info.json         # RGB K、深度 K、单位、时间约定和来源
  mount.json               # 腕部到光学相机的固定外参及是否实测
  states.jsonl             # 每帧时间、关节角/速度、控制量、相机位姿
  episode/                 # 与真实 iPhone 录制相同的 NPZ/JSON episode
  preview_points.ply       # 第一帧的相机坐标点云
  run.json                 # 实际帧数、仿真时间、运行耗时
```

同一编号的 RGB、深度、状态是一帧。RGB PNG 保存无损 RGB；深度 PNG 保存数值，**不要当作 8 位灰度预览来读取**。

| 数据 | 约定 |
|---|---|
| RGB | `uint8 (480, 640, 3)`，RGB 通道，原始相机方向 |
| 原始深度 PNG | `uint16 (192, 256)`，默认每单位 `0.0001 m` |
| 米制深度 | `depth_m = raw_depth × meters_per_unit`，无效值为 NaN |
| 无返回深度 | PNG 中为 0；65535 保留为饱和码，不生成有效深度 |
| 深度定义 | 光轴 Z 深度，单位米 |
| 相机局部坐标 | X 向右、Y 向下、Z 向前 |
| `world_from_camera` | 从上述光学坐标到 MuJoCo Z-up 世界的 4×4 矩阵 |
| 时间 | 仿真秒，与关节状态的 `simulation_time` 相同 |

读取 PNG：

```python
import json
import numpy as np
from PIL import Image

root = "recordings/wrist-panda"
info = json.load(open(root + "/camera_info.json"))
rgb = np.array(Image.open(root + "/rgb/00000000.png"))
raw_depth = np.array(Image.open(root + "/depth/00000000.png"))
depth_m = raw_depth.astype(np.float32) * info["depth"]["meters_per_unit"]
depth_m[raw_depth == 0] = np.nan
```

推荐通过统一数据接口读取，这样真实和仿真都得到 `RGBDFrame`：

```python
from iphone_mujoco.recording import ReplaySource

with ReplaySource("recordings/wrist-panda/episode") as source:
    frame = source.poll().rgbd
    rgb, depth_m = frame.rgb, frame.depth_m
    K_rgb = frame.calibration.matrix
    K_depth = frame.calibration.depth_calibration().matrix
    camera_pose = frame.world_from_camera
    qpos = frame.metadata["robot_state"]["qpos"]
    cloud = frame.point_cloud(world=True)
```

这里 `ReplaySource` 默认逐帧读取、不等待。如果设置 `realtime=True`，它按当时本机
接收/写盘间隔等待；如需按仿真速度播放，应使用每帧 `timestamp` 中的仿真时间。

如需已有 PyPI `asmagic` SDK 接收仿真相机，可在采集命令增加
`--publish tcp://127.0.0.1:18000`。接收仍使用
`asmagic.ARDataSubscriber("127.0.0.1", port=18000)`；复用既有协议和 SDK。
此路径的 JPEG 使用有损编码，而数据集 RGB PNG 保持无损。

## 安装外参与传感器模型

`mount.json` 的 `parent_from_camera` 表示
`T_world_camera = T_world_parent × T_parent_camera`。
默认 Panda 挂在 `hand`，UR5e 挂在 `wrist_3_link`；都朝向初始工作台。
默认固定外参标记为 `measured=false`，是演示安装。
真实 iPhone 安装到手腕后，需要测量/手眼标定，再用 `--mount measured-mount.json` 读取。

只挂光学 rig，不添加近似手机外壳，以免遮住镜头或给机器人加入未经测量的质量。
如需真实手机的负载动力学，应另外加入测量后的外形、质量和惯量。

当前与 iPhone 对齐的是数据接口、K、图像尺寸、深度量化单位和坐标约定。
RGB-D 是理想渲染，未模拟 LiDAR 噪声、滚动快门、曝光、畸变与真实置信度。
本地渲染测试验证了同帧关联和单位；真实安装外参及真实传感器误差需要实测。
