# 从可见的虚拟 iPhone 开始使用 MuJoCo

MuJoCo 中的虚拟 iPhone 是一个可复用相机模组：手机外观、具名相机
`iphone_camera`、RGB 内参与深度分辨率。MuJoCo 负责渲染虚拟世界，采集接口输出
与 asMagic iPhone 适配器相同的 `RGBDFrame`。

## 先打开实验台

本机双击项目根目录的 **Open Phone Lab.command**。
也可以双击 **run.command → 1**。不需要连接真机，不需要下载机械臂。

```bash
cd iphone-mujoco
.venv/bin/mjpython -m iphone_mujoco phone-lab --profile profiles/my-iphone.json
```

macOS 窗口使用 `mjpython`，Linux/Windows 可用 `python -m iphone_mujoco`。
主画面显示手机、支架、实验台和相机视锥，右侧同时显示镜头 RGB 与米制深度预览。
滚轮缩放、鼠标拖动观察场景；镜头预览不会跟着总览视角变化。

| 按键 | 动作 |
|---|---|
| 方向键 | 在桌面水平面上移动虚拟手机 |
| Q / E | 降低 / 升高手机 |
| J / L | 左右转动手机 |
| I / K | 改变镜头俯仰 |
| 1 / 2 | 总览 / 手机第一视角 |
| P | 保存当前 RGB-D、标定、位姿和点云 |
| R | 手机回到默认位置 |
| B | 开关缓慢横向扫描 |
| V / F | 开关右侧预览 / 视锥 |
| 关闭窗口 | 结束运行 |

默认入口使用标为 synthetic 的演示配置。用 `--profile profiles/my-iphone.json` 载入自己的真机标定；换手机或拍摄配置后重新 `calibrate`。
别人首次使用时不需要此文件：不传 `--profile` 会使用明确标记的 synthetic 示例。

## 场景里可以做什么

实验台宽 1.10 m、深 0.80 m，放有红色立方体、蓝色球、绿色圆柱、
40/80/120 mm 高度台阶、35 mm 格子的棋盘板、100 mm 间距刻度与遮挡柱。
它是观察 RGB-D 几何和测试采集程序的静态场景，没有假装机器人完成抓取。

- 移动手机观察遮挡变化，并按 P 保存不同视角的数据。
- 将深度回投影成点云，用相机世界位姿检查物体尺寸与台阶高度。
- 在棋盘板上验证自己的角点检测、相机投影与标定程序。
- 更换内参，比较视场变化，再将模组接到其他桌面或机器人场景。

每次 P 会在 `recordings/phone-lab-时间/capture-编号/` 保存：

```text
rgb.png                 无损 RGB
depth_m.npy             float32 光轴 Z 深度，单位米，NaN 为无返回
depth_uint16.png        uint16 深度，每单位 0.0001 m，0 为无返回
depth_preview.png       .15–2 m 的颜色预览，不能作为深度数值
calibration.json        RGB K 与深度尺寸
capture.json            相机世界位姿、时间、来源和单位
points.ply              光学相机坐标点云
points_world.ply        MuJoCo 世界坐标点云
episode/                可由 ReplaySource 读取的统一数据格式
```

远于 UInt16 协议可表示范围的深度会作为无返回处理；PNG 与米制数组使用同一量化。
实际 LiDAR 噪声、反射、曝光与置信度未建模。手机外形和镜头安装位置是近似外观，
没有加入未测量的质量、惯量或碰撞模型。

## 把整个场景交给别人

```bash
iphone-mujoco export-lab --output my-phone-lab
```

生成 `scene.xml`、`iphone.xml`、`calibration.json` 和 `scene_info.json`，
另附可固定到机器人上的 `iphone_fixed.xml` 与其标定 JSON。
文件全部使用相对引用，整个目录复制到其他计算机后可直接加载 **scene.xml**：

```bash
python -m mujoco.viewer --mjcf my-phone-lab/scene.xml
```

只有普通 MuJoCo 也能显示这个完整场景；RGB-D 面板和 P 保存功能属于本包的
`phone-lab` 入口。标准 MuJoCo 查看器可以使用其相机列表切换到 `iphone_camera`。
默认公开示例使用 synthetic K；导出自己的标定时增加 `--profile my-iphone.json`。

## 只把 iPhone 模组导入自己的场景

```bash
iphone-mujoco export-phone --profile my-iphone.json --output iphone.xml
```

生成可复用的 `iphone.xml` 与 `iphone.calibration.json`。后者保存深度尺寸，
因为 MJCF 相机自身只包含 RGB 成像参数。在自己的 MJCF 中引用：

```xml
<mujoco>
  <worldbody>
    <body name="camera_mount" pos="0 -0.8 1.0" euler="1.3 0 0">
      <include file="iphone.xml"/>
    </body>
  </worldbody>
</mujoco>
```

`iphone.xml` 是模组 include，完整场景 `scene.xml` 才是可以单独打开的文件。
将 include 放入机器人末端 body 内就会跟随机器人；这种固定安装不使用 `--mocap`。
多相机使用不同名称，例如 `--body-name phone_left --camera-name camera_left`。

程序中直接调用：

```python
from iphone_mujoco import CameraCalibration, MujocoRGBDCamera, write_phone_module

k = CameraCalibration.load("my-iphone.json")
write_phone_module("iphone.xml", k)
# 将这个 include 加入自己的 MJCF，得到 model/data 后：
with MujocoRGBDCamera(model, k, camera="iphone_camera") as camera:
    frame = camera.render(data)
    rgb, depth_m = frame.rgb, frame.depth_m
    cloud = frame.point_cloud()
```

安装包的核心演示只依赖 MuJoCo、NumPy 和 Pillow。
连接真实 iPhone 时再安装 `[live]`，仍使用已有 PyPI `asmagic` SDK。
