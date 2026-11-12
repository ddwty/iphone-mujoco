"""iPhone RGB-D cameras, recordings and robot adapters for MuJoCo."""

__version__ = "0.6.0"

from .camera import CameraCalibration, MujocoRGBDCamera, RGBDFrame
from .profiles import DeviceProfile, phone_mjcf, configured_phone_mjcf, write_phone_module
from .devices import PhoneHardware, device_ids, get_device
from .configuration import CameraConfiguration, configure_camera, bind_runtime_calibration
from .wrist import WristCameraMount, WristRGBDRecorder, wrist_camera_mjcf, configured_wrist_mount
from .mobile import MobileRobotSpec, DifferentialDriveController, load_mobile_scene
from .experiments import SceneRGBDRecorder, load_camera_scene, advance_scene, collect_camera_episode

__all__ = ["CameraCalibration", "MujocoRGBDCamera", "RGBDFrame", "DeviceProfile", "phone_mjcf", "configured_phone_mjcf", "write_phone_module",
           "PhoneHardware", "device_ids", "get_device", "CameraConfiguration", "configure_camera", "bind_runtime_calibration",
           "WristCameraMount", "WristRGBDRecorder", "wrist_camera_mjcf", "configured_wrist_mount",
           "MobileRobotSpec", "DifferentialDriveController", "load_mobile_scene", "SceneRGBDRecorder",
           "load_camera_scene", "advance_scene", "collect_camera_episode", "__version__"]
