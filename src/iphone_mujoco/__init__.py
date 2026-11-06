"""iPhone RGB-D cameras, recordings and robot adapters for MuJoCo."""

__version__ = "0.4.1"

from .camera import CameraCalibration, MujocoRGBDCamera, RGBDFrame
from .profiles import DeviceProfile, phone_mjcf, write_phone_module
from .wrist import WristCameraMount, WristRGBDRecorder, wrist_camera_mjcf

__all__ = ["CameraCalibration", "MujocoRGBDCamera", "RGBDFrame", "DeviceProfile", "phone_mjcf", "write_phone_module",
           "WristCameraMount", "WristRGBDRecorder", "wrist_camera_mjcf", "__version__"]
