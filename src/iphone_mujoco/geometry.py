"""Rigid geometry with explicit iPhone, optical-camera and simulation bases.

All distances are meters. NumPy transforms multiply column vectors; their last
column is translation. CV cameras look along +Z with +X right and +Y down.
ARKit and MuJoCo camera-local axes look along -Z with +Y up.
"""
from __future__ import annotations

import numpy as np


CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])
CV_TO_MUJOCO_CAMERA = CV_TO_ARKIT.copy()
# Match CoordinateFrameTransform.roboticsRotation in the asMagic app:
# ARKit world -> robotics world (+X forward, +Y left, +Z up).
ARKIT_WORLD_TO_MUJOCO = np.array(
    [[0., 0., -1., 0.], [-1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]]
)


def validate_transform(value: np.ndarray) -> np.ndarray:
    """Return a finite SE(3) transform, rejecting scale and reflections."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Transform must be a finite 4 x 4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Transform must have homogeneous last row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5):
        raise ValueError("Transform rotation must be orthonormal and right handed")
    return matrix


def pose_from_xyzw(values: np.ndarray) -> np.ndarray:
    """Decode the current asMagic [tx,ty,tz,qx,qy,qz,qw] wire pose."""
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Pose must contain seven finite values [tx,ty,tz,qx,qy,qz,qw]")
    norm = np.linalg.norm(pose[3:])
    if norm < 1e-10:
        raise ValueError("Pose quaternion cannot be zero")
    x, y, z, w = pose[3:] / norm
    result = np.eye(4)
    result[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    result[:3, 3] = pose[:3]
    return result


def convert_iphone_pose(
    pose: np.ndarray, pose_convention: str, *, world_from_sender: np.ndarray | None = None
) -> np.ndarray:
    """Convert sender-world <- ARKit-camera to simulation-world <- CV-camera.

    ``arkit`` means asMagic's unmodified World coordinate mode; ``robotics``
    (or ``mujoco``) means a Z-up sender world. ``custom`` requires the measured
    simulation-world <- sender-world registration. App rotations and offsets
    are not on the wire and cannot be inferred. A supplied registration replaces
    the preset world conversion, including its origin.
    """
    pose = validate_transform(pose)
    if pose_convention not in {"arkit", "robotics", "mujoco", "custom"}:
        raise ValueError("pose_convention must be arkit, robotics, mujoco or custom")
    if world_from_sender is not None:
        basis = validate_transform(world_from_sender)
    elif pose_convention == "arkit":
        basis = ARKIT_WORLD_TO_MUJOCO
    elif pose_convention == "custom":
        raise ValueError("custom pose convention requires world_from_sender")
    else:
        basis = np.eye(4)
    return basis @ pose @ CV_TO_ARKIT


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform N x 3 points without changing their order."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    transform = validate_transform(transform)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(points: np.ndarray, calibration) -> np.ndarray:
    """Project CV-camera points to top-left integer-center pixel coordinates.

    Points behind the camera yield NaN. Coordinates are not clipped to the
    image, allowing a caller to distinguish off-image projections.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    pixels = np.full((len(points), 2), np.nan)
    pixels[valid, 0] = calibration.fx * points[valid, 0] / points[valid, 2] + calibration.cx
    pixels[valid, 1] = calibration.fy * points[valid, 1] / points[valid, 2] + calibration.cy
    return pixels
