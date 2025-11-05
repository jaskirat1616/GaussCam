"""
Camera Utilities

Camera intrinsics, extrinsics, and projection utilities.
"""

import numpy as np
from typing import Tuple, Optional


class CameraIntrinsics:
    """Camera intrinsic parameters."""
    
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ):
        """
        Initialize camera intrinsics.
        
        Args:
            fx: Focal length in x
            fy: Focal length in y
            cx: Principal point x
            cy: Principal point y
            width: Image width
            height: Image height
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
    
    @classmethod
    def from_fov(cls, fov_degrees: float, width: int, height: int) -> "CameraIntrinsics":
        """
        Create intrinsics from field of view.
        
        Args:
            fov_degrees: Field of view in degrees
            width: Image width
            height: Image height
        """
        fov_rad = np.radians(fov_degrees)
        fx = fy = width / (2.0 * np.tan(fov_rad / 2.0))
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx, fy, cx, cy, width, height)
    
    @classmethod
    def default(cls, width: int, height: int, fov: float = 60.0) -> "CameraIntrinsics":
        """Create default intrinsics (typical webcam FOV ~60 degrees)."""
        return cls.from_fov(fov, width, height)
    
    def get_matrix(self) -> np.ndarray:
        """Get intrinsic matrix K."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float32)
    
    def backproject(self, depth: np.ndarray) -> np.ndarray:
        """
        Backproject depth map to 3D point cloud.
        
        Args:
            depth: Depth map (H, W) in meters or normalized
        
        Returns:
            Point cloud (H, W, 3) in camera coordinates
        """
        h, w = depth.shape
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        u, v = np.meshgrid(u, v)
        
        # Convert to camera coordinates
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        
        points = np.stack([x, y, z], axis=-1)
        return points


class CameraPose:
    """Camera pose (extrinsics)."""
    
    def __init__(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
    ):
        """
        Initialize camera pose.
        
        Args:
            rotation: Rotation matrix (3, 3) or quaternion (4,)
            translation: Translation vector (3,)
        """
        if rotation.shape == (4,):
            # Quaternion to rotation matrix
            self.rotation = self._quaternion_to_matrix(rotation)
        else:
            self.rotation = rotation
        
        self.translation = translation
    
    def get_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T
    
    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
        ], dtype=np.float32)
        return R
    
    @classmethod
    def identity(cls) -> "CameraPose":
        """Create identity pose."""
        return cls(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))


def project_points(points: np.ndarray, intrinsics: CameraIntrinsics, pose: Optional[CameraPose] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points: 3D points (N, 3) in world or camera coordinates
        intrinsics: Camera intrinsics
        pose: Optional camera pose (if points are in world coordinates)
    
    Returns:
        (uv, mask) where uv is (N, 2) image coordinates and mask is valid projection mask
    """
    if pose is not None:
        # Transform to camera coordinates
        points_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
        T = pose.get_matrix()
        points_cam = (np.linalg.inv(T) @ points_hom.T).T[:, :3]
    else:
        points_cam = points
    
    # Project
    K = intrinsics.get_matrix()
    points_proj = (K @ points_cam.T).T
    
    # Normalize
    uv = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-8)
    
    # Check validity
    valid = (points_proj[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < intrinsics.width) & \
            (uv[:, 1] >= 0) & (uv[:, 1] < intrinsics.height)
    
    return uv, valid

