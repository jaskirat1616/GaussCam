"""
Novel View Utilities

Novel view interpolation and camera control.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R
from backend.utils.camera import CameraPose


class NovelViewController:
    """Controller for novel view rendering."""
    
    def __init__(
        self,
        base_pose: Optional[np.ndarray] = None,
        base_intrinsics: Optional[np.ndarray] = None,
    ):
        """
        Initialize novel view controller.
        
        Args:
            base_pose: Base camera pose matrix (4, 4)
            base_intrinsics: Base camera intrinsics matrix (3, 3)
        """
        if base_pose is None:
            self.base_pose = np.eye(4, dtype=np.float32)
        else:
            self.base_pose = base_pose.copy()
        
        if base_intrinsics is None:
            self.base_intrinsics = np.eye(3, dtype=np.float32)
        else:
            self.base_intrinsics = base_intrinsics.copy()
        
        # Novel view parameters
        self.rotation_x = 0.0  # Degrees
        self.rotation_y = 0.0  # Degrees
        self.rotation_z = 0.0  # Degrees
        self.translation_x = 0.0  # Meters
        self.translation_y = 0.0  # Meters
        self.translation_z = 0.0  # Meters
        self.scale = 1.0
    
    def set_rotation(self, x: float, y: float, z: float = 0.0) -> None:
        """
        Set rotation angles.
        
        Args:
            x: Rotation around X axis (degrees)
            y: Rotation around Y axis (degrees)
            z: Rotation around Z axis (degrees)
        """
        self.rotation_x = x
        self.rotation_y = y
        self.rotation_z = z
    
    def set_translation(self, x: float, y: float, z: float = 0.0) -> None:
        """
        Set translation.
        
        Args:
            x: Translation in X (meters)
            y: Translation in Y (meters)
            z: Translation in Z (meters)
        """
        self.translation_x = x
        self.translation_y = y
        self.translation_z = z
    
    def set_scale(self, scale: float) -> None:
        """
        Set scale factor.
        
        Args:
            scale: Scale factor
        """
        self.scale = scale
    
    def get_pose(self) -> np.ndarray:
        """
        Get current novel view pose.
        
        Returns:
            Camera pose matrix (4, 4)
        """
        # Create rotation matrix from Euler angles
        rotation = R.from_euler('xyz', [self.rotation_x, self.rotation_y, self.rotation_z], degrees=True)
        rotation_matrix = rotation.as_matrix()
        
        # Create translation
        translation = np.array([self.translation_x, self.translation_y, self.translation_z])
        
        # Build transformation matrix
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation * self.scale
        
        # Apply to base pose
        novel_pose = self.base_pose @ transform
        
        return novel_pose
    
    def get_intrinsics(self) -> np.ndarray:
        """
        Get current intrinsics (can be modified by zoom).
        
        Returns:
            Camera intrinsics matrix (3, 3)
        """
        # Scale intrinsics by scale factor (zoom)
        intrinsics = self.base_intrinsics.copy()
        intrinsics[:2, :2] *= self.scale
        return intrinsics
    
    def reset(self) -> None:
        """Reset to base pose."""
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.translation_z = 0.0
        self.scale = 1.0
    
    def interpolate_pose(
        self,
        pose1: np.ndarray,
        pose2: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Interpolate between two poses.
        
        Args:
            pose1: First pose matrix (4, 4)
            pose2: Second pose matrix (4, 4)
            t: Interpolation factor [0, 1]
        
        Returns:
            Interpolated pose matrix (4, 4)
        """
        # Extract rotation and translation
        R1 = pose1[:3, :3]
        t1 = pose1[:3, 3]
        R2 = pose2[:3, :3]
        t2 = pose2[:3, 3]
        
        # Interpolate rotation using SLERP
        R1_quat = R.from_matrix(R1).as_quat()
        R2_quat = R.from_matrix(R2).as_quat()
        R_interp = R.from_quat(R1_quat).slerp(R2_quat, t)
        R_interp_matrix = R_interp.as_matrix()
        
        # Interpolate translation
        t_interp = (1 - t) * t1 + t * t2
        
        # Build interpolated pose
        pose_interp = np.eye(4, dtype=np.float32)
        pose_interp[:3, :3] = R_interp_matrix
        pose_interp[:3, 3] = t_interp
        
        return pose_interp
    
    def smooth_pose(
        self,
        target_pose: np.ndarray,
        current_pose: np.ndarray,
        alpha: float = 0.1,
    ) -> np.ndarray:
        """
        Smoothly interpolate towards target pose.
        
        Args:
            target_pose: Target pose matrix (4, 4)
            current_pose: Current pose matrix (4, 4)
            alpha: Smoothing factor [0, 1]
        
        Returns:
            Smoothed pose matrix (4, 4)
        """
        return self.interpolate_pose(current_pose, target_pose, alpha)


class MouseController:
    """Mouse/trackpad controller for novel view."""
    
    def __init__(self, controller: NovelViewController):
        """
        Initialize mouse controller.
        
        Args:
            controller: Novel view controller
        """
        self.controller = controller
        self.last_pos = None
        self.is_dragging = False
        self.sensitivity = 0.5
    
    def on_mouse_press(self, x: float, y: float) -> None:
        """
        Handle mouse press.
        
        Args:
            x: Mouse X position
            y: Mouse Y position
        """
        self.last_pos = (x, y)
        self.is_dragging = True
    
    def on_mouse_release(self) -> None:
        """Handle mouse release."""
        self.is_dragging = False
        self.last_pos = None
    
    def on_mouse_move(self, x: float, y: float) -> None:
        """
        Handle mouse move.
        
        Args:
            x: Mouse X position
            y: Mouse Y position
        """
        if not self.is_dragging or self.last_pos is None:
            return
        
        dx = x - self.last_pos[0]
        dy = y - self.last_pos[1]
        
        # Update rotation
        current_x = self.controller.rotation_x
        current_y = self.controller.rotation_y
        
        self.controller.set_rotation(
            current_x + dy * self.sensitivity,
            current_y + dx * self.sensitivity,
        )
        
        self.last_pos = (x, y)
    
    def on_wheel(self, delta: float) -> None:
        """
        Handle mouse wheel (zoom).
        
        Args:
            delta: Wheel delta (positive = zoom in, negative = zoom out)
        """
        zoom_factor = 1.0 + delta * 0.1
        new_scale = self.controller.scale * zoom_factor
        new_scale = np.clip(new_scale, 0.1, 10.0)
        self.controller.set_scale(new_scale)

