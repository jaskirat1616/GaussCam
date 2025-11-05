"""
Tests for Camera Utilities
"""

import pytest
import numpy as np
from backend.utils.camera import CameraIntrinsics, CameraPose


def test_camera_intrinsics():
    """Test camera intrinsics."""
    width, height = 640, 480
    intrinsics = CameraIntrinsics.default(width, height, fov=60.0)
    
    assert intrinsics.width == width
    assert intrinsics.height == height
    assert intrinsics.fx > 0
    assert intrinsics.fy > 0
    assert intrinsics.cx == width / 2.0
    assert intrinsics.cy == height / 2.0


def test_camera_intrinsics_matrix():
    """Test intrinsics matrix."""
    intrinsics = CameraIntrinsics.default(640, 480)
    K = intrinsics.get_matrix()
    
    assert K.shape == (3, 3)
    assert K[0, 0] == intrinsics.fx
    assert K[1, 1] == intrinsics.fy
    assert K[0, 2] == intrinsics.cx
    assert K[1, 2] == intrinsics.cy


def test_backproject():
    """Test depth backprojection."""
    intrinsics = CameraIntrinsics.default(640, 480)
    
    # Create test depth map
    depth = np.ones((480, 640), dtype=np.float32) * 5.0
    
    # Backproject
    points = intrinsics.backproject(depth)
    
    assert points.shape == (480, 640, 3)
    assert np.all(points[:, :, 2] == depth)  # Z should equal depth


def test_camera_pose():
    """Test camera pose."""
    pose = CameraPose.identity()
    T = pose.get_matrix()
    
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], np.zeros(3))

