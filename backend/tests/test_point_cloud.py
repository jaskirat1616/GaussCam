"""
Tests for Point Cloud Utilities
"""

import pytest
import numpy as np
from backend.utils.point_cloud import depth_to_point_cloud, downsample_point_cloud, filter_point_cloud
from backend.utils.camera import CameraIntrinsics


def test_depth_to_point_cloud():
    """Test depth to point cloud conversion."""
    intrinsics = CameraIntrinsics.default(640, 480)
    
    # Create test depth and RGB
    depth = np.ones((480, 640), dtype=np.float32) * 5.0
    rgb = np.random.rand(480, 640, 3).astype(np.float32)
    
    # Convert to point cloud
    points, colors = depth_to_point_cloud(depth, rgb, intrinsics, depth_scale=1.0)
    
    assert len(points) > 0
    assert len(colors) > 0
    assert points.shape[1] == 3
    assert colors.shape[1] == 3
    assert len(points) == len(colors)


def test_downsample_point_cloud():
    """Test point cloud downsampling."""
    num_points = 1000
    points = np.random.randn(num_points, 3).astype(np.float32) * 0.5
    colors = np.random.rand(num_points, 3).astype(np.float32)
    
    # Downsample
    points_ds, colors_ds = downsample_point_cloud(points, colors, voxel_size=0.1)
    
    assert len(points_ds) <= len(points)
    assert len(points_ds) == len(colors_ds)


def test_filter_point_cloud():
    """Test point cloud filtering."""
    num_points = 1000
    points = np.random.randn(num_points, 3).astype(np.float32) * 0.5
    colors = np.random.rand(num_points, 3).astype(np.float32)
    
    # Filter
    points_filt, colors_filt = filter_point_cloud(
        points, colors, min_depth=0.1, max_depth=10.0, remove_outliers=False
    )
    
    assert len(points_filt) <= len(points)
    assert len(points_filt) == len(colors_filt)

