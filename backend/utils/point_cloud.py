"""
Point Cloud Utilities

Depth-to-point-cloud conversion with RGB fusion.
"""

import numpy as np
from typing import Tuple, Optional
from backend.utils.camera import CameraIntrinsics


def depth_to_point_cloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float = 1.0,
    max_depth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth map to colored point cloud.
    
    Args:
        depth: Depth map (H, W) in meters or normalized [0, 1]
        rgb: RGB image (H, W, 3) in [0, 1] or [0, 255]
        intrinsics: Camera intrinsics
        depth_scale: Scale factor for depth (if depth is normalized)
        max_depth: Maximum depth threshold (meters)
    
    Returns:
        (points, colors) where points is (N, 3) and colors is (N, 3)
    """
    # Normalize RGB to [0, 1]
    if rgb.max() > 1.0:
        rgb = rgb.astype(np.float32) / 255.0
    
    # Scale depth if needed
    if depth.max() <= 1.0:
        depth = depth * depth_scale
    
    # Apply depth threshold
    if max_depth is not None:
        depth = np.clip(depth, 0, max_depth)
    
    # Backproject to 3D
    points_3d = intrinsics.backproject(depth)
    
    # Flatten
    h, w = depth.shape
    points = points_3d.reshape(-1, 3)  # (H*W, 3)
    colors = rgb.reshape(-1, 3)  # (H*W, 3)
    
    # Filter invalid points (zero depth, NaN, Inf)
    valid_mask = (
        (depth.flatten() > 0) &
        np.isfinite(points).all(axis=1) &
        (points[:, 2] > 0)  # Positive depth
    )
    
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    return points, colors


def downsample_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud using voxel grid.
    
    Args:
        points: Point cloud (N, 3)
        colors: Colors (N, 3)
        voxel_size: Voxel size in meters
    
    Returns:
        (downsampled_points, downsampled_colors)
    """
    if len(points) == 0:
        return points, colors
    
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Get unique voxels
    unique_voxels, inverse_indices = np.unique(
        voxel_indices, axis=0, return_inverse=True
    )
    
    # Average points and colors within each voxel
    downsampled_points = np.zeros((len(unique_voxels), 3), dtype=np.float32)
    downsampled_colors = np.zeros((len(unique_voxels), 3), dtype=np.float32)
    
    for i in range(len(unique_voxels)):
        mask = inverse_indices == i
        downsampled_points[i] = points[mask].mean(axis=0)
        downsampled_colors[i] = colors[mask].mean(axis=0)
    
    return downsampled_points, downsampled_colors


def filter_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    remove_outliers: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter point cloud by depth and optionally remove outliers.
    
    Args:
        points: Point cloud (N, 3)
        colors: Colors (N, 3)
        min_depth: Minimum depth threshold
        max_depth: Maximum depth threshold
        remove_outliers: Remove statistical outliers
    
    Returns:
        (filtered_points, filtered_colors)
    """
    # Depth filtering
    depths = np.linalg.norm(points, axis=1)
    depth_mask = (depths >= min_depth) & (depths <= max_depth)
    
    points = points[depth_mask]
    colors = colors[depth_mask]
    
    if remove_outliers and len(points) > 0:
        # Statistical outlier removal
        from scipy.spatial.distance import cdist
        
        # Compute distances to k nearest neighbors
        k = min(10, len(points) - 1)
        if k > 0:
            distances = cdist(points, points)
            distances.sort(axis=1)
            k_distances = distances[:, 1:k+1]  # Exclude self
            mean_distances = k_distances.mean(axis=1)
            std_distances = k_distances.std(axis=1)
            
            # Remove points beyond 2 standard deviations
            threshold = mean_distances.mean() + 2 * std_distances.mean()
            outlier_mask = mean_distances < threshold
            points = points[outlier_mask]
            colors = colors[outlier_mask]
    
    return points, colors

