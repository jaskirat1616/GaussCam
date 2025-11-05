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
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth map to colored point cloud (GPU-accelerated if available).
    
    Args:
        depth: Depth map (H, W) in meters or normalized [0, 1]
        rgb: RGB image (H, W, 3) in [0, 1] or [0, 255]
        intrinsics: Camera intrinsics
        depth_scale: Scale factor for depth (if depth is normalized)
        max_depth: Maximum depth threshold (meters)
        use_gpu: Use GPU acceleration if available
    
    Returns:
        (points, colors) where points is (N, 3) and colors is (N, 3)
    """
    # Try GPU-accelerated version if available
    if use_gpu:
        try:
            from backend.utils.optimization import fast_depth_to_point_cloud_gpu
            from backend.utils.gpu_detection import is_cuda, is_mps
            
            if is_cuda() or is_mps():
                return fast_depth_to_point_cloud_gpu(
                    depth, rgb, intrinsics, depth_scale, max_depth
                )
        except:
            pass  # Fall back to CPU version
    
    # CPU version (optimized)
    # Normalize RGB to [0, 1]
    if rgb.max() > 1.0:
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)
    
    # Scale depth if needed
    if depth.max() <= 1.0:
        depth = depth * depth_scale
    
    # Apply depth threshold
    if max_depth is not None:
        depth = np.clip(depth, 0, max_depth)
    
    # Backproject to 3D (vectorized)
    points_3d = intrinsics.backproject(depth)
    
    # Flatten
    h, w = depth.shape
    points = points_3d.reshape(-1, 3)  # (H*W, 3)
    colors = rgb.reshape(-1, 3)  # (H*W, 3)
    
    # Filter invalid points (vectorized)
    depth_flat = depth.flatten()
    valid_mask = (
        (depth_flat > 0) &
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
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud using voxel grid (GPU-accelerated if available).
    
    Args:
        points: Point cloud (N, 3)
        colors: Colors (N, 3)
        voxel_size: Voxel size in meters
        use_gpu: Use GPU acceleration if available
    
    Returns:
        (downsampled_points, downsampled_colors)
    """
    if len(points) == 0:
        return points, colors
    
    # Try GPU-accelerated version if available
    if use_gpu:
        try:
            from backend.utils.optimization import vectorized_downsample_point_cloud
            from backend.utils.gpu_detection import get_device, is_cuda, is_mps
            
            if is_cuda() or is_mps():
                return vectorized_downsample_point_cloud(
                    points, colors, voxel_size, device=get_device()
                )
        except:
            pass  # Fall back to CPU version
    
    # CPU version (optimized with vectorization)
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Get unique voxels
    unique_voxels, inverse_indices = np.unique(
        voxel_indices, axis=0, return_inverse=True
    )
    
    # Vectorized averaging using numpy indexing
    num_voxels = len(unique_voxels)
    downsampled_points = np.zeros((num_voxels, 3), dtype=np.float32)
    downsampled_colors = np.zeros((num_voxels, 3), dtype=np.float32)
    counts = np.zeros(num_voxels, dtype=np.int32)
    
    # Use bincount for fast accumulation
    for i in range(3):
        downsampled_points[:, i] = np.bincount(
            inverse_indices, weights=points[:, i], minlength=num_voxels
        )
        downsampled_colors[:, i] = np.bincount(
            inverse_indices, weights=colors[:, i], minlength=num_voxels
        )
    
    counts = np.bincount(inverse_indices, minlength=num_voxels)
    counts = np.maximum(counts, 1)  # Avoid division by zero
    
    # Average by dividing by counts
    downsampled_points /= counts[:, np.newaxis]
    downsampled_colors /= counts[:, np.newaxis]
    
    return downsampled_points, downsampled_colors


def filter_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    remove_outliers: bool = True,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter point cloud by depth and optionally remove outliers.
    
    Args:
        points: Point cloud (N, 3)
        colors: Colors (N, 3)
        min_depth: Minimum depth threshold
        max_depth: Maximum depth threshold
        remove_outliers: Remove statistical outliers
        use_gpu: Use GPU acceleration if available
    
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
        try:
            if use_gpu:
                # Try GPU-accelerated outlier removal
                try:
                    import torch
                    from backend.utils.gpu_detection import get_device, is_cuda, is_mps
                    
                    if is_cuda() or is_mps():
                        device = get_device()
                        points_t = torch.from_numpy(points).float().to(device)
                        
                        # Compute distances to k nearest neighbors (GPU)
                        k = min(10, len(points) - 1)
                        if k > 0:
                            # Use efficient distance computation
                            # Compute pairwise distances (only for small point clouds)
                            if len(points) < 10000:  # Only for smaller point clouds
                                distances = torch.cdist(points_t, points_t)
                                distances = torch.sort(distances, dim=1)[0]
                                k_distances = distances[:, 1:k+1]
                                mean_distances = k_distances.mean(dim=1)
                                std_distances = k_distances.std(dim=1)
                                
                                # Remove points beyond 2 standard deviations
                                threshold = mean_distances.mean() + 2 * std_distances.mean()
                                outlier_mask = (mean_distances < threshold).cpu().numpy()
                                points = points[outlier_mask]
                                colors = colors[outlier_mask]
                            else:
                                # For large point clouds, use simpler filtering
                                # Filter by depth consistency
                                z_values = points[:, 2]
                                z_mean = np.mean(z_values)
                                z_std = np.std(z_values)
                                outlier_mask = np.abs(z_values - z_mean) < 2 * z_std
                                points = points[outlier_mask]
                                colors = colors[outlier_mask]
                except:
                    # Fall back to CPU version
                    pass
            
            # CPU version (fallback or if GPU not available)
            if len(points) > 0:
                from scipy.spatial.distance import cdist
                
                # For large point clouds, use sampling for efficiency
                if len(points) > 10000:
                    # Sample points for distance computation
                    sample_size = min(5000, len(points))
                    sample_indices = np.random.choice(len(points), sample_size, replace=False)
                    sample_points = points[sample_indices]
                    
                    # Compute distances from all points to sample
                    distances = cdist(points, sample_points)
                    min_distances = distances.min(axis=1)
                    
                    # Remove points with very large minimum distances
                    threshold = np.percentile(min_distances, 95)
                    outlier_mask = min_distances < threshold
                    points = points[outlier_mask]
                    colors = colors[outlier_mask]
                else:
                    # For smaller point clouds, use full distance matrix
                    k = min(10, len(points) - 1)
                    if k > 0:
                        distances = cdist(points, points)
                        distances.sort(axis=1)
                        k_distances = distances[:, 1:k+1]
                        mean_distances = k_distances.mean(axis=1)
                        std_distances = k_distances.std(axis=1)
                        
                        # Remove points beyond 2 standard deviations
                        threshold = mean_distances.mean() + 2 * std_distances.mean()
                        outlier_mask = mean_distances < threshold
                        points = points[outlier_mask]
                        colors = colors[outlier_mask]
        except Exception as e:
            # If outlier removal fails, continue without it
            import warnings
            warnings.warn(f"Outlier removal failed: {e}")
    
    return points, colors

