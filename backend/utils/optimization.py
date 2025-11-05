"""
Advanced Optimization Utilities

GPU-accelerated operations and performance optimizations.
"""

import numpy as np
import torch
from typing import Tuple, Optional
from backend.utils.gpu_detection import get_device, is_cuda, is_mps


def vectorized_downsample_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float = 0.01,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated voxel downsampling using PyTorch.
    
    Args:
        points: Point cloud (N, 3)
        colors: Colors (N, 3)
        voxel_size: Voxel size in meters
        device: PyTorch device (auto-detected if None)
    
    Returns:
        (downsampled_points, downsampled_colors)
    """
    if len(points) == 0:
        return points, colors
    
    if device is None:
        device = get_device()
    
    # Convert to tensors
    points_t = torch.from_numpy(points).float().to(device)
    colors_t = torch.from_numpy(colors).float().to(device)
    
    # Compute voxel indices
    voxel_indices = (points_t / voxel_size).floor().long()
    
    # Get unique voxels
    voxel_indices_np = voxel_indices.cpu().numpy()
    unique_voxels, inverse_indices = np.unique(
        voxel_indices_np, axis=0, return_inverse=True
    )
    
    # Use scatter_add for fast averaging (GPU-accelerated)
    num_voxels = len(unique_voxels)
    downsampled_points = torch.zeros((num_voxels, 3), device=device)
    downsampled_colors = torch.zeros((num_voxels, 3), device=device)
    counts = torch.zeros(num_voxels, device=device)
    
    inverse_indices_t = torch.from_numpy(inverse_indices).long().to(device)
    
    # Scatter add for accumulation
    if len(points_t) > 0:
        downsampled_points.index_add_(0, inverse_indices_t, points_t)
        downsampled_colors.index_add_(0, inverse_indices_t, colors_t)
        counts.index_add_(0, inverse_indices_t, torch.ones(len(points_t), device=device))
    
    # Average by dividing by counts
    counts = counts.clamp(min=1).unsqueeze(-1)
    downsampled_points = downsampled_points / counts
    downsampled_colors = downsampled_colors / counts
    
    # Convert back to numpy
    return (
        downsampled_points.cpu().numpy(),
        downsampled_colors.cpu().numpy(),
    )


def fast_depth_to_point_cloud_gpu(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics,
    depth_scale: float = 1.0,
    max_depth: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated depth to point cloud conversion.
    
    Args:
        depth: Depth map (H, W)
        rgb: RGB image (H, W, 3)
        intrinsics: Camera intrinsics
        depth_scale: Scale factor for depth
        max_depth: Maximum depth threshold
        device: PyTorch device
    
    Returns:
        (points, colors)
    """
    if device is None:
        device = get_device()
    
    h, w = depth.shape
    
    # Normalize RGB
    if rgb.max() > 1.0:
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)
    
    # Scale depth
    if depth.max() <= 1.0:
        depth = depth * depth_scale
    
    # Apply threshold
    if max_depth is not None:
        depth = np.clip(depth, 0, max_depth)
    
    # Convert to tensors
    depth_t = torch.from_numpy(depth).float().to(device)
    rgb_t = torch.from_numpy(rgb).float().to(device)
    
    # Create pixel grid
    u = torch.arange(w, device=device).float()
    v = torch.arange(h, device=device).float()
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
    
    # Backproject using intrinsics
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy
    
    x = (u_grid - cx) * depth_t / fx
    y = (v_grid - cy) * depth_t / fy
    z = depth_t
    
    # Stack into points
    points_3d = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    
    # Flatten and filter
    points = points_3d.reshape(-1, 3)
    colors = rgb_t.reshape(-1, 3)
    
    # Filter valid points (vectorized)
    valid_mask = (
        (depth_t.flatten() > 0) &
        torch.isfinite(points).all(dim=1) &
        (points[:, 2] > 0)
    )
    
    points_valid = points[valid_mask].cpu().numpy()
    colors_valid = colors[valid_mask].cpu().numpy()
    
    return points_valid, colors_valid


def adaptive_quality_settings(fps: float, target_fps: float = 15.0) -> dict:
    """
    Adaptively adjust quality settings based on current FPS.
    
    Args:
        fps: Current FPS
        target_fps: Target FPS
    
    Returns:
        Dictionary with optimized settings
    """
    ratio = fps / target_fps if target_fps > 0 else 1.0
    
    if ratio < 0.7:  # Too slow
        return {
            "depth_skip_frames": 20,
            "frame_skip": 10,
            "max_gaussians": 2000,
            "target_size": 256,
            "target_pixels": 15000,
            "voxel_size": 0.15,
        }
    elif ratio < 0.9:  # Slightly slow
        return {
            "depth_skip_frames": 15,
            "frame_skip": 8,
            "max_gaussians": 3000,
            "target_size": 320,
            "target_pixels": 20000,
            "voxel_size": 0.12,
        }
    elif ratio > 1.3:  # Fast enough, can increase quality
        return {
            "depth_skip_frames": 5,
            "frame_skip": 2,
            "max_gaussians": 8000,
            "target_size": 384,
            "target_pixels": 30000,
            "voxel_size": 0.08,
        }
    else:  # Balanced
        return {
            "depth_skip_frames": 10,
            "frame_skip": 5,
            "max_gaussians": 5000,
            "target_size": 320,
            "target_pixels": 20000,
            "voxel_size": 0.10,
        }


def batch_process_gaussians(
    gaussians_list: list,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Batch process multiple Gaussian sets for efficient GPU usage.
    
    Args:
        gaussians_list: List of Gaussian objects
        batch_size: Batch size for processing
        device: PyTorch device
    
    Returns:
        Batched Gaussian tensor
    """
    if device is None:
        device = get_device()
    
    # TODO: Implement batch processing
    # This would allow processing multiple frames simultaneously
    pass

