"""
Memory Management Utilities

GPU memory pooling and efficient memory management.
"""

import torch
from typing import Dict, Optional, Any
from collections import deque


class GPUMemoryPool:
    """GPU memory pool for efficient tensor allocation."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize GPU memory pool.
        
        Args:
            device: PyTorch device (auto-detected if None)
        """
        if device is None:
            from backend.utils.gpu_detection import get_device
            device = get_device()
        
        self.device = device
        self.pools: Dict[str, deque] = {}
        self.max_pool_size = 10
    
    def get_tensor(
        self,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
        key: str = "default",
    ) -> torch.Tensor:
        """
        Get a tensor from pool or allocate new one.
        
        Args:
            shape: Tensor shape
            dtype: Tensor dtype
            key: Pool key for reuse
        
        Returns:
            Tensor
        """
        pool_key = f"{key}_{shape}_{dtype}"
        
        if pool_key not in self.pools:
            self.pools[pool_key] = deque()
        
        pool = self.pools[pool_key]
        
        # Try to reuse from pool
        while pool:
            tensor = pool.popleft()
            if tensor.shape == shape and tensor.dtype == dtype:
                tensor.zero_()  # Clear tensor
                return tensor
        
        # Allocate new tensor
        return torch.zeros(shape, dtype=dtype, device=self.device)
    
    def return_tensor(self, tensor: torch.Tensor, key: str = "default") -> None:
        """
        Return tensor to pool for reuse.
        
        Args:
            tensor: Tensor to return
            key: Pool key
        """
        if tensor.device != self.device:
            return  # Don't pool tensors on different devices
        
        pool_key = f"{key}_{tensor.shape}_{tensor.dtype}"
        
        if pool_key not in self.pools:
            self.pools[pool_key] = deque()
        
        pool = self.pools[pool_key]
        
        # Add to pool if not full
        if len(pool) < self.max_pool_size:
            pool.append(tensor.detach())
    
    def clear(self) -> None:
        """Clear all pools."""
        self.pools.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


class FrustumCuller:
    """Frustum culling for Gaussian visibility."""
    
    def __init__(self, camera_pose: Optional[Any] = None, intrinsics: Optional[Any] = None):
        """
        Initialize frustum culler.
        
        Args:
            camera_pose: Camera pose matrix (4, 4)
            intrinsics: Camera intrinsics matrix (3, 3)
        """
        self.camera_pose = camera_pose
        self.intrinsics = intrinsics
    
    def cull(
        self,
        points: torch.Tensor,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        near: float = 0.1,
        far: float = 100.0,
    ) -> torch.Tensor:
        """
        Cull points outside frustum.
        
        Args:
            points: 3D points (N, 3)
            camera_pose: Camera pose matrix (4, 4)
            intrinsics: Camera intrinsics matrix (3, 3)
            near: Near plane distance
            far: Far plane distance
        
        Returns:
            Boolean mask (N,) indicating visible points
        """
        # Transform to camera space
        points_hom = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=1)
        camera_pose_inv = torch.inverse(camera_pose)
        points_cam = (camera_pose_inv @ points_hom.T).T[:, :3]
        
        # Project to image space
        points_proj = (intrinsics @ points_cam.T).T
        depths = points_proj[:, 2]
        
        # Normalize to image coordinates
        uv = points_proj[:, :2] / (depths.unsqueeze(1) + 1e-8)
        
        # Check visibility
        width = intrinsics[0, 2] * 2
        height = intrinsics[1, 2] * 2
        
        visible = (
            (depths > near) &
            (depths < far) &
            (uv[:, 0] >= -width) &
            (uv[:, 0] < width) &
            (uv[:, 1] >= -height) &
            (uv[:, 1] < height)
        )
        
        return visible
    
    def cull_gaussians(
        self,
        gaussians: Any,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        near: float = 0.1,
        far: float = 100.0,
    ) -> Any:
        """
        Cull Gaussians outside frustum.
        
        Args:
            gaussians: Gaussian object
            camera_pose: Camera pose matrix (4, 4)
            intrinsics: Camera intrinsics matrix (3, 3)
            near: Near plane distance
            far: Far plane distance
        
        Returns:
            Culled Gaussian object
        """
        from backend.gaussian.fitter import Gaussian
        
        # Convert to torch
        centroids = torch.from_numpy(gaussians.centroids).float()
        if camera_pose.device != centroids.device:
            centroids = centroids.to(camera_pose.device)
        
        # Cull
        visible_mask = self.cull(centroids, camera_pose, intrinsics, near, far)
        
        # Convert mask to numpy
        visible_mask_np = visible_mask.cpu().numpy()
        
        # Filter Gaussians
        culled_gaussians = Gaussian(
            centroids=gaussians.centroids[visible_mask_np],
            covariances=None,
            colors=gaussians.colors[visible_mask_np],
            opacity=gaussians.opacity[visible_mask_np],
            scales=gaussians.scales[visible_mask_np],
            rotations=gaussians.rotations[visible_mask_np],
        )
        
        return culled_gaussians

