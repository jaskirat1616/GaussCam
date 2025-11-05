"""
MPS Renderer

Gaussian Splatting renderer using PyTorch MPS operations for Apple Silicon.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from backend.renderer.base import Renderer
from backend.gaussian.fitter import Gaussian
from backend.utils.gpu_detection import get_device, is_mps


class MPSRenderer(Renderer):
    """MPS-based Gaussian Splatting renderer using PyTorch."""
    
    def __init__(self, width: int = 640, height: int = 480, device: Optional[torch.device] = None):
        """
        Initialize MPS renderer.
        
        Args:
            width: Render width
            height: Render height
            device: PyTorch device (auto-detected if None)
        """
        super().__init__(width, height)
        
        if device is None:
            device = get_device()
        
        if not is_mps():
            raise RuntimeError("MPS not available. Cannot use MPSRenderer.")
        
        self.device = device
        # Defer background color tensor creation to avoid Metal conflicts during init
        self._background_color = None
        
        print(f"MPSRenderer initialized: {width}x{height} on {device}")
    
    @property
    def background_color(self) -> torch.Tensor:
        """Lazy initialization of background color tensor."""
        if self._background_color is None:
            self._background_color = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        return self._background_color
    
    def render(
        self,
        gaussians: Gaussian,
        camera_pose: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render Gaussians using PyTorch MPS operations.
        
        Args:
            gaussians: Gaussian object
            camera_pose: Camera pose matrix (4, 4) or None for identity
            intrinsics: Camera intrinsics matrix (3, 3) or None for default
        
        Returns:
            Rendered image (H, W, 3) RGB in [0, 1]
        """
        # Convert Gaussians to torch tensors
        gaussian_dict = gaussians.to_torch(self.device)
        
        # Get camera parameters
        if camera_pose is None:
            camera_pose = np.eye(4, dtype=np.float32)
        if intrinsics is None:
            intrinsics = np.array([
                [self.width, 0, self.width / 2],
                [0, self.width, self.height / 2],
                [0, 0, 1],
            ], dtype=np.float32)
        
        # Convert to torch
        camera_pose_t = torch.from_numpy(camera_pose).float().to(self.device)
        intrinsics_t = torch.from_numpy(intrinsics).float().to(self.device)
        
        # Extract Gaussian parameters
        means = gaussian_dict["centroids"]  # (N, 3)
        scales = gaussian_dict["scales"]  # (N, 3)
        rotations = gaussian_dict["rotations"]  # (N, 4) quaternions
        colors = gaussian_dict["colors"]  # (N, 3)
        opacities = gaussian_dict["opacity"]  # (N,)
        
        # Transform to camera space
        means_cam = self._transform_to_camera_space(means, camera_pose_t)
        
        # Project to image space
        uv, depths = self._project_points(means_cam, intrinsics_t)
        
        # Render using alpha blending
        image = self._render_gaussians(
            uv, depths, scales, rotations, colors, opacities, self.width, self.height
        )
        
        # Convert to numpy
        image_np = image.cpu().numpy()
        image_np = np.clip(image_np, 0.0, 1.0)
        
        return image_np
    
    def _transform_to_camera_space(
        self,
        points: torch.Tensor,
        camera_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Transform points from world to camera space."""
        # Convert to homogeneous coordinates
        points_hom = torch.cat([points, torch.ones((points.shape[0], 1), device=self.device)], dim=1)
        
        # Inverse transform (camera pose)
        camera_pose_inv = torch.inverse(camera_pose)
        
        # Transform
        points_cam = (camera_pose_inv @ points_hom.T).T[:, :3]
        
        return points_cam
    
    def _project_points(
        self,
        points_cam: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3D points to 2D image coordinates."""
        # Project
        points_proj = (intrinsics @ points_cam.T).T
        
        # Extract depths
        depths = points_proj[:, 2]
        
        # Normalize to image coordinates
        uv = points_proj[:, :2] / (depths.unsqueeze(1) + 1e-8)
        
        return uv, depths
    
    def _render_gaussians(
        self,
        uv: torch.Tensor,
        depths: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Render Gaussians using alpha blending with 2D Gaussian projection."""
        # Initialize render buffer
        image = torch.zeros((height, width, 3), device=self.device)
        alpha_buffer = torch.zeros((height, width), device=self.device)
        
        # Sort by depth (back to front)
        sorted_indices = torch.argsort(depths, descending=True)
        
        # Render each Gaussian
        sorted_uv = uv[sorted_indices]
        sorted_depths = depths[sorted_indices]
        sorted_scales = scales[sorted_indices]
        sorted_rotations = rotations[sorted_indices]
        sorted_colors = colors[sorted_indices]
        sorted_opacities = opacities[sorted_indices]
        
        # Create valid mask (wider bounds for 2D Gaussian rendering)
        valid_mask = (
            (sorted_uv[:, 0] >= -width) & (sorted_uv[:, 0] < 2 * width) &
            (sorted_uv[:, 1] >= -height) & (sorted_uv[:, 1] < 2 * height) &
            (sorted_depths > 0)
        )
        
        # Apply valid mask
        valid_indices_tensor = torch.where(valid_mask)[0]
        
        if valid_indices_tensor.numel() == 0:
            return image
        
        num_valid = valid_indices_tensor.numel()
        
        # If too many Gaussians, use simpler point-based rendering for speed
        if num_valid > 5000:  # Lower threshold for faster rendering
            # Fallback to faster point-based rendering
            max_render = min(num_valid, 10000)  # Limit to 10K for speed
            return self._render_points_fast(
                sorted_uv, sorted_depths, sorted_colors, sorted_opacities, 
                width, height, max_render
            )
        
        # Limit rendering for 2D Gaussian projection (slower but higher quality)
        max_render = min(num_valid, 5000)
        
        # Render Gaussians with 2D projection (for smaller counts)
        for i in range(min(max_render, num_valid)):
            idx = valid_indices_tensor[i].item()
            
            u_center = sorted_uv[idx, 0].item()
            v_center = sorted_uv[idx, 1].item()
            depth = sorted_depths[idx].item()
            scale = sorted_scales[idx]
            rotation = sorted_rotations[idx]
            color = sorted_colors[idx]
            opacity = sorted_opacities[idx].item()
            
            # Project 3D Gaussian to 2D (simplified - use scale as 2D radius)
            # Compute 2D Gaussian radius from 3D scale and depth
            radius_2d = max(scale[0].item(), scale[1].item(), scale[2].item()) * width / depth
            radius_2d = min(radius_2d, width / 2.0)  # Limit radius
            
            # Compute bounding box
            u_min = max(0, int(u_center - radius_2d))
            u_max = min(width, int(u_center + radius_2d) + 1)
            v_min = max(0, int(v_center - radius_2d))
            v_max = min(height, int(v_center + radius_2d) + 1)
            
            if u_max <= u_min or v_max <= v_min:
                continue
            
            # Render 2D Gaussian ellipse in bounding box
            u_grid = torch.arange(u_min, u_max, device=self.device).float() + 0.5
            v_grid = torch.arange(v_min, v_max, device=self.device).float() + 0.5
            
            # Create meshgrid - shape will be [u_size, v_size]
            u_mesh, v_mesh = torch.meshgrid(u_grid, v_grid, indexing='ij')
            
            # Distance from center
            du = u_mesh - u_center
            dv = v_mesh - v_center
            dist_sq = (du * du + dv * dv) / (radius_2d * radius_2d + 1e-8)
            
            # Gaussian weight (2D Gaussian) - shape [u_size, v_size]
            weight = torch.exp(-dist_sq * 0.5)
            weight = weight * opacity
            
            # Get slice shapes
            u_size = u_max - u_min
            v_size = v_max - v_min
            
            # meshgrid with indexing='ij' gives [u_size, v_size], but we need [v_size, u_size] for image indexing
            # Transpose to match image buffer which is indexed as [height, width] = [v, u]
            weight = weight.T  # Now shape [v_size, u_size]
            
            # Get current alpha values for this region - shape [v_size, u_size]
            alpha_region = alpha_buffer[v_min:v_max, u_min:u_max]
            
            # Compute new alpha contribution - shape [v_size, u_size]
            alpha_new = weight * (1.0 - alpha_region)
            
            # Apply to alpha buffer and image
            alpha_buffer[v_min:v_max, u_min:u_max] += alpha_new
            image[v_min:v_max, u_min:u_max] += color.unsqueeze(0).unsqueeze(0) * alpha_new.unsqueeze(-1)
        
        # Normalize by alpha
        alpha_mask = alpha_buffer > 1e-8
        image[alpha_mask] /= alpha_buffer[alpha_mask].unsqueeze(-1)
        
        # Blend with background
        image = image + self.background_color.unsqueeze(0).unsqueeze(0) * (1.0 - alpha_buffer.unsqueeze(-1))
        
        return image
    
    def _render_points_fast(
        self,
        uv: torch.Tensor,
        depths: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        width: int,
        height: int,
        max_points: int,
    ) -> torch.Tensor:
        """Fast point-based rendering for large Gaussian counts."""
        # Initialize render buffer
        image = torch.zeros((height, width, 3), device=self.device)
        alpha_buffer = torch.zeros((height, width), device=self.device)
        
        # Sort by depth (back to front)
        sorted_indices = torch.argsort(depths, descending=True)
        
        # Limit to max_points
        sorted_indices = sorted_indices[:max_points]
        
        # Get sorted values
        sorted_uv = uv[sorted_indices]
        sorted_depths = depths[sorted_indices]
        sorted_colors = colors[sorted_indices]
        sorted_opacities = opacities[sorted_indices]
        
        # Create valid mask
        valid_mask = (
            (sorted_uv[:, 0] >= 0) & (sorted_uv[:, 0] < width) &
            (sorted_uv[:, 1] >= 0) & (sorted_uv[:, 1] < height) &
            (sorted_depths > 0)
        )
        
        # Apply valid mask
        valid_indices = torch.where(valid_mask)[0]
        
        if valid_indices.numel() == 0:
            # Blend with background
            image = image + self.background_color.unsqueeze(0).unsqueeze(0) * (1.0 - alpha_buffer.unsqueeze(-1))
            return image
        
        # Get valid coordinates
        u_coords = sorted_uv[valid_indices, 0].clamp(0, width - 1).long()
        v_coords = sorted_uv[valid_indices, 1].clamp(0, height - 1).long()
        valid_colors = sorted_colors[valid_indices]
        valid_opacities = sorted_opacities[valid_indices]
        
        # Limit to max_points for performance
        num_to_render = min(valid_indices.numel(), max_points)
        
        # Use vectorized scatter operations for faster rendering
        if num_to_render > 0:
            u_coords_limited = u_coords[:num_to_render]
            v_coords_limited = v_coords[:num_to_render]
            colors_limited = valid_colors[:num_to_render]
            opacities_limited = valid_opacities[:num_to_render]
            
            # Get current alpha values for these pixels
            pixel_indices = v_coords_limited * width + u_coords_limited
            current_alpha = alpha_buffer.flatten()[pixel_indices]
            
            # Compute new alpha contribution
            new_alpha = opacities_limited * (1.0 - current_alpha)
            
            # Update alpha buffer (vectorized)
            alpha_buffer.flatten()[pixel_indices] += new_alpha
            
            # Update image (vectorized)
            image_flat = image.view(-1, 3)
            for c in range(3):
                image_flat[pixel_indices, c] += colors_limited[:, c] * new_alpha
        
        # Normalize by alpha
        alpha_mask = alpha_buffer > 1e-8
        image[alpha_mask] /= alpha_buffer[alpha_mask].unsqueeze(-1)
        
        # Blend with background
        image = image + self.background_color.unsqueeze(0).unsqueeze(0) * (1.0 - alpha_buffer.unsqueeze(-1))
        
        return image
    
    def resize(self, width: int, height: int) -> None:
        """Resize render buffer."""
        self.width = width
        self.height = height
    
    def clear(self) -> None:
        """Clear render buffer."""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

