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
from backend.gaussian.four_d import Gaussian4D
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
        self._offset_cache: dict[int, torch.Tensor] = {}
        self._max_kernel_radius: int = 32
        
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
        **kwargs,
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
            uv, depths, scales, colors, opacities, self.width, self.height
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

    def _get_offset_grid(self, radius: int) -> torch.Tensor:
        """Get cached integer offset grid for a given radius."""
        radius = int(max(1, min(radius, self._max_kernel_radius)))
        cached = self._offset_cache.get(radius)
        if cached is not None:
            return cached

        coords = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.int64)
        u_offsets, v_offsets = torch.meshgrid(coords, coords, indexing='ij')
        offsets = torch.stack((u_offsets.reshape(-1), v_offsets.reshape(-1)), dim=1)
        self._offset_cache[radius] = offsets
        return offsets
    
    def _render_gaussians(
        self,
        uv: torch.Tensor,
        depths: torch.Tensor,
        scales: torch.Tensor,
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
        sorted_uv = uv[sorted_indices]
        sorted_depths = depths[sorted_indices]
        sorted_scales = scales[sorted_indices]
        sorted_colors = colors[sorted_indices]
        sorted_opacities = opacities[sorted_indices]

        valid_mask = (
            (sorted_uv[:, 0] >= -width) & (sorted_uv[:, 0] < 2 * width) &
            (sorted_uv[:, 1] >= -height) & (sorted_uv[:, 1] < 2 * height) &
            (sorted_depths > 0)
        )

        valid_indices_tensor = torch.where(valid_mask)[0]
        if valid_indices_tensor.numel() == 0:
            return image

        num_valid = valid_indices_tensor.numel()
        # Use fast point rendering for large counts or if explicitly requested
        fast_threshold = 8000  # Lower threshold for faster rendering
        if num_valid > fast_threshold:
            max_render = min(num_valid, 8000)
            return self._render_points_fast(
                sorted_uv, sorted_depths, sorted_colors, sorted_opacities,
                width, height, max_render
            )

        alpha_flat = alpha_buffer.view(-1)
        image_flat = image.view(-1, 3)
        max_render = min(num_valid, 6000)  # Lower max for faster rendering

        for idx_tensor in valid_indices_tensor[:max_render]:
            idx = idx_tensor.item()

            depth_val = torch.clamp(sorted_depths[idx], min=1e-4)
            scale = sorted_scales[idx]
            color = sorted_colors[idx]
            opacity = sorted_opacities[idx]

            radius_unclamped = torch.max(scale) * width / depth_val
            radius_float = torch.clamp(radius_unclamped, min=1.0, max=float(max(width, height)))
            radius_px = int(torch.round(torch.clamp(radius_float, max=float(self._max_kernel_radius))).item())
            radius_px = max(1, min(radius_px, self._max_kernel_radius))

            offsets = self._get_offset_grid(radius_px)

            u_center = sorted_uv[idx, 0]
            v_center = sorted_uv[idx, 1]

            base_u = torch.round(u_center).to(torch.int64)
            base_v = torch.round(v_center).to(torch.int64)

            pixel_u = base_u + offsets[:, 0]
            pixel_v = base_v + offsets[:, 1]

            valid_pixel_mask = (
                (pixel_u >= 0) & (pixel_u < width) &
                (pixel_v >= 0) & (pixel_v < height)
            )

            if not torch.any(valid_pixel_mask):
                continue

            pixel_u = pixel_u[valid_pixel_mask]
            pixel_v = pixel_v[valid_pixel_mask]
            pixel_indices = (pixel_v * width + pixel_u).to(torch.int64)

            pixel_center_u = pixel_u.to(torch.float32) + 0.5
            pixel_center_v = pixel_v.to(torch.float32) + 0.5

            du = pixel_center_u - u_center
            dv = pixel_center_v - v_center

            radius_sq = radius_float * radius_float + 1e-6
            dist_sq = (du * du + dv * dv) / radius_sq
            weights = torch.exp(-0.5 * dist_sq)
            weights = torch.clamp(weights * opacity, min=0.0, max=1.0)

            if weights.numel() == 0:
                continue

            current_alpha = alpha_flat.index_select(0, pixel_indices)
            alpha_contrib = weights * (1.0 - current_alpha)

            if torch.all(alpha_contrib <= 1e-7):
                continue

            alpha_flat.index_add_(0, pixel_indices, alpha_contrib)

            color_contrib = alpha_contrib.unsqueeze(-1) * color.unsqueeze(0)
            image_flat.index_add_(0, pixel_indices, color_contrib)

        image = image_flat.view(height, width, 3)
        alpha_buffer = alpha_flat.view(height, width)

        alpha_mask = alpha_buffer > 1e-8
        if torch.any(alpha_mask):
            image[alpha_mask] /= alpha_buffer[alpha_mask].unsqueeze(-1)

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
    
    def render_dynamic(
        self,
        gaussians: Gaussian4D,
        time_offset: float = 0.0,
        camera_pose: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        # Apply simple Euler integration for centroid motion as placeholder.
        motion = gaussians.motion.translation * time_offset
        animated = Gaussian(
            centroids=gaussians.gaussian.centroids + motion,
            covariances=None,
            colors=gaussians.gaussian.colors,
            opacity=gaussians.gaussian.opacity,
            scales=gaussians.gaussian.scales,
            rotations=gaussians.gaussian.rotations,
        )
        return self.render(animated, camera_pose, intrinsics, **kwargs)

    def resize(self, width: int, height: int) -> None:
        """Resize render buffer."""
        self.width = width
        self.height = height
    
    def clear(self) -> None:
        """Clear render buffer."""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @property
    def capabilities(self):
        caps = super().capabilities.copy()
        caps.update({"dynamic_gaussians": True})
        return caps

