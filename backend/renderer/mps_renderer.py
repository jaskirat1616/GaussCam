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
        """Render Gaussians using alpha blending (simplified)."""
        # Initialize render buffer
        image = torch.zeros((height, width, 3), device=self.device)
        alpha_buffer = torch.zeros((height, width), device=self.device)
        
        # Sort by depth (back to front)
        sorted_indices = torch.argsort(depths, descending=True)
        
        # Render each Gaussian
        # Use vectorized operations for better performance
        sorted_uv = uv[sorted_indices]
        sorted_depths = depths[sorted_indices]
        sorted_colors = colors[sorted_indices]
        sorted_opacities = opacities[sorted_indices]
        
        # Clamp UV coordinates to image bounds
        u_coords = sorted_uv[:, 0].clamp(0, width - 1).long()
        v_coords = sorted_uv[:, 1].clamp(0, height - 1).long()
        
        # Create valid mask
        valid_mask = (
            (sorted_uv[:, 0] >= 0) & (sorted_uv[:, 0] < width) &
            (sorted_uv[:, 1] >= 0) & (sorted_uv[:, 1] < height) &
            (sorted_depths > 0)
        )
        
        # Apply valid mask and render
        valid_indices_tensor = torch.where(valid_mask)[0]
        
        if valid_indices_tensor.numel() > 0:
            u_valid = u_coords[valid_indices_tensor]
            v_valid = v_coords[valid_indices_tensor]
            colors_valid = sorted_colors[valid_indices_tensor]
            opacities_valid = sorted_opacities[valid_indices_tensor]
            
            # Render valid Gaussians (limit to first N for speed)
            num_valid = valid_indices_tensor.numel()
            max_render = min(num_valid, 50000)  # Limit rendering for speed
            
            # Use vectorized operations where possible
            u_valid_limited = u_valid[:max_render]
            v_valid_limited = v_valid[:max_render]
            colors_valid_limited = colors_valid[:max_render]
            opacities_valid_limited = opacities_valid[:max_render]
            
            for i in range(max_render):
                u_int = u_valid_limited[i].item()
                v_int = v_valid_limited[i].item()
                color = colors_valid_limited[i]
                opacity = opacities_valid_limited[i].item()
                
                # Alpha blending
                if alpha_buffer[v_int, u_int] < 1.0:
                    alpha = opacity * (1.0 - alpha_buffer[v_int, u_int])
                    image[v_int, u_int] += color * alpha
                    alpha_buffer[v_int, u_int] += alpha
        
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

