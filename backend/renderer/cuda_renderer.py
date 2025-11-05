"""
CUDA Renderer

Gaussian Splatting renderer using gsplat library for NVIDIA GPUs.
"""

import numpy as np
import torch
from typing import Optional
from backend.renderer.base import Renderer
from backend.gaussian.fitter import Gaussian
from backend.utils.gpu_detection import get_device, is_cuda

try:
    import gsplat
    from gsplat import rasterization_gaussians
    GSplat_AVAILABLE = True
except ImportError:
    GSplat_AVAILABLE = False
    print("Warning: gsplat not available. Install with: pip install gsplat")


class CUDARenderer(Renderer):
    """CUDA-based Gaussian Splatting renderer using gsplat."""
    
    def __init__(self, width: int = 640, height: int = 480, device: Optional[torch.device] = None):
        """
        Initialize CUDA renderer.
        
        Args:
            width: Render width
            height: Render height
            device: PyTorch device (auto-detected if None)
        """
        super().__init__(width, height)
        
        if not GSplat_AVAILABLE:
            raise ImportError("gsplat library required for CUDA rendering. Install with: pip install gsplat")
        
        if device is None:
            device = get_device()
        
        if not is_cuda():
            raise RuntimeError("CUDA not available. Cannot use CUDARenderer.")
        
        self.device = device
        self.background_color = torch.tensor([0.0, 0.0, 0.0], device=device)
        
        print(f"CUDARenderer initialized: {width}x{height} on {device}")
    
    def render(
        self,
        gaussians: Gaussian,
        camera_pose: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render Gaussians using gsplat.
        
        Args:
            gaussians: Gaussian object
            camera_pose: Camera pose matrix (4, 4) or None for identity
            intrinsics: Camera intrinsics matrix (3, 3) or None for default
        
        Returns:
            Rendered image (H, W, 3) RGB in [0, 1]
        """
        if not GSplat_AVAILABLE:
            raise RuntimeError("gsplat not available")
        
        # Convert Gaussians to torch tensors
        gaussian_dict = gaussians.to_torch(self.device)
        
        # Get camera parameters
        if camera_pose is None:
            camera_pose = np.eye(4, dtype=np.float32)
        if intrinsics is None:
            # Default intrinsics
            intrinsics = np.array([
                [self.width, 0, self.width / 2],
                [0, self.width, self.height / 2],
                [0, 0, 1],
            ], dtype=np.float32)
        
        # Convert to torch
        camera_pose_t = torch.from_numpy(camera_pose).float().to(self.device)
        intrinsics_t = torch.from_numpy(intrinsics).float().to(self.device)
        
        # Extract camera parameters
        # gsplat expects: camera_center, viewmatrix, focal, tanfovx, tanfovy
        # We need to extract these from pose and intrinsics
        
        # Camera center (translation)
        camera_center = camera_pose_t[:3, 3]
        
        # View matrix (inverse of pose)
        viewmatrix = torch.inverse(camera_pose_t)[:3, :]
        
        # Focal length
        fx = intrinsics_t[0, 0]
        fy = intrinsics_t[1, 1]
        
        # Tan FOV
        tanfovx = (self.width / 2.0) / fx
        tanfovy = (self.height / 2.0) / fy
        
        # Render using gsplat
        try:
            # gsplat rasterization interface
            # Note: API may vary, adjust based on actual gsplat version
            rendered_image = rasterization_gaussians(
                means=gaussian_dict["centroids"],
                quats=gaussian_dict["rotations"],
                scales=gaussian_dict["scales"],
                opacities=gaussian_dict["opacity"],
                colors=gaussian_dict["colors"],
                viewmats=viewmatrix.unsqueeze(0),
                Ks=intrinsics_t.unsqueeze(0),
                width=self.width,
                height=self.height,
            )
            
            # Extract rendered image (gsplat returns [B, H, W, 3] or [H, W, 3])
            if rendered_image.dim() == 4:
                rendered_image = rendered_image[0]
            
            # Convert to numpy
            image = rendered_image.cpu().numpy()
            
            # Clamp to [0, 1]
            image = np.clip(image, 0.0, 1.0)
            
            return image
        
        except Exception as e:
            print(f"Rendering error: {e}")
            # Return black image on error
            return np.zeros((self.height, self.width, 3), dtype=np.float32)
    
    def resize(self, width: int, height: int) -> None:
        """Resize render buffer."""
        self.width = width
        self.height = height
    
    def clear(self) -> None:
        """Clear render buffer."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

