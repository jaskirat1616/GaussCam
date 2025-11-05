"""
Base Renderer Interface

Abstract renderer for Gaussian Splatting with CUDA and MPS backends.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple
from backend.gaussian.fitter import Gaussian


class Renderer(ABC):
    """Abstract renderer interface."""
    
    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize renderer.
        
        Args:
            width: Render width
            height: Render height
        """
        self.width = width
        self.height = height
    
    @abstractmethod
    def render(
        self,
        gaussians: Gaussian,
        camera_pose: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render Gaussians.
        
        Args:
            gaussians: Gaussian object
            camera_pose: Camera pose matrix (4, 4) or None for identity
            intrinsics: Camera intrinsics matrix (3, 3) or None for default
        
        Returns:
            Rendered image (H, W, 3) RGB in [0, 1]
        """
        raise NotImplementedError
    
    @abstractmethod
    def resize(self, width: int, height: int) -> None:
        """Resize render buffer."""
        raise NotImplementedError
    
    def clear(self) -> None:
        """Clear render buffer (optional)."""
        pass

