"""
Base Renderer Interface

Abstract renderer for Gaussian Splatting with CUDA and MPS backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from backend.gaussian.fitter import Gaussian
from backend.gaussian.four_d import Gaussian4D


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
        **kwargs: Any,
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
    
    def render_dynamic(
        self,
        gaussians: Gaussian4D,
        time_offset: float = 0.0,
        camera_pose: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Render dynamic (4D) Gaussians. Default falls back to static render."""

        return self.render(
            gaussians=gaussians.gaussian,
            camera_pose=camera_pose,
            intrinsics=intrinsics,
            **kwargs,
        )

    @abstractmethod
    def resize(self, width: int, height: int) -> None:
        """Resize render buffer."""
        raise NotImplementedError
    
    def clear(self) -> None:
        """Clear render buffer (optional)."""
        pass

    @property
    def capabilities(self) -> Dict[str, bool]:
        """Report backend capabilities (dynamic splatting, web streaming, etc.)."""

        return {
            "dynamic_gaussians": False,
            "multi_channel": False,
            "web_stream": False,
        }

