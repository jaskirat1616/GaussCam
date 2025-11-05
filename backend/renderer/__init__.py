"""Gaussian Splatting renderers across CUDA, MPS, and WebGPU backends."""

from .base import Renderer
from .cuda_renderer import CUDARenderer
from .mps_renderer import MPSRenderer
from .webgpu_renderer import WebGPURenderer

__all__ = [
    "Renderer",
    "CUDARenderer",
    "MPSRenderer",
    "WebGPURenderer",
]
