"""Renderer manager that selects best backend based on hardware capabilities."""

from __future__ import annotations

from typing import Optional

from backend.renderer.cuda_renderer import CUDARenderer
from backend.renderer.mps_renderer import MPSRenderer
from backend.renderer.webgpu_renderer import WebGPURenderer
from backend.utils.gpu_detection import is_cuda, is_mps


def create_renderer(
    preferred: Optional[str] = None,
    width: int = 640,
    height: int = 480,
):
    """Factory helper selecting renderer backend."""

    preferred = (preferred or "auto").lower()

    if preferred == "cuda" and is_cuda():
        return CUDARenderer(width=width, height=height)
    if preferred == "mps" and is_mps():
        return MPSRenderer(width=width, height=height)
    if preferred == "webgpu":
        return WebGPURenderer(width=width, height=height)

    if is_cuda():
        return CUDARenderer(width=width, height=height)
    if is_mps():
        return MPSRenderer(width=width, height=height)

    return WebGPURenderer(width=width, height=height)

