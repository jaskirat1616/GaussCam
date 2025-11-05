"""WebGPU renderer stub preparing data for browser-based playback."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from backend.gaussian.fitter import Gaussian
from backend.gaussian.four_d import Gaussian4D
from backend.renderer.base import Renderer


class WebGPURenderer(Renderer):
    """Renderer that serializes Gaussian data for a WebGPU frontend."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        super().__init__(width, height)
        self.last_payload: Dict[str, Any] = {}

    def render(
        self,
        gaussians: Gaussian,
        camera_pose: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        payload = self._serialize_gaussians(gaussians, camera_pose, intrinsics)
        self.last_payload = payload
        return np.zeros((self.height, self.width, 3), dtype=np.float32)

    def render_dynamic(
        self,
        gaussians: Gaussian4D,
        time_offset: float = 0.0,
        camera_pose: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        payload = self._serialize_gaussians(
            gaussians.gaussian,
            camera_pose,
            intrinsics,
            extra={
                "motion": {
                    "translation": gaussians.motion.translation.tolist(),
                    "rotation_axis_angle": gaussians.motion.rotation_axis_angle.tolist(),
                    "scale_velocity": gaussians.motion.scale_velocity.tolist(),
                    "timestamps": gaussians.timestamps.tolist(),
                }
            },
        )
        self.last_payload = payload
        return np.zeros((self.height, self.width, 3), dtype=np.float32)

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def _serialize_gaussians(
        self,
        gaussians: Gaussian,
        camera_pose: Optional[np.ndarray],
        intrinsics: Optional[np.ndarray],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = {
            "centroids": gaussians.centroids.tolist(),
            "scales": gaussians.scales.tolist() if gaussians.scales is not None else None,
            "rotations": gaussians.rotations.tolist() if gaussians.rotations is not None else None,
            "colors": gaussians.colors.tolist(),
            "opacity": gaussians.opacity.tolist(),
            "camera_pose": camera_pose.tolist() if camera_pose is not None else None,
            "intrinsics": intrinsics.tolist() if intrinsics is not None else None,
            "resolution": [self.width, self.height],
        }
        if extra:
            data.update(extra)
        return data

    @property
    def capabilities(self):
        caps = super().capabilities.copy()
        caps.update({"web_stream": True, "dynamic_gaussians": True})
        return caps

