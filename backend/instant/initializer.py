"""Feed-forward initialization of Gaussians from sparse views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from backend.gaussian.fitter import Gaussian, GaussianFitter
from backend.utils.point_cloud import depth_to_point_cloud


@dataclass
class ViewSample:
    rgb: np.ndarray
    depth: np.ndarray


class InstantGaussianInitializer:
    """PixelSplat-inspired feed-forward Gaussian initializer."""

    def __init__(self, max_gaussians: int = 50000) -> None:
        self.max_gaussians = max_gaussians
        self.fitter = GaussianFitter(k_neighbors=6, initial_opacity=0.85)

    def initialize(
        self,
        views: Sequence[ViewSample],
        intrinsics,
        depth_scale: float = 5.0,
        max_depth: float = 10.0,
    ) -> Gaussian:
        points_list = []
        colors_list = []
        weights_list = []

        for sample in views:
            points, colors = depth_to_point_cloud(
                sample.depth,
                sample.rgb,
                intrinsics,
                depth_scale=depth_scale,
                max_depth=max_depth,
            )

            if len(points) == 0:
                continue

            weights = self._compute_importance(sample.rgb, sample.depth)
            weights = np.clip(weights[: len(points)], 1e-3, 1.0)

            points_list.append(points)
            colors_list.append(colors)
            weights_list.append(weights)

        if not points_list:
            raise ValueError("No valid points extracted from views")

        points = np.concatenate(points_list, axis=0)
        colors = np.concatenate(colors_list, axis=0)
        weights = np.concatenate(weights_list, axis=0)

        if points.shape[0] > self.max_gaussians:
            top_indices = np.argsort(-weights)[: self.max_gaussians]
            points = points[top_indices]
            colors = colors[top_indices]

        return self.fitter.fit(points, colors, method="pca")

    def _compute_importance(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        laplacian = cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
        depth_var = cv2.Laplacian(depth.astype(np.float32), cv2.CV_32F)
        lap_norm = np.abs(laplacian)
        depth_norm = np.abs(depth_var)

        importance = lap_norm.flatten() + depth_norm.flatten()
        if importance.size == 0:
            return np.ones(rgb.shape[0] * rgb.shape[1], dtype=np.float32)

        importance = importance / (importance.max() + 1e-6)
        return importance.astype(np.float32)

