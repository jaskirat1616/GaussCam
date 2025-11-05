"""Gaussian compression primitives: pruning, quantization, and codebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from backend.gaussian.fitter import Gaussian


@dataclass
class CompressionStats:
    pruned: int
    quantized: bool
    codebook_size: int


def prune_gaussians(gaussians: Gaussian, importance_scores: np.ndarray, target_count: int) -> Gaussian:
    """Prune Gaussians based on importance scores."""

    if gaussians.num_gaussians <= target_count:
        return gaussians

    indices = np.argsort(-importance_scores)[:target_count]
    return Gaussian(
        centroids=gaussians.centroids[indices],
        covariances=None,
        colors=gaussians.colors[indices],
        opacity=gaussians.opacity[indices],
        scales=gaussians.scales[indices],
        rotations=gaussians.rotations[indices],
    )


def quantize_attributes(gaussians: Gaussian, bits: int = 8) -> Dict[str, np.ndarray]:
    """Quantize Gaussian attributes to reduce storage."""

    scale = 2 ** bits - 1
    colors = (gaussians.colors * scale).astype(np.uint8)
    opacity = (gaussians.opacity * scale).astype(np.uint8)

    return {
        "centroids": gaussians.centroids.astype(np.float16),
        "colors_q": colors,
        "opacity_q": opacity,
        "scales": gaussians.scales.astype(np.float16),
        "rotations": gaussians.rotations.astype(np.float16),
    }


def build_codebook(colors: np.ndarray, size: int = 256) -> Dict[str, np.ndarray]:
    """Construct a simple k-means codebook for color compression."""

    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters=size, random_state=42)
    assignments = kmeans.fit_predict(colors)
    codebook = kmeans.cluster_centers_

    return {"codebook": codebook.astype(np.float32), "assignments": assignments.astype(np.int32)}


def compress_gaussians(
    gaussians: Gaussian,
    importance_scores: Optional[np.ndarray] = None,
    target_count: Optional[int] = None,
    codebook_size: int = 256,
) -> Dict[str, object]:
    """End-to-end compression returning quantized payload and stats."""

    compressed_gaussians = gaussians
    if target_count is not None and importance_scores is not None:
        compressed_gaussians = prune_gaussians(gaussians, importance_scores, target_count)

    quantized = quantize_attributes(compressed_gaussians)
    codebook = build_codebook(compressed_gaussians.colors, size=min(codebook_size, compressed_gaussians.num_gaussians))

    stats = CompressionStats(
        pruned=gaussians.num_gaussians - compressed_gaussians.num_gaussians,
        quantized=True,
        codebook_size=codebook["codebook"].shape[0],
    )

    return {
        "quantized": quantized,
        "codebook": codebook,
        "stats": stats,
    }

