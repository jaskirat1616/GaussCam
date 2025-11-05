"""Gaussian fitting, merging, temporal hierarchy, and LOD management."""

from .fitter import Gaussian, GaussianFitter
from .merger import GaussianMerger, LODManager
from .four_d import (
    Gaussian4D,
    GaussianDeformation,
    GaussianMotion,
    TemporalGaussianCluster,
    TemporalHierarchy,
)

__all__ = [
    "Gaussian",
    "GaussianFitter",
    "GaussianMerger",
    "LODManager",
    "GaussianMotion",
    "GaussianDeformation",
    "Gaussian4D",
    "TemporalGaussianCluster",
    "TemporalHierarchy",
]
