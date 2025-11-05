"""
Tests for Gaussian Fitting and Merging
"""

import pytest
import numpy as np
from backend.gaussian.fitter import GaussianFitter, Gaussian
from backend.gaussian.merger import GaussianMerger, LODManager


def test_gaussian_initialization():
    """Test Gaussian initialization."""
    num = 100
    centroids = np.random.randn(num, 3).astype(np.float32)
    scales = np.ones((num, 3), dtype=np.float32) * 0.01
    rotations = np.zeros((num, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w = 1
    colors = np.random.rand(num, 3).astype(np.float32)
    opacity = np.ones(num, dtype=np.float32) * 0.9
    
    gaussians = Gaussian(
        centroids=centroids,
        covariances=None,
        colors=colors,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
    )
    
    assert gaussians.num_gaussians == num
    assert gaussians.centroids.shape == (num, 3)
    assert gaussians.colors.shape == (num, 3)


def test_gaussian_fitter():
    """Test Gaussian fitter."""
    fitter = GaussianFitter(k_neighbors=8, initial_opacity=0.9)
    
    # Create test point cloud
    num_points = 1000
    points = np.random.randn(num_points, 3).astype(np.float32) * 0.5
    colors = np.random.rand(num_points, 3).astype(np.float32)
    
    # Fit Gaussians
    gaussians = fitter.fit(points, colors, method="uniform")
    
    assert gaussians.num_gaussians == num_points
    assert gaussians.centroids.shape == (num_points, 3)
    assert gaussians.scales.shape == (num_points, 3)


def test_gaussian_merger():
    """Test Gaussian merger."""
    merger = GaussianMerger(merge_threshold=0.01, max_gaussians=1000)
    
    # Create test Gaussians
    fitter = GaussianFitter()
    num_points = 100
    points = np.random.randn(num_points, 3).astype(np.float32) * 0.5
    colors = np.random.rand(num_points, 3).astype(np.float32)
    
    gaussians1 = fitter.fit(points, colors, method="uniform")
    gaussians2 = fitter.fit(points + 0.1, colors, method="uniform")
    
    # Merge
    merged = merger.merge(gaussians1, merge_strategy="weighted")
    assert merged.num_gaussians >= gaussians1.num_gaussians
    
    merged2 = merger.merge(gaussians2, merge_strategy="weighted")
    assert merged2.num_gaussians >= merged.num_gaussians


def test_lod_manager():
    """Test LOD manager."""
    lod_manager = LODManager(levels=4)
    
    # Create test Gaussians
    num = 1000
    centroids = np.random.randn(num, 3).astype(np.float32)
    scales = np.ones((num, 3), dtype=np.float32) * 0.01
    rotations = np.zeros((num, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    colors = np.random.rand(num, 3).astype(np.float32)
    opacity = np.ones(num, dtype=np.float32) * 0.9
    
    gaussians = Gaussian(
        centroids=centroids,
        covariances=None,
        colors=colors,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
    )
    
    # Get LOD
    lod_gaussians = lod_manager.get_gaussians(gaussians, level=2)
    assert lod_gaussians.num_gaussians <= gaussians.num_gaussians

