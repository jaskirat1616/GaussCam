"""
Gaussian Fitting Module

Compute Gaussian parameters (centroids, covariances, colors, opacity) from point clouds.
"""

import numpy as np
import torch
from typing import Tuple, Optional
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from backend.utils.gpu_detection import get_device, is_cuda, is_mps


class Gaussian:
    """Gaussian splat representation."""
    
    def __init__(
        self,
        centroids: np.ndarray,  # (N, 3)
        covariances: np.ndarray,  # (N, 3, 3) or (N, 6) for quaternion representation
        colors: np.ndarray,  # (N, 3) RGB
        opacity: np.ndarray,  # (N,)
        scales: Optional[np.ndarray] = None,  # (N, 3)
        rotations: Optional[np.ndarray] = None,  # (N, 4) quaternions
    ):
        """
        Initialize Gaussians.
        
        Args:
            centroids: Gaussian centers (N, 3)
            covariances: Covariance matrices (N, 3, 3) or quaternion representation (N, 6)
            colors: RGB colors (N, 3) in [0, 1]
            opacity: Opacity values (N,) in [0, 1]
            scales: Scale factors (N, 3) if using quaternion representation
            rotations: Rotation quaternions (N, 4) if using quaternion representation
        """
        self.centroids = centroids
        self.covariances = covariances
        self.colors = colors
        self.opacity = opacity
        self.scales = scales
        self.rotations = rotations
        self.num_gaussians = len(centroids)
    
    def to_torch(self, device: Optional[torch.device] = None) -> dict:
        """Convert to PyTorch tensors."""
        if device is None:
            device = get_device()
        
        result = {
            "centroids": torch.from_numpy(self.centroids).float().to(device),
            "colors": torch.from_numpy(self.colors).float().to(device),
            "opacity": torch.from_numpy(self.opacity).float().to(device),
        }
        
        if self.scales is not None and self.rotations is not None:
            # Quaternion representation (used by gsplat)
            result["scales"] = torch.from_numpy(self.scales).float().to(device)
            result["rotations"] = torch.from_numpy(self.rotations).float().to(device)
        else:
            # Covariance matrix representation
            result["covariances"] = torch.from_numpy(self.covariances).float().to(device)
        
        return result


class GaussianFitter:
    """Fit Gaussians from point clouds."""
    
    def __init__(
        self,
        k_neighbors: int = 8,
        min_scale: float = 0.001,
        max_scale: float = 0.1,
        initial_opacity: float = 0.9,
    ):
        """
        Initialize Gaussian fitter.
        
        Args:
            k_neighbors: Number of neighbors for covariance estimation
            min_scale: Minimum Gaussian scale
            max_scale: Maximum Gaussian scale
            initial_opacity: Initial opacity value
        """
        self.k_neighbors = k_neighbors
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.initial_opacity = initial_opacity
    
    def fit(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        method: str = "pca",
    ) -> Gaussian:
        """
        Fit Gaussians from point cloud.
        
        Args:
            points: Point cloud (N, 3)
            colors: Colors (N, 3) in [0, 1]
            method: Fitting method ('pca', 'knn', 'uniform')
        
        Returns:
            Gaussian object
        """
        if method == "pca":
            return self._fit_pca(points, colors)
        elif method == "knn":
            return self._fit_knn(points, colors)
        elif method == "uniform":
            return self._fit_uniform(points, colors)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _fit_uniform(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> Gaussian:
        """Fit uniform Gaussians (simple approach)."""
        num_points = len(points)
        
        # Centroids are the points themselves
        centroids = points.astype(np.float32)
        
        # Uniform scale
        scale = np.mean(np.std(points, axis=0))
        scale = np.clip(scale, self.min_scale, self.max_scale)
        
        # Identity quaternions (no rotation)
        rotations = np.zeros((num_points, 4), dtype=np.float32)
        rotations[:, 0] = 1.0  # w = 1
        
        # Uniform scales
        scales = np.ones((num_points, 3), dtype=np.float32) * scale
        
        # Colors
        colors_normalized = colors.astype(np.float32)
        if colors_normalized.max() > 1.0:
            colors_normalized = colors_normalized / 255.0
        
        # Initial opacity
        opacity = np.ones(num_points, dtype=np.float32) * self.initial_opacity
        
        return Gaussian(
            centroids=centroids,
            covariances=None,  # Using quaternion representation
            colors=colors_normalized,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
        )
    
    def _fit_knn(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> Gaussian:
        """Fit Gaussians using k-nearest neighbors for covariance."""
        num_points = len(points)
        
        if num_points == 0:
            raise ValueError("Empty point cloud")
        
        # Centroids
        centroids = points.astype(np.float32)
        
        # Compute k-nearest neighbors for each point
        k = min(self.k_neighbors, num_points - 1)
        if k == 0:
            return self._fit_uniform(points, colors)
        
        # Compute distances
        distances = cdist(points, points)
        distances.sort(axis=1)
        
        # Get k nearest neighbors (excluding self)
        neighbor_indices = np.argsort(distances, axis=1)[:, 1:k+1]
        
        # Compute covariances from neighbors
        scales = np.zeros((num_points, 3), dtype=np.float32)
        rotations = np.zeros((num_points, 4), dtype=np.float32)
        rotations[:, 0] = 1.0  # w = 1 (identity rotation)
        
        for i in range(num_points):
            neighbors = points[neighbor_indices[i]]
            center = points[i]
            
            # Compute covariance
            diff = neighbors - center
            if len(diff) > 0:
                cov = np.cov(diff.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # Scale from eigenvalues
                scales[i] = np.sqrt(np.maximum(eigenvalues, self.min_scale**2))
                scales[i] = np.clip(scales[i], self.min_scale, self.max_scale)
                
                # Rotation from eigenvectors (simplified - use PCA)
                # For now, use identity rotation
            else:
                scales[i] = np.ones(3) * self.min_scale
        
        # Colors
        colors_normalized = colors.astype(np.float32)
        if colors_normalized.max() > 1.0:
            colors_normalized = colors_normalized / 255.0
        
        # Opacity
        opacity = np.ones(num_points, dtype=np.float32) * self.initial_opacity
        
        return Gaussian(
            centroids=centroids,
            covariances=None,
            colors=colors_normalized,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
        )
    
    def _fit_pca(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> Gaussian:
        """Fit Gaussians using PCA for local covariance estimation."""
        num_points = len(points)
        
        if num_points == 0:
            raise ValueError("Empty point cloud")
        
        # Centroids
        centroids = points.astype(np.float32)
        
        # Compute k-nearest neighbors
        k = min(self.k_neighbors, num_points - 1)
        if k == 0:
            return self._fit_uniform(points, colors)
        
        # Compute distances
        distances = cdist(points, points)
        neighbor_indices = np.argsort(distances, axis=1)[:, 1:k+1]
        
        # Fit Gaussians using PCA
        scales = np.zeros((num_points, 3), dtype=np.float32)
        rotations = np.zeros((num_points, 4), dtype=np.float32)
        rotations[:, 0] = 1.0  # w = 1
        
        for i in range(num_points):
            neighbors = points[neighbor_indices[i]]
            center = points[i]
            
            if len(neighbors) > 2:
                # PCA on neighbors
                pca = PCA(n_components=3)
                pca.fit(neighbors)
                
                # Scales from explained variance
                scales[i] = np.sqrt(np.maximum(pca.explained_variance_, self.min_scale**2))
                scales[i] = np.clip(scales[i], self.min_scale, self.max_scale)
            else:
                # Fallback to uniform scale
                scale = np.mean(np.std(points, axis=0))
                scales[i] = np.clip(scale, self.min_scale, self.max_scale)
        
        # Colors
        colors_normalized = colors.astype(np.float32)
        if colors_normalized.max() > 1.0:
            colors_normalized = colors_normalized / 255.0
        
        # Opacity
        opacity = np.ones(num_points, dtype=np.float32) * self.initial_opacity
        
        return Gaussian(
            centroids=centroids,
            covariances=None,
            colors=colors_normalized,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
        )
    
    def fit_downsampled(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        max_gaussians: int = 100000,
        voxel_size: Optional[float] = None,
        method: str = "pca",
        use_gpu: bool = True,
    ) -> Gaussian:
        """
        Fit Gaussians with downsampling to limit number (GPU-accelerated if available).
        
        Args:
            points: Point cloud (N, 3)
            colors: Colors (N, 3)
            max_gaussians: Maximum number of Gaussians
            voxel_size: Voxel size for downsampling (auto if None)
            method: Fitting method
            use_gpu: Use GPU acceleration for downsampling
        
        Returns:
            Gaussian object
        """
        if len(points) <= max_gaussians:
            return self.fit(points, colors, method=method)
        
        # Downsample if needed
        if voxel_size is None:
            # Auto-compute voxel size
            bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
            voxel_size = np.max(bbox_size) / np.cbrt(max_gaussians)
        
        from backend.utils.point_cloud import downsample_point_cloud
        
        # Use GPU-accelerated downsampling if available
        points_ds, colors_ds = downsample_point_cloud(points, colors, voxel_size=voxel_size, use_gpu=use_gpu)
        
        # Fit on downsampled points
        return self.fit(points_ds, colors_ds, method=method)

