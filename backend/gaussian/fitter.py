"""
Gaussian Fitting Module

Compute Gaussian parameters (centroids, covariances, colors, opacity) from point clouds.
"""

import numpy as np
import torch
from typing import Optional
from scipy.spatial import cKDTree
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

    # ------------------------------------------------------------------
    # Internal helpers

    def _normalize_colors(self, colors: np.ndarray) -> np.ndarray:
        """Ensure colors are float32 in [0, 1]."""
        if colors.size == 0:
            return colors.astype(np.float32, copy=False)
        colors_float = colors.astype(np.float32, copy=False)
        if colors_float.max() > 1.0:
            colors_float = colors_float / 255.0
        return np.clip(colors_float, 0.0, 1.0)

    def _compute_neighbors(self, points: np.ndarray, k: int) -> np.ndarray:
        """Return indices for k nearest neighbours using cKDTree."""
        num_points = len(points)
        if num_points == 0:
            return np.empty((0, 0), dtype=np.int32)

        k = min(k, num_points - 1)
        if k <= 0:
            return np.empty((num_points, 0), dtype=np.int32)

        tree = cKDTree(points)
        # Query k+1 neighbours to include the point itself; workers=-1 enables multithreading
        distances, indices = tree.query(points, k=k + 1, workers=-1)

        if k + 1 == 1:
            # Query returns 1D arrays when k == 0 in practice; normalise shapes
            indices = indices[:, np.newaxis]

        return indices[:, 1:].astype(np.int32, copy=False)
    
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
        colors_normalized = self._normalize_colors(colors)
        
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
        if k <= 0:
            return self._fit_uniform(points, colors)

        neighbor_indices = self._compute_neighbors(points, k)
        
        # Compute covariances from neighbors
        scales = np.zeros((num_points, 3), dtype=np.float32)
        rotations = np.zeros((num_points, 4), dtype=np.float32)
        rotations[:, 0] = 1.0  # w = 1 (identity rotation)
        
        for i in range(num_points):
            neighbor_idx = neighbor_indices[i]
            if neighbor_idx.size == 0:
                scales[i] = np.ones(3, dtype=np.float32) * self.min_scale
                continue

            neighbors = points[neighbor_idx]
            diff = neighbors - points[i]

            if diff.shape[0] >= 3:
                cov = np.cov(diff, rowvar=False, bias=True)
            elif diff.shape[0] > 0:
                cov = (diff.T @ diff) / max(diff.shape[0], 1)
            else:
                cov = np.eye(3, dtype=np.float32) * (self.min_scale ** 2)

            cov = np.asarray(cov, dtype=np.float32)
            cov += np.eye(3, dtype=np.float32) * 1e-6  # Regularize for numerical stability

            try:
                eigenvalues = np.linalg.eigvalsh(cov)
            except np.linalg.LinAlgError:
                eigenvalues = np.array([self.min_scale ** 2] * 3, dtype=np.float32)

            scales[i] = np.sqrt(np.maximum(eigenvalues, self.min_scale ** 2))
            scales[i] = np.clip(scales[i], self.min_scale, self.max_scale)
        
        # Colors
        colors_normalized = self._normalize_colors(colors)
        
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
        if k <= 0:
            return self._fit_uniform(points, colors)
        
        neighbor_indices = self._compute_neighbors(points, k)
        
        # Fit Gaussians using PCA
        scales = np.zeros((num_points, 3), dtype=np.float32)
        rotations = np.zeros((num_points, 4), dtype=np.float32)
        rotations[:, 0] = 1.0  # w = 1
        
        for i in range(num_points):
            neighbor_idx = neighbor_indices[i]
            if neighbor_idx.size > 2:
                neighbors = points[neighbor_idx]
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
        colors_normalized = self._normalize_colors(colors)
        
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
    
    def fit_from_depth(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        intrinsics,
        max_gaussians: int = 100000,
        voxel_size: Optional[float] = None,
        method: str = "uniform",
        depth_scale: float = 5.0,
        max_depth: float = 10.0,
        use_gpu: bool = True,
    ) -> Gaussian:
        """
        Fit Gaussians directly from depth map (skips explicit point cloud creation).
        
        Args:
            depth: Depth map (H, W) normalized [0, 1] or in meters
            rgb: RGB image (H, W, 3) in [0, 1] or [0, 255]
            intrinsics: Camera intrinsics
            max_gaussians: Maximum number of Gaussians
            voxel_size: Voxel size for downsampling (auto if None)
            method: Fitting method ('uniform', 'pca', 'knn')
            depth_scale: Scale factor for depth
            max_depth: Maximum depth threshold
            use_gpu: Use GPU acceleration if available
        
        Returns:
            Gaussian object
        """
        # Convert depth to points directly with downsampling
        # Use GPU-accelerated version if available
        if use_gpu:
            try:
                from backend.utils.optimization import fast_depth_to_gaussians_gpu
                from backend.utils.gpu_detection import is_cuda, is_mps
                
                if is_cuda() or is_mps():
                    return fast_depth_to_gaussians_gpu(
                        depth, rgb, intrinsics, self, max_gaussians, voxel_size, method, depth_scale, max_depth
                    )
            except:
                pass  # Fall back to CPU version
        
        # CPU version: convert depth to points with downsampling
        from backend.utils.point_cloud import downsample_point_cloud
        
        # Normalize RGB
        if rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0
        else:
            rgb = rgb.astype(np.float32)
        
        # Scale depth
        if depth.max() <= 1.0:
            depth_scaled = depth * depth_scale
        else:
            depth_scaled = depth.astype(np.float32)
        
        # Apply threshold
        depth_scaled = np.clip(depth_scaled, 0, max_depth)
        
        # Backproject to 3D (vectorized)
        points_3d = intrinsics.backproject(depth_scaled)
        
        # Flatten and filter
        h, w = depth.shape
        points = points_3d.reshape(-1, 3)
        colors = rgb.reshape(-1, 3)
        
        depth_flat = depth_scaled.flatten()
        valid_mask = (
            (depth_flat > 0.1) &
            np.isfinite(points).all(axis=1) &
            (points[:, 2] > 0.1)
        )
        
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        # Downsample if needed (before fitting)
        if len(points) > max_gaussians:
            if voxel_size is None:
                bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
                voxel_size = np.max(bbox_size) / np.cbrt(max_gaussians)
            
            points, colors = downsample_point_cloud(points, colors, voxel_size=voxel_size, use_gpu=use_gpu)
        
        # Fit Gaussians
        return self.fit(points, colors, method=method)

