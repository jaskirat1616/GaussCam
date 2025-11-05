"""
Gaussian Merging Module

Temporal Gaussian merging for frame-to-frame consistency and LOD management.
"""

import numpy as np
from typing import Optional, List
from scipy.spatial import cKDTree
from backend.gaussian.fitter import Gaussian


class GaussianMerger:
    """Merge Gaussians temporally and manage LOD."""
    
    def __init__(
        self,
        merge_threshold: float = 0.01,
        max_gaussians: int = 1000000,
        lod_levels: int = 4,
    ):
        """
        Initialize Gaussian merger.
        
        Args:
            merge_threshold: Distance threshold for merging (meters)
            max_gaussians: Maximum number of Gaussians
            lod_levels: Number of LOD levels
        """
        self.merge_threshold = merge_threshold
        self.max_gaussians = max_gaussians
        self.lod_levels = lod_levels
        self.accumulated_gaussians: Optional[Gaussian] = None
        self.frame_count = 0
    
    def merge(
        self,
        new_gaussians: Gaussian,
        merge_strategy: str = "weighted",
    ) -> Gaussian:
        """
        Merge new Gaussians with accumulated ones.
        
        Args:
            new_gaussians: New Gaussians from current frame
            merge_strategy: Merge strategy ('weighted', 'nearest', 'average')
        
        Returns:
            Merged Gaussian object
        """
        if self.accumulated_gaussians is None:
            # First frame
            self.accumulated_gaussians = new_gaussians
            self.frame_count = 1
            return self.accumulated_gaussians
        
        # Merge with accumulated
        if merge_strategy == "weighted":
            merged = self._merge_weighted(new_gaussians)
        elif merge_strategy == "nearest":
            merged = self._merge_nearest(new_gaussians)
        elif merge_strategy == "average":
            merged = self._merge_average(new_gaussians)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Update accumulated
        self.accumulated_gaussians = merged
        self.frame_count += 1
        
        # Limit number of Gaussians
        if merged.num_gaussians > self.max_gaussians:
            merged = self._downsample(merged, self.max_gaussians)
            self.accumulated_gaussians = merged
        
        return merged
    
    def _merge_weighted(self, new_gaussians: Gaussian) -> Gaussian:
        """Weighted merge based on distance."""
        old = self.accumulated_gaussians
        new = new_gaussians
        
        if new.num_gaussians == 0:
            return old

        tree = cKDTree(new.centroids)
        distances, matches = tree.query(
            old.centroids,
            k=1,
            distance_upper_bound=self.merge_threshold,
            workers=-1,
        )
        matches = matches.astype(np.int64, copy=False)

        merge_mask = np.isfinite(distances)
        
        # Create merged arrays
        merged_centroids = []
        merged_scales = []
        merged_rotations = []
        merged_colors = []
        merged_opacity = []
        
        # Merge matched Gaussians
        for i in range(len(old.centroids)):
            if merge_mask[i]:
                j = matches[i]
                # Weighted average (weight by opacity)
                w_old = old.opacity[i]
                w_new = new.opacity[j]
                w_total = w_old + w_new + 1e-8
                
                merged_centroids.append(
                    (old.centroids[i] * w_old + new.centroids[j] * w_new) / w_total
                )
                merged_scales.append(
                    (old.scales[i] * w_old + new.scales[j] * w_new) / w_total
                )
                merged_rotations.append(old.rotations[i])  # Keep old rotation
                merged_colors.append(
                    (old.colors[i] * w_old + new.colors[j] * w_new) / w_total
                )
                merged_opacity.append(
                    np.clip((w_old + w_new) / 2.0, 0.0, 1.0)
                )
            else:
                # Keep old Gaussian
                merged_centroids.append(old.centroids[i])
                merged_scales.append(old.scales[i])
                merged_rotations.append(old.rotations[i])
                merged_colors.append(old.colors[i])
                merged_opacity.append(old.opacity[i] * 0.95)  # Slight decay
        
        # Add unmatched new Gaussians
        matched_new = set(matches[merge_mask])
        for j in range(len(new.centroids)):
            if j not in matched_new:
                merged_centroids.append(new.centroids[j])
                merged_scales.append(new.scales[j])
                merged_rotations.append(new.rotations[j])
                merged_colors.append(new.colors[j])
                merged_opacity.append(new.opacity[j])
        
        # Convert to arrays
        merged_centroids = np.array(merged_centroids, dtype=np.float32)
        merged_scales = np.array(merged_scales, dtype=np.float32)
        merged_rotations = np.array(merged_rotations, dtype=np.float32)
        merged_colors = np.array(merged_colors, dtype=np.float32)
        merged_opacity = np.array(merged_opacity, dtype=np.float32)
        
        return Gaussian(
            centroids=merged_centroids,
            covariances=None,
            colors=merged_colors,
            opacity=merged_opacity,
            scales=merged_scales,
            rotations=merged_rotations,
        )
    
    def _merge_nearest(self, new_gaussians: Gaussian) -> Gaussian:
        """Merge by replacing nearest Gaussians."""
        # Simplified: just append new Gaussians
        old = self.accumulated_gaussians
        new = new_gaussians
        
        # Concatenate
        merged_centroids = np.concatenate([old.centroids, new.centroids], axis=0)
        merged_scales = np.concatenate([old.scales, new.scales], axis=0)
        merged_rotations = np.concatenate([old.rotations, new.rotations], axis=0)
        merged_colors = np.concatenate([old.colors, new.colors], axis=0)
        merged_opacity = np.concatenate([old.opacity, new.opacity], axis=0)
        
        return Gaussian(
            centroids=merged_centroids,
            covariances=None,
            colors=merged_colors,
            opacity=merged_opacity,
            scales=merged_scales,
            rotations=merged_rotations,
        )
    
    def _merge_average(self, new_gaussians: Gaussian) -> Gaussian:
        """Average merge (same as weighted with equal weights)."""
        return self._merge_weighted(new_gaussians)
    
    def _downsample(self, gaussians: Gaussian, max_count: int) -> Gaussian:
        """Downsample Gaussians to target count."""
        if gaussians.num_gaussians <= max_count:
            return gaussians
        
        # Simple uniform downsampling
        indices = np.linspace(0, gaussians.num_gaussians - 1, max_count, dtype=np.int32)
        
        return Gaussian(
            centroids=gaussians.centroids[indices],
            covariances=None,
            colors=gaussians.colors[indices],
            opacity=gaussians.opacity[indices],
            scales=gaussians.scales[indices],
            rotations=gaussians.rotations[indices],
        )
    
    def get_lod(self, level: int) -> Gaussian:
        """
        Get Gaussians at specified LOD level.
        
        Args:
            level: LOD level (0 = finest, lod_levels-1 = coarsest)
        
        Returns:
            Downsampled Gaussian object
        """
        if self.accumulated_gaussians is None:
            return None
        
        if level == 0:
            return self.accumulated_gaussians
        
        # Compute target count
        factor = 2 ** level
        target_count = max(1, self.accumulated_gaussians.num_gaussians // factor)
        
        return self._downsample(self.accumulated_gaussians, target_count)
    
    def reset(self) -> None:
        """Reset accumulated Gaussians."""
        self.accumulated_gaussians = None
        self.frame_count = 0


class LODManager:
    """Level-of-Detail manager for progressive rendering."""
    
    def __init__(
        self,
        levels: int = 4,
        target_counts: Optional[List[int]] = None,
    ):
        """
        Initialize LOD manager.
        
        Args:
            levels: Number of LOD levels
            target_counts: Target Gaussian counts per level
        """
        self.levels = levels
        self.target_counts = target_counts or [1000000, 500000, 250000, 100000]
        self.current_level = 0
    
    def get_gaussians(self, gaussians: Gaussian, level: Optional[int] = None) -> Gaussian:
        """
        Get Gaussians at specified LOD level.
        
        Args:
            gaussians: Full Gaussian object
            level: LOD level (None for current level)
        
        Returns:
            Downsampled Gaussian object
        """
        if level is None:
            level = self.current_level
        
        level = np.clip(level, 0, self.levels - 1)
        
        if level == 0:
            return gaussians
        
        target_count = self.target_counts[level] if level < len(self.target_counts) else \
                       gaussians.num_gaussians // (2 ** level)
        
        # Downsample
        if gaussians.num_gaussians <= target_count:
            return gaussians
        
        indices = np.linspace(0, gaussians.num_gaussians - 1, target_count, dtype=np.int32)
        
        return Gaussian(
            centroids=gaussians.centroids[indices],
            covariances=None,
            colors=gaussians.colors[indices],
            opacity=gaussians.opacity[indices],
            scales=gaussians.scales[indices],
            rotations=gaussians.rotations[indices],
        )
    
    def set_level(self, level: int) -> None:
        """Set current LOD level."""
        self.current_level = np.clip(level, 0, self.levels - 1)

