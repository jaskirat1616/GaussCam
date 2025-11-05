"""Gaussian merging utilities including temporal-aware variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from backend.gaussian.fitter import Gaussian
from backend.gaussian.four_d import Gaussian4D, GaussianMotion, TemporalHierarchy


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


@dataclass
class TemporalMergeStats:
    """Statistics captured during temporal merges."""

    spawned_children: int = 0
    pruned_nodes: int = 0
    average_residual: float = 0.0


class TemporalGaussianMerger:
    """Temporal merger managing deformable Gaussians and hierarchy updates."""

    def __init__(
        self,
        hierarchy: TemporalHierarchy,
        residual_threshold: float = 0.02,
        psnr_target: float = 35.0,
        importance_update: Optional[Callable[[float, float], float]] = None,
    ) -> None:
        self.hierarchy = hierarchy
        self.residual_threshold = residual_threshold
        self.psnr_target = psnr_target
        self.importance_update = importance_update

    def accumulate(
        self,
        cluster_id: int,
        new_gaussian: Gaussian4D,
        frame_psnr: float,
        frame_ate: float,
        level: int = 0,
    ) -> TemporalMergeStats:
        cluster = self.hierarchy._clusters.get(cluster_id)
        stats = TemporalMergeStats()

        if cluster is None:
            self.hierarchy.register_cluster(
                gaussian=new_gaussian,
                importance_score=self._evaluate_importance(frame_psnr, frame_ate),
                keyframe_time=float(np.mean(new_gaussian.timestamps)),
                level=level,
            )
            return stats

        base_gaussian = cluster.gaussian
        merged_gaussian, residual = self._merge_gaussian4d(base_gaussian, new_gaussian)

        cluster.gaussian = merged_gaussian
        average_residual = float(np.mean(residual)) if residual.size else 0.0
        stats.average_residual = average_residual

        importance = self._evaluate_importance(frame_psnr, frame_ate)
        cluster.update_importance(
            psnr_delta=importance - self.psnr_target,
            ate_delta=-frame_ate,
        )

        if average_residual > self.residual_threshold:
            child_cluster = self.hierarchy.register_cluster(
                gaussian=new_gaussian,
                importance_score=importance,
                keyframe_time=float(np.max(new_gaussian.timestamps)),
                level=level + 1,
                parent_cluster=cluster_id,
            )
            stats.spawned_children += 1
            cluster.child_ids.append(child_cluster.cluster_id)

        return stats

    def prune(self, target_counts: Optional[dict] = None) -> TemporalMergeStats:
        stats = TemporalMergeStats()
        if target_counts:
            pruned = self.hierarchy.rebalance_levels(target_counts)
            stats.pruned_nodes = len(pruned)
        return stats

    def _merge_gaussian4d(self, base: Gaussian4D, new: Gaussian4D) -> Tuple[Gaussian4D, np.ndarray]:
        gaussian = self._merge_static(base.gaussian, new.gaussian)
        motion = self._blend_motion(base.motion, new.motion)
        timestamps = np.maximum(base.timestamps, new.timestamps)
        hierarchy_ids = base.hierarchy_ids
        parent_ids = base.parent_ids
        residual = self._compute_residual(base, new)

        merged = Gaussian4D(
            gaussian=gaussian,
            motion=motion,
            timestamps=timestamps,
            hierarchy_ids=hierarchy_ids,
            parent_ids=parent_ids,
            deformation=base.deformation or new.deformation,
            residual_error=residual,
        )

        return merged, residual

    def _merge_static(self, base: Gaussian, new: Gaussian) -> Gaussian:
        if base.num_gaussians != new.num_gaussians:
            combined_centroids = np.concatenate([base.centroids, new.centroids], axis=0)
            combined_colors = np.concatenate([base.colors, new.colors], axis=0)
            combined_opacity = np.concatenate([base.opacity, new.opacity], axis=0)
            combined_scales = np.concatenate([base.scales, new.scales], axis=0)
            combined_rotations = np.concatenate([base.rotations, new.rotations], axis=0)
            return Gaussian(
                centroids=combined_centroids,
                covariances=None,
                colors=combined_colors,
                opacity=combined_opacity,
                scales=combined_scales,
                rotations=combined_rotations,
            )

        blend = 0.5
        centroids = blend * base.centroids + (1.0 - blend) * new.centroids
        colors = blend * base.colors + (1.0 - blend) * new.colors
        opacity = np.clip(blend * base.opacity + (1.0 - blend) * new.opacity, 0.0, 1.0)
        scales = blend * base.scales + (1.0 - blend) * new.scales
        rotations = base.rotations

        return Gaussian(
            centroids=centroids.astype(np.float32),
            covariances=None,
            colors=colors.astype(np.float32),
            opacity=opacity.astype(np.float32),
            scales=scales.astype(np.float32),
            rotations=rotations.astype(np.float32),
        )

    def _blend_motion(self, base: GaussianMotion, new: GaussianMotion) -> GaussianMotion:
        alpha = 0.5
        translation = alpha * base.translation + (1 - alpha) * new.translation
        rotation = alpha * base.rotation_axis_angle + (1 - alpha) * new.rotation_axis_angle
        scale = alpha * base.scale_velocity + (1 - alpha) * new.scale_velocity

        if base.deformation_weights is None and new.deformation_weights is None:
            deformation = None
        else:
            base_weights = base.deformation_weights or np.zeros_like(new.deformation_weights)
            new_weights = new.deformation_weights or np.zeros_like(base_weights)
            deformation = alpha * base_weights + (1 - alpha) * new_weights

        return GaussianMotion(
            translation=translation,
            rotation_axis_angle=rotation,
            scale_velocity=scale,
            deformation_weights=deformation,
        )

    def _compute_residual(self, base: Gaussian4D, new: Gaussian4D) -> np.ndarray:
        position_residual = np.linalg.norm(base.gaussian.centroids - new.gaussian.centroids, axis=-1)
        color_residual = np.linalg.norm(base.gaussian.colors - new.gaussian.colors, axis=-1)
        motion_residual = np.linalg.norm(base.motion.translation - new.motion.translation, axis=-1)
        residual = 0.5 * position_residual + 0.3 * color_residual + 0.2 * motion_residual
        return residual.astype(np.float32)

    def _evaluate_importance(self, psnr: float, ate: float) -> float:
        if self.importance_update is not None:
            return float(self.importance_update(psnr, ate))
        return psnr - ate * 10.0

