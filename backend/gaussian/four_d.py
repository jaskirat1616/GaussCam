"""4D Gaussian representations with temporal hierarchy support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from backend.gaussian.fitter import Gaussian
from backend.utils.gpu_detection import get_device


@dataclass
class GaussianMotion:
    """Per-Gaussian motion estimates in SE(3) + scale space."""

    translation: np.ndarray  # (N, 3)
    rotation_axis_angle: np.ndarray  # (N, 3)
    scale_velocity: np.ndarray  # (N, 3)
    deformation_weights: Optional[np.ndarray] = None  # (N, K)

    def __post_init__(self) -> None:
        self.translation = np.asarray(self.translation, dtype=np.float32)
        self.rotation_axis_angle = np.asarray(self.rotation_axis_angle, dtype=np.float32)
        self.scale_velocity = np.asarray(self.scale_velocity, dtype=np.float32)
        if self.deformation_weights is not None:
            self.deformation_weights = np.asarray(self.deformation_weights, dtype=np.float32)

    @property
    def count(self) -> int:
        return self.translation.shape[0]


@dataclass
class GaussianDeformation:
    """Low-rank deformation basis for dynamic Gaussians."""

    basis: np.ndarray  # (K, 3, 3)
    rest_covariances: np.ndarray  # (N, 3, 3)

    def __post_init__(self) -> None:
        self.basis = np.asarray(self.basis, dtype=np.float32)
        self.rest_covariances = np.asarray(self.rest_covariances, dtype=np.float32)


@dataclass
class Gaussian4D:
    """Temporal Gaussian container supporting deformable dynamics."""

    gaussian: Gaussian
    motion: GaussianMotion
    timestamps: np.ndarray  # (N,)
    hierarchy_ids: np.ndarray  # (N,)
    parent_ids: np.ndarray  # (N,)
    deformation: Optional[GaussianDeformation] = None
    residual_error: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float32)
        self.hierarchy_ids = np.asarray(self.hierarchy_ids, dtype=np.int32)
        self.parent_ids = np.asarray(self.parent_ids, dtype=np.int32)
        if self.residual_error is not None:
            self.residual_error = np.asarray(self.residual_error, dtype=np.float32)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        n = self.gaussian.num_gaussians
        if self.motion.count != n:
            raise ValueError(f"Motion count ({self.motion.count}) does not match Gaussian count ({n}).")
        if self.timestamps.shape[0] != n:
            raise ValueError("Timestamp count mismatch.")
        if self.hierarchy_ids.shape[0] != n or self.parent_ids.shape[0] != n:
            raise ValueError("Hierarchy or parent id count mismatch.")
        if self.deformation is not None:
            if self.deformation.rest_covariances.shape[0] != n:
                raise ValueError("Rest covariance count mismatch.")

    def to_torch(self, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Convert representation to torch tensors for GPU kernels."""

        if device is None:
            device = get_device()

        payload = self.gaussian.to_torch(device=device)
        payload.update(
            {
                "timestamps": torch.from_numpy(self.timestamps).float().to(device),
                "translation_velocity": torch.from_numpy(self.motion.translation).float().to(device),
                "rotation_axis_angle": torch.from_numpy(self.motion.rotation_axis_angle).float().to(device),
                "scale_velocity": torch.from_numpy(self.motion.scale_velocity).float().to(device),
                "hierarchy_ids": torch.from_numpy(self.hierarchy_ids).int().to(device),
                "parent_ids": torch.from_numpy(self.parent_ids).int().to(device),
            }
        )

        if self.motion.deformation_weights is not None:
            payload["deformation_weights"] = torch.from_numpy(self.motion.deformation_weights).float().to(device)

        if self.deformation is not None:
            payload["deformation_basis"] = torch.from_numpy(self.deformation.basis).float().to(device)
            payload["rest_covariances"] = torch.from_numpy(self.deformation.rest_covariances).float().to(device)

        if self.residual_error is not None:
            payload["residual_error"] = torch.from_numpy(self.residual_error).float().to(device)

        return payload


@dataclass
class TemporalGaussianCluster:
    """Cluster of Gaussians sharing temporal coherence."""

    cluster_id: int
    gaussian: Gaussian4D
    importance_score: float
    keyframe_time: float
    child_ids: List[int] = field(default_factory=list)

    def update_importance(self, psnr_delta: float, ate_delta: float, smoothing: float = 0.9) -> None:
        metric = 0.5 * (psnr_delta + ate_delta)
        self.importance_score = smoothing * self.importance_score + (1.0 - smoothing) * metric


class TemporalHierarchy:
    """Manages temporal clusters and multi-level pruning queues."""

    def __init__(self) -> None:
        self._clusters: Dict[int, TemporalGaussianCluster] = {}
        self._levels: Dict[int, List[int]] = {}
        self._next_cluster_id = 0

    def register_cluster(
        self,
        gaussian: Gaussian4D,
        importance_score: float,
        keyframe_time: float,
        level: int = 0,
        parent_cluster: Optional[int] = None,
    ) -> TemporalGaussianCluster:
        cluster_id = self._next_cluster_id
        self._next_cluster_id += 1

        cluster = TemporalGaussianCluster(
            cluster_id=cluster_id,
            gaussian=gaussian,
            importance_score=importance_score,
            keyframe_time=keyframe_time,
        )

        if parent_cluster is not None and parent_cluster in self._clusters:
            self._clusters[parent_cluster].child_ids.append(cluster_id)

        self._clusters[cluster_id] = cluster
        self._levels.setdefault(level, []).append(cluster_id)
        return cluster

    def iter_clusters(self, level: Optional[int] = None) -> Iterable[TemporalGaussianCluster]:
        if level is None:
            return (self._clusters[idx] for idx in self._clusters.keys())
        for idx in self._levels.get(level, []):
            yield self._clusters[idx]

    def find_least_important(self, level: int) -> Optional[TemporalGaussianCluster]:
        candidates = self._levels.get(level, [])
        if not candidates:
            return None
        least_id = min(candidates, key=lambda idx: self._clusters[idx].importance_score)
        return self._clusters[least_id]

    def prune_cluster(self, cluster_id: int) -> None:
        if cluster_id not in self._clusters:
            return
        cluster = self._clusters.pop(cluster_id)
        for level, ids in list(self._levels.items()):
            if cluster_id in ids:
                ids.remove(cluster_id)
                if not ids:
                    self._levels.pop(level, None)
        for parent in self._clusters.values():
            if cluster_id in parent.child_ids:
                parent.child_ids.remove(cluster_id)
        for child_id in cluster.child_ids:
            self.prune_cluster(child_id)

    def rebalance_levels(self, target_counts: Dict[int, int]) -> List[int]:
        pruned: List[int] = []
        for level, target in target_counts.items():
            ids = self._levels.get(level, [])
            while len(ids) > target:
                least = self.find_least_important(level)
                if least is None:
                    break
                pruned.append(least.cluster_id)
                self.prune_cluster(least.cluster_id)
                ids = self._levels.get(level, [])
        return pruned


