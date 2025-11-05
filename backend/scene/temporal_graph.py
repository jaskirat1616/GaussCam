"""Temporal scene graph for dynamic Gaussian splatting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class TemporalNode:
    """Represents a captured frame/keyframe within the SLAM graph."""

    node_id: int
    timestamp: float
    pose: np.ndarray  # (4, 4)
    gaussian_cluster: Optional[int] = None
    importance: float = 1.0
    residual_psnr: float = 0.0
    residual_ate: float = 0.0
    observations: Dict[str, np.ndarray] = field(default_factory=dict)

    def update_metrics(self, psnr: float, ate: float, momentum: float = 0.9) -> None:
        self.residual_psnr = momentum * self.residual_psnr + (1.0 - momentum) * psnr
        self.residual_ate = momentum * self.residual_ate + (1.0 - momentum) * ate
        self.importance = self.residual_psnr - 10.0 * self.residual_ate


@dataclass
class TemporalEdge:
    """Pose constraint between two nodes."""

    from_id: int
    to_id: int
    relative_pose: np.ndarray  # (4, 4)
    information: np.ndarray  # (6, 6)
    is_loop_closure: bool = False


class TemporalGraph:
    """Maintains temporal nodes/edges for dynamic SLAM integration."""

    def __init__(self) -> None:
        self._nodes: Dict[int, TemporalNode] = {}
        self._edges: List[TemporalEdge] = []
        self._next_node_id = 0

    def add_node(
        self,
        timestamp: float,
        pose: np.ndarray,
        gaussian_cluster: Optional[int] = None,
        observations: Optional[Dict[str, np.ndarray]] = None,
    ) -> TemporalNode:
        node_id = self._next_node_id
        self._next_node_id += 1

        node = TemporalNode(
            node_id=node_id,
            timestamp=timestamp,
            pose=np.asarray(pose, dtype=np.float32),
            gaussian_cluster=gaussian_cluster,
            observations=observations or {},
        )

        self._nodes[node_id] = node
        return node

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        relative_pose: np.ndarray,
        information: Optional[np.ndarray] = None,
        is_loop_closure: bool = False,
    ) -> TemporalEdge:
        information = information if information is not None else np.eye(6, dtype=np.float32)
        edge = TemporalEdge(
            from_id=from_id,
            to_id=to_id,
            relative_pose=np.asarray(relative_pose, dtype=np.float32),
            information=np.asarray(information, dtype=np.float32),
            is_loop_closure=is_loop_closure,
        )
        self._edges.append(edge)
        return edge

    def iter_nodes(self) -> Iterable[TemporalNode]:
        return self._nodes.values()

    def iter_edges(self) -> Iterable[TemporalEdge]:
        return iter(self._edges)

    def get_latest_nodes(self, count: int = 5) -> List[TemporalNode]:
        nodes = sorted(self._nodes.values(), key=lambda n: n.timestamp, reverse=True)
        return nodes[:count]

    def link_cluster(self, node_id: int, cluster_id: int) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].gaussian_cluster = cluster_id

    def prune_low_importance(self, threshold: float) -> List[int]:
        removed: List[int] = []
        for node_id, node in list(self._nodes.items()):
            if node.importance < threshold:
                removed.append(node_id)
                self._nodes.pop(node_id)
        if removed:
            self._edges = [e for e in self._edges if e.from_id not in removed and e.to_id not in removed]
        return removed

    def compute_loop_candidates(self, max_temporal_gap: float = 1.5) -> List[Tuple[int, int]]:
        candidates: List[Tuple[int, int]] = []
        nodes = list(self._nodes.values())
        nodes.sort(key=lambda n: n.timestamp)
        for anchor in nodes:
            for target in nodes:
                if target.node_id <= anchor.node_id:
                    continue
                if target.timestamp - anchor.timestamp < max_temporal_gap:
                    continue
                candidates.append((anchor.node_id, target.node_id))
        return candidates


