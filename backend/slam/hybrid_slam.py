"""Hybrid SLAM stack integrating depth estimation, RAFT tracking, and temporal graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import cv2

from backend.depth.depth_anything_v2_wrapper import DepthAnythingV2Estimator
from backend.gaussian.four_d import Gaussian4D, GaussianMotion, TemporalHierarchy
from backend.instant.initializer import InstantGaussianInitializer, ViewSample
from backend.scene.temporal_graph import TemporalGraph
from backend.utils.logging_config import get_logger
from backend.utils.optimization import dynamic_pixel_downsampling, optimize_gaussian4d
from backend.utils.tracking import RAFTTracker, SimpleTracker


logger = get_logger(__name__)


@dataclass
class FrameState:
    """State bundle produced for each processed frame."""

    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    flow: Optional[np.ndarray]
    importance: float


class HybridSLAM:
    """Hybrid SLAM pipeline orchestrating depth, tracking, and Gaussian updates."""

    def __init__(
        self,
        model_size: str = "large",
        use_transformers: bool = True,
        target_pixels: int = 640 * 480,
        depth_estimator: Optional[DepthAnythingV2Estimator] = None,
        tracker: Optional[object] = None,
    ) -> None:
        self.depth_estimator = depth_estimator or DepthAnythingV2Estimator(
            model_size=model_size,
            use_transformers=use_transformers,
        )

        if tracker is not None:
            self.tracker = tracker
        else:
            try:
                self.tracker = RAFTTracker()
                logger.info("Using RAFT tracker for SLAM pipeline")
            except Exception as error:
                logger.warning(f"RAFT unavailable ({error}); falling back to SimpleTracker")
                self.tracker = SimpleTracker()

        self.temporal_graph = TemporalGraph()
        self.hierarchy = TemporalHierarchy()
        self.instant_initializer = InstantGaussianInitializer()
        self.target_pixels = target_pixels
        self.prev_rgb: Optional[np.ndarray] = None
        self.prev_depth: Optional[np.ndarray] = None
        self.prev_pose: Optional[np.ndarray] = None
        self.prev_timestamp: Optional[float] = None
        self.current_pixels = target_pixels

    def process_frame(
        self,
        rgb: np.ndarray,
        timestamp: float,
        intrinsics,
        gpu_utilization: float = 0.5,
    ) -> FrameState:
        """Process an incoming frame, updating SLAM state and returning frame metadata."""

        h, w = rgb.shape[:2]
        self.current_pixels = dynamic_pixel_downsampling(self.current_pixels, gpu_utilization)

        resize_needed = (h * w) > self.current_pixels
        if resize_needed:
            scale = np.sqrt(self.current_pixels / (h * w))
            target_size = (max(16, int(w * scale)), max(16, int(h * scale)))
            rgb_down = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
        else:
            rgb_down = rgb

        depth_est = self.depth_estimator.estimate_depth(rgb_down, postprocess=True, postprocess_method="improved")

        if resize_needed:
            depth = cv2.resize(depth_est, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            depth = depth_est

        pose = np.eye(4, dtype=np.float32)
        flow = None

        if self.prev_rgb is not None:
            flow, pose = self.tracker.track(
                self.prev_rgb,
                rgb,
                intrinsics=intrinsics,
                depth1=self.prev_depth,
                depth2=depth,
            )

        node = self.temporal_graph.add_node(
            timestamp=timestamp,
            pose=pose,
            gaussian_cluster=None,
        )

        if flow is not None:
            avg_flow = float(np.linalg.norm(flow.reshape(-1, 2), axis=1).mean())
        else:
            avg_flow = 0.0

        importance = 1.0 / (1.0 + np.exp(-avg_flow))
        node.importance = importance

        self.prev_rgb = rgb
        self.prev_depth = depth
        self.prev_pose = pose
        self.prev_timestamp = timestamp

        return FrameState(
            timestamp=timestamp,
            rgb=rgb,
            depth=depth,
            pose=pose,
            flow=flow,
            importance=importance,
        )

    def initialize_from_views(
        self,
        views: Sequence[FrameState],
        intrinsics,
    ) -> Gaussian4D:
        view_samples = [ViewSample(rgb=view.rgb, depth=view.depth) for view in views]
        gaussian = self.instant_initializer.initialize(view_samples, intrinsics=intrinsics)
        return self.build_gaussian4d(gaussian)

    def integrate_gaussians(
        self,
        gaussian4d: Gaussian4D,
        frame_psnr: float,
        frame_ate: float,
        target_counts: Optional[dict] = None,
    ) -> dict:
        metrics = optimize_gaussian4d(
            gaussian4d=gaussian4d,
            temporal_graph=self.temporal_graph,
            hierarchy=self.hierarchy,
            frame_psnr=frame_psnr,
            frame_ate=frame_ate,
            target_counts=target_counts,
        )

        # Attach cluster to latest node for downstream consumers
        latest_nodes = self.temporal_graph.get_latest_nodes(count=1)
        if latest_nodes and "cluster_id" in metrics:
            latest_nodes[0].gaussian_cluster = int(metrics["cluster_id"])

        return metrics

    def build_gaussian4d(
        self,
        gaussian,
        velocity_scale: float = 1.0,
    ) -> Gaussian4D:
        count = gaussian.num_gaussians
        motion = GaussianMotion(
            translation=np.zeros((count, 3), dtype=np.float32),
            rotation_axis_angle=np.zeros((count, 3), dtype=np.float32),
            scale_velocity=np.zeros((count, 3), dtype=np.float32),
        )
        timestamps = np.zeros(count, dtype=np.float32)
        hierarchy_ids = np.zeros(count, dtype=np.int32)
        parent_ids = -np.ones(count, dtype=np.int32)

        return Gaussian4D(
            gaussian=gaussian,
            motion=motion,
            timestamps=timestamps,
            hierarchy_ids=hierarchy_ids,
            parent_ids=parent_ids,
        )


