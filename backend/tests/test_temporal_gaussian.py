import numpy as np

from backend.gaussian.fitter import Gaussian
from backend.gaussian.four_d import Gaussian4D, GaussianMotion, TemporalHierarchy
from backend.utils.optimization import optimize_gaussian4d
from backend.scene.temporal_graph import TemporalGraph


def _make_dummy_gaussian(count: int = 8) -> Gaussian:
    centroids = np.random.rand(count, 3).astype(np.float32)
    colors = np.random.rand(count, 3).astype(np.float32)
    opacity = np.ones(count, dtype=np.float32)
    scales = np.ones((count, 3), dtype=np.float32) * 0.01
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    return Gaussian(
        centroids=centroids,
        covariances=None,
        colors=colors,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
    )


def _make_motion(count: int = 8) -> GaussianMotion:
    return GaussianMotion(
        translation=np.zeros((count, 3), dtype=np.float32),
        rotation_axis_angle=np.zeros((count, 3), dtype=np.float32),
        scale_velocity=np.zeros((count, 3), dtype=np.float32),
    )


def test_optimize_gaussian4d_registers_cluster():
    base = _make_dummy_gaussian()
    motion = _make_motion()
    timestamps = np.zeros(base.num_gaussians, dtype=np.float32)
    hierarchy_ids = np.zeros(base.num_gaussians, dtype=np.int32)
    parent_ids = -np.ones(base.num_gaussians, dtype=np.int32)

    gaussian4d = Gaussian4D(
        gaussian=base,
        motion=motion,
        timestamps=timestamps,
        hierarchy_ids=hierarchy_ids,
        parent_ids=parent_ids,
    )

    temporal_graph = TemporalGraph()
    hierarchy = TemporalHierarchy()

    metrics = optimize_gaussian4d(
        gaussian4d=gaussian4d,
        temporal_graph=temporal_graph,
        hierarchy=hierarchy,
        frame_psnr=30.0,
        frame_ate=0.02,
    )

    assert "cluster_id" in metrics
    assert len(list(hierarchy.iter_clusters())) == 1


