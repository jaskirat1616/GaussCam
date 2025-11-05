import numpy as np

from backend.slam.hybrid_slam import HybridSLAM


class DummyDepthEstimator:
    def estimate_depth(self, image, postprocess=True, postprocess_method="improved"):
        h, w = image.shape[:2]
        return np.ones((h, w), dtype=np.float32) * 0.5


class DummyTracker:
    def track(self, image1, image2, intrinsics, depth1=None, depth2=None):
        h, w = image2.shape[:2]
        return np.zeros((h, w, 2), dtype=np.float32), np.eye(4, dtype=np.float32)


class DummyIntrinsics:
    fx = fy = 500.0
    cx = cy = 128.0


def test_hybrid_slam_process_frame():
    slam = HybridSLAM(
        model_size="large",
        depth_estimator=DummyDepthEstimator(),
        tracker=DummyTracker(),
    )

    rgb = np.ones((64, 64, 3), dtype=np.uint8) * 128
    state = slam.process_frame(rgb=rgb, timestamp=0.0, intrinsics=DummyIntrinsics())

    assert state.depth.shape == (64, 64)
    assert state.pose.shape == (4, 4)

