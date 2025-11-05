import numpy as np

from backend.instant.initializer import InstantGaussianInitializer, ViewSample
from backend.utils.camera import CameraIntrinsics


def test_instant_initializer_creates_gaussians():
    initializer = InstantGaussianInitializer(max_gaussians=128)

    rgb = np.ones((64, 64, 3), dtype=np.uint8) * 127
    depth = np.linspace(0.1, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)

    intrinsics = CameraIntrinsics.default(width=64, height=64, fov=60.0)

    gaussian = initializer.initialize(
        views=[ViewSample(rgb=rgb, depth=depth)],
        intrinsics=intrinsics,
    )

    assert gaussian.num_gaussians > 0

