import json
import numpy as np

from backend.gaussian.fitter import Gaussian
from backend.compress import compress_gaussians


def _mock_gaussian(count: int = 16) -> Gaussian:
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


def test_compress_gaussians_outputs_payload(tmp_path):
    gaussian = _mock_gaussian()
    importance = np.ones(gaussian.num_gaussians, dtype=np.float32)
    payload = compress_gaussians(gaussian, importance_scores=importance, target_count=8)

    assert payload["stats"].pruned >= 0
    assert payload["quantized"]["centroids"].dtype == np.float16

    export_path = tmp_path / "compressed.json"
    from backend.utils.export import export_compressed

    export_compressed(gaussian, str(export_path), target_count=8)
    data = json.loads(export_path.read_text())
    assert "quantized" in data

