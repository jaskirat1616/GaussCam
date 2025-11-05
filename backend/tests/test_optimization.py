"""Tests for optimization utilities."""

import numpy as np
import torch

from backend.gaussian.fitter import GaussianFitter
from backend.utils.optimization import batch_process_gaussians


def _make_gaussian(fitter: GaussianFitter, num_points: int) -> object:
    points = np.random.randn(num_points, 3).astype(np.float32)
    colors = np.random.rand(num_points, 3).astype(np.float32)
    return fitter.fit(points, colors, method="uniform")


def test_batch_process_gaussians_padding_and_masking():
    fitter = GaussianFitter()
    g1 = _make_gaussian(fitter, 120)
    g2 = _make_gaussian(fitter, 45)

    batches = batch_process_gaussians([g1, g2], batch_size=2, device=torch.device("cpu"))

    assert len(batches) == 1
    batch = batches[0]

    assert batch["centroids"].shape[0] == 2
    assert batch["centroids"].shape[1] == g1.num_gaussians
    assert batch["mask"].dtype == torch.bool
    assert batch["mask"][0].sum().item() == g1.num_gaussians
    assert batch["mask"][1].sum().item() == g2.num_gaussians
    assert batch["counts"].tolist() == [g1.num_gaussians, g2.num_gaussians]


def test_batch_process_gaussians_empty():
    assert batch_process_gaussians([], batch_size=4) == []

