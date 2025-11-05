"""
Tests for GPU Detection
"""

from backend.utils.gpu_detection import (
    GPUBackend,
    GPUDetector,
    get_gpu_detector,
    get_device,
    is_cuda,
    is_mps,
    reset_gpu_detector,
)


def test_gpu_detector_initialization():
    """Test GPU detector initialization."""
    detector = GPUDetector()
    assert detector is not None
    assert detector.backend in ["cuda", "mps", "cpu"]
    assert detector.device_name is not None


def test_get_gpu_detector():
    """Test global GPU detector."""
    detector1 = get_gpu_detector()
    detector2 = get_gpu_detector()
    assert detector1 is detector2  # Should be singleton


def test_get_device():
    """Test device getter."""
    device = get_device()
    assert device is not None


def test_backend_checkers():
    """Test backend checker functions."""
    # These should not raise errors
    is_cuda()
    is_mps()


def test_device_info():
    """Test device info retrieval."""
    detector = GPUDetector()
    info = detector.get_device_info()
    assert "backend" in info
    assert "device_name" in info
    assert "platform" in info


def test_available_backends_contains_cpu():
    detector = GPUDetector()
    assert GPUBackend.CPU in detector.get_available_backends()


def test_forced_cpu_backend(monkeypatch):
    monkeypatch.setenv("GAUSSCAM_BACKEND", "cpu")
    reset_gpu_detector()
    detector = get_gpu_detector()
    assert detector.get_backend() == GPUBackend.CPU
    assert detector.device_name == "CPU"
    # Cleanup
    monkeypatch.delenv("GAUSSCAM_BACKEND", raising=False)
    reset_gpu_detector()

