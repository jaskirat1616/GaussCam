"""
GPU Detection Utility

Auto-detects CUDA (NVIDIA) or MPS (Apple Silicon) and provides backend selection.
"""

import platform
import sys
from typing import Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


class GPUBackend:
    """GPU backend types."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class GPUDetector:
    """Detects available GPU backend and provides device selection."""
    
    def __init__(self):
        self.backend: Optional[str] = None
        self.device: Optional[torch.device] = None
        self.device_name: Optional[str] = None
        self._detect_backend()
    
    def _detect_backend(self) -> None:
        """Detect available GPU backend."""
        if torch is None:
            print("Warning: PyTorch not installed. Falling back to CPU.")
            self.backend = GPUBackend.CPU
            self.device = None
            return
        
        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            self.backend = GPUBackend.CUDA
            self.device = torch.device("cuda")
            self.device_name = torch.cuda.get_device_name(0)
            print(f"CUDA backend detected: {self.device_name}")
            return
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.backend = GPUBackend.MPS
            self.device = torch.device("mps")
            self.device_name = "Apple Silicon (MPS)"
            print(f"MPS backend detected: {self.device_name}")
            return
        
        # Fallback to CPU
        self.backend = GPUBackend.CPU
        self.device = torch.device("cpu")
        self.device_name = "CPU"
        print("No GPU detected. Falling back to CPU.")
    
    def get_device(self) -> torch.device:
        """Get the appropriate PyTorch device."""
        if self.device is None:
            return torch.device("cpu")
        return self.device
    
    def get_backend(self) -> str:
        """Get the backend type as string."""
        return self.backend or GPUBackend.CPU
    
    def is_cuda(self) -> bool:
        """Check if CUDA backend is available."""
        return self.backend == GPUBackend.CUDA
    
    def is_mps(self) -> bool:
        """Check if MPS backend is available."""
        return self.backend == GPUBackend.MPS
    
    def is_cpu(self) -> bool:
        """Check if using CPU backend."""
        return self.backend == GPUBackend.CPU
    
    def get_device_info(self) -> dict:
        """Get detailed device information."""
        info = {
            "backend": self.backend,
            "device_name": self.device_name,
            "platform": platform.system(),
            "architecture": platform.machine(),
        }
        
        if self.is_cuda() and torch:
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        
        return info


# Global detector instance
_detector: Optional[GPUDetector] = None


def get_gpu_detector() -> GPUDetector:
    """Get or create the global GPU detector instance."""
    global _detector
    if _detector is None:
        _detector = GPUDetector()
    return _detector


def get_device() -> torch.device:
    """Get the appropriate PyTorch device for the current system."""
    return get_gpu_detector().get_device()


def get_backend() -> str:
    """Get the backend type (cuda, mps, or cpu)."""
    return get_gpu_detector().get_backend()


def is_cuda() -> bool:
    """Check if CUDA backend is available."""
    return get_gpu_detector().is_cuda()


def is_mps() -> bool:
    """Check if MPS backend is available."""
    return get_gpu_detector().is_mps()


def is_cpu() -> bool:
    """Check if using CPU backend."""
    return get_gpu_detector().is_cpu()


if __name__ == "__main__":
    """Test GPU detection."""
    print("=" * 60)
    print("GaussCam GPU Detection")
    print("=" * 60)
    
    detector = GPUDetector()
    info = detector.get_device_info()
    
    print("\nDevice Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nBackend: {detector.get_backend()}")
    print(f"Device: {detector.get_device()}")
    print(f"CUDA Available: {detector.is_cuda()}")
    print(f"MPS Available: {detector.is_mps()}")
    print(f"Using CPU: {detector.is_cpu()}")
    print("=" * 60)

