"""
GPU Detection Utility

Auto-detects CUDA (NVIDIA) or MPS (Apple Silicon) and provides backend selection.
Supports environment overrides for deterministic behaviour in production.
"""

import os
import platform
from typing import List, Optional

try:
    import torch
except ImportError:  # pragma: no cover - torch may be absent during docs builds
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
        self.device_index: Optional[int] = None
        # Allow forcing backend selection via environment (useful for CI/testing)
        self._force_backend = (os.getenv("GAUSSCAM_BACKEND") or "").strip().lower()
        cuda_device_env = os.getenv("GAUSSCAM_CUDA_DEVICE")
        self._preferred_cuda_index: Optional[int] = None
        if cuda_device_env is not None:
            try:
                self._preferred_cuda_index = int(cuda_device_env)
            except ValueError:
                print(f"Warning: Invalid GAUSSCAM_CUDA_DEVICE value '{cuda_device_env}'. Ignoring.")
        self._detect_backend()
    
    def _detect_backend(self) -> None:
        """Detect available GPU backend."""
        if torch is None:
            print("Warning: PyTorch not installed. Falling back to CPU.")
            self._set_cpu(reason="pytorch-missing")
            return
        
        # Honour forced backend if requested
        if self._apply_forced_backend():
            return

        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            self._set_cuda()
            return
        
        # Check for MPS (Apple Silicon)
        if self._mps_available():
            self._set_mps()
            return
        
        # Fallback to CPU
        self._set_cpu(reason="no-gpu")

    def _apply_forced_backend(self) -> bool:
        """Apply forced backend selection via environment variables."""
        if not self._force_backend:
            return False

        backend = self._force_backend
        if backend not in {GPUBackend.CUDA, GPUBackend.MPS, GPUBackend.CPU}:
            print(f"Warning: Unsupported GAUSSCAM_BACKEND='{backend}'. Ignoring.")
            return False

        try:
            if backend == GPUBackend.CUDA:
                if torch.cuda.is_available():
                    self._set_cuda(index=self._preferred_cuda_index, forced=True)
                    return True
                print("Warning: GAUSSCAM_BACKEND=cuda requested but CUDA is unavailable. Falling back to auto-detect.")
                return False
            if backend == GPUBackend.MPS:
                if self._mps_available():
                    self._set_mps(forced=True)
                    return True
                print("Warning: GAUSSCAM_BACKEND=mps requested but MPS is unavailable. Falling back to auto-detect.")
                return False
            # CPU forced
            self._set_cpu(reason="forced")
            return True
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: Failed to apply forced backend '{backend}': {exc}")
            return False

    def _set_cuda(self, index: Optional[int] = None, forced: bool = False) -> None:
        """Configure CUDA backend."""
        assert torch is not None  # for type checkers
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        device_index = index
        if device_index is None:
            device_index = torch.cuda.current_device()

        # Clamp index into valid range
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("CUDA reported no devices")
        if device_index < 0 or device_index >= device_count:
            print(f"Warning: Requested CUDA device {device_index} out of range (0..{device_count-1}). Using device 0.")
            device_index = 0

        torch.cuda.set_device(device_index)
        self.backend = GPUBackend.CUDA
        self.device = torch.device(f"cuda:{device_index}")
        self.device_name = torch.cuda.get_device_name(device_index)
        self.device_index = device_index
        prefix = "CUDA backend detected" if not forced else "CUDA backend forced"
        print(f"{prefix}: {self.device_name} (device {device_index})")

    def _mps_available(self) -> bool:
        """Return True if PyTorch MPS backend is available and built."""
        if torch is None:
            return False
        has_mps = hasattr(torch.backends, "mps")
        if not has_mps:
            return False
        is_available = torch.backends.mps.is_available()
        # Some builds expose is_built() for clarity
        is_built_fn = getattr(torch.backends.mps, "is_built", None)
        is_built = is_built_fn() if callable(is_built_fn) else True
        return bool(is_available and is_built)

    def _set_mps(self, forced: bool = False) -> None:
        """Configure MPS backend."""
        if not self._mps_available():
            raise RuntimeError("MPS backend not available")
        assert torch is not None
        self.backend = GPUBackend.MPS
        self.device = torch.device("mps")
        self.device_name = "Apple Silicon (MPS)"
        self.device_index = None
        prefix = "MPS backend detected" if not forced else "MPS backend forced"
        print(f"{prefix}: {self.device_name}")

    def _set_cpu(self, reason: str = "") -> None:
        """Configure CPU backend."""
        self.backend = GPUBackend.CPU
        if torch is not None:
            self.device = torch.device("cpu")
        else:
            self.device = None
        suffix = f" ({reason})" if reason else ""
        self.device_name = "CPU"
        self.device_index = None
        print(f"CPU backend selected{suffix}.")
    
    def get_device(self) -> torch.device:
        """Get the appropriate PyTorch device."""
        if self.device is None:
            if torch is None:
                raise RuntimeError("PyTorch is not available, cannot construct device")
            return torch.device("cpu")
        return self.device
    
    def get_backend(self) -> str:
        """Get the backend type as string."""
        return self.backend or GPUBackend.CPU

    def get_available_backends(self) -> List[str]:
        """Return the list of detected, usable backends."""
        backends: List[str] = [GPUBackend.CPU]
        if torch is None:
            return backends
        if torch.cuda.is_available():
            backends.append(GPUBackend.CUDA)
        if self._mps_available():
            backends.append(GPUBackend.MPS)
        return backends
    
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
            "device_index": self.device_index,
        }
        
        if self.is_cuda() and torch:
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        
        return info

    def synchronize(self) -> None:
        """Synchronize the active device (best-effort)."""
        if torch is None:
            return
        if self.is_cuda() and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.is_mps():
            try:
                torch.mps.synchronize()
            except AttributeError:
                pass  # Older PyTorch versions may not expose synchronize


# Global detector instance
_detector: Optional[GPUDetector] = None


def get_gpu_detector() -> GPUDetector:
    """Get or create the global GPU detector instance."""
    global _detector
    if _detector is None:
        _detector = GPUDetector()
    return _detector


def reset_gpu_detector() -> None:
    """Reset the cached GPU detector (primarily for testing)."""
    global _detector
    _detector = None


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

