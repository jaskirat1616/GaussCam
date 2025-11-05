"""
Input Validation

Comprehensive input validation for production use.
"""

from pathlib import Path
from typing import Union, Optional, List
import numpy as np
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Validation error."""
    pass


def validate_file_path(
    path: Union[str, Path],
    must_exist: bool = True,
    allowed_extensions: Optional[List[str]] = None,
) -> Path:
    """
    Validate file path.
    
    Args:
        path: File path to validate
        must_exist: Whether file must exist
        allowed_extensions: List of allowed file extensions (e.g., ['.mp4', '.avi'])
    
    Returns:
        Path object
    
    Raises:
        ValidationError: If validation fails
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {path}")
    
    if not must_exist and path.exists():
        raise ValidationError(f"File already exists: {path}")
    
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValidationError(
                f"File extension not allowed: {path.suffix}. "
                f"Allowed: {allowed_extensions}"
            )
    
    return path


def validate_video_file(path: Union[str, Path]) -> Path:
    """
    Validate video file path.
    
    Args:
        path: Video file path
    
    Returns:
        Path object
    
    Raises:
        ValidationError: If validation fails
    """
    return validate_file_path(
        path,
        must_exist=True,
        allowed_extensions=['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']
    )


def validate_resolution(width: int, height: int) -> tuple:
    """
    Validate resolution values.
    
    Args:
        width: Width in pixels
        height: Height in pixels
    
    Returns:
        (width, height) tuple
    
    Raises:
        ValidationError: If validation fails
    """
    if width < 1 or width > 7680:  # Max 8K
        raise ValidationError(f"Invalid width: {width}. Must be between 1 and 7680")
    
    if height < 1 or height > 4320:  # Max 8K
        raise ValidationError(f"Invalid height: {height}. Must be between 1 and 4320")
    
    return (width, height)


def validate_frame(frame: np.ndarray) -> np.ndarray:
    """
    Validate video frame.
    
    Args:
        frame: Frame as numpy array
    
    Returns:
        Validated frame
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(frame, np.ndarray):
        raise ValidationError(f"Frame must be numpy array, got {type(frame)}")
    
    if len(frame.shape) != 3:
        raise ValidationError(f"Frame must be 3D (H, W, C), got shape {frame.shape}")
    
    h, w, c = frame.shape
    
    if c not in [1, 3, 4]:
        raise ValidationError(f"Frame must have 1, 3, or 4 channels, got {c}")
    
    if h < 1 or w < 1:
        raise ValidationError(f"Frame dimensions must be positive, got {h}x{w}")
    
    if frame.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValidationError(f"Frame dtype must be uint8, float32, or float64, got {frame.dtype}")
    
    return frame


def validate_webcam_device(device_id: int) -> int:
    """
    Validate webcam device ID.
    
    Args:
        device_id: Webcam device ID
    
    Returns:
        Device ID
    
    Raises:
        ValidationError: If validation fails
    """
    if device_id < 0:
        raise ValidationError(f"Device ID must be non-negative, got {device_id}")
    
    if device_id > 100:  # Reasonable upper limit
        raise ValidationError(f"Device ID too large: {device_id}")
    
    return device_id


def validate_performance_mode(mode: str) -> str:
    """
    Validate performance mode.
    
    Args:
        mode: Performance mode string
    
    Returns:
        Validated mode
    
    Raises:
        ValidationError: If validation fails
    """
    valid_modes = ['Fast', 'Balanced', 'Quality']
    if mode not in valid_modes:
        raise ValidationError(
            f"Invalid performance mode: {mode}. Must be one of {valid_modes}"
        )
    return mode


def validate_gaussian_count(count: int, max_count: int = 1000000) -> int:
    """
    Validate Gaussian count.
    
    Args:
        count: Number of Gaussians
        max_count: Maximum allowed count
    
    Returns:
        Validated count
    
    Raises:
        ValidationError: If validation fails
    """
    if count < 0:
        raise ValidationError(f"Gaussian count must be non-negative, got {count}")
    
    if count > max_count:
        raise ValidationError(
            f"Gaussian count exceeds maximum: {count} > {max_count}"
        )
    
    return count


__all__ = [
    'ValidationError',
    'validate_file_path',
    'validate_video_file',
    'validate_resolution',
    'validate_frame',
    'validate_webcam_device',
    'validate_performance_mode',
    'validate_gaussian_count',
]

