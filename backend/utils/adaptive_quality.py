"""
Adaptive Quality Management

Dynamically adjusts quality settings based on performance metrics.
"""

import time
from typing import Dict, Optional
from collections import deque


class AdaptiveQualityManager:
    """Manages adaptive quality settings based on performance."""
    
    def __init__(
        self,
        target_fps: float = 15.0,
        min_fps: float = 10.0,
        max_fps: float = 30.0,
        history_size: int = 10,
    ):
        """
        Initialize adaptive quality manager.
        
        Args:
            target_fps: Target FPS to maintain
            min_fps: Minimum acceptable FPS
            max_fps: Maximum FPS before increasing quality
            history_size: Number of FPS measurements to track
        """
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.fps_history = deque(maxlen=history_size)
        self.frame_times = deque(maxlen=history_size)
        self.last_time = time.time()
        self.current_settings = self._get_default_settings()
    
    def update(self, frame_time: Optional[float] = None) -> None:
        """
        Update performance metrics.
        
        Args:
            frame_time: Time taken to process frame (seconds)
        """
        current_time = time.time()
        if frame_time is None:
            frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        if frame_time > 0:
            fps = 1.0 / frame_time
            self.fps_history.append(fps)
    
    def get_current_fps(self) -> float:
        """Get current average FPS."""
        if len(self.fps_history) == 0:
            return self.target_fps
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_optimal_settings(self) -> Dict:
        """
        Get optimal settings based on current performance.
        
        Returns:
            Dictionary with optimized settings
        """
        current_fps = self.get_current_fps()
        
        if current_fps < self.min_fps:
            # Too slow - reduce quality aggressively
            return {
                "depth_skip_frames": 20,
                "frame_skip": 10,
                "max_gaussians": 2000,
                "target_size": 256,
                "target_pixels": 15000,
                "voxel_size": 0.15,
                "interpolation": "linear",
            }
        elif current_fps < self.target_fps * 0.8:
            # Slow - reduce quality
            return {
                "depth_skip_frames": 15,
                "frame_skip": 8,
                "max_gaussians": 3000,
                "target_size": 288,
                "target_pixels": 18000,
                "voxel_size": 0.12,
                "interpolation": "linear",
            }
        elif current_fps > self.max_fps:
            # Fast enough - can increase quality
            return {
                "depth_skip_frames": 5,
                "frame_skip": 2,
                "max_gaussians": 8000,
                "target_size": 384,
                "target_pixels": 30000,
                "voxel_size": 0.08,
                "interpolation": "bilinear",
            }
        else:
            # Balanced
            return self._get_default_settings()
    
    def _get_default_settings(self) -> Dict:
        """Get default balanced settings."""
        return {
            "depth_skip_frames": 10,
            "frame_skip": 5,
            "max_gaussians": 5000,
            "target_size": 320,
            "target_pixels": 20000,
            "voxel_size": 0.10,
            "interpolation": "linear",
        }
    
    def should_adapt(self) -> bool:
        """Check if settings should be adapted."""
        if len(self.fps_history) < 3:
            return False
        
        current_fps = self.get_current_fps()
        return current_fps < self.min_fps or current_fps > self.max_fps

