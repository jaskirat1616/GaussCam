"""
Progress Tracking

Progress tracking for long-running operations.
"""

import time
from typing import Optional, Callable
from dataclasses import dataclass
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProgressInfo:
    """Progress information."""
    current: int
    total: Optional[int]
    percentage: float
    elapsed_time: float
    estimated_remaining: Optional[float] = None
    rate: Optional[float] = None  # items per second


class ProgressTracker:
    """Progress tracker for operations."""
    
    def __init__(
        self,
        total: Optional[int] = None,
        update_interval: float = 0.5,  # Update every 0.5 seconds
        callback: Optional[Callable[[ProgressInfo], None]] = None,
    ):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items (None for indefinite)
            update_interval: Minimum time between updates (seconds)
            callback: Callback function for progress updates
        """
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.update_interval = update_interval
        self.callback = callback
        self.last_value = 0
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Throttle updates
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Calculate metrics
        percentage = (self.current / self.total * 100) if self.total else 0.0
        rate = self.current / elapsed if elapsed > 0 else 0.0
        
        estimated_remaining = None
        if self.total and rate > 0:
            remaining_items = self.total - self.current
            estimated_remaining = remaining_items / rate
        
        info = ProgressInfo(
            current=self.current,
            total=self.total,
            percentage=percentage,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining,
            rate=rate,
        )
        
        # Call callback if provided
        if self.callback:
            self.callback(info)
        
        # Log progress
        if self.total:
            logger.info(
                f"Progress: {self.current}/{self.total} ({percentage:.1f}%) - "
                f"Rate: {rate:.1f}/s - Elapsed: {elapsed:.1f}s"
            )
        else:
            logger.info(f"Progress: {self.current} items - Rate: {rate:.1f}/s - Elapsed: {elapsed:.1f}s")
    
    def finish(self) -> None:
        """Mark progress as finished."""
        elapsed = time.time() - self.start_time
        logger.info(f"Completed: {self.current} items in {elapsed:.2f} seconds")
    
    def reset(self) -> None:
        """Reset progress tracker."""
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = time.time()


class FrameProgressTracker(ProgressTracker):
    """Specialized progress tracker for frame processing."""
    
    def __init__(self, total_frames: Optional[int] = None, **kwargs):
        """Initialize frame progress tracker."""
        super().__init__(total=total_frames, **kwargs)
        self.frames_processed = 0
        self.frames_skipped = 0
    
    def update_frame(self, processed: bool = True) -> None:
        """Update frame progress."""
        if processed:
            self.frames_processed += 1
        else:
            self.frames_skipped += 1
        self.update(1)
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        elapsed = time.time() - self.start_time
        return {
            'total_frames': self.current,
            'processed': self.frames_processed,
            'skipped': self.frames_skipped,
            'elapsed_time': elapsed,
            'fps': self.frames_processed / elapsed if elapsed > 0 else 0.0,
        }


__all__ = ['ProgressTracker', 'FrameProgressTracker', 'ProgressInfo']

