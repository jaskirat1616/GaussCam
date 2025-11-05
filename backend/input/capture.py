"""
Webcam and Video Capture Module

Handles live webcam input and offline video file reading with preprocessing.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Iterator, Callable
from queue import Queue, Empty
from threading import Thread, Event
import time


class FrameCapture:
    """Base class for frame capture (webcam or video file)."""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize frame capture.
        
        Args:
            width: Target frame width
            height: Target frame height
            fps: Target frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.is_running = False
        self.frame_count = 0
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame. Returns (success, frame)."""
        raise NotImplementedError
    
    def release(self) -> None:
        """Release capture resources."""
        raise NotImplementedError
    
    def get_fps(self) -> float:
        """Get actual FPS."""
        raise NotImplementedError


class WebcamCapture(FrameCapture):
    """Webcam capture using OpenCV."""
    
    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        """
        Initialize webcam capture.
        
        Args:
            device_id: Camera device ID (usually 0)
            width: Target frame width
            height: Target frame height
            fps: Target frames per second
        """
        super().__init__(width, height, fps)
        self.device_id = device_id
        self.cap: Optional[cv2.VideoCapture] = None
        self._open()
    
    def _open(self) -> None:
        """Open webcam device."""
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam device {self.device_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Webcam opened: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from webcam."""
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            # Resize if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
        return ret, frame
    
    def release(self) -> None:
        """Release webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_fps(self) -> float:
        """Get actual FPS."""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)


class VideoCapture(FrameCapture):
    """Video file capture using OpenCV."""
    
    def __init__(
        self,
        video_path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        loop: bool = False,
    ):
        """
        Initialize video file capture.
        
        Args:
            video_path: Path to video file
            width: Target frame width (None to keep original)
            height: Target frame height (None to keep original)
            loop: Whether to loop the video
        """
        super().__init__(width or 640, height or 480, 30)
        self.video_path = video_path
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self._open()
    
    def _open(self) -> None:
        """Open video file."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.width is None:
            self.width = actual_width
        if self.height is None:
            self.height = actual_height
        
        print(f"Video opened: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")
        print(f"Total frames: {self.total_frames}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video."""
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret and self.loop:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            # Resize if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
        
        return ret, frame
    
    def release(self) -> None:
        """Release video."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_fps(self) -> float:
        """Get video FPS."""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_progress(self) -> float:
        """Get playback progress (0.0 to 1.0)."""
        if self.total_frames == 0:
            return 0.0
        return self.frame_count / self.total_frames


class FramePreprocessor:
    """Frame preprocessing pipeline."""
    
    def __init__(
        self,
        normalize: bool = True,
        denoise: bool = False,
        to_rgb: bool = True,
    ):
        """
        Initialize preprocessor.
        
        Args:
            normalize: Normalize to [0, 1] range
            denoise: Apply denoising
            to_rgb: Convert BGR to RGB
        """
        self.normalize = normalize
        self.denoise = denoise
        self.to_rgb = to_rgb
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            Processed frame
        """
        # Convert BGR to RGB
        if self.to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Denoise
        if self.denoise:
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        # Normalize to [0, 1]
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame


class AsyncFrameCapture:
    """Asynchronous frame capture with queue for processing pipeline."""
    
    def __init__(
        self,
        capture: FrameCapture,
        preprocessor: Optional[FramePreprocessor] = None,
        queue_size: int = 2,
    ):
        """
        Initialize async capture.
        
        Args:
            capture: Frame capture instance
            preprocessor: Optional frame preprocessor
            queue_size: Maximum queue size
        """
        self.capture = capture
        self.preprocessor = preprocessor or FramePreprocessor()
        self.queue = Queue(maxsize=queue_size)
        self.stop_event = Event()
        self.thread: Optional[Thread] = None
        self.is_running = False
    
    def start(self) -> None:
        """Start async capture thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        """Stop async capture thread."""
        self.is_running = False
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.capture.release()
    
    def _capture_loop(self) -> None:
        """Capture loop running in background thread."""
        while not self.stop_event.is_set():
            ret, frame = self.capture.read()
            
            if not ret:
                break
            
            # Preprocess frame
            processed = self.preprocessor.process(frame)
            
            # Put in queue (non-blocking, drop frame if queue full)
            try:
                self.queue.put_nowait(processed)
            except:
                # Queue full, skip frame
                pass
    
    def read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Read next processed frame from queue.
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            Processed frame or None if timeout
        """
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None
    
    def has_frames(self) -> bool:
        """Check if queue has frames."""
        return not self.queue.empty()


def list_webcam_devices(max_devices: int = 10) -> list:
    """
    List available webcam devices.
    
    Args:
        max_devices: Maximum number of devices to check
    
    Returns:
        List of available device IDs
    """
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

