"""
Render Widget

PySide6 widget for Gaussian Splatting rendering display.
"""

import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtCore import QTimer, Qt, Signal
from typing import Optional


class RenderWidget(QWidget):
    """Widget for displaying rendered Gaussian Splatting output."""
    
    frame_updated = Signal(np.ndarray)  # Signal emitted when frame is updated
    
    def __init__(self, parent=None, width: int = 640, height: int = 480):
        """
        Initialize render widget.
        
        Args:
            parent: Parent widget
            width: Display width
            height: Display height
        """
        super().__init__(parent)
        # Don't store width/height as attributes - they shadow QWidget methods
        # Use setMinimumSize instead
        self.current_frame: Optional[np.ndarray] = None
        self.setMinimumSize(width, height)
        
        # Update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(33)  # ~30 FPS update rate
    
    def set_frame(self, frame: np.ndarray) -> None:
        """
        Set rendered frame to display.
        
        Args:
            frame: Rendered image (H, W, 3) RGB in [0, 1] or [0, 255]
        """
        # Normalize to [0, 255] if needed
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        
        # Ensure correct shape
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            self.current_frame = frame
            self.frame_updated.emit(frame)
            self.update()
    
    def paintEvent(self, event) -> None:
        """Paint event handler."""
        if self.current_frame is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Convert numpy array to QImage
        h, w, c = self.current_frame.shape
        qimage = QImage(
            self.current_frame.data,
            w,
            h,
            w * c,
            QImage.Format.Format_RGB888,
        )
        
        # Scale to widget size
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        
        # Draw centered
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)
    
    def resizeEvent(self, event) -> None:
        """Resize event handler."""
        # Note: Don't override self.width/height as they conflict with QWidget methods
        # Store as separate attributes if needed
        super().resizeEvent(event)

