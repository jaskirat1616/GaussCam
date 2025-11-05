"""
3D Viewer Widget

Interactive 3D viewer for Gaussian Splatting scene using OpenGL.
"""

import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QOpenGLContext, QSurfaceFormat
from typing import Optional
import math

try:
    from OpenGL import GL
    from OpenGL.GL import (
        glClear, glClearColor, glEnable, glDisable,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
        GL_POINTS, glPointSize, glBegin, glEnd, glVertex3f,
        glColor3f, glMatrixMode, glLoadIdentity, glViewport,
        GL_PROJECTION, GL_MODELVIEW, GLfloat
    )
    try:
        from OpenGL import GLU
        GLU_AVAILABLE = True
    except ImportError:
        GLU_AVAILABLE = False
        logger.warning("PyOpenGL GLU not available. 3D viewer may have limited functionality.")
    OPENGL_AVAILABLE = True
except ImportError as e:
    OPENGL_AVAILABLE = False
    GL = None
    logger.warning(f"PyOpenGL not available. 3D viewer disabled. Install with: pip install PyOpenGL PyOpenGL-accelerate")

from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class Viewer3DWidget(QOpenGLWidget):
    """3D viewer widget for Gaussian Splatting scene."""
    
    def __init__(self, parent=None):
        """Initialize 3D viewer."""
        super().__init__(parent)
        
        if not OPENGL_AVAILABLE:
            logger.warning("OpenGL not available. 3D viewer will not work.")
            self.is_available = False
            return
        
        self.is_available = True
        
        # Set OpenGL format
        # Use Compatibility Profile to support deprecated fixed-function pipeline
        # (glMatrixMode, glBegin, glEnd, etc.)
        fmt = QSurfaceFormat()
        fmt.setVersion(2, 1)  # Use OpenGL 2.1 for compatibility
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)  # Compatibility profile
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)  # Anti-aliasing
        self.setFormat(fmt)
        
        # Camera parameters
        self.camera_distance = 5.0
        self.camera_angle_x = 0.0
        self.camera_angle_y = 0.0
        self.camera_target = np.array([0.0, 0.0, 0.0])
        
        # Mouse interaction
        self.last_mouse_pos = None
        self.is_mouse_pressed = False
        
        # Gaussian data
        self.gaussians = None
        self.positions = None
        self.colors = None
        self.max_points = 50000  # Limit for performance in real-time
        
        # Update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(33)  # ~30 FPS
        
        # Enable mouse tracking
        self.setMouseTracking(True)
    
    def initializeGL(self):
        """Initialize OpenGL."""
        if not self.is_available:
            return
        
        try:
            glClearColor(0.1, 0.1, 0.1, 1.0)
            glEnable(GL_DEPTH_TEST)
            glPointSize(2.0)
            logger.info("3D viewer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenGL: {e}", exc_info=True)
            self.is_available = False
    
    def resizeGL(self, width: int, height: int):
        """Handle resize."""
        if not self.is_available:
            return
        
        try:
            glViewport(0, 0, width, height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            
            # Calculate aspect ratio
            aspect = width / height if height > 0 else 1.0
            
            # Set perspective projection
            if GLU_AVAILABLE:
                GLU.gluPerspective(45.0, aspect, 0.1, 100.0)
            else:
                # Manual perspective projection
                fov = 45.0
                f = 1.0 / math.tan(math.radians(fov) / 2.0)
                near = 0.1
                far = 100.0
                # Set up perspective matrix manually if needed
                pass
            
            glMatrixMode(GL_MODELVIEW)
        except Exception as e:
            logger.error(f"Resize error: {e}", exc_info=True)
    
    def paintGL(self):
        """Render scene."""
        if not self.is_available or self.positions is None:
            return
        
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            # Set up camera
            self._setup_camera()
            
            # Draw Gaussians as points
            if len(self.positions) > 0:
                glBegin(GL_POINTS)
                
                # Limit points for performance
                num_points = min(len(self.positions), self.max_points)
                if self.colors is not None and len(self.colors) > 0:
                    for i in range(num_points):
                        pos = self.positions[i]
                        if i < len(self.colors):
                            col = self.colors[i]
                            # Ensure colors are in [0, 1]
                            if col.max() > 1.0:
                                col = col / 255.0
                            glColor3f(col[0], col[1], col[2])
                        else:
                            glColor3f(1.0, 1.0, 1.0)
                        glVertex3f(pos[0], pos[1], pos[2])
                else:
                    glColor3f(1.0, 1.0, 1.0)
                    for i in range(num_points):
                        pos = self.positions[i]
                        glVertex3f(pos[0], pos[1], pos[2])
                
                glEnd()
        except Exception as e:
            logger.error(f"Render error: {e}", exc_info=True)
    
    def _setup_camera(self):
        """Set up camera view."""
        # Calculate camera position
        cam_x = self.camera_distance * math.cos(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_x))
        cam_y = self.camera_distance * math.sin(math.radians(self.camera_angle_x))
        cam_z = self.camera_distance * math.sin(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_x))
        
        cam_pos = np.array([cam_x, cam_y, cam_z])
        cam_pos = cam_pos + self.camera_target
        
        # Look at target
        if GLU_AVAILABLE:
            GLU.gluLookAt(
                cam_pos[0], cam_pos[1], cam_pos[2],
                self.camera_target[0], self.camera_target[1], self.camera_target[2],
                0.0, 1.0, 0.0
            )
        else:
            # Manual look-at if GLU not available
            forward = self.camera_target - cam_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # Set view matrix (simplified)
            pass
    
    def set_gaussians(self, gaussians):
        """Set Gaussian data to display."""
        if gaussians is None:
            self.positions = None
            self.colors = None
            return
        
        try:
            self.gaussians = gaussians
            # Use centroids as positions
            self.positions = gaussians.centroids
            
            if hasattr(gaussians, 'colors'):
                self.colors = gaussians.colors
            else:
                self.colors = None
            
            # Update camera target to center of scene
            if len(self.positions) > 0:
                self.camera_target = np.mean(self.positions, axis=0)
                # Adjust camera distance based on scene size
                bounds = np.max(self.positions, axis=0) - np.min(self.positions, axis=0)
                self.camera_distance = np.max(bounds) * 2.0
            
            self.update()
        except Exception as e:
            logger.error(f"Failed to set Gaussians: {e}", exc_info=True)
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            self.is_mouse_pressed = True
            self.last_mouse_pos = event.position()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self.is_mouse_pressed = False
            self.last_mouse_pos = None
    
    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self.is_mouse_pressed and self.last_mouse_pos is not None:
            current_pos = event.position()
            dx = current_pos.x() - self.last_mouse_pos.x()
            dy = current_pos.y() - self.last_mouse_pos.y()
            
            # Rotate camera
            self.camera_angle_y += dx * 0.5
            self.camera_angle_x += dy * 0.5
            
            # Clamp vertical angle
            self.camera_angle_x = max(-90, min(90, self.camera_angle_x))
            
            self.last_mouse_pos = current_pos
            self.update()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y() / 120.0
        self.camera_distance *= (1.0 - delta * 0.1)
        self.camera_distance = max(0.1, min(100.0, self.camera_distance))
        self.update()
    
    def reset_view(self):
        """Reset camera view."""
        self.camera_angle_x = 0.0
        self.camera_angle_y = 0.0
        self.camera_distance = 5.0
        if len(self.positions) > 0:
            self.camera_target = np.mean(self.positions, axis=0)
        self.update()

