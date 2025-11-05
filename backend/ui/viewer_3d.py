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
        GL_POINTS, GL_LINES, glPointSize, glLineWidth,
        glBegin, glEnd, glVertex3f, glColor3f, glColor4f,
        glMatrixMode, glLoadIdentity, glViewport,
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
        self.scales = None
        self.opacities = None
        self.max_points = 100000  # Increased limit for better visualization
        self.point_size_scale = 1.0  # Scale factor for point sizes
        
        # Scene bounds for visualization
        self.scene_bounds = None
        self.scene_center = None
        self.show_grid = True
        self.show_axes = True
        
        # Update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(33)  # ~30 FPS
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
        # Set tooltip for user interaction
        self.setToolTip("3D Gaussian Splat Environment Viewer\n\n"
                       "Controls:\n"
                       "• Left-click and drag: Rotate camera\n"
                       "• Mouse wheel: Zoom in/out\n"
                       "\n"
                       "Features:\n"
                       "• Ground grid for spatial reference\n"
                       "• Coordinate axes (RGB = XYZ)\n"
                       "• Bounding box shows scene extent\n"
                       "• Gaussians rendered with colors, scales, and opacity")
    
    def initializeGL(self):
        """Initialize OpenGL."""
        if not self.is_available:
            return
        
        try:
            glClearColor(0.1, 0.1, 0.1, 1.0)
            glEnable(GL_DEPTH_TEST)
            # Enable blending for opacity support
            try:
                from OpenGL.GL import GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, glBlendFunc
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            except:
                pass
            glPointSize(3.0)
            logger.info("3D Gaussian Splat viewer initialized")
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
        """Render Gaussian splats in 3D."""
        if not self.is_available:
            return
        
        if self.positions is None or len(self.positions) == 0:
            # Draw empty scene
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            return
        
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            # Set up camera
            self._setup_camera()
            
            # Draw scene reference (grid, axes, bounding box) first
            if self.show_grid:
                self._draw_grid()
            if self.show_axes:
                self._draw_axes()
            self._draw_bounding_box()
            
            # Draw Gaussian splats
            if len(self.positions) > 0:
                # Limit points for performance
                num_points = min(len(self.positions), self.max_points)
                
                # Use larger point size for better scene visibility
                base_point_size = 5.0 if num_points < 10000 else 4.0
                glPointSize(base_point_size * self.point_size_scale)
                
                # Enable point smoothing for better appearance
                try:
                    from OpenGL.GL import GL_POINT_SMOOTH, glEnable
                    glEnable(GL_POINT_SMOOTH)
                except:
                    pass
                
                glBegin(GL_POINTS)
                
                # Draw each Gaussian with proper color and size
                for i in range(num_points):
                    pos = self.positions[i]
                    
                    # Set color based on Gaussian color
                    if self.colors is not None and i < len(self.colors):
                        col = self.colors[i].copy()
                        # Ensure colors are in [0, 1]
                        if col.max() > 1.0:
                            col = col / 255.0
                        # Apply opacity if available (brighten/darken based on opacity)
                        alpha = 1.0
                        if self.opacities is not None and i < len(self.opacities):
                            alpha = float(self.opacities[i])
                            # Apply opacity as color intensity (simpler than alpha blending)
                            col = col * (0.5 + 0.5 * alpha)
                        try:
                            glColor4f(col[0], col[1], col[2], alpha)
                        except:
                            # Fallback to RGB if 4f not available
                            glColor3f(col[0], col[1], col[2])
                    else:
                        glColor3f(1.0, 1.0, 1.0)
                    
                    # Draw point at Gaussian centroid
                    glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
                
                glEnd()
                
                # Draw scale indicators if available (as small lines)
                if self.scales is not None and len(self.scales) > 0:
                    glLineWidth(1.0)
                    glBegin(GL_LINES)
                    glColor3f(0.5, 0.5, 0.5)
                    
                    for i in range(min(num_points, len(self.scales))):
                        pos = self.positions[i]
                        scale = self.scales[i]
                        # Draw scale as small line segments
                        scale_mag = np.mean(scale) * 0.1 * self.point_size_scale
                        glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
                        glVertex3f(float(pos[0] + scale_mag), float(pos[1]), float(pos[2]))
                        glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
                        glVertex3f(float(pos[0]), float(pos[1] + scale_mag), float(pos[2]))
                        glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
                        glVertex3f(float(pos[0]), float(pos[1]), float(pos[2] + scale_mag))
                    
                    glEnd()
        except Exception as e:
            logger.error(f"Render error: {e}", exc_info=True)
    
    def _draw_grid(self):
        """Draw ground grid for spatial reference."""
        if self.scene_bounds is None or self.scene_center is None:
            return
        
        try:
            # Compute grid size based on scene bounds
            grid_size = max(np.max(self.scene_bounds), 1.0)
            grid_spacing = max(grid_size / 15.0, 0.1)
            grid_extent = grid_size * 1.2
            
            glLineWidth(1.0)
            glBegin(GL_LINES)
            glColor3f(0.4, 0.4, 0.4)  # Dark gray grid
            
            # Draw grid lines on ground plane (Y = scene center Y or minimum Y)
            ground_y = float(self.scene_center[1])
            center_x = float(self.scene_center[0])
            center_z = float(self.scene_center[2])
            
            # Draw lines parallel to X axis (varying Z)
            num_lines = int(grid_extent / grid_spacing) * 2 + 1
            for i in range(-num_lines//2, num_lines//2 + 1):
                z = center_z + i * grid_spacing
                glVertex3f(center_x - grid_extent, ground_y, z)
                glVertex3f(center_x + grid_extent, ground_y, z)
            
            # Draw lines parallel to Z axis (varying X)
            for i in range(-num_lines//2, num_lines//2 + 1):
                x = center_x + i * grid_spacing
                glVertex3f(x, ground_y, center_z - grid_extent)
                glVertex3f(x, ground_y, center_z + grid_extent)
            
            glEnd()
        except Exception as e:
            logger.debug(f"Grid drawing error: {e}")
    
    def _draw_axes(self):
        """Draw coordinate axes for reference."""
        if self.scene_center is None:
            origin = np.array([0.0, 0.0, 0.0])
        else:
            origin = self.scene_center.copy()
        
        try:
            axis_length = max(np.max(self.scene_bounds) if self.scene_bounds is not None else 1.0, 1.0) * 0.3
            
            glLineWidth(2.0)
            glBegin(GL_LINES)
            
            # X axis (red)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(float(origin[0]), float(origin[1]), float(origin[2]))
            glVertex3f(float(origin[0] + axis_length), float(origin[1]), float(origin[2]))
            
            # Y axis (green)
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(float(origin[0]), float(origin[1]), float(origin[2]))
            glVertex3f(float(origin[0]), float(origin[1] + axis_length), float(origin[2]))
            
            # Z axis (blue)
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(float(origin[0]), float(origin[1]), float(origin[2]))
            glVertex3f(float(origin[0]), float(origin[1]), float(origin[2] + axis_length))
            
            glEnd()
        except Exception as e:
            logger.debug(f"Axes drawing error: {e}")
    
    def _draw_bounding_box(self):
        """Draw bounding box around the scene to show 3D environment extent."""
        if self.scene_bounds is None or self.scene_center is None:
            return
        
        try:
            center = self.scene_center
            half_size = self.scene_bounds / 2.0
            
            # Compute box corners
            min_corner = center - half_size
            max_corner = center + half_size
            
            glLineWidth(1.5)
            glBegin(GL_LINES)
            glColor3f(0.6, 0.6, 0.6)  # Light gray for bounding box
            
            # Bottom face (4 edges)
            glVertex3f(float(min_corner[0]), float(min_corner[1]), float(min_corner[2]))
            glVertex3f(float(max_corner[0]), float(min_corner[1]), float(min_corner[2]))
            glVertex3f(float(max_corner[0]), float(min_corner[1]), float(min_corner[2]))
            glVertex3f(float(max_corner[0]), float(min_corner[1]), float(max_corner[2]))
            glVertex3f(float(max_corner[0]), float(min_corner[1]), float(max_corner[2]))
            glVertex3f(float(min_corner[0]), float(min_corner[1]), float(max_corner[2]))
            glVertex3f(float(min_corner[0]), float(min_corner[1]), float(max_corner[2]))
            glVertex3f(float(min_corner[0]), float(min_corner[1]), float(min_corner[2]))
            
            # Top face (4 edges)
            glVertex3f(float(min_corner[0]), float(max_corner[1]), float(min_corner[2]))
            glVertex3f(float(max_corner[0]), float(max_corner[1]), float(min_corner[2]))
            glVertex3f(float(max_corner[0]), float(max_corner[1]), float(min_corner[2]))
            glVertex3f(float(max_corner[0]), float(max_corner[1]), float(max_corner[2]))
            glVertex3f(float(max_corner[0]), float(max_corner[1]), float(max_corner[2]))
            glVertex3f(float(min_corner[0]), float(max_corner[1]), float(max_corner[2]))
            glVertex3f(float(min_corner[0]), float(max_corner[1]), float(max_corner[2]))
            glVertex3f(float(min_corner[0]), float(max_corner[1]), float(min_corner[2]))
            
            # Vertical edges (4 edges)
            glVertex3f(float(min_corner[0]), float(min_corner[1]), float(min_corner[2]))
            glVertex3f(float(min_corner[0]), float(max_corner[1]), float(min_corner[2]))
            glVertex3f(float(max_corner[0]), float(min_corner[1]), float(min_corner[2]))
            glVertex3f(float(max_corner[0]), float(max_corner[1]), float(min_corner[2]))
            glVertex3f(float(min_corner[0]), float(min_corner[1]), float(max_corner[2]))
            glVertex3f(float(min_corner[0]), float(max_corner[1]), float(max_corner[2]))
            glVertex3f(float(max_corner[0]), float(min_corner[1]), float(max_corner[2]))
            glVertex3f(float(max_corner[0]), float(max_corner[1]), float(max_corner[2]))
            
            glEnd()
        except Exception as e:
            logger.debug(f"Bounding box drawing error: {e}")
    
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
        """Set Gaussian splat data to display in 3D."""
        if gaussians is None:
            self.positions = None
            self.colors = None
            self.scales = None
            self.opacities = None
            return
        
        try:
            self.gaussians = gaussians
            # Use centroids as positions
            self.positions = gaussians.centroids
            
            # Extract colors
            if hasattr(gaussians, 'colors'):
                self.colors = gaussians.colors
            else:
                self.colors = None
            
            # Extract scales for visualization
            if hasattr(gaussians, 'scales') and gaussians.scales is not None:
                self.scales = gaussians.scales
            else:
                self.scales = None
            
            # Extract opacities for visualization
            if hasattr(gaussians, 'opacity') and gaussians.opacity is not None:
                self.opacities = gaussians.opacity
            else:
                self.opacities = None
            
            # Compute scene bounds and center for 3D environment visualization
            if len(self.positions) > 0:
                # Compute scene bounds
                min_pos = np.min(self.positions, axis=0)
                max_pos = np.max(self.positions, axis=0)
                self.scene_bounds = max_pos - min_pos
                self.scene_center = (min_pos + max_pos) / 2.0
                
                # Update camera target to center of scene
                self.camera_target = self.scene_center.copy()
                
                # Adjust camera distance based on scene size
                max_bound = np.max(self.scene_bounds)
                if max_bound > 0:
                    self.camera_distance = max_bound * 3.5  # Further back to see more of the environment
                    # Reset camera angles for better initial view
                    self.camera_angle_x = 20.0  # Slight angle down
                    self.camera_angle_y = 45.0  # Angled view
                else:
                    self.camera_distance = 5.0
                
                # Adjust point size scale based on scene scale
                if max_bound > 0:
                    self.point_size_scale = max(0.8, min(3.0, 1.0 / (max_bound * 0.03)))
            else:
                self.scene_bounds = None
                self.scene_center = None
            
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
        """Reset camera view to default angled view of the scene."""
        self.camera_angle_x = 20.0  # Slight angle down
        self.camera_angle_y = 45.0  # Angled view
        if self.scene_center is not None:
            self.camera_target = self.scene_center.copy()
            if self.scene_bounds is not None:
                max_bound = np.max(self.scene_bounds)
                if max_bound > 0:
                    self.camera_distance = max_bound * 3.5
                else:
                    self.camera_distance = 5.0
            else:
                self.camera_distance = 5.0
        elif self.positions is not None and len(self.positions) > 0:
            self.camera_target = np.mean(self.positions, axis=0)
            bounds = np.max(self.positions, axis=0) - np.min(self.positions, axis=0)
            max_bound = np.max(bounds)
            if max_bound > 0:
                self.camera_distance = max_bound * 3.5
            else:
                self.camera_distance = 5.0
        else:
            self.camera_distance = 5.0
            self.camera_target = np.array([0.0, 0.0, 0.0])
        self.update()

