"""
Main Window

PySide6 main window for GaussCam application.
"""

import numpy as np
import cv2
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
    QProgressBar,
    QStatusBar,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QKeySequence
from typing import Optional, Dict, Any
import sys
import time

from backend.ui.render_widget import RenderWidget
from backend.utils.gpu_detection import get_gpu_detector, is_cuda, is_mps
from backend.renderer.base import Renderer
from backend.renderer.cuda_renderer import CUDARenderer
from backend.renderer.mps_renderer import MPSRenderer
from backend.utils.logging_config import get_logger
from backend.utils.validation import (
    validate_video_file,
    validate_webcam_device,
    validate_performance_mode,
    ValidationError,
)
from backend.utils.progress import FrameProgressTracker
from backend.utils.config import get_config

logger = get_logger(__name__)


class ProcessingThread(QThread):
    """Background thread for processing frames."""
    
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    
    def __init__(
        self,
        input_source: str,
        video_path: Optional[str] = None,
        renderer=None,
        webcam_device_id: int = 0,
        parent=None,
    ):
        """
        Initialize processing thread.
        
        Args:
            input_source: 'webcam' or 'video'
            video_path: Path to video file (if input_source is 'video')
            renderer: Renderer instance
            webcam_device_id: Webcam device ID (default: 0)
            parent: Parent widget
        """
        super().__init__(parent)
        self.input_source = input_source
        self.video_path = video_path
        self.renderer = renderer
        self.webcam_device_id = webcam_device_id
        self.is_running = False
        self.stop_flag = False
        
        # Components (will be initialized in run)
        self.capture = None
        self.async_capture = None
        self.depth_estimator = None
        self.intrinsics = None
        self.gaussian_fitter = None
        self.gaussian_merger = None
    
    def run(self) -> None:
        """Run processing loop."""
        try:
            self.is_running = True
            
            # Initialize components
            from backend.input.capture import (
                WebcamCapture,
                VideoCapture,
                AsyncFrameCapture,
                FramePreprocessor,
            )
            from backend.depth.midas_wrapper import MiDaSDepthEstimator
            from backend.utils.camera import CameraIntrinsics
            from backend.utils.point_cloud import depth_to_point_cloud
            from backend.gaussian.fitter import GaussianFitter
            from backend.gaussian.merger import GaussianMerger
            
            # Setup capture
            width, height = 640, 480
            if self.input_source == "webcam":
                self.capture = WebcamCapture(device_id=self.webcam_device_id, width=width, height=height, fps=30)
            elif self.input_source == "video" and self.video_path:
                self.capture = VideoCapture(self.video_path, width=width, height=height)
                width = self.capture.width
                height = self.capture.height
            else:
                self.error_occurred.emit("Invalid input source")
                return
            
            # Frame preprocessor
            preprocessor = FramePreprocessor(normalize=True, to_rgb=True)
            self.async_capture = AsyncFrameCapture(self.capture, preprocessor, queue_size=2)
            self.async_capture.start()
            
            # Depth estimator (use smaller model for speed)
            print("Loading MiDaS depth model...")
            # Use hybrid model for better speed/quality balance
            try:
                self.depth_estimator = MiDaSDepthEstimator(model_name="Intel/dpt-hybrid-midas")
                print("MiDaS hybrid model loaded (faster)")
            except:
                # Fallback to large model
                self.depth_estimator = MiDaSDepthEstimator(model_name="Intel/dpt-large")
                print("MiDaS large model loaded")
            
            # Camera intrinsics
            self.intrinsics = CameraIntrinsics.default(width, height, fov=60.0)
            
            # Gaussian fitter and merger
            self.gaussian_fitter = GaussianFitter(k_neighbors=8, initial_opacity=0.9)
            self.gaussian_merger = GaussianMerger(merge_threshold=0.01, max_gaussians=500000)
            
            print(f"Processing started: {self.input_source}")
            
            # Processing loop with optimizations
            frame_count = 0
            processing_start_time = time.time()
            
            # Get performance mode from parent
            performance_mode = "Balanced"
            if hasattr(self.parent(), 'performance_mode'):
                performance_mode = self.parent().performance_mode
            
            # Initialize adaptive quality manager
            from backend.utils.adaptive_quality import AdaptiveQualityManager
            adaptive_manager = AdaptiveQualityManager(target_fps=15.0)
            
            # Track consecutive None frames to prevent early exit
            consecutive_none_frames = 0
            max_none_frames = 10  # Allow up to 10 consecutive None frames before checking
            
            # Adjust parameters based on performance mode and input source
            # Video files can be processed faster since they're not real-time
            is_video = self.input_source == "video"
            
            if performance_mode == "Fast":
                depth_skip_frames = 20 if is_video else 15  # More skipping for video
                frame_skip = 15 if is_video else 8  # Process every 16th frame for video
                max_gaussians = 3000
                target_size = 256
                target_pixels = 15000
                voxel_size = 0.15
            elif performance_mode == "Quality":
                depth_skip_frames = 10 if is_video else 5
                frame_skip = 5 if is_video else 2
                max_gaussians = 10000
                target_size = 384
                target_pixels = 30000
                voxel_size = 0.08
            else:  # Balanced
                depth_skip_frames = 15 if is_video else 10
                frame_skip = 10 if is_video else 5  # Process every 11th frame for video
                max_gaussians = 5000
                target_size = 320
                target_pixels = 20000
                voxel_size = 0.10
            
            last_depth = None
            last_rendered = None  # Cache last rendered frame
            
            while not self.stop_flag:
                # Read frame
                frame = self.async_capture.read(timeout=0.1)
                if frame is None:
                    if self.input_source == "video":
                        # End of video
                        print("Video ended")
                        break
                    # For webcam, keep trying - don't exit immediately
                    consecutive_none_frames += 1
                    if consecutive_none_frames > max_none_frames:
                        print(f"Warning: {consecutive_none_frames} consecutive None frames, continuing...")
                        consecutive_none_frames = 0  # Reset counter
                    self.msleep(100)
                    continue
                
                # Reset counter when we get a valid frame
                consecutive_none_frames = 0
                
                # Skip frames for performance - only process every Nth frame
                if frame_count % (frame_skip + 1) != 0:
                    # Emit cached frame if available
                    if last_rendered is not None:
                        self.frame_ready.emit(last_rendered)
                    frame_count += 1
                    # For video, skip faster (no sleep needed)
                    if not is_video:
                        self.msleep(33)  # ~30 FPS display rate for webcam
                    continue
                
                print(f"Processing frame {frame_count}: shape={frame.shape if hasattr(frame, 'shape') else 'unknown'}")
                
                # Convert to numpy if needed
                import torch
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                
                # Ensure frame is in correct format
                if frame.dtype != np.float32:
                    if frame.max() > 1.0:
                        frame = frame.astype(np.float32) / 255.0
                    else:
                        frame = frame.astype(np.float32)
                
                # Display original frame first for immediate feedback
                if frame_count == 0:
                    # Show original frame immediately
                    display_frame = frame.copy()
                    if display_frame.max() <= 1.0:
                        display_frame = (display_frame * 255).astype(np.uint8)
                    else:
                        display_frame = display_frame.astype(np.uint8)
                    self.frame_ready.emit(display_frame)
                    print(f"Displayed initial frame: shape={display_frame.shape}")
                
                # Estimate depth (skip frames for speed)
                if frame_count % depth_skip_frames == 0 or last_depth is None:
                    depth_start_time = time.time()
                    print(f"Estimating depth for frame {frame_count}...")
                    try:
                        # Always resize for faster depth estimation
                        h, w = frame.shape[:2]
                        # Use adaptive target size based on performance mode
                        if h > target_size or w > target_size:
                            scale = min(target_size / h, target_size / w)
                            small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
                            small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                            depth = self.depth_estimator.estimate_depth(small_frame, postprocess=True)
                            # Resize depth back to original size (use linear for speed)
                            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
                        else:
                            depth = self.depth_estimator.estimate_depth(frame, postprocess=True)
                        last_depth = depth
                        depth_time = time.time() - depth_start_time
                        print(f"Depth estimated: shape={depth.shape}, min={depth.min():.3f}, max={depth.max():.3f}, time={depth_time:.3f}s")
                    except Exception as e:
                        print(f"Depth estimation error: {e}")
                        import traceback
                        traceback.print_exc()
                        if last_depth is not None:
                            depth = last_depth
                        else:
                            self.msleep(100)
                            continue
                else:
                    # Reuse last depth
                    depth = last_depth
                
                # Convert to point cloud (with aggressive downsampling)
                # Skip point cloud conversion if depth hasn't changed (for video)
                if is_video and last_depth is not None and frame_count % depth_skip_frames != 0:
                    # Reuse previous point cloud if depth hasn't changed
                    print(f"Skipping point cloud conversion (reusing depth)...")
                    # Use cached points from last frame if available
                    if hasattr(self, '_last_points') and hasattr(self, '_last_colors'):
                        points = self._last_points
                        colors = self._last_colors
                    else:
                        # Skip this frame entirely if no cached data
                        frame_count += 1
                        self.msleep(50)
                        continue
                else:
                    print(f"Converting to point cloud...")
                    try:
                        # Always downsample depth map first for speed
                        h, w = depth.shape[:2]
                        # Use adaptive target pixels based on performance mode
                        if h * w > target_pixels:
                            scale = np.sqrt(target_pixels / (h * w))
                            small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
                            small_depth = cv2.resize(depth, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                            small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                            # Update intrinsics for smaller size
                            scale_factor = small_w / w
                            from backend.utils.camera import CameraIntrinsics
                            small_intrinsics = CameraIntrinsics(
                                fx=self.intrinsics.fx * scale_factor,
                                fy=self.intrinsics.fy * scale_factor,
                                cx=self.intrinsics.cx * scale_factor,
                                cy=self.intrinsics.cy * scale_factor,
                                width=small_w,
                                height=small_h,
                            )
                            # Use GPU-accelerated version if available
                            points, colors = depth_to_point_cloud(
                                small_depth, small_frame, small_intrinsics, depth_scale=5.0, max_depth=10.0, use_gpu=True
                            )
                        else:
                            # Use GPU-accelerated version if available
                            points, colors = depth_to_point_cloud(
                                depth, frame, self.intrinsics, depth_scale=5.0, max_depth=10.0, use_gpu=True
                            )
                        print(f"Point cloud: {len(points)} points")
                        
                        # Cache points for video reuse
                        if is_video:
                            self._last_points = points
                            self._last_colors = colors
                        
                        # Aggressive downsampling for speed (GPU-accelerated)
                        if len(points) > 3000:  # Even lower threshold
                            from backend.utils.point_cloud import downsample_point_cloud
                            # Use adaptive voxel size based on performance mode
                            points, colors = downsample_point_cloud(points, colors, voxel_size=voxel_size, use_gpu=True)
                            print(f"Downsampled to {len(points)} points")
                            
                            # Update cache
                            if is_video:
                                self._last_points = points
                                self._last_colors = colors
                    except Exception as e:
                        print(f"Point cloud conversion error: {e}")
                        import traceback
                        traceback.print_exc()
                        # Skip this frame and continue
                        frame_count += 1
                        if not is_video:
                            self.msleep(100)
                        continue
                
                if len(points) > 0:
                    try:
                        # Fit Gaussians (use uniform method for speed)
                        print(f"Fitting Gaussians from {len(points)} points...")
                        # Use performance mode setting with GPU acceleration
                        gaussians = self.gaussian_fitter.fit_downsampled(
                            points, colors, max_gaussians=max_gaussians, method="uniform", use_gpu=True
                        )
                        print(f"Fitted {gaussians.num_gaussians} Gaussians")
                        
                        # Merge with accumulated
                        print(f"Merging Gaussians...")
                        merged_gaussians = self.gaussian_merger.merge(
                            gaussians, merge_strategy="weighted"
                        )
                        print(f"Merged to {merged_gaussians.num_gaussians} Gaussians")
                        
                        # Render
                        if self.renderer is not None and merged_gaussians.num_gaussians > 0:
                            try:
                                print(f"Rendering {merged_gaussians.num_gaussians} Gaussians...")
                                rendered = self.renderer.render(merged_gaussians)
                                print(f"Rendered frame: shape={rendered.shape}, min={rendered.min():.3f}, max={rendered.max():.3f}")
                                # Cache rendered frame
                                last_rendered = rendered.copy()
                                # Emit frame for display
                                self.frame_ready.emit(rendered)
                                print(f"Frame emitted to UI")
                            except Exception as e:
                                print(f"Rendering error: {e}")
                                import traceback
                                traceback.print_exc()
                                # Don't exit on rendering error - continue processing
                                if last_rendered is not None:
                                    self.frame_ready.emit(last_rendered)
                    except Exception as e:
                        print(f"Gaussian fitting/merging error: {e}")
                        import traceback
                        traceback.print_exc()
                        # Don't exit on fitting error - continue processing
                        if last_rendered is not None:
                            self.frame_ready.emit(last_rendered)
                else:
                    # Emit cached frame or last rendered frame if no points
                    if last_rendered is not None:
                        self.frame_ready.emit(last_rendered)
                    elif self.renderer is not None and self.gaussian_merger.accumulated_gaussians is not None:
                        try:
                            rendered = self.renderer.render(self.gaussian_merger.accumulated_gaussians)
                            last_rendered = rendered.copy()
                            self.frame_ready.emit(rendered)
                        except:
                            pass
                
                frame_count += 1
                
                # Update adaptive quality manager
                frame_time = time.time() - processing_start_time
                adaptive_manager.update(frame_time)
                processing_start_time = time.time()
                
                # Adapt settings if needed (every 10 frames)
                if frame_count % 10 == 0 and adaptive_manager.should_adapt():
                    optimal = adaptive_manager.get_optimal_settings()
                    depth_skip_frames = optimal["depth_skip_frames"]
                    frame_skip = optimal["frame_skip"]
                    max_gaussians = optimal["max_gaussians"]
                    target_size = optimal["target_size"]
                    target_pixels = optimal["target_pixels"]
                    voxel_size = optimal["voxel_size"]
                    print(f"Adapted settings: FPS={adaptive_manager.get_current_fps():.1f}, max_gaussians={max_gaussians}")
                
                # Small delay to prevent CPU spinning
                # Less delay for video (no real-time requirement)
                if is_video:
                    self.msleep(50)  # Minimal delay for video processing
                else:
                    self.msleep(200)  # Delay for webcam to balance performance
        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
            self.error_occurred.emit("Processing interrupted by user")
        except Exception as e:
            import traceback
            error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
        finally:
            # Cleanup
            print("Cleaning up processing thread...")
            if self.async_capture is not None:
                try:
                    self.async_capture.stop()
                except:
                    pass
            if self.capture is not None:
                try:
                    self.capture.release()
                except:
                    pass
            self.is_running = False
            print("Processing thread finished")
    
    def stop(self) -> None:
        """Stop processing thread."""
        self.stop_flag = True
        self.wait(3000)  # Wait up to 3 seconds


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        logger.info("Initializing MainWindow...")
        
        # Load configuration
        try:
            self.config = get_config()
            logger.info("Loaded application configuration")
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
            from backend.utils.config import AppConfig
            self.config = AppConfig()
        
        self.setWindowTitle("GaussCam - Gaussian Splatting Renderer")
        self.setMinimumSize(
            self.config.window_width,
            self.config.window_height
        )
        
        # Initialize components
        self.renderer: Optional[Renderer] = None
        self.processing_thread: Optional[ProcessingThread] = None
        self.video_path: Optional[str] = None
        self.selected_webcam_id: int = 0
        self.performance_mode: str = "Balanced"
        
        # GPU detection (defer to avoid Metal conflicts during PySide6 init)
        self.gpu_detector = None
        self.backend = "unknown"
        
        # Initialize progress tracker
        self.progress_tracker: Optional[FrameProgressTracker] = None
        
        # Setup status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar.addPermanentWidget(self.progress_bar)
        
        # Setup UI first
        self._setup_ui()
        self._setup_menu()
        
        # Initialize GPU detection after UI is ready
        self._init_gpu_detection()
        
        self._update_status("Ready")
        logger.info("MainWindow initialized successfully")
    
    def _setup_ui(self) -> None:
        """Setup user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Render widget (will be initialized after GPU detection)
        self.render_widget = None
        self.render_placeholder = QLabel("Initializing GPU detection...")
        self.render_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.render_placeholder.setMinimumSize(640, 480)
        self.render_placeholder.setStyleSheet("background-color: #1e1e1e; color: white; font-size: 14px;")
        main_layout.addWidget(self.render_placeholder, stretch=3)
        
        # Control panel
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)
    
    def _create_control_panel(self) -> QWidget:
        """Create control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Input source
        input_group = QGroupBox("Input Source")
        input_layout = QVBoxLayout()
        
        self.input_combo = QComboBox()
        self.input_combo.addItems(["Webcam", "Video File"])
        self.input_combo.currentTextChanged.connect(self._on_input_changed)
        input_layout.addWidget(QLabel("Source:"))
        input_layout.addWidget(self.input_combo)
        
        # Webcam selection
        webcam_label = QLabel("Webcam Device:")
        input_layout.addWidget(webcam_label)
        
        self.webcam_combo = QComboBox()
        self._populate_webcam_devices()
        self.webcam_combo.currentIndexChanged.connect(self._on_webcam_changed)
        input_layout.addWidget(self.webcam_combo)
        
        self.video_button = QPushButton("Select Video File")
        self.video_button.clicked.connect(self._select_video_file)
        self.video_button.setEnabled(False)
        input_layout.addWidget(self.video_button)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Rendering settings
        render_group = QGroupBox("Rendering Settings")
        render_layout = QVBoxLayout()
        
        self.lod_slider = QSlider(Qt.Orientation.Horizontal)
        self.lod_slider.setMinimum(0)
        self.lod_slider.setMaximum(3)
        self.lod_slider.setValue(0)
        self.lod_slider.valueChanged.connect(self._on_lod_changed)
        render_layout.addWidget(QLabel("LOD Level:"))
        render_layout.addWidget(self.lod_slider)
        
        # Performance settings
        perf_label = QLabel("Performance Mode:")
        render_layout.addWidget(perf_label)
        
        self.performance_combo = QComboBox()
        self.performance_combo.addItems(["Fast", "Balanced", "Quality"])
        self.performance_combo.setCurrentText("Balanced")
        self.performance_combo.currentTextChanged.connect(self._on_performance_changed)
        render_layout.addWidget(self.performance_combo)
        
        self.gaussian_count_label = QLabel("Gaussians: 0")
        render_layout.addWidget(self.gaussian_count_label)
        
        self.fps_label = QLabel("FPS: 0")
        render_layout.addWidget(self.fps_label)
        
        render_group.setLayout(render_layout)
        layout.addWidget(render_group)
        
        # Novel view controls
        view_group = QGroupBox("Novel View")
        view_layout = QVBoxLayout()
        
        self.rotation_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_x_slider.setMinimum(-180)
        self.rotation_x_slider.setMaximum(180)
        self.rotation_x_slider.setValue(0)
        view_layout.addWidget(QLabel("Rotation X:"))
        view_layout.addWidget(self.rotation_x_slider)
        
        self.rotation_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_y_slider.setMinimum(-180)
        self.rotation_y_slider.setMaximum(180)
        self.rotation_y_slider.setValue(0)
        view_layout.addWidget(QLabel("Rotation Y:"))
        view_layout.addWidget(self.rotation_y_slider)
        
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)
        
        # Control buttons
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        layout.addStretch()
        
        # Status label (will be updated after GPU detection)
        self.status_label = QLabel("Backend: Initializing...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        return panel
    
    def _setup_menu(self) -> None:
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Video...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._select_video_file)
        file_menu.addAction(open_action)
        
        # Export menu
        export_menu = file_menu.addMenu("Export")
        
        export_gaussians_action = QAction("Save Gaussians...", self)
        export_gaussians_action.triggered.connect(self._export_gaussians)
        export_menu.addAction(export_gaussians_action)
        
        load_gaussians_action = QAction("Load Gaussians...", self)
        load_gaussians_action.triggered.connect(self._load_gaussians)
        export_menu.addAction(load_gaussians_action)
        
        export_video_action = QAction("Export Rendered Video...", self)
        export_video_action.triggered.connect(self._export_video)
        export_menu.addAction(export_video_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        reset_view_action = QAction("Reset View", self)
        reset_view_action.triggered.connect(self._reset_view)
        view_menu.addAction(reset_view_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _init_gpu_detection(self) -> None:
        """Initialize GPU detection after UI is ready."""
        try:
            self.gpu_detector = get_gpu_detector()
            self.backend = self.gpu_detector.get_backend()
            # Update status label
            status_text = f"Backend: {self.backend.upper()}\nDevice: {self.gpu_detector.device_name}"
            if hasattr(self, 'status_label'):
                self.status_label.setText(status_text)
            
            # Update placeholder text
            if hasattr(self, 'render_placeholder'):
                self.render_placeholder.setText(f"GPU: {self.backend.upper()}\nReady to render")
            
            # Defer render widget initialization using QTimer to avoid Metal conflicts
            # This ensures the event loop has started before creating RenderWidget
            QTimer.singleShot(100, self._init_render_widget)
        except Exception as e:
            print(f"Warning: GPU detection failed: {e}")
            self.backend = "cpu"
            if hasattr(self, 'render_placeholder'):
                self.render_placeholder.setText(f"GPU detection failed: {e}")
    
    def _init_render_widget(self) -> None:
        """Initialize render widget after GPU detection."""
        if self.render_widget is not None:
            return  # Already initialized
        
        if not hasattr(self, 'render_placeholder') or self.render_placeholder is None:
            print("Warning: render_placeholder not found")
            return
        
        try:
            # Get the parent widget and its layout
            parent_widget = self.render_placeholder.parent()
            if parent_widget is None:
                print("Warning: render_placeholder has no parent")
                return
            
            layout = parent_widget.layout()
            if layout is None:
                print("Warning: parent has no layout")
                return
            
            # Remove placeholder from layout
            layout.removeWidget(self.render_placeholder)
            self.render_placeholder.hide()
            self.render_placeholder.deleteLater()
            
            # Create and add render widget
            self.render_widget = RenderWidget(self, width=640, height=480)
            layout.addWidget(self.render_widget, stretch=3)
            
            # Clear placeholder reference
            self.render_placeholder = None
            
            print("RenderWidget initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize RenderWidget: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'render_placeholder') and self.render_placeholder is not None:
                self.render_placeholder.setText(f"Render widget initialization failed: {e}")
    
    def _setup_renderer(self) -> None:
        """Setup renderer based on GPU backend (lazy initialization)."""
        if self.renderer is not None:
            return  # Already initialized
        
        try:
            if is_cuda():
                self.renderer = CUDARenderer(width=640, height=480)
            elif is_mps():
                # Initialize MPS renderer with error handling
                # Defer MPS tensor creation to avoid Metal conflicts
                self.renderer = MPSRenderer(width=640, height=480)
            else:
                QMessageBox.warning(
                    self,
                    "No GPU",
                    "No GPU detected. Falling back to CPU (not implemented).",
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Renderer Error",
                f"Failed to initialize renderer: {str(e)}",
            )
            self.renderer = None
    
    def _populate_webcam_devices(self) -> None:
        """Populate webcam device list."""
        try:
            from backend.input.capture import list_webcam_devices
            devices = list_webcam_devices(max_devices=10)
            self.webcam_combo.clear()
            for device_id in devices:
                self.webcam_combo.addItem(f"Camera {device_id}", device_id)
            if len(devices) == 0:
                self.webcam_combo.addItem("No cameras found", -1)
                self.webcam_combo.setEnabled(False)
            else:
                self.selected_webcam_id = devices[0]
        except Exception as e:
            print(f"Error listing webcam devices: {e}")
            self.webcam_combo.addItem("Camera 0", 0)
            self.selected_webcam_id = 0
    
    def _on_webcam_changed(self, index: int) -> None:
        """Handle webcam device selection change."""
        if self.webcam_combo.itemData(index) is not None:
            self.selected_webcam_id = self.webcam_combo.itemData(index)
            print(f"Selected webcam device: {self.selected_webcam_id}")
    
    def _on_input_changed(self, text: str) -> None:
        """Handle input source change."""
        is_video = text == "Video File"
        self.video_button.setEnabled(is_video)
        self.webcam_combo.setEnabled(not is_video)
    
    def _on_performance_changed(self, mode: str) -> None:
        """Handle performance mode change."""
        self.performance_mode = mode
        print(f"Performance mode changed to: {mode}")
        # Update processing thread parameters if running
        if self.processing_thread is not None and self.processing_thread.is_running:
            # Will apply on next frame
            pass
    
    def _on_lod_changed(self, value: int) -> None:
        """Handle LOD level change."""
        # Update LOD level (will be used in next render)
        if hasattr(self, 'processing_thread') and self.processing_thread is not None:
            if hasattr(self.processing_thread, 'gaussian_merger') and \
               self.processing_thread.gaussian_merger is not None:
                # Apply LOD to accumulated Gaussians
                # This will take effect on next frame
                pass
    
    def _select_video_file(self) -> None:
        """Select video file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Video File",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;All Files (*)",
            )
            
            if file_path:
                # Validate video file
                try:
                    validate_video_file(file_path)
                    self.video_path = file_path
                    # Update input combo to video
                    self.input_combo.setCurrentText("Video File")
                    logger.info(f"Video file selected: {file_path}")
                    from pathlib import Path
                    self._update_status(f"Video file: {Path(file_path).name}")
                except ValidationError as e:
                    logger.error(f"Invalid video file: {e}")
                    QMessageBox.critical(
                        self,
                        "Invalid Video File",
                        f"Invalid video file:\n{str(e)}",
                    )
        except Exception as e:
            logger.error(f"Error selecting video file: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "File Selection Error",
                f"Failed to select video file:\n{str(e)}",
            )
    
    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        try:
            # Initialize renderer if not already done
            if self.renderer is None:
                logger.info("Initializing renderer...")
                self._setup_renderer()
            
            if self.renderer is None:
                error_msg = "Failed to initialize renderer. Cannot start processing."
                logger.error(error_msg)
                QMessageBox.warning(
                    self,
                    "No Renderer",
                    error_msg,
                )
                return
            
            # Get input source
            input_source = self.input_combo.currentText().lower()
            if input_source == "video file":
                input_source = "video"
                if not self.video_path:
                    error_msg = "Please select a video file first."
                    logger.warning(error_msg)
                    QMessageBox.warning(
                        self,
                        "No Video File",
                        error_msg,
                    )
                    return
                
                # Validate video file
                try:
                    validate_video_file(self.video_path)
                    logger.info(f"Validated video file: {self.video_path}")
                except ValidationError as e:
                    logger.error(f"Video file validation failed: {e}")
                    QMessageBox.critical(
                        self,
                        "Invalid Video File",
                        f"Invalid video file:\n{str(e)}",
                    )
                    return
            elif input_source == "webcam":
                input_source = "webcam"
                # Validate webcam device
                try:
                    validate_webcam_device(self.selected_webcam_id)
                    logger.info(f"Validated webcam device: {self.selected_webcam_id}")
                except ValidationError as e:
                    logger.error(f"Webcam validation failed: {e}")
                    QMessageBox.critical(
                        self,
                        "Invalid Webcam",
                        f"Invalid webcam device:\n{str(e)}",
                    )
                    return
            else:
                error_msg = "Please select an input source (Webcam or Video File)."
                logger.warning(error_msg)
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    error_msg,
                )
                return
            
            # Validate performance mode
            try:
                performance_mode = validate_performance_mode(self.performance_mode)
                logger.info(f"Using performance mode: {performance_mode}")
            except ValidationError as e:
                logger.warning(f"Invalid performance mode, using default: {e}")
                performance_mode = "Balanced"
            
            # Stop existing thread if running
            if self.processing_thread is not None and self.processing_thread.isRunning():
                logger.info("Stopping existing processing thread...")
                self.processing_thread.stop()
            
            # Initialize progress tracker for video
            if input_source == "video":
                try:
                    import cv2
                    cap = cv2.VideoCapture(self.video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    self.progress_tracker = FrameProgressTracker(total_frames=total_frames)
                    logger.info(f"Video has {total_frames} frames")
                except Exception as e:
                    logger.warning(f"Could not determine video frame count: {e}")
                    self.progress_tracker = FrameProgressTracker(total_frames=None)
            else:
                self.progress_tracker = FrameProgressTracker(total_frames=None)
            
            # Create and start processing thread
            logger.info(f"Creating processing thread for {input_source}...")
            self.processing_thread = ProcessingThread(
                input_source=input_source,
                video_path=self.video_path if input_source == "video" else None,
                renderer=self.renderer,
                webcam_device_id=self.selected_webcam_id,
                parent=self,
            )
            self.processing_thread.frame_ready.connect(self._on_frame_ready)
            self.processing_thread.error_occurred.connect(self._on_processing_error)
            self.processing_thread.finished.connect(self._on_processing_finished)
            
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.processing_thread.start()
            
            logger.info(f"Started processing: {input_source}")
            self._update_status(f"Processing: {input_source}")
        except Exception as e:
            logger.error(f"Error starting processing: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Start Error",
                f"Failed to start processing:\n{str(e)}",
            )
    
    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        logger.info("Stopping processing...")
        if self.processing_thread is not None and self.processing_thread.isRunning():
            self.processing_thread.stop()
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._update_status("Stopped")
        logger.info("Processing stopped")
    
    def _on_frame_ready(self, frame: np.ndarray) -> None:
        """Handle frame ready signal from processing thread."""
        try:
            if self.render_widget is not None:
                self.render_widget.set_frame(frame)
            
            # Update Gaussian count
            if hasattr(self, 'gaussian_count_label') and self.processing_thread is not None:
                if hasattr(self.processing_thread, 'gaussian_merger') and \
                   self.processing_thread.gaussian_merger is not None:
                    gaussians = self.processing_thread.gaussian_merger.accumulated_gaussians
                    if gaussians is not None:
                        count = gaussians.num_gaussians
                        self.gaussian_count_label.setText(f"Gaussians: {count}")
                        
                        # Update progress if available
                        if self.progress_tracker is not None:
                            self.progress_tracker.update_frame(processed=True)
                            self._update_progress(
                                self.progress_tracker.current,
                                self.progress_tracker.total
                            )
                    else:
                        self.gaussian_count_label.setText("Gaussians: 0")
        except Exception as e:
            logger.error(f"Error handling frame ready: {e}", exc_info=True)
    
    def _on_processing_error(self, error_msg: str) -> None:
        """Handle processing error."""
        logger.error(f"Processing error: {error_msg}")
        QMessageBox.critical(
            self,
            "Processing Error",
            f"An error occurred during processing:\n{error_msg}",
        )
        self._update_status("Error occurred")
        self.progress_bar.setVisible(False)
        self._on_stop_clicked()
    
    def _on_processing_finished(self) -> None:
        """Handle processing finished."""
        print("Processing finished")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Show completion message
        if self.processing_thread is not None:
            if hasattr(self.processing_thread, 'gaussian_merger') and \
               self.processing_thread.gaussian_merger is not None:
                gaussians = self.processing_thread.gaussian_merger.accumulated_gaussians
                if gaussians is not None:
                    QMessageBox.information(
                        self,
                        "Processing Complete",
                        f"Processing finished successfully!\n\n"
                        f"Total Gaussians: {gaussians.num_gaussians}\n"
                        f"Frames processed: {self.processing_thread.gaussian_merger.frame_count}\n\n"
                        f"You can now:\n"
                        f"- Export Gaussians (File > Export > Save Gaussians)\n"
                        f"- Export rendered video (File > Export > Export Rendered Video)\n"
                        f"- Adjust novel view controls",
                    )
    
    def _export_gaussians(self) -> None:
        """Export Gaussians to file."""
        if self.processing_thread is None or not hasattr(self.processing_thread, 'gaussian_merger'):
            QMessageBox.warning(
                self,
                "No Data",
                "No Gaussians to export. Please process a video or webcam first.",
            )
            return
        
        gaussians = self.processing_thread.gaussian_merger.accumulated_gaussians
        if gaussians is None:
            QMessageBox.warning(
                self,
                "No Data",
                "No Gaussians to export. Please process a video or webcam first.",
            )
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Gaussians",
            "",
            "Pickle Files (*.pkl);;All Files (*)",
        )
        
        if file_path:
            try:
                import pickle
                data = {
                    "centroids": gaussians.centroids,
                    "scales": gaussians.scales,
                    "rotations": gaussians.rotations,
                    "colors": gaussians.colors,
                    "opacity": gaussians.opacity,
                }
                with open(file_path, "wb") as f:
                    pickle.dump(data, f)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Saved {gaussians.num_gaussians} Gaussians to:\n{file_path}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export Gaussians:\n{str(e)}",
                )
    
    def _load_gaussians(self) -> None:
        """Load Gaussians from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Gaussians",
            "",
            "Pickle Files (*.pkl);;All Files (*)",
        )
        
        if file_path:
            try:
                import pickle
                from backend.gaussian.fitter import Gaussian
                
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                
                gaussians = Gaussian(
                    centroids=data["centroids"],
                    covariances=None,
                    colors=data["colors"],
                    opacity=data["opacity"],
                    scales=data["scales"],
                    rotations=data["rotations"],
                )
                
                # Initialize merger with loaded Gaussians
                if self.processing_thread is None:
                    from backend.gaussian.merger import GaussianMerger
                    merger = GaussianMerger(merge_threshold=0.01, max_gaussians=500000)
                    merger.accumulated_gaussians = gaussians
                    merger.frame_count = 1
                    
                    # Create a dummy processing thread for rendering
                    self.processing_thread = type('obj', (object,), {
                        'gaussian_merger': merger,
                        'isRunning': lambda: False,
                    })()
                
                # Render loaded Gaussians
                if self.renderer is None:
                    self._setup_renderer()
                
                if self.renderer is not None:
                    rendered = self.renderer.render(gaussians)
                    if self.render_widget is not None:
                        self.render_widget.set_frame(rendered)
                    
                    QMessageBox.information(
                        self,
                        "Load Successful",
                        f"Loaded {gaussians.num_gaussians} Gaussians from:\n{file_path}",
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Failed to load Gaussians:\n{str(e)}",
                )
    
    def _export_video(self) -> None:
        """Export rendered video."""
        if self.processing_thread is None or not hasattr(self.processing_thread, 'gaussian_merger'):
            QMessageBox.warning(
                self,
                "No Data",
                "No Gaussians to export. Please process a video or webcam first.",
            )
            return
        
        gaussians = self.processing_thread.gaussian_merger.accumulated_gaussians
        if gaussians is None:
            QMessageBox.warning(
                self,
                "No Data",
                "No Gaussians to export. Please process a video or webcam first.",
            )
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Rendered Video",
            "",
            "Video Files (*.mp4);;All Files (*)",
        )
        
        if file_path:
            try:
                import cv2
                
                if self.renderer is None:
                    self._setup_renderer()
                
                if self.renderer is None:
                    QMessageBox.warning(
                        self,
                        "No Renderer",
                        "Failed to initialize renderer for export.",
                    )
                    return
                
                # Render single frame
                rendered = self.renderer.render(gaussians)
                h, w = rendered.shape[:2]
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30
                video_writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))
                
                # Convert and write frame
                output_frame = (rendered * 255).astype(np.uint8)
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(output_frame)
                video_writer.release()
                
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Exported rendered frame to:\n{file_path}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export video:\n{str(e)}",
                )
    
    def _reset_view(self) -> None:
        """Reset novel view."""
        self.rotation_x_slider.setValue(0)
        self.rotation_y_slider.setValue(0)
    
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About GaussCam",
            "GaussCam - Gaussian Splatting Renderer\n\n"
            "Version 0.1.0\n\n"
            "Supports CUDA (NVIDIA) and MPS (Apple Silicon)",
        )
    
    def _update_status(self, message: str, timeout: int = 0) -> None:
        """
        Update status bar message.
        
        Args:
            message: Status message
            timeout: Message timeout in milliseconds (0 = permanent)
        """
        self.statusBar.showMessage(message, timeout)
        logger.debug(f"Status: {message}")
    
    def _update_progress(self, current: int, total: Optional[int] = None) -> None:
        """
        Update progress bar.
        
        Args:
            current: Current progress value
            total: Total progress value (None for indefinite)
        """
        if total is None:
            self.progress_bar.setRange(0, 0)  # Indeterminate
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        
        self.progress_bar.setVisible(True)
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        if self.processing_thread is not None:
            self.processing_thread.stop()
        if self.renderer is not None:
            self.renderer.clear()
        event.accept()


def main():
    """Main entry point."""
    from PySide6.QtWidgets import QApplication
    
    try:
        logger.info("Starting GaussCam application...")
        app = QApplication(sys.argv)
        app.setApplicationName("GaussCam")
        app.setApplicationVersion("0.1.0")
        
        window = MainWindow()
        window.show()
        
        logger.info("GaussCam application started successfully")
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


