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
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QKeySequence
from typing import Optional
import sys

from backend.ui.render_widget import RenderWidget
from backend.utils.gpu_detection import get_gpu_detector, is_cuda, is_mps
from backend.renderer.base import Renderer
from backend.renderer.cuda_renderer import CUDARenderer
from backend.renderer.mps_renderer import MPSRenderer


class ProcessingThread(QThread):
    """Background thread for processing frames."""
    
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    
    def __init__(
        self,
        input_source: str,
        video_path: Optional[str] = None,
        renderer=None,
        parent=None,
    ):
        """
        Initialize processing thread.
        
        Args:
            input_source: 'webcam' or 'video'
            video_path: Path to video file (if input_source is 'video')
            renderer: Renderer instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.input_source = input_source
        self.video_path = video_path
        self.renderer = renderer
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
                self.capture = WebcamCapture(device_id=0, width=width, height=height, fps=30)
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
            depth_skip_frames = 5  # Process depth every N frames (increased for speed)
            frame_skip = 2  # Skip N frames between processing (process every Nth frame)
            last_depth = None
            
            while not self.stop_flag:
                # Read frame
                frame = self.async_capture.read(timeout=0.1)
                if frame is None:
                    if self.input_source == "video":
                        # End of video
                        break
                    self.msleep(10)
                    continue
                
                print(f"Frame {frame_count} read: shape={frame.shape if hasattr(frame, 'shape') else 'unknown'}")
                
                # Skip frames for performance
                if frame_count % (frame_skip + 1) != 0:
                    frame_count += 1
                    self.msleep(10)
                    continue
                
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
                    print(f"Estimating depth for frame {frame_count}...")
                    try:
                        # Resize frame for faster depth estimation (if too large)
                        h, w = frame.shape[:2]
                        if h > 480 or w > 640:
                            scale = min(480 / h, 640 / w)
                            small_h, small_w = int(h * scale), int(w * scale)
                            small_frame = cv2.resize(frame, (small_w, small_h))
                            depth = self.depth_estimator.estimate_depth(small_frame, postprocess=True)
                            depth = cv2.resize(depth, (w, h))
                        else:
                            depth = self.depth_estimator.estimate_depth(frame, postprocess=True)
                        last_depth = depth
                        print(f"Depth estimated: shape={depth.shape}, min={depth.min():.3f}, max={depth.max():.3f}")
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
                print(f"Converting to point cloud...")
                try:
                    # Downsample depth map first for speed
                    h, w = depth.shape[:2]
                    if h * w > 50000:  # If too many pixels, downsample first
                        scale = np.sqrt(50000 / (h * w))
                        small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
                        small_depth = cv2.resize(depth, (small_w, small_h))
                        small_frame = cv2.resize(frame, (small_w, small_h))
                        points, colors = depth_to_point_cloud(
                            small_depth, small_frame, self.intrinsics, depth_scale=5.0, max_depth=10.0
                        )
                    else:
                        points, colors = depth_to_point_cloud(
                            depth, frame, self.intrinsics, depth_scale=5.0, max_depth=10.0
                        )
                    print(f"Point cloud: {len(points)} points")
                    
                    # Aggressive downsampling for speed
                    if len(points) > 5000:  # Lower threshold
                        from backend.utils.point_cloud import downsample_point_cloud
                        voxel_size = 0.08  # Larger voxels for more aggressive downsampling
                        points, colors = downsample_point_cloud(points, colors, voxel_size=voxel_size)
                        print(f"Downsampled to {len(points)} points")
                except Exception as e:
                    print(f"Point cloud conversion error: {e}")
                    import traceback
                    traceback.print_exc()
                    self.msleep(100)
                    continue
                
                if len(points) > 0:
                    try:
                        # Fit Gaussians (use uniform method for speed)
                        print(f"Fitting Gaussians from {len(points)} points...")
                        gaussians = self.gaussian_fitter.fit_downsampled(
                            points, colors, max_gaussians=10000, method="uniform"  # Faster uniform method
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
                                # Emit frame for display
                                self.frame_ready.emit(rendered)
                                print(f"Frame emitted to UI")
                            except Exception as e:
                                print(f"Rendering error: {e}")
                                import traceback
                                traceback.print_exc()
                    except Exception as e:
                        print(f"Gaussian fitting/merging error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Emit last frame or blank frame if no points
                    if self.renderer is not None and self.gaussian_merger.accumulated_gaussians is not None:
                        try:
                            rendered = self.renderer.render(self.gaussian_merger.accumulated_gaussians)
                            self.frame_ready.emit(rendered)
                        except:
                            pass
                
                frame_count += 1
                
                # Small delay to prevent CPU spinning
                self.msleep(100)  # Increased delay for better performance balance
        
        except Exception as e:
            import traceback
            error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
        finally:
            # Cleanup
            if self.async_capture is not None:
                self.async_capture.stop()
            if self.capture is not None:
                self.capture.release()
            self.is_running = False
    
    def stop(self) -> None:
        """Stop processing thread."""
        self.stop_flag = True
        self.wait(3000)  # Wait up to 3 seconds


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.setWindowTitle("GaussCam - Gaussian Splatting")
        self.setGeometry(100, 100, 1280, 720)
        
        # Initialize components
        self.renderer: Optional[Renderer] = None
        self.processing_thread: Optional[ProcessingThread] = None
        self.video_path: Optional[str] = None
        
        # GPU detection (defer to avoid Metal conflicts during PySide6 init)
        self.gpu_detector = None
        self.backend = "unknown"
        
        # Setup UI first
        self._setup_ui()
        self._setup_menu()
        
        # Initialize GPU detection after UI is ready
        self._init_gpu_detection()
    
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
        
        self.gaussian_count_label = QLabel("Gaussians: 0")
        render_layout.addWidget(self.gaussian_count_label)
        
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
    
    def _on_input_changed(self, text: str) -> None:
        """Handle input source change."""
        self.video_button.setEnabled(text == "Video File")
    
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
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if file_path:
            self.video_path = file_path
            # Update input combo to video
            self.input_combo.setCurrentText("Video File")
            print(f"Video file selected: {file_path}")
    
    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        # Initialize renderer if not already done
        if self.renderer is None:
            self._setup_renderer()
        
        if self.renderer is None:
            QMessageBox.warning(
                self,
                "No Renderer",
                "Failed to initialize renderer. Cannot start processing.",
            )
            return
        
        # Get input source
        input_source = self.input_combo.currentText().lower()
        if input_source == "video file":
            input_source = "video"
            if not self.video_path:
                QMessageBox.warning(
                    self,
                    "No Video File",
                    "Please select a video file first.",
                )
                return
        elif input_source == "webcam":
            input_source = "webcam"
        else:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please select an input source (Webcam or Video File).",
            )
            return
        
        # Stop existing thread if running
        if self.processing_thread is not None and self.processing_thread.isRunning():
            self.processing_thread.stop()
        
        # Create and start processing thread
        self.processing_thread = ProcessingThread(
            input_source=input_source,
            video_path=self.video_path if input_source == "video" else None,
            renderer=self.renderer,
            parent=self,
        )
        self.processing_thread.frame_ready.connect(self._on_frame_ready)
        self.processing_thread.error_occurred.connect(self._on_processing_error)
        self.processing_thread.finished.connect(self._on_processing_finished)
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.processing_thread.start()
        
        print(f"Started processing: {input_source}")
    
    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        if self.processing_thread is not None and self.processing_thread.isRunning():
            self.processing_thread.stop()
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        print("Stopped processing")
    
    def _on_frame_ready(self, frame: np.ndarray) -> None:
        """Handle frame ready signal from processing thread."""
        if self.render_widget is not None:
            self.render_widget.set_frame(frame)
        
        # Update Gaussian count
        if hasattr(self, 'gaussian_count_label') and self.processing_thread is not None:
            if hasattr(self.processing_thread, 'gaussian_merger') and \
               self.processing_thread.gaussian_merger is not None:
                gaussians = self.processing_thread.gaussian_merger.accumulated_gaussians
                if gaussians is not None:
                    self.gaussian_count_label.setText(f"Gaussians: {gaussians.num_gaussians}")
                else:
                    self.gaussian_count_label.setText("Gaussians: 0")
    
    def _on_processing_error(self, error_msg: str) -> None:
        """Handle processing error."""
        print(f"Processing error: {error_msg}")
        QMessageBox.critical(
            self,
            "Processing Error",
            f"An error occurred during processing:\n{error_msg}",
        )
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
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


