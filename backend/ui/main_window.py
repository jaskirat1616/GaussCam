"""
Main Window

PySide6 main window for GaussCam application.
"""

import numpy as np
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
from PySide6.QtCore import Qt, QThread, Signal
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
    
    def __init__(self, parent=None):
        """Initialize processing thread."""
        super().__init__(parent)
        self.is_running = False
        self.stop_flag = False
    
    def run(self) -> None:
        """Run processing loop."""
        self.is_running = True
        while not self.stop_flag:
            # Processing loop
            # TODO: Implement actual processing
            pass
        self.is_running = False
    
    def stop(self) -> None:
        """Stop processing thread."""
        self.stop_flag = True
        self.wait()


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
        
        # GPU detection
        self.gpu_detector = get_gpu_detector()
        self.backend = self.gpu_detector.get_backend()
        
        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._setup_renderer()
    
    def _setup_ui(self) -> None:
        """Setup user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Render widget
        self.render_widget = RenderWidget(self, width=640, height=480)
        main_layout.addWidget(self.render_widget, stretch=3)
        
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
        
        # Status label
        status_text = f"Backend: {self.backend.upper()}\nDevice: {self.gpu_detector.device_name}"
        self.status_label = QLabel(status_text)
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
    
    def _setup_renderer(self) -> None:
        """Setup renderer based on GPU backend."""
        try:
            if is_cuda():
                self.renderer = CUDARenderer(width=640, height=480)
            elif is_mps():
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
    
    def _on_input_changed(self, text: str) -> None:
        """Handle input source change."""
        self.video_button.setEnabled(text == "Video File")
    
    def _on_lod_changed(self, value: int) -> None:
        """Handle LOD level change."""
        # TODO: Update renderer LOD
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
            # TODO: Load video file
            pass
    
    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # TODO: Start processing
        pass
    
    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        # TODO: Stop processing
        pass
    
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

