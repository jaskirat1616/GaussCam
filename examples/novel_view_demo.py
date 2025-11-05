"""
Novel View Demo

Interactive novel view control demo.
"""

import sys
import numpy as np
import cv2
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.input.capture import WebcamCapture, AsyncFrameCapture, FramePreprocessor
from backend.depth.midas_wrapper import MiDaSDepthEstimator
from backend.utils.camera import CameraIntrinsics
from backend.utils.point_cloud import depth_to_point_cloud
from backend.gaussian.fitter import GaussianFitter
from backend.gaussian.merger import GaussianMerger
from backend.renderer.base import Renderer
from backend.renderer.cuda_renderer import CUDARenderer
from backend.renderer.mps_renderer import MPSRenderer
from backend.utils.novel_view import NovelViewController
from backend.utils.gpu_detection import is_cuda, is_mps


class NovelViewDemo:
    """Novel view demo class."""
    
    def __init__(self):
        """Initialize demo."""
        self.width, self.height = 640, 480
        self.setup_components()
        self.setup_novel_view()
        self.setup_controls()
    
    def setup_components(self):
        """Setup components."""
        # Webcam capture
        self.webcam = WebcamCapture(device_id=0, width=self.width, height=self.height, fps=30)
        preprocessor = FramePreprocessor(normalize=True, to_rgb=True)
        self.async_capture = AsyncFrameCapture(self.webcam, preprocessor, queue_size=2)
        self.async_capture.start()
        
        # Depth estimator
        self.depth_estimator = MiDaSDepthEstimator(model_name="Intel/dpt-large")
        
        # Camera intrinsics
        self.intrinsics = CameraIntrinsics.default(self.width, self.height, fov=60.0)
        
        # Gaussian fitter
        self.gaussian_fitter = GaussianFitter(k_neighbors=8, initial_opacity=0.9)
        
        # Gaussian merger
        self.gaussian_merger = GaussianMerger(merge_threshold=0.01, max_gaussians=500000)
        
        # Renderer
        if is_cuda():
            self.renderer: Renderer = CUDARenderer(width=self.width, height=self.height)
        elif is_mps():
            self.renderer: Renderer = MPSRenderer(width=self.width, height=self.height)
        else:
            raise RuntimeError("No GPU available")
        
        self.gaussians = None
    
    def setup_novel_view(self):
        """Setup novel view controller."""
        # Base pose (identity)
        base_pose = np.eye(4, dtype=np.float32)
        base_intrinsics = self.intrinsics.get_matrix()
        
        self.novel_view = NovelViewController(
            base_pose=base_pose,
            base_intrinsics=base_intrinsics,
        )
    
    def setup_controls(self):
        """Setup keyboard controls."""
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.scale = 1.0
    
    def update_novel_view(self):
        """Update novel view parameters."""
        self.novel_view.set_rotation(self.rotation_x, self.rotation_y)
        self.novel_view.set_scale(self.scale)
    
    def run(self):
        """Run demo."""
        print("GaussCam Novel View Demo")
        print("=" * 60)
        print("Controls:")
        print("  W/S: Rotate X axis")
        print("  A/D: Rotate Y axis")
        print("  +/-: Zoom in/out")
        print("  R: Reset view")
        print("  Q: Quit")
        print("=" * 60)
        
        frame_count = 0
        
        try:
            while True:
                # Read frame
                frame = self.async_capture.read(timeout=0.1)
                if frame is None:
                    continue
                
                # Convert to numpy if needed
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                
                # Estimate depth
                depth = self.depth_estimator.estimate_depth(frame, postprocess=True)
                
                # Convert to point cloud
                points, colors = depth_to_point_cloud(
                    depth, frame, self.intrinsics, depth_scale=5.0, max_depth=10.0
                )
                
                if len(points) > 0:
                    # Fit Gaussians
                    gaussians = self.gaussian_fitter.fit_downsampled(
                        points, colors, max_gaussians=50000, method="pca"
                    )
                    
                    # Merge with accumulated
                    self.gaussians = self.gaussian_merger.merge(
                        gaussians, merge_strategy="weighted"
                    )
                
                # Update novel view
                self.update_novel_view()
                
                # Get novel view pose
                novel_pose = self.novel_view.get_pose()
                novel_intrinsics = self.novel_view.get_intrinsics()
                
                # Render with novel view
                if self.gaussians is not None:
                    rendered = self.renderer.render(
                        self.gaussians,
                        camera_pose=novel_pose,
                        intrinsics=novel_intrinsics,
                    )
                    
                    # Display
                    display_frame = (rendered * 255).astype(np.uint8)
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                    
                    # Add info
                    info_text = (
                        f"Gaussians: {self.gaussians.num_gaussians if self.gaussians else 0}, "
                        f"Rotation: ({self.rotation_x:.1f}, {self.rotation_y:.1f}), "
                        f"Scale: {self.scale:.2f}"
                    )
                    cv2.putText(
                        display_frame,
                        info_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    
                    cv2.imshow("GaussCam Novel View Demo", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('w'):
                    self.rotation_x += 5.0
                elif key == ord('s'):
                    self.rotation_x -= 5.0
                elif key == ord('a'):
                    self.rotation_y -= 5.0
                elif key == ord('d'):
                    self.rotation_y += 5.0
                elif key == ord('+') or key == ord('='):
                    self.scale *= 1.1
                elif key == ord('-'):
                    self.scale *= 0.9
                elif key == ord('r'):
                    self.rotation_x = 0.0
                    self.rotation_y = 0.0
                    self.scale = 1.0
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.async_capture.stop()
            self.webcam.release()
            cv2.destroyAllWindows()
            self.renderer.clear()
            print("\nDemo finished")


def main():
    """Main function."""
    demo = NovelViewDemo()
    demo.run()


if __name__ == "__main__":
    main()

