"""
Webcam Demo

Live webcam Gaussian Splatting demo.
"""

import sys
import numpy as np
import cv2
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
from backend.utils.gpu_detection import is_cuda, is_mps


def main():
    """Main demo function."""
    print("GaussCam Webcam Demo")
    print("=" * 60)
    
    # GPU detection
    from backend.utils.gpu_detection import get_gpu_detector
    detector = get_gpu_detector()
    print(f"Backend: {detector.get_backend()}")
    print(f"Device: {detector.device_name}")
    
    # Initialize components
    width, height = 640, 480
    
    # Webcam capture
    print("\nInitializing webcam...")
    webcam = WebcamCapture(device_id=0, width=width, height=height, fps=30)
    
    # Frame preprocessor
    preprocessor = FramePreprocessor(normalize=True, to_rgb=True)
    
    # Async capture
    async_capture = AsyncFrameCapture(webcam, preprocessor, queue_size=2)
    async_capture.start()
    
    # Depth estimator
    print("Loading MiDaS depth model...")
    depth_estimator = MiDaSDepthEstimator(model_name="Intel/dpt-large")
    
    # Camera intrinsics
    intrinsics = CameraIntrinsics.default(width, height, fov=60.0)
    
    # Gaussian fitter
    gaussian_fitter = GaussianFitter(k_neighbors=8, initial_opacity=0.9)
    
    # Gaussian merger
    gaussian_merger = GaussianMerger(merge_threshold=0.01, max_gaussians=500000)
    
    # Renderer
    print("Initializing renderer...")
    if is_cuda():
        renderer: Renderer = CUDARenderer(width=width, height=height)
    elif is_mps():
        renderer: Renderer = MPSRenderer(width=width, height=height)
    else:
        print("Error: No GPU available")
        return
    
    print("\nStarting webcam feed...")
    print("Press 'q' to quit")
    
    frame_count = 0
    
    try:
        while True:
            # Read frame
            frame = async_capture.read(timeout=0.1)
            if frame is None:
                continue
            
            # Convert to numpy if needed
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            
            # Estimate depth
            depth = depth_estimator.estimate_depth(frame, postprocess=True)
            
            # Convert to point cloud
            points, colors = depth_to_point_cloud(
                depth, frame, intrinsics, depth_scale=5.0, max_depth=10.0
            )
            
            if len(points) > 0:
                # Fit Gaussians
                gaussians = gaussian_fitter.fit_downsampled(
                    points, colors, max_gaussians=50000, method="pca"
                )
                
                # Merge with accumulated
                merged_gaussians = gaussian_merger.merge(gaussians, merge_strategy="weighted")
                
                # Render
                rendered = renderer.render(merged_gaussians)
                
                # Display
                display_frame = (rendered * 255).astype(np.uint8)
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                
                # Add info
                info_text = f"Gaussians: {merged_gaussians.num_gaussians}, Frame: {frame_count}"
                cv2.putText(
                    display_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                
                cv2.imshow("GaussCam Webcam Demo", display_frame)
            
            frame_count += 1
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        async_capture.stop()
        webcam.release()
        cv2.destroyAllWindows()
        renderer.clear()
        print("\nDemo finished")


if __name__ == "__main__":
    main()

