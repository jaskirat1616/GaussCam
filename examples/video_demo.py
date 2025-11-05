"""
Video Demo

Offline video Gaussian Splatting demo.
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.input.capture import VideoCapture, FramePreprocessor
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
    parser = argparse.ArgumentParser(description="GaussCam Video Demo")
    parser.add_argument("--input", type=str, required=True, help="Input video file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video file")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--max-gaussians", type=int, default=50000, help="Maximum Gaussians per frame")
    
    args = parser.parse_args()
    
    print("GaussCam Video Demo")
    print("=" * 60)
    
    # GPU detection
    from backend.utils.gpu_detection import get_gpu_detector
    detector = get_gpu_detector()
    print(f"Backend: {detector.get_backend()}")
    print(f"Device: {detector.device_name}")
    
    # Initialize components
    print(f"\nLoading video: {args.input}")
    video = VideoCapture(args.input, loop=False)
    width, height = video.width, video.height
    
    # Frame preprocessor
    preprocessor = FramePreprocessor(normalize=True, to_rgb=True)
    
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
    
    # Video writer
    print(f"\nCreating output video: {args.output}")
    fps = int(video.get_fps())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print("\nProcessing video...")
    frame_count = 0
    max_frames = args.max_frames or video.total_frames
    
    try:
        with tqdm(total=max_frames, desc="Processing frames") as pbar:
            while frame_count < max_frames:
                # Read frame
                ret, frame = video.read()
                if not ret:
                    break
                
                # Preprocess
                processed_frame = preprocessor.process(frame)
                
                # Estimate depth
                depth = depth_estimator.estimate_depth(processed_frame, postprocess=True)
                
                # Convert to point cloud
                points, colors = depth_to_point_cloud(
                    depth, processed_frame, intrinsics, depth_scale=5.0, max_depth=10.0
                )
                
                if len(points) > 0:
                    # Fit Gaussians
                    gaussians = gaussian_fitter.fit_downsampled(
                        points, colors, max_gaussians=args.max_gaussians, method="pca"
                    )
                    
                    # Merge with accumulated
                    merged_gaussians = gaussian_merger.merge(gaussians, merge_strategy="weighted")
                    
                    # Render
                    rendered = renderer.render(merged_gaussians)
                    
                    # Convert to BGR for OpenCV
                    output_frame = (rendered * 255).astype(np.uint8)
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    
                    # Write frame
                    video_writer.write(output_frame)
                
                frame_count += 1
                pbar.update(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        video.release()
        video_writer.release()
        renderer.clear()
        print(f"\nVideo processing finished. Output saved to: {args.output}")


if __name__ == "__main__":
    main()

