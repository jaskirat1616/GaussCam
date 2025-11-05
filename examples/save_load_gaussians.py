"""
Save/Load Gaussians Demo

Demonstrates saving and loading Gaussian splats.
"""

import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

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


def save_gaussians(gaussians, filepath: str):
    """
    Save Gaussians to file.
    
    Args:
        gaussians: Gaussian object
        filepath: Output file path
    """
    data = {
        "centroids": gaussians.centroids,
        "scales": gaussians.scales,
        "rotations": gaussians.rotations,
        "colors": gaussians.colors,
        "opacity": gaussians.opacity,
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved {gaussians.num_gaussians} Gaussians to {filepath}")


def load_gaussians(filepath: str):
    """
    Load Gaussians from file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Gaussian object
    """
    from backend.gaussian.fitter import Gaussian
    
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    gaussians = Gaussian(
        centroids=data["centroids"],
        covariances=None,
        colors=data["colors"],
        opacity=data["opacity"],
        scales=data["scales"],
        rotations=data["rotations"],
    )
    
    print(f"Loaded {gaussians.num_gaussians} Gaussians from {filepath}")
    return gaussians


def process_video(input_path: str, output_path: str, max_frames: int = None):
    """
    Process video and save Gaussians.
    
    Args:
        input_path: Input video file
        output_path: Output Gaussians file
        max_frames: Maximum frames to process
    """
    print("GaussCam Save/Load Gaussians Demo")
    print("=" * 60)
    
    # GPU detection
    from backend.utils.gpu_detection import get_gpu_detector
    detector = get_gpu_detector()
    print(f"Backend: {detector.get_backend()}")
    print(f"Device: {detector.device_name}")
    
    # Initialize components
    print(f"\nLoading video: {input_path}")
    video = VideoCapture(input_path, loop=False)
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
    gaussian_merger = GaussianMerger(merge_threshold=0.01, max_gaussians=1000000)
    
    print("\nProcessing video...")
    frame_count = 0
    max_frames = max_frames or video.total_frames
    
    try:
        from tqdm import tqdm
        
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
                        points, colors, max_gaussians=50000, method="pca"
                    )
                    
                    # Merge with accumulated
                    merged_gaussians = gaussian_merger.merge(gaussians, merge_strategy="weighted")
                
                frame_count += 1
                pbar.update(1)
        
        # Save final Gaussians
        final_gaussians = gaussian_merger.accumulated_gaussians
        if final_gaussians is not None:
            save_gaussians(final_gaussians, output_path)
        else:
            print("Error: No Gaussians to save")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        # Save partial Gaussians
        final_gaussians = gaussian_merger.accumulated_gaussians
        if final_gaussians is not None:
            save_gaussians(final_gaussians, output_path)
    
    finally:
        video.release()
        print(f"\nProcessing finished. Gaussians saved to: {output_path}")


def render_gaussians(gaussians_path: str, output_video: str = None):
    """
    Load and render saved Gaussians.
    
    Args:
        gaussians_path: Path to saved Gaussians file
        output_video: Optional output video file
    """
    print(f"\nLoading Gaussians from: {gaussians_path}")
    gaussians = load_gaussians(gaussians_path)
    
    # Renderer
    print("Initializing renderer...")
    width, height = 640, 480
    
    if is_cuda():
        renderer: Renderer = CUDARenderer(width=width, height=height)
    elif is_mps():
        renderer: Renderer = MPSRenderer(width=width, height=height)
    else:
        print("Error: No GPU available")
        return
    
    # Render
    print("Rendering Gaussians...")
    rendered = renderer.render(gaussians)
    
    # Display
    import cv2
    display_frame = (rendered * 255).astype(np.uint8)
    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Loaded Gaussians", display_frame)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    renderer.clear()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="GaussCam Save/Load Gaussians Demo")
    parser.add_argument("--input", type=str, help="Input video file")
    parser.add_argument("--save", type=str, help="Save Gaussians to file")
    parser.add_argument("--load", type=str, help="Load Gaussians from file")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    if args.input and args.save:
        # Process video and save
        process_video(args.input, args.save, max_frames=args.max_frames)
    elif args.load:
        # Load and render
        render_gaussians(args.load)
    else:
        print("Usage:")
        print("  Process video: --input <video> --save <output.pkl>")
        print("  Load Gaussians: --load <input.pkl>")


if __name__ == "__main__":
    main()

