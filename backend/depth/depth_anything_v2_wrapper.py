"""
Depth Anything V2 Wrapper

Depth estimation using Depth Anything V2 models (Small, Base, Large).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import cv2
from pathlib import Path

try:
    from transformers import pipeline
    from PIL import Image
except ImportError:
    pipeline = None
    Image = None

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_ANYTHING_V2_AVAILABLE = True
except ImportError:
    DEPTH_ANYTHING_V2_AVAILABLE = False
    DepthAnythingV2 = None

from backend.utils.gpu_detection import get_device, is_cuda, is_mps
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class DepthAnythingV2Estimator:
    """Depth Anything V2 depth estimation model wrapper."""
    
    def __init__(
        self,
        model_size: str = "small",  # 'small', 'base', 'large'
        use_transformers: bool = True,  # Use HuggingFace Transformers
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Depth Anything V2 depth estimator.
        
        Args:
            model_size: Model size ('small', 'base', 'large')
            use_transformers: Use HuggingFace Transformers (recommended)
            device: PyTorch device (auto-detected if None)
        """
        self.model_size = model_size.lower()
        self.use_transformers = use_transformers
        self.device = device or get_device()
        
        self.model = None
        self.pipeline = None
        self.depth_anything_model = None
        
        if use_transformers:
            self._load_transformers()
        else:
            self._load_direct()
    
    def _load_transformers(self) -> None:
        """Load model using HuggingFace Transformers."""
        if pipeline is None or Image is None:
            raise ImportError(
                "transformers and Pillow required. Install with: pip install transformers pillow"
            )
        
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf',
        }
        
        if self.model_size not in model_map:
            raise ValueError(
                f"Invalid model size: {self.model_size}. Must be 'small', 'base', or 'large'"
            )
        
        model_name = model_map[self.model_size]
        
        logger.info(f"Loading Depth Anything V2 ({self.model_size}) via Transformers...")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            self.pipeline = pipeline(
                task="depth-estimation",
                model=model_name,
                device=0 if is_cuda() else -1 if is_mps() else -1,
            )
            logger.info("Depth Anything V2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Depth Anything V2 via Transformers: {e}")
            raise
    
    def _load_direct(self) -> None:
        """Load model directly (requires depth_anything_v2 package)."""
        if not DEPTH_ANYTHING_V2_AVAILABLE:
            raise ImportError(
                "depth_anything_v2 package required. "
                "Install with: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
            )
        
        model_configs = {
            'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        if self.model_size not in model_configs:
            raise ValueError(
                f"Invalid model size: {self.model_size}. Must be 'small', 'base', or 'large'"
            )
        
        config = model_configs[self.model_size]
        encoder = config['encoder']
        
        logger.info(f"Loading Depth Anything V2 ({self.model_size}) directly...")
        logger.info(f"Encoder: {encoder}")
        logger.info(f"Device: {self.device}")
        
        try:
            self.depth_anything_model = DepthAnythingV2(**config)
            
            # Load checkpoint
            checkpoint_path = f"checkpoints/depth_anything_v2_{encoder}.pth"
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}. "
                    "Please download from https://github.com/DepthAnything/Depth-Anything-V2"
                )
            
            self.depth_anything_model.load_state_dict(
                torch.load(checkpoint_path, map_location='cpu')
            )
            self.depth_anything_model = self.depth_anything_model.to(self.device).eval()
            
            logger.info("Depth Anything V2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Depth Anything V2 directly: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for depth estimation.
        
        Args:
            image: RGB image as numpy array (H, W, 3) in [0, 1] or [0, 255]
        
        Returns:
            Preprocessed image
        """
        # Ensure image is in [0, 255] range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # OpenCV uses BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _predict_transformers(self, image: np.ndarray) -> np.ndarray:
        """Predict depth using HuggingFace Transformers."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call _load_transformers() first.")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Predict depth
        result = self.pipeline(pil_image)
        depth = result["depth"]
        
        # Convert to numpy array
        depth = np.array(depth, dtype=np.float32)
        
        # Normalize to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        depth_range = depth_max - depth_min + 1e-8
        depth = (depth - depth_min) / depth_range
        
        return depth
    
    def _predict_direct(self, image: np.ndarray) -> np.ndarray:
        """Predict depth using direct model loading."""
        if self.depth_anything_model is None:
            raise RuntimeError("Model not loaded. Call _load_direct() first.")
        
        # Use the model's infer_image method
        depth = self.depth_anything_model.infer_image(image)
        
        # Normalize to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        depth_range = depth_max - depth_min + 1e-8
        depth = (depth - depth_min) / depth_range
        
        return depth
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth map from RGB image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            Depth map (H, W) normalized to [0, 1]
        """
        image = self.preprocess(image)
        
        if self.use_transformers:
            depth = self._predict_transformers(image)
        else:
            depth = self._predict_direct(image)
        
        return depth
    
    def postprocess(self, depth: np.ndarray, method: str = "normalize") -> np.ndarray:
        """
        Post-process depth map.
        
        Args:
            depth: Raw depth map
            method: Post-processing method ('normalize', 'invert', 'smooth', 'improved')
        
        Returns:
            Processed depth map
        """
        if method == "normalize":
            # Already normalized in predict
            return depth
        
        elif method == "invert":
            # Invert depth (far objects become bright)
            return 1.0 - depth
        
        elif method == "smooth":
            # Apply bilateral filtering for smoothness while preserving edges
            depth_uint8 = (depth * 255).astype(np.uint8)
            smoothed = cv2.bilateralFilter(depth_uint8, d=9, sigmaColor=75, sigmaSpace=75)
            return smoothed.astype(np.float32) / 255.0
        
        elif method == "improved":
            # Post-processing for better accuracy
            # 1. Bilateral filtering for edge-preserving smoothing
            depth_uint8 = (depth * 255).astype(np.uint8)
            smoothed = cv2.bilateralFilter(depth_uint8, d=9, sigmaColor=75, sigmaSpace=75)
            
            # 2. Remove noise with median filter
            denoised = cv2.medianBlur(smoothed, ksize=5)
            
            # 3. Normalize back to [0, 1]
            result = denoised.astype(np.float32) / 255.0
            
            # 4. Ensure valid range
            result = np.clip(result, 0.0, 1.0)
            
            return result
        
        return depth
    
    def estimate_depth(
        self,
        image: np.ndarray,
        postprocess: bool = True,
        postprocess_method: str = "improved",
    ) -> np.ndarray:
        """
        Estimate depth map from RGB image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            postprocess: Apply post-processing
            postprocess_method: Post-processing method ('smooth', 'improved')
        
        Returns:
            Depth map (H, W) normalized to [0, 1]
        """
        depth = self.predict(image)
        
        if postprocess:
            depth = self.postprocess(depth, method=postprocess_method)
        
        return depth
    
    def __del__(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
        if self.pipeline is not None:
            del self.pipeline
        if self.depth_anything_model is not None:
            del self.depth_anything_model

