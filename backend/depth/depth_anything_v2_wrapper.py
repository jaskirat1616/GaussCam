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
        use_transformers: bool = False,  # Prefer direct loading from repo
        device: Optional[torch.device] = None,
        checkpoint_path: Optional[str] = None,  # Path to checkpoint file
    ):
        """
        Initialize Depth Anything V2 depth estimator.
        
        Args:
            model_size: Model size ('small', 'base', 'large')
            use_transformers: Use HuggingFace Transformers (fallback if direct not available)
            device: PyTorch device (auto-detected if None)
            checkpoint_path: Optional path to checkpoint file (if None, tries to download)
        """
        self.model_size = model_size.lower()
        self.use_transformers = use_transformers
        self.device = device or get_device()
        self.checkpoint_path = checkpoint_path
        
        self.model = None
        self.pipeline = None
        self.depth_anything_model = None
        
        # Try direct loading first (preferred), fallback to Transformers
        if DEPTH_ANYTHING_V2_AVAILABLE and not use_transformers:
            try:
                self._load_direct()
                logger.info(f"Loaded Depth Anything V2 ({model_size}) directly from repository")
            except Exception as e:
                logger.warning(f"Direct loading failed: {e}")
                logger.info("Falling back to HuggingFace Transformers...")
                if pipeline is not None:
                    self._load_transformers()
                else:
                    raise ImportError("Neither direct loading nor Transformers available. Install with: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git")
        elif use_transformers or (pipeline is not None and not DEPTH_ANYTHING_V2_AVAILABLE):
            self._load_transformers()
        else:
            raise ImportError("Depth Anything V2 not available. Install with: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git")
    
    def _load_transformers(self) -> None:
        """Load model using HuggingFace Transformers."""
        if pipeline is None or Image is None:
            raise ImportError(
                "transformers and Pillow required. Install with: pip install transformers pillow"
            )
        
        # Try LiheYoung models first (more reliable on HuggingFace)
        model_map = {
            'small': 'LiheYoung/Depth-Anything-V2-Small-hf',
            'base': 'LiheYoung/Depth-Anything-V2-Base-hf',
            'large': 'LiheYoung/Depth-Anything-V2-Large-hf',
        }
        
        # Alternative model names if the above don't work
        alt_model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf',
        }
        
        if self.model_size not in model_map:
            raise ValueError(
                f"Invalid model size: {self.model_size}. Must be 'small', 'base', or 'large'"
            )
        
        model_name = model_map.get(self.model_size)
        if model_name is None:
            raise ValueError(f"Invalid model size: {self.model_size}")
        
        logger.info(f"Loading Depth Anything V2 ({self.model_size}) via Transformers...")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            # Models are automatically downloaded from HuggingFace on first use
            # and cached locally for future use
            logger.info(f"Downloading model from HuggingFace (first time only)...")
            logger.info(f"Model will be cached locally after download")
            
            # Try primary model name first
            try:
                # Use device=-1 for CPU/MPS (MPS doesn't work well with pipeline)
                device_id = 0 if is_cuda() else -1
                logger.debug(f"Using device_id={device_id} for pipeline")
                
                self.pipeline = pipeline(
                    task="depth-estimation",
                    model=model_name,
                    device=device_id,
                )
                logger.info("Depth Anything V2 model loaded successfully")
            except KeyError as ke:
                # KeyError during model loading - likely model config issue
                error_msg = str(ke)
                logger.warning(f"KeyError loading {model_name}: {error_msg}")
                logger.warning("This suggests the model may not be properly configured on HuggingFace.")
                logger.warning("Trying alternative model name...")
                
                # Try alternative model name
                alt_model_name = alt_model_map.get(self.model_size)
                if alt_model_name:
                    try:
                        self.pipeline = pipeline(
                            task="depth-estimation",
                            model=alt_model_name,
                            device=device_id,
                        )
                        logger.info(f"Depth Anything V2 model loaded successfully (using alternative: {alt_model_name})")
                    except Exception as e2:
                        logger.error(f"Alternative model also failed: {e2}")
                        raise e2
                else:
                    raise ke
            except Exception as e1:
                # Try alternative model name if primary fails
                logger.warning(f"Failed to load {model_name}: {e1}")
                logger.warning("Trying alternative model name...")
                alt_model_name = alt_model_map.get(self.model_size)
                if alt_model_name:
                    try:
                        self.pipeline = pipeline(
                            task="depth-estimation",
                            model=alt_model_name,
                            device=device_id,
                        )
                        logger.info(f"Depth Anything V2 model loaded successfully (using alternative: {alt_model_name})")
                    except Exception as e2:
                        logger.error(f"Alternative model also failed: {e2}")
                        raise e2
                else:
                    raise e1
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load Depth Anything V2 via Transformers: {error_msg}")
            
            # Check if it's a KeyError with 'depth_anything' - this might be a model config issue
            if "'depth_anything'" in error_msg or "depth_anything" in error_msg.lower():
                logger.error("This appears to be a model configuration issue.")
                logger.error("The Depth Anything V2 models may not be available on HuggingFace.")
                logger.error("Note: Depth Anything V2 may require direct loading from the repository.")
                logger.error("Try using MiDaS models instead, or install Depth Anything V2 directly:")
                logger.error("  pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git")
            else:
                logger.error("This may be due to:")
                logger.error("  - Network connectivity issues")
                logger.error("  - Model not available on HuggingFace")
                logger.error("  - HuggingFace authentication required")
            
            logger.error("Models are downloaded automatically from HuggingFace on first use.")
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
        
        # Convert to PIL Image (already in RGB format from preprocess)
        pil_image = Image.fromarray(image)
        
        # Predict depth
        try:
            result = self.pipeline(pil_image)
            
            # Debug: log result type and keys if dict
            if isinstance(result, dict):
                logger.debug(f"Pipeline returned dict with keys: {list(result.keys())}")
            
            # Handle different result formats
            # Some pipelines return dict with 'depth', others return PIL Image directly
            if isinstance(result, dict):
                # Try common keys in order of preference
                depth = None
                for key in ["depth", "predicted_depth", "depth_map", "depth_image"]:
                    if key in result:
                        depth = result[key]
                        logger.debug(f"Found depth in key: {key}")
                        break
                
                if depth is None:
                    # Try to get the first value if it's a dict
                    if result:
                        depth = list(result.values())[0]
                        logger.debug(f"Using first value from dict: {type(depth)}")
                    else:
                        raise ValueError("Pipeline returned empty dictionary")
            elif hasattr(result, 'mode'):  # PIL Image
                depth = result
                logger.debug("Pipeline returned PIL Image")
            else:
                depth = result
                logger.debug(f"Pipeline returned: {type(result)}")
            
            if depth is None:
                raise ValueError("Pipeline returned None or empty result")
                
        except KeyError as e:
            logger.error(f"Depth estimation KeyError: {e}")
            if isinstance(result, dict):
                logger.error(f"Available keys in result: {list(result.keys())}")
            raise ValueError(f"Depth estimation failed - missing key: {e}")
        except Exception as e:
            logger.error(f"Depth estimation error: {e}", exc_info=True)
            if 'result' in locals():
                logger.error(f"Result type: {type(result)}, Result: {result}")
            raise
        
        # Convert to numpy array
        # Transformers pipeline returns PIL Image, convert to numpy
        if hasattr(depth, 'numpy'):
            depth = depth.numpy()
        elif hasattr(depth, 'mode'):  # PIL Image
            depth = np.array(depth, dtype=np.float32)
        else:
            depth = np.array(depth, dtype=np.float32)
        
        # Ensure 2D array
        if len(depth.shape) == 3:
            depth = depth.squeeze()
        if len(depth.shape) != 2:
            raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")
        
        # Normalize to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        depth_range = depth_max - depth_min + 1e-8
        depth = (depth - depth_min) / depth_range
        
        return depth.astype(np.float32)
    
    def _predict_direct(self, image: np.ndarray) -> np.ndarray:
        """Predict depth using direct model from repository."""
        if self.depth_anything_model is None:
            raise RuntimeError("Model not loaded. Call _load_direct() first.")
        
        # Preprocess image (convert to RGB if needed)
        # The model's infer_image expects BGR image from OpenCV
        # But we've already converted to RGB in preprocess, so convert back
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Use the model's infer_image method (as per repository usage)
        with torch.no_grad():
            depth = self.depth_anything_model.infer_image(image_bgr)
        
        # Depth is returned as numpy array (H, W)
        # Normalize to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        depth_range = depth_max - depth_min + 1e-8
        depth = (depth - depth_min) / depth_range
        
        return depth.astype(np.float32)
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            Depth map as numpy array (H, W) normalized to [0, 1]
        """
        return self.predict(image)
    
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

