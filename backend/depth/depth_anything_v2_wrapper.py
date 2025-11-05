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
    from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
    from transformers import __version__ as transformers_version
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
    
    # Check if transformers version supports depth_anything architecture
    # Depth Anything V2 support was added in transformers 4.40.0+
    try:
        version_parts = transformers_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        TRANSFORMERS_VERSION_OK = (major > 4) or (major == 4 and minor >= 40)
    except (ValueError, IndexError):
        # If version parsing fails, assume it might be OK (could be dev version)
        TRANSFORMERS_VERSION_OK = True
except ImportError:
    pipeline = None
    AutoImageProcessor = None
    AutoModelForDepthEstimation = None
    Image = None
    transformers_version = None
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_VERSION_OK = False

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
        self.processor = None
        self.pipeline = None
        self.depth_anything_model = None
        
        # Try direct loading first (preferred), fallback to Transformers
        if DEPTH_ANYTHING_V2_AVAILABLE and not use_transformers:
            try:
                logger.info(f"Attempting direct loading from repository...")
                self._load_direct()
                # Verify model was actually loaded
                if self.depth_anything_model is None:
                    raise RuntimeError("Direct loading completed but model is None")
                logger.info(f"Loaded Depth Anything V2 ({model_size}) directly from repository")
                self.use_transformers = False  # Ensure flag is set correctly
            except Exception as e:
                logger.warning(f"Direct loading failed: {e}", exc_info=True)
                logger.info("Falling back to HuggingFace Transformers...")
                
                # Reset direct loading state
                self.depth_anything_model = None
                
                # Check if transformers version is compatible before trying
                if TRANSFORMERS_AVAILABLE and TRANSFORMERS_VERSION_OK:
                    if pipeline is not None:
                        self.use_transformers = True  # Switch to Transformers mode
                        try:
                            self._load_transformers()
                            # Verify Transformers was actually loaded
                            if self.pipeline is None and (self.model is None or self.processor is None):
                                raise RuntimeError("Transformers loading completed but model is None")
                            logger.info("Successfully loaded via Transformers")
                        except Exception as te:
                            logger.error(f"Transformers loading also failed: {te}")
                            raise ImportError(
                                f"Both direct loading and Transformers failed. "
                                f"Direct error: {e}. Transformers error: {te}. "
                                f"Install Depth Anything V2: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
                            )
                    else:
                        raise ImportError(
                            f"Neither direct loading nor Transformers available. "
                            f"Direct loading error: {e}. "
                            f"Install with: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
                        )
                else:
                    # Transformers version too old or not available
                    if TRANSFORMERS_AVAILABLE and not TRANSFORMERS_VERSION_OK:
                        logger.error(f"Transformers version {transformers_version} is too old for Depth Anything V2")
                        logger.error("Depth Anything V2 requires transformers >= 4.40.0")
                        logger.error("")
                        logger.error("Please update transformers:")
                        logger.error("  pip install --upgrade transformers")
                        logger.error("  or")
                        logger.error("  pip install git+https://github.com/huggingface/transformers.git")
                        logger.error("")
                        logger.error("Or install Depth Anything V2 directly from repository:")
                        logger.error("  pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git")
                        raise ImportError(
                            f"Transformers version too old for Depth Anything V2. "
                            f"Current version: {transformers_version}, Required: >= 4.40.0. "
                            f"Update with: pip install --upgrade transformers"
                        )
                    else:
                        raise ImportError(
                            f"Neither direct loading nor Transformers available. "
                            f"Direct loading error: {e}. "
                            f"Install with: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
                        )
        elif use_transformers:
            # Check version before attempting Transformers loading
            if TRANSFORMERS_AVAILABLE and not TRANSFORMERS_VERSION_OK:
                logger.error(f"Transformers version {transformers_version} is too old for Depth Anything V2")
                logger.error("Depth Anything V2 requires transformers >= 4.40.0")
                logger.error("")
                logger.error("Please update transformers:")
                logger.error("  pip install --upgrade transformers")
                logger.error("  or")
                logger.error("  pip install git+https://github.com/huggingface/transformers.git")
                logger.error("")
                logger.error("Or install Depth Anything V2 directly from repository:")
                logger.error("  pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git")
                raise ImportError(
                    f"Transformers version too old for Depth Anything V2. "
                    f"Current version: {transformers_version}, Required: >= 4.40.0. "
                    f"Update with: pip install --upgrade transformers"
                )
            self._load_transformers()
        elif pipeline is not None and not DEPTH_ANYTHING_V2_AVAILABLE:
            # Only try Transformers if version is OK
            if TRANSFORMERS_VERSION_OK:
                self._load_transformers()
            else:
                raise ImportError(
                    f"Transformers version too old for Depth Anything V2. "
                    f"Current version: {transformers_version}, Required: >= 4.40.0. "
                    f"Update with: pip install --upgrade transformers"
                )
        else:
            raise ImportError("Depth Anything V2 not available. Install with: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git")
    
    def _load_transformers(self) -> None:
        """Load model using HuggingFace Transformers."""
        if not TRANSFORMERS_AVAILABLE or Image is None:
            raise ImportError(
                "transformers and Pillow required. Install with: pip install transformers pillow"
            )
        
        # Check transformers version for depth_anything support
        if not TRANSFORMERS_VERSION_OK:
            logger.warning(f"Current transformers version: {transformers_version}")
            logger.warning("Depth Anything V2 requires transformers >= 4.40.0")
            logger.warning("Please update transformers:")
            logger.warning("  pip install --upgrade transformers")
            logger.warning("  or")
            logger.warning("  pip install git+https://github.com/huggingface/transformers.git")
            logger.warning("Attempting to load anyway (may fail)...")
        
        # Use official depth-anything organization models from HuggingFace
        # According to https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
        # These are the official Transformers-compatible models
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf',
        }
        
        if self.model_size not in model_map:
            raise ValueError(
                f"Invalid model size: {self.model_size}. Must be 'small', 'base', or 'large'"
            )
        
        model_name = model_map.get(self.model_size)
        logger.info(f"Loading Depth Anything V2 ({self.model_size}) via Transformers...")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {self.device}")
        if transformers_version:
            logger.info(f"Transformers version: {transformers_version}")
        
        try:
            # Try pipeline first (simpler API)
            # If pipeline fails, try AutoModelForDepthEstimation (more control)
            logger.info(f"Downloading model from HuggingFace (first time only)...")
            logger.info(f"Model will be cached locally after download")
            
            # Use device=-1 for CPU/MPS (MPS doesn't work well with pipeline)
            device_id = 0 if is_cuda() else -1
            logger.debug(f"Using device_id={device_id} for pipeline")
            
            try:
                # Try pipeline approach first
                self.pipeline = pipeline(
                    task="depth-estimation",
                    model=model_name,
                    device=device_id,
                )
                logger.info("Depth Anything V2 model loaded successfully via Transformers pipeline")
            except Exception as pipeline_error:
                error_msg = str(pipeline_error)
                logger.warning(f"Pipeline loading failed: {error_msg}")
                logger.info("Trying AutoModelForDepthEstimation approach...")
                
                # Fallback to AutoModelForDepthEstimation (more reliable)
                try:
                    self.processor = AutoImageProcessor.from_pretrained(model_name)
                    self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info("Depth Anything V2 model loaded successfully via AutoModelForDepthEstimation")
                except Exception as auto_error:
                    error_msg = str(auto_error)
                    logger.error(f"Failed to load {model_name}: {error_msg}")
                    
                    # Check if it's a model not found error
                    if "not a valid model identifier" in error_msg or "not a local folder" in error_msg:
                        logger.error(f"Model {model_name} is not available on HuggingFace.")
                        logger.error("This may be due to:")
                        logger.error("  - Model name is incorrect")
                        logger.error("  - Model requires authentication (private repository)")
                        logger.error("  - Network connectivity issues")
                        logger.error("  - Transformers version too old (need latest version)")
                        logger.error("")
                        logger.error("Try updating transformers:")
                        logger.error("  pip install --upgrade transformers")
                        logger.error("  or")
                        logger.error("  pip install git+https://github.com/huggingface/transformers.git")
                        logger.error("")
                        logger.error("Or install Depth Anything V2 directly from repository:")
                        logger.error("  pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git")
                        raise ImportError(
                            f"Depth Anything V2 model {model_name} not available on HuggingFace. "
                            f"Try updating transformers or install from repository: "
                            f"pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
                        )
                    else:
                        # Check for architecture recognition issues
                        if "'depth_anything'" in error_msg or "depth_anything" in error_msg.lower():
                            logger.error("Transformers library doesn't recognize 'depth_anything' architecture.")
                            logger.error("This means your transformers version is too old.")
                            logger.error("Update transformers to the latest version:")
                            logger.error("  pip install --upgrade transformers")
                            logger.error("  or")
                            logger.error("  pip install git+https://github.com/huggingface/transformers.git")
                        raise auto_error
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load Depth Anything V2 via Transformers: {error_msg}")
            raise
    
    def _load_direct(self) -> None:
        """Load model directly from repository (requires depth_anything_v2 package)."""
        if not DEPTH_ANYTHING_V2_AVAILABLE:
            raise ImportError(
                "depth_anything_v2 package not available. "
                "Install with: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
            )
        
        # Model configurations from the repository (as per README)
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
        
        logger.info(f"Loading Depth Anything V2 ({self.model_size}) directly from repository...")
        logger.info(f"Encoder: {encoder}")
        logger.info(f"Device: {self.device}")
        
        # Create model (following official repository usage)
        # According to https://github.com/DepthAnything/Depth-Anything-V2:
        # model = DepthAnythingV2(**model_configs[encoder])
        self.depth_anything_model = DepthAnythingV2(**config)
        
        # Load checkpoint
        # Checkpoint name format from repository: depth_anything_v2_{encoder}.pth
        # e.g., depth_anything_v2_vits.pth, depth_anything_v2_vitb.pth, depth_anything_v2_vitl.pth
        checkpoint_name = f"depth_anything_v2_{encoder}.pth"
        
        # Try to find checkpoint in common locations
        checkpoint_paths = []
        if self.checkpoint_path:
            checkpoint_paths.append(Path(self.checkpoint_path))
        checkpoint_paths.extend([
            Path("checkpoints") / checkpoint_name,  # Local checkpoints directory
            Path.home() / ".cache" / "depth_anything_v2" / checkpoint_name,  # Cache directory
        ])
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if path and path.exists():
                checkpoint_path = path
                logger.info(f"Found checkpoint at: {checkpoint_path}")
                break
        
        if checkpoint_path is None:
            # Try to download from HuggingFace
            # According to https://github.com/DepthAnything/Depth-Anything-V2
            # Checkpoints are at: https://huggingface.co/depth-anything/Depth-Anything-V2-{Size}/resolve/main/depth_anything_v2_{encoder}.pth
            model_name_capitalized = self.model_size.capitalize()
            # Use the exact URL format from the repository README
            checkpoint_url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_name_capitalized}/resolve/main/{checkpoint_name}"
            logger.info(f"Checkpoint not found locally. Downloading from HuggingFace...")
            logger.info(f"URL: {checkpoint_url}")
            logger.info(f"This matches the official repository: https://github.com/DepthAnything/Depth-Anything-V2")
            
            # Download checkpoint
            import urllib.request
            checkpoint_dir = Path.home() / ".cache" / "depth_anything_v2"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / checkpoint_name
            
            try:
                logger.info(f"Downloading checkpoint (this may take a while)...")
                urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
                logger.info(f"Checkpoint downloaded to: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to download checkpoint: {e}")
                raise FileNotFoundError(
                    f"Checkpoint not found and download failed. "
                    f"Please download {checkpoint_name} from "
                    f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_name_capitalized}/resolve/main/{checkpoint_name} "
                    f"and place it in the 'checkpoints' directory or provide the path."
                )
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
            
            self.depth_anything_model.load_state_dict(checkpoint, strict=False)
            self.depth_anything_model = self.depth_anything_model.to(self.device).eval()
            
            logger.info("Depth Anything V2 model loaded successfully (direct from repository)")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
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
        # Use pipeline if available, otherwise use AutoModel
        if self.pipeline is not None:
            return self._predict_pipeline(image)
        elif self.model is not None and self.processor is not None:
            return self._predict_automodel(image)
        else:
            raise RuntimeError("Model not loaded. Call _load_transformers() first.")
    
    def _predict_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Predict depth using Transformers pipeline."""
        # Convert to PIL Image (already in RGB format from preprocess)
        pil_image = Image.fromarray(image)
        
        # Predict depth using pipeline
        # According to https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf:
        # depth = pipe(image)["depth"]
        try:
            result = self.pipeline(pil_image)
            
            # Debug: log result type and keys if dict
            if isinstance(result, dict):
                logger.debug(f"Pipeline returned dict with keys: {list(result.keys())}")
            
            # Handle different result formats
            # According to the HuggingFace model card, pipeline returns dict with "depth" key
            if isinstance(result, dict):
                # Try "depth" key first (as per official documentation)
                if "depth" in result:
                    depth = result["depth"]
                    logger.debug("Found depth in 'depth' key")
                else:
                    # Try other common keys as fallback
                    depth = None
                    for key in ["predicted_depth", "depth_map", "depth_image"]:
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
    
    def _predict_automodel(self, image: np.ndarray) -> np.ndarray:
        """Predict depth using AutoModelForDepthEstimation."""
        # Convert to PIL Image (already in RGB format from preprocess)
        pil_image = Image.fromarray(image)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict depth
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        h, w = image.shape[:2]
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy and normalize
        depth = prediction.squeeze().cpu().numpy()
        depth_min = depth.min()
        depth_max = depth.max()
        depth_range = depth_max - depth_min + 1e-8
        depth = (depth - depth_min) / depth_range
        
        return depth.astype(np.float32)
    
    def _predict_direct(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth using direct model from repository.
        
        Following official repository usage from https://github.com/DepthAnything/Depth-Anything-V2:
        raw_img = cv2.imread('your/image/path')  # BGR format from OpenCV
        depth = model.infer_image(raw_img)  # HxW raw depth map in numpy
        """
        if self.depth_anything_model is None:
            raise RuntimeError("Model not loaded. Call _load_direct() first.")
        
        # The model's infer_image expects BGR image from OpenCV (as per repository)
        # Our preprocess converts to RGB, so we need to convert back to BGR
        # According to repository: "raw_img = cv2.imread('your/image/path')" - OpenCV uses BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Use the model's infer_image method (as per repository usage)
        # According to repository: "depth = model.infer_image(raw_img) # HxW raw depth map in numpy"
        with torch.no_grad():
            depth = self.depth_anything_model.infer_image(image_bgr)
        
        # Depth is returned as numpy array (H, W) - raw depth map
        # Normalize to [0, 1] for consistency with other depth estimators
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
        
        # Determine which method to use based on what's actually loaded
        if self.pipeline is not None or (self.model is not None and self.processor is not None):
            # Transformers is loaded
            depth = self._predict_transformers(image)
        elif self.depth_anything_model is not None:
            # Direct model is loaded
            depth = self._predict_direct(image)
        else:
            raise RuntimeError(
                "No model loaded. "
                "Either direct loading or Transformers loading must succeed. "
                "Check initialization logs for errors."
            )
        
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

