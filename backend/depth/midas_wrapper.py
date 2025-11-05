"""
MiDaS Depth Estimation Wrapper

Depth estimation using MiDaS-large model with PyTorch, ONNX Runtime, and CoreML support.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import cv2
from pathlib import Path

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    AutoImageProcessor = None
    AutoModelForDepthEstimation = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from backend.utils.gpu_detection import get_device, is_cuda, is_mps


class MiDaSDepthEstimator:
    """MiDaS-large depth estimation model wrapper."""
    
    def __init__(
        self,
        model_name: str = "Intel/dpt-large",
        use_onnx: bool = False,
        use_coreml: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize MiDaS depth estimator.
        
        Args:
            model_name: HuggingFace model name (Intel/dpt-large or Intel/dpt-hybrid-midas)
            use_onnx: Use ONNX Runtime for inference (faster on CPU/CUDA)
            use_coreml: Use CoreML for Apple Silicon (not yet implemented)
            device: PyTorch device (auto-detected if None)
        """
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.use_coreml = use_coreml
        self.device = device or get_device()
        
        self.model = None
        self.processor = None
        self.onnx_session = None
        
        if use_onnx:
            self._load_onnx()
        else:
            self._load_pytorch()
    
    def _load_pytorch(self) -> None:
        """Load PyTorch model."""
        if AutoImageProcessor is None or AutoModelForDepthEstimation is None:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
        
        print(f"Loading MiDaS model: {self.model_name}")
        print(f"Device: {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("MiDaS model loaded successfully")
    
    def _load_onnx(self) -> None:
        """Load ONNX model (placeholder for future implementation)."""
        if ort is None:
            raise ImportError(
                "onnxruntime required for ONNX inference. Install with: pip install onnxruntime"
            )
        
        # TODO: Implement ONNX model loading and conversion
        # For now, fall back to PyTorch
        print("ONNX loading not yet implemented, falling back to PyTorch")
        self.use_onnx = False
        self._load_pytorch()
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for depth estimation.
        
        Args:
            image: RGB image as numpy array (H, W, 3) in [0, 1] or [0, 255]
        
        Returns:
            Preprocessed tensor
        """
        # Convert to [0, 255] uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Process with MiDaS processor
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth map from image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            Depth map as numpy array (H, W) normalized to [0, 1]
        """
        if self.use_onnx and self.onnx_session is not None:
            return self._predict_onnx(image)
        else:
            return self._predict_pytorch(image)
    
    def _predict_pytorch(self, image: np.ndarray) -> np.ndarray:
        """Predict depth using PyTorch model with optimizations."""
        with torch.no_grad():
            # Use torch.compile for faster inference if available (PyTorch 2.0+)
            if not hasattr(self, '_model_compiled'):
                try:
                    # Try to compile model for faster inference
                    if hasattr(torch, 'compile'):
                        self.model = torch.compile(self.model, mode="reduce-overhead")
                        self._model_compiled = True
                except:
                    self._model_compiled = False
            
            inputs = self.preprocess(image)
            
            # Use inference mode for additional speed
            with torch.inference_mode():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Interpolate to original image size (use bilinear for speed)
            h, w = image.shape[:2]
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bilinear",  # Faster than bicubic
                align_corners=False,
            )
            
            # Convert to numpy and normalize (vectorized)
            depth = prediction.squeeze().cpu().numpy()
            depth_min = depth.min()
            depth_max = depth.max()
            depth_range = depth_max - depth_min + 1e-8
            depth = (depth - depth_min) / depth_range
            
            return depth
    
    def _predict_onnx(self, image: np.ndarray) -> np.ndarray:
        """Predict depth using ONNX Runtime (placeholder)."""
        # TODO: Implement ONNX inference
        return self._predict_pytorch(image)
    
    def postprocess(self, depth: np.ndarray, method: str = "normalize") -> np.ndarray:
        """
        Post-process depth map with accuracy improvements.
        
        Args:
            depth: Raw depth map
            method: Post-processing method ('normalize', 'invert', 'smooth', 'enhanced')
        
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
            # Use larger kernel for better smoothness
            smoothed = cv2.bilateralFilter(depth_uint8, d=9, sigmaColor=75, sigmaSpace=75)
            return smoothed.astype(np.float32) / 255.0
        
        elif method == "enhanced":
            # Enhanced post-processing for better accuracy
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
        postprocess_method: str = "enhanced",
    ) -> np.ndarray:
        """
        Estimate depth map from RGB image with accuracy improvements.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            postprocess: Apply post-processing
            postprocess_method: Post-processing method ('smooth', 'enhanced')
        
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
        if self.onnx_session is not None:
            del self.onnx_session


class AsyncDepthEstimator:
    """Asynchronous depth estimation for processing pipeline."""
    
    def __init__(
        self,
        estimator: MiDaSDepthEstimator,
        queue_size: int = 2,
    ):
        """
        Initialize async depth estimator.
        
        Args:
            estimator: MiDaS depth estimator instance
            queue_size: Maximum queue size
        """
        self.estimator = estimator
        self.queue_size = queue_size
        self.input_queue = None
        self.output_queue = None
        self.stop_event = None
        self.thread = None
        self.is_running = False
    
    def start(self) -> None:
        """Start async depth estimation thread."""
        if self.is_running:
            return
        
        from queue import Queue
        from threading import Thread, Event
        
        self.input_queue = Queue(maxsize=self.queue_size)
        self.output_queue = Queue(maxsize=self.queue_size)
        self.stop_event = Event()
        self.is_running = True
        
        self.thread = Thread(target=self._estimation_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        """Stop async depth estimation thread."""
        self.is_running = False
        if self.stop_event:
            self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _estimation_loop(self) -> None:
        """Depth estimation loop running in background thread."""
        while not self.stop_event.is_set():
            try:
                image = self.input_queue.get(timeout=0.1)
                depth = self.estimator.estimate_depth(image)
                self.output_queue.put(depth)
            except:
                pass
    
    def estimate(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth (non-blocking, returns None if queue full)."""
        if not self.is_running:
            return None
        
        try:
            self.input_queue.put_nowait(image)
        except:
            return None
        
        try:
            return self.output_queue.get_nowait()
        except:
            return None

