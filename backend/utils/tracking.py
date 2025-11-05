"""
Camera Tracking Module

RAFT optical flow for frame-to-frame camera pose estimation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from backend.utils.gpu_detection import get_device

try:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    from torchvision.utils import flow_to_image
    RAFT_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False
    print("Warning: RAFT not available. Install torchvision>=0.15.0")


class RAFTTracker:
    """RAFT optical flow tracker for camera pose estimation."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize RAFT tracker.
        
        Args:
            device: PyTorch device (auto-detected if None)
        """
        if not RAFT_AVAILABLE:
            raise ImportError("RAFT not available. Install torchvision>=0.15.0")
        
        if device is None:
            device = get_device()
        
        self.device = device
        self.model = None
        self.transforms = None
        self._load_model()
        
        print(f"RAFT tracker initialized on {device}")
    
    def _load_model(self) -> None:
        """Load RAFT model."""
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = raft_large(weights=weights, progress=True)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def estimate_flow(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate optical flow between two images.
        
        Args:
            image1: First image (H, W, 3) RGB in [0, 1] or [0, 255]
            image2: Second image (H, W, 3) RGB in [0, 1] or [0, 255]
        
        Returns:
            (flow, flow_image) where flow is (H, W, 2) and flow_image is (H, W, 3) for visualization
        """
        if self.model is None:
            raise RuntimeError("RAFT model not loaded")
        
        # Normalize to [0, 1] if needed
        if image1.max() > 1.0:
            image1 = image1.astype(np.float32) / 255.0
        if image2.max() > 1.0:
            image2 = image2.astype(np.float32) / 255.0
        
        # Convert to torch tensors
        img1_t = torch.from_numpy(image1).float().to(self.device)
        img2_t = torch.from_numpy(image2).float().to(self.device)
        
        # Convert to (C, H, W) format
        if img1_t.dim() == 3:
            img1_t = img1_t.permute(2, 0, 1).unsqueeze(0)
            img2_t = img2_t.permute(2, 0, 1).unsqueeze(0)
        
        # Apply transforms
        img1_t, img2_t = self.transforms(img1_t, img2_t)
        
        # Estimate flow
        with torch.no_grad():
            flow_list = self.model(img1_t, img2_t)
            flow = flow_list[-1]  # Use final flow
        
        # Convert to numpy
        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
        
        # Convert to visualization image
        flow_img = flow_to_image(flow[0])
        flow_img_np = flow_img.permute(1, 2, 0).cpu().numpy()
        flow_img_np = flow_img_np / 255.0
        
        return flow_np, flow_img_np
    
    def estimate_pose(
        self,
        flow: np.ndarray,
        intrinsics: np.ndarray,
        depth1: Optional[np.ndarray] = None,
        depth2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate camera pose from optical flow.
        
        Args:
            flow: Optical flow (H, W, 2)
            intrinsics: Camera intrinsics matrix (3, 3)
            depth1: Depth map for first image (H, W) optional
            depth2: Depth map for second image (H, W) optional
        
        Returns:
            Camera pose transformation matrix (4, 4)
        """
        # Simplified pose estimation from flow
        # For full implementation, would use depth + flow for 3D-2D correspondence
        
        # For now, return identity (would need proper implementation)
        # This is a placeholder - actual implementation would:
        # 1. Use flow + depth to compute 3D correspondences
        # 2. Use PnP or similar to estimate pose
        
        if depth1 is not None and depth2 is not None:
            # Use depth + flow for pose estimation
            # TODO: Implement full pose estimation
            pass
        
        # Return identity for now
        return np.eye(4, dtype=np.float32)
    
    def track(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        intrinsics: np.ndarray,
        depth1: Optional[np.ndarray] = None,
        depth2: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track camera pose between two frames.
        
        Args:
            image1: First image (H, W, 3)
            image2: Second image (H, W, 3)
            intrinsics: Camera intrinsics matrix (3, 3)
            depth1: Optional depth map for first image
            depth2: Optional depth map for second image
        
        Returns:
            (flow, pose) where flow is (H, W, 2) and pose is (4, 4)
        """
        # Estimate flow
        flow, flow_img = self.estimate_flow(image1, image2)
        
        # Estimate pose
        pose = self.estimate_pose(flow, intrinsics, depth1, depth2)
        
        return flow, pose


class SimpleTracker:
    """Simple frame-to-frame tracker using feature matching (fallback)."""
    
    def __init__(self):
        """Initialize simple tracker."""
        import cv2
        
        # ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def track(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        intrinsics: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track camera pose using feature matching.
        
        Args:
            image1: First image (H, W, 3)
            image2: Second image (H, W, 3)
            intrinsics: Camera intrinsics matrix (3, 3)
        
        Returns:
            (flow, pose) where flow is (H, W, 2) and pose is (4, 4)
        """
        import cv2
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if len(image1.shape) == 3 else image1
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if len(image2.shape) == 3 else image2
        
        # Detect features
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return np.zeros((image1.shape[0], image1.shape[1], 2), dtype=np.float32), np.eye(4, dtype=np.float32)
        
        # Match features
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        if len(pts1) < 8:
            return np.zeros((image1.shape[0], image1.shape[1], 2), dtype=np.float32), np.eye(4, dtype=np.float32)
        
        # Compute homography or essential matrix
        # For now, use homography (simplified)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        
        if H is None:
            return np.zeros((image1.shape[0], image1.shape[1], 2), dtype=np.float32), np.eye(4, dtype=np.float32)
        
        # Compute flow from homography
        h, w = image1.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        pts = np.stack([x.flatten(), y.flatten()], axis=1)
        pts_hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        pts2_proj = (H @ pts_hom.T).T
        pts2_proj = pts2_proj[:, :2] / (pts2_proj[:, 2:3] + 1e-8)
        flow = (pts2_proj - pts).reshape(h, w, 2)
        
        # Estimate pose from homography (simplified)
        # For full implementation, would decompose homography
        pose = np.eye(4, dtype=np.float32)
        
        return flow, pose

