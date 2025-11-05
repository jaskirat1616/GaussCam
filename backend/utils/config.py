"""
Configuration Management

Centralized configuration management with validation and defaults.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    # Frame skipping
    frame_skip_fast: int = 15
    frame_skip_balanced: int = 10
    frame_skip_quality: int = 5
    
    # Depth estimation
    depth_skip_fast: int = 20
    depth_skip_balanced: int = 15
    depth_skip_quality: int = 10
    
    # Gaussian fitting
    max_gaussians_fast: int = 3000
    max_gaussians_balanced: int = 5000
    max_gaussians_quality: int = 10000
    
    # Point cloud
    target_pixels_fast: int = 15000
    target_pixels_balanced: int = 20000
    target_pixels_quality: int = 30000
    
    voxel_size_fast: float = 0.15
    voxel_size_balanced: float = 0.10
    voxel_size_quality: float = 0.08
    
    # Resolution
    target_size_fast: int = 256
    target_size_balanced: int = 320
    target_size_quality: int = 384
    
    # Video-specific
    video_frame_skip_multiplier: float = 1.5  # Multiply skip for video
    video_delay_ms: int = 50
    webcam_delay_ms: int = 200


@dataclass
class RendererConfig:
    """Renderer configuration."""
    width: int = 640
    height: int = 480
    background_color: tuple = (0.0, 0.0, 0.0)
    max_gaussians_2d_projection: int = 5000
    max_gaussians_fast_render: int = 10000


@dataclass
class AppConfig:
    """Application configuration."""
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    
    # Logging
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    
    # UI
    window_width: int = 1200
    window_height: int = 800
    
    # Performance
    enable_gpu: bool = True
    enable_adaptive_quality: bool = True
    target_fps: float = 15.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """Create from dictionary."""
        processing = ProcessingConfig(**data.get('processing', {}))
        renderer = RendererConfig(**data.get('renderer', {}))
        
        return cls(
            processing=processing,
            renderer=renderer,
            log_level=data.get('log_level', 'INFO'),
            log_dir=data.get('log_dir'),
            window_width=data.get('window_width', 1200),
            window_height=data.get('window_height', 800),
            enable_gpu=data.get('enable_gpu', True),
            enable_adaptive_quality=data.get('enable_adaptive_quality', True),
            target_fps=data.get('target_fps', 15.0),
        )


class ConfigManager:
    """Configuration manager with file persistence."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config file (default: config.json in project root)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.json"
        
        self.config_path = config_path
        self.config = AppConfig()
        
        # Load existing config if available
        if config_path.exists():
            try:
                self.load()
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
    
    def load(self) -> None:
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.info(f"Config file not found: {self.config_path}, using defaults")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix == '.yaml' or self.config_path.suffix == '.yml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self.config = AppConfig.from_dict(data)
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise
    
    def get(self) -> AppConfig:
        """Get current configuration."""
        return self.config
    
    def update(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")
        
        self.save()


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> AppConfig:
    """Get global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.get()


def save_config() -> None:
    """Save global configuration."""
    global _config_manager
    if _config_manager is not None:
        _config_manager.save()


__all__ = ['AppConfig', 'ProcessingConfig', 'RendererConfig', 'ConfigManager', 'get_config', 'save_config']

