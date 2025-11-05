# Changelog

All notable changes to GaussCam will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- OUTPUT.md documentation with comprehensive output guide
- Fast point-based rendering fallback for large Gaussian counts (>5000)
- Export capabilities: Save/load Gaussians, export rendered videos
- Pre-downsampling of depth maps for performance optimization
- Frame skipping for real-time processing
- Debug logging throughout processing pipeline

### Changed
- Optimized rendering performance: reduced threshold from 10K to 5K Gaussians
- Switched to uniform Gaussian fitting for better speed
- Reduced max Gaussians from 30K to 10K for real-time performance
- Increased depth skip frames from 3 to 5
- More aggressive point cloud downsampling (voxel size 0.08)

### Fixed
- Fixed shape mismatch in MPS renderer 2D Gaussian projection
- Fixed rendering hanging on large Gaussian counts
- Improved error handling and tracebacks

## [0.1.0] - 2024-11-04

### Added
- Initial release with CUDA and MPS support
- Webcam and video file input support
- MiDaS depth estimation integration
- Gaussian Splatting rendering pipeline
- PySide6 desktop GUI application
- Real-time processing with frame capture
- Novel view rendering capabilities
- Temporal Gaussian merging across frames
- Level of Detail (LOD) controls
- Cross-platform support (Windows + macOS)

### Performance
- Real-time rendering at 15-30 FPS on mid-range GPUs
- Memory-efficient Gaussian representation
- GPU-accelerated processing pipeline

