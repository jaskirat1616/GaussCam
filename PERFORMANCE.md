# Performance Optimization Guide

## Overview

GaussCam is optimized for real-time performance on both CUDA (NVIDIA) and MPS (Apple Silicon) platforms. This document outlines performance tuning strategies and benchmarks.

## Performance Metrics

### Target Performance
- **Input Resolution**: 640x480 (configurable)
- **Processing Speed**: 15-30 FPS on mid-range GPUs
- **Gaussian Count**: 5,000-10,000 (adjustable)
- **Memory Usage**: 100-500 MB GPU memory
- **Latency**: <100ms end-to-end

### Current Optimizations

1. **Frame Skipping**: Process every 3rd frame
2. **Depth Skip**: Process depth every 5 frames
3. **Pre-downsampling**: Downsample depth maps before point cloud conversion
4. **Aggressive Downsampling**: Voxel-based point cloud downsampling
5. **Fast Rendering**: Point-based rendering for >5000 Gaussians
6. **Uniform Fitting**: Faster Gaussian fitting method

## Tuning Parameters

### For Better Performance (Lower Quality)
- Increase `frame_skip` to 3-5
- Increase `depth_skip_frames` to 10
- Reduce `max_gaussians` to 5000
- Use `method="uniform"` for Gaussian fitting
- Reduce input resolution to 320x240

### For Better Quality (Lower Performance)
- Decrease `frame_skip` to 1
- Decrease `depth_skip_frames` to 2
- Increase `max_gaussians` to 20000
- Use `method="pca"` for Gaussian fitting
- Increase input resolution to 1280x720

## Platform-Specific Notes

### macOS (MPS)
- MPS has different performance characteristics than CUDA
- Point-based rendering is faster than 2D Gaussian projection on MPS
- Memory bandwidth may be a bottleneck

### Windows/Linux (CUDA)
- CUDA has better parallelization
- 2D Gaussian projection can be used for smaller counts
- GPU memory is less of a constraint

## Benchmarking

Run performance benchmarks:
```bash
python examples/webcam_demo.py --benchmark
```

## Troubleshooting

### Slow Performance
- Check GPU utilization: `nvidia-smi` or Activity Monitor
- Reduce Gaussian count
- Increase frame skipping
- Check if depth estimation is the bottleneck

### High Memory Usage
- Reduce max Gaussians
- Increase downsampling voxel size
- Process fewer frames

