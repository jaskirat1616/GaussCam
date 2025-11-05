# GaussCam

A production-ready real-time monocular Gaussian Splatting renderer supporting both CUDA (NVIDIA GPUs) and MPS (Apple Silicon). Features live webcam input, offline video processing, depth estimation, and novel view rendering.

## Features

- **Real-time Gaussian Splatting**: GPU-accelerated rendering with CUDA (NVIDIA) and MPS (Apple Silicon)
- **Monocular Input**: Live webcam feed and offline video file support
- **Depth Estimation**: MiDaS-large model for accurate monocular depth estimation
- **Camera Tracking**: RAFT optical flow for frame-to-frame pose estimation
- **Novel View Rendering**: Interactive camera control for arbitrary viewpoints
- **Cross-Platform**: Windows (CUDA) and macOS (MPS) support

## Requirements

### Windows (CUDA)

- Python 3.10 or higher
- NVIDIA GPU with CUDA 12.0+ support
- CUDA Toolkit 12.0+ (optional, for compilation)
- Windows 10/11

### macOS (MPS)

- Python 3.10 or higher
- Apple Silicon (M1/M2/M3) Mac
- macOS 12.0+ (Monterey or later)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/GaussCam.git
cd GaussCam
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS:**
```bash
source venv/bin/activate
```

### 3. Install PyTorch

#### Windows (CUDA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### macOS (MPS)

```bash
pip install torch torchvision
```

Note: PyTorch on macOS automatically supports MPS for Apple Silicon.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify GPU detection

Run the GPU detection script:

```bash
python -m backend.utils.gpu_detection
```

This should detect either CUDA or MPS depending on your platform.

## Usage

### Webcam Mode

Run the live webcam demo:

```bash
python examples/webcam_demo.py
```

### Offline Video Mode

Process a video file:

```bash
python examples/video_demo.py --input path/to/video.mp4 --output output.mp4
```

### Interactive Novel View

Launch the GUI with novel view controls:

```bash
python examples/novel_view_demo.py
```

### Save/Load Gaussians

Example script for Gaussian persistence:

```bash
python examples/save_load_gaussians.py --input video.mp4 --save gaussians.pkl
```

## Architecture

```
GaussCam/
├── backend/
│   ├── renderer/       # CUDA and MPS rendering backends
│   ├── depth/          # MiDaS depth estimation
│   ├── gaussian/       # Point cloud → Gaussian conversion
│   ├── input/          # Webcam/video capture
│   ├── ui/             # PySide6 GUI
│   ├── utils/          # GPU detection, transforms, helpers
│   └── tests/          # Unit tests
├── examples/           # Demo scripts
└── requirements.txt
```

## Performance

- **Real-time**: 15-60 FPS on mid-range GPUs
- **Webcam latency**: <100ms end-to-end
- **Offline video**: 30+ FPS processing
- **GPU memory**: Efficient pooling for 1M+ Gaussians

## Troubleshooting

### CUDA not detected (Windows)

1. Verify CUDA installation: `nvcc --version`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure GPU drivers are up to date

### MPS not available (macOS)

1. Verify Apple Silicon: `uname -m` should show `arm64`
2. Check PyTorch MPS: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. Ensure macOS 12.0+ (Monterey or later)

### Import errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

## Development

### Running tests

```bash
pytest backend/tests/
```

### Code formatting

```bash
black backend/
```

## License

MIT License

## Acknowledgments

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Original paper and implementation
- [gsplat](https://github.com/nerfstudio-project/gsplat) - CUDA-accelerated Gaussian Splatting library
- [MiDaS](https://github.com/isl-org/MiDaS) - Monocular depth estimation
- [RAFT](https://github.com/princeton-vl/RAFT) - Optical flow estimation

