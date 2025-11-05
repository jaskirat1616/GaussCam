# GaussCam Usage Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the application:**
   ```bash
   python3 main.py
   ```

## Using the GUI

### Webcam Mode

1. Select "Webcam" from the Input Source dropdown
2. Click "Start" button
3. The application will:
   - Capture frames from your webcam
   - Estimate depth using MiDaS
   - Convert to 3D point cloud
   - Fit Gaussians
   - Render Gaussian Splatting output
   - Display in the render widget

### Video File Mode

1. Select "Video File" from the Input Source dropdown
2. Click "Select Video File" button and choose a video file
3. Click "Start" button
4. The application will process the video frame by frame

### Controls

- **LOD Level Slider**: Adjust level of detail (0 = finest, 3 = coarsest)
- **Rotation X/Y Sliders**: Adjust novel view rotation (not yet connected)
- **Start/Stop Buttons**: Control processing

### Status Display

- **Backend**: Shows detected GPU backend (CUDA or MPS)
- **Device**: Shows GPU device name
- **Gaussians**: Shows current number of Gaussians in the scene

## Troubleshooting

### No frames appearing

- Ensure webcam is connected and accessible
- Check that video file path is valid
- Check console for error messages

### Slow performance

- Reduce Gaussian count limit in processing thread
- Use lower resolution input
- Adjust LOD level

### Depth estimation errors

- Ensure transformers library is installed
- Check that MiDaS model downloads successfully
- Verify input frame format is correct

### Rendering errors

- Check GPU availability (CUDA or MPS)
- Verify renderer initialization succeeds
- Check console for specific error messages

## Example Scripts

See `examples/` directory for standalone scripts:
- `webcam_demo.py` - Webcam processing
- `video_demo.py` - Video file processing
- `novel_view_demo.py` - Interactive novel view
- `save_load_gaussians.py` - Gaussian persistence

