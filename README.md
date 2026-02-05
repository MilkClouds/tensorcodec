# avdec

**Batch video frame loader for ML training.**

Drop-in TorchCodec API in pure Python — NumPy output, no PyTorch required.

## Features

- **TorchCodec API**: `get_frames_at()`, `get_frames_played_at()`, exact mode semantics
- **Pure Python**: pip install, no compilation, no version conflicts
- **Runs anywhere**: Linux, macOS, Windows, ARM, AMD ROCm
- **Optimized for training**: Efficient sequential decoding, minimal memory copies (benchmarks TBD)

## Installation

```bash
pip install avdec

# Optional: Fast MP4 metadata reading
pip install avdec[mp4]
```

## Quick Start

```python
from avdec import VideoDecoder

# Open video
decoder = VideoDecoder("video.mp4")

# Get frames by index (like TorchCodec)
frames = decoder.get_frames_at([0, 10, 20, 30])
print(frames.data.shape)  # (4, H, W, 3)

# Get frames by timestamp
frames = decoder.get_frames_played_at([0.0, 1.0, 2.0])

# Array-like indexing
first_frame = decoder[0]
every_10th = decoder[::10]

# Context manager
with VideoDecoder("video.mp4") as decoder:
    batch = decoder.get_frames_at([0, 100, 200])

decoder.close()
```

## TorchCodec Compatibility

avdec implements TorchCodec's **exact mode** semantics:

- **Playback frame semantics**: `frame[i].pts <= timestamp < frame[i+1].pts`
- Pre-scans all packets to build accurate frame index
- Same API: `get_frames_at()`, `get_frames_played_at()`

### Key Differences

| Feature | TorchCodec | avdec |
|---------|------------|-------|
| Installation | Complex (PyTorch + FFmpeg version matching) | `pip install avdec` |
| PyTorch dependency | Required (specific version) | None |
| GPU decoding | NVDEC support | CPU only |
| Output format | torch.Tensor (NCHW) | numpy.ndarray (NHWC) |
| Platforms | x86_64 Linux/Mac/Windows | All platforms |

## Seek Modes

```python
from avdec import VideoDecoder, SeekMode

# Exact mode (default): Scans all packets for accurate timing
decoder = VideoDecoder("video.mp4", seek_mode=SeekMode.EXACT)

# Approximate mode: Uses average FPS (faster init, less accurate for VFR)
decoder = VideoDecoder("video.mp4", seek_mode=SeekMode.APPROXIMATE)
```

## Why avdec?

TorchCodec provides excellent ML-optimized video decoding, but suffers from:

1. **PyTorch version lock-in**: Each TorchCodec version requires exact PyTorch version
2. **FFmpeg binary dependencies**: Complex version matching required
3. **Platform limitations**: No ARM binaries, no AMD ROCm support
4. **ABI compatibility issues**: Segfaults with PyTorch 2.9.1+

avdec solves these by using PyAV (Python FFmpeg bindings) with runtime linking:
- No compile-time dependencies
- Works with any PyTorch version (or without PyTorch)
- Runs on all platforms where FFmpeg works

## License

MIT

