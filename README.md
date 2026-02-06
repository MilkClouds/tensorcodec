# avdec

**Pure-Python video decoder for ML training — NumPy output, no PyTorch required.**

## Features

- **Playback-frame semantics**: `get_frames_played_at()`, `get_frames_played_in_range()`
- **NCHW output**: `(N, 3, H, W)` uint8 NumPy arrays, ready for ML pipelines
- **Pure Python**: `pip install avdec` — no compilation, no version conflicts
- **Video diagnostics**: `doctor()` analyzes videos for ML training suitability

## Installation

```bash
pip install avdec
```

Dependencies: `av>=15.0`, `numpy>=1.20`. Python 3.9+.

## Quick Start

```python
from avdec import VideoDecoder

with VideoDecoder("video.mp4") as decoder:
    # Frames at specific timestamps
    batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
    print(batch.data.shape)  # (3, 3, H, W) — NCHW uint8

    # All frames in a time range
    batch = decoder.get_frames_played_in_range(0.0, 5.0)

    # Resample to a fixed FPS
    batch = decoder.get_frames_played_in_range(0.0, 5.0, fps=10.0)

    # Metadata
    print(decoder.metadata.duration_seconds)
    print(decoder.metadata.average_rate)
```

## API

### `VideoDecoder(source)`

Opens a video file for decoding.

- **`get_frames_played_at(seconds: list[float]) -> FrameBatch`** — Returns frames using playback semantics: frame *i* where `frame[i].pts <= timestamp < frame[i+1].pts`.
- **`get_frames_played_in_range(start, stop, fps=None) -> FrameBatch`** — Returns all frames in `[start, stop)`. Pass `fps` to resample to a fixed frame rate.
- **`metadata`** — `VideoStreamMetadata` with `num_frames`, `duration_seconds`, `average_rate`, `width`, `height`, `begin_stream_seconds`, `end_stream_seconds`.
- **`close()`** / context manager — Releases resources.

### `FrameBatch`

Dataclass returned by both methods:

- `data`: `np.ndarray` — `(N, C, H, W)` uint8
- `pts_seconds`: `np.ndarray` — `(N,)` float64
- `duration_seconds`: `np.ndarray` — `(N,)` float64

### `doctor(path) -> DiagnosticReport`

Analyzes a video for ML training suitability — keyframe intervals, frame rate consistency, moov atom position.

```python
import avdec

report = avdec.doctor("video.mp4")
print(report)
```

## vs TorchCodec

Both share **playback-frame semantics** (`frame[i].pts <= timestamp < frame[i+1].pts`) and the same core API (`get_frames_played_at`, `get_frames_played_in_range`).

| Feature | TorchCodec | avdec |
|---------|------------|-------|
| Dependencies | PyTorch + FFmpeg version matching | `pip install avdec` |
| Output format | `torch.Tensor` (NCHW) | `numpy.ndarray` (NCHW) |
| GPU decoding | NVDEC support | CPU only |

## License

MIT
