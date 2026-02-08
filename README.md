# avdec

[![PyPI](https://img.shields.io/pypi/v/avdec.svg)](https://pypi.org/project/avdec/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/avdec/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MilkClouds/avdec/blob/main/LICENSE)

[**Installation**](#installation) | [**Quick Start**](#quick-start) | [**API Reference**](#api) | [**Benchmarks**](#benchmarks) | [**Contributing**](#contributing)

**Video decoder for ML training — just `pip install`, NumPy output, no PyTorch required.**

```bash
pip install avdec     # That's it. No build step, all deps come from PyPI.
```

```python
from avdec import VideoDecoder

with VideoDecoder("video.mp4") as decoder:
    batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
    print(batch.data.shape)  # (3, 3, H, W) — NCHW uint8
```

![benchmark](benchmarks/results/readme.png)

> VLA-style random seek + clip read on 64 × 5 min videos (640×480, H.264, 30 fps, GOP 10). [Full results →](benchmarks/RESULTS.md)

---

## Why avdec?

Most video decoders are built for **playback** or **sequential processing**. ML training — especially VLA (Vision-Language-Action) and robotics — does something fundamentally different: **random seek + short clip read**, thousands of times per epoch.

avdec is built around this access pattern:

1. **Truly self-contained `pip install`.** One command, any platform. PyAV wheels [bundle FFmpeg](https://pyav.org/docs/develop/overview/installation.html) — no system packages, no `conda install ffmpeg`, no version matrix. Unlike TorchCodec (which requires a separate FFmpeg installation), avdec's entire dependency tree comes from PyPI.

2. **Playback-frame semantics.** `get_frames_played_at(seconds)` returns the frame that *would be displayed* at each timestamp, matching the semantics of TorchCodec. No off-by-one frame surprises.

3. **I/O-efficient seeking.** On random-access workloads, avdec reads **20 KB/frame** from disk — on par with TorchCodec approximate mode, and **47× less** than TorchCodec exact mode. This matters on network storage and shared clusters.

4. **Video diagnostics built in.** `avdec.doctor()` inspects videos *before* training and reports issues (sparse keyframes, VFR, misplaced moov atom) with concrete `ffmpeg` fix commands.

## Installation

```bash
pip install avdec
```

**Requirements:** Python 3.9+ · `av>=15.0` · `numpy>=1.20`

That's the entire dependency tree — FFmpeg is bundled inside the PyAV wheel. No system packages, no `apt-get`, no `conda install ffmpeg`. Works on **Linux**, **macOS**, and **Windows**.

## Quick Start

### Decode frames at specific timestamps

```python
from avdec import VideoDecoder

with VideoDecoder("video.mp4") as decoder:
    # Frames at specific timestamps (playback semantics)
    batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
    print(batch.data.shape)  # (3, 3, H, W) — NCHW uint8
    print(batch.pts_seconds) # [0.0, 1.0, 2.0]
```

### Decode a time range

```python
with VideoDecoder("video.mp4") as decoder:
    # All frames in [start, stop)
    batch = decoder.get_frames_played_in_range(0.0, 5.0)

    # Resample to a fixed FPS
    batch = decoder.get_frames_played_in_range(0.0, 5.0, fps=10.0)
```

### Inspect video metadata

```python
with VideoDecoder("video.mp4") as decoder:
    m = decoder.metadata
    print(f"{m.num_frames} frames, {m.duration_seconds}s, "
          f"{m.width}×{m.height} @ {float(m.average_rate):.1f} fps")
```

### Diagnose videos for training

```python
import avdec

report = avdec.doctor("video.mp4")
print(report)
```

```
┌ video.mp4
│ mov / h264 / 1920×1080
│ 9000 frames / 300.0s / 30.00fps
└ 648.2 MB

⚠ Random frame access is slow
   To read a random frame from this video,
   you must first decode 150 frames on average.
   ...
   Recommendation: For training, re-encode with keyframe interval of 1 sec

   $ ffmpeg -i "video.mp4" -c:v libx264 -g 30 -c:a copy "video_fixed.mp4"
```

## API

### `VideoDecoder(source, *, stream_index=None)`

Opens a video file for decoding.

- **`get_frames_played_at(seconds: list[float]) -> FrameBatch`** — Returns frames using playback semantics: frame *i* where `frame[i].pts <= timestamp < frame[i+1].pts`.
- **`get_frames_played_in_range(start, stop, fps=None) -> FrameBatch`** — Returns all frames in `[start, stop)`. Pass `fps` to resample to a fixed frame rate.
- **`metadata`** — `VideoStreamMetadata` with `num_frames`, `duration_seconds`, `average_rate`, `width`, `height`, `begin_stream_seconds`, `end_stream_seconds`.
- **`close()`** / context manager — Releases resources.

### `FrameBatch`

Dataclass returned by both methods:

| Field | Type | Description |
|-------|------|-------------|
| `data` | `np.ndarray` | `(N, C, H, W)` uint8 |
| `pts_seconds` | `np.ndarray` | `(N,)` float64 — presentation timestamps |
| `duration_seconds` | `np.ndarray` | `(N,)` float64 — per-frame durations |

### `doctor(path) -> DiagnosticReport`

Analyzes a video for ML training suitability — keyframe intervals, frame rate consistency, moov atom position. Returns actionable findings with `ffmpeg` fix commands.

## Benchmarks

### Decode speed (random seek + clip read)

| Decoder | FPS | Notes |
|---------|----:|-------|
| torchcodec (approximate, thr=1) | **276** | Fastest — requires PyTorch + FFmpeg |
| torchcodec (exact, thr=1) | 239 | |
| **avdec** | **188** | Single `pip install`, no config |
| torchcodec (gpu, approximate) | 174 | Requires CUDA + `set_cuda_backend("beta")` |
| decord | 88 | Unmaintained since 2021 |
| opencv | 58 | |
| torchvision-pyav | 28 | |

### Disk I/O per frame (random seek)

| Decoder | Bytes/Frame | Relative |
|---------|------------:|---------:|
| **avdec** | **20.1 KB** | 1.0× |
| torchcodec (approximate) | 21.4 KB | 1.1× |
| torchcodec (exact) | 952.7 KB | **47.4×** |

For sequential decode, torchcodec with multi-threading dominates (2,325 FPS vs avdec's 654 FPS). avdec is optimized for seek-heavy workloads, not sequential throughput.

→ [**Full benchmark methodology and results**](benchmarks/RESULTS.md)

## Comparison with TorchCodec

Both share **playback-frame semantics** and the same core API (`get_frames_played_at`, `get_frames_played_in_range`). Choose based on your constraints:

| | TorchCodec | avdec |
|---|---|---|
| **Install** | `pip install torchcodec` + separate FFmpeg install | `pip install avdec` (FFmpeg bundled in wheel) |
| **Wheel size** | ~183 MB CPU / ~805 MB CUDA (torch + torchcodec) | ~56 MB (av + numpy) |
| **FFmpeg** | Must install separately (system pkg or conda) | Bundled inside PyAV wheel |
| **Output** | `torch.Tensor` (NCHW) | `numpy.ndarray` (NCHW) |
| **Random seek speed** | 276 FPS (approximate) | 188 FPS |
| **Sequential speed** | 2,325 FPS | 654 FPS |
| **GPU decoding** | ✅ NVDEC | ❌ CPU only |
| **Disk I/O (seek)** | 21.4 KB/frame (approx) | 20.1 KB/frame |
| **Video diagnostics** | ❌ | ✅ `doctor()` |

**Pick avdec when** you want a single `pip install` with no system dependencies, don't need GPU decode, or your pipeline uses NumPy/JAX.
**Pick TorchCodec when** you need maximum throughput and your stack is already PyTorch.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/MilkClouds/avdec).

```bash
git clone https://github.com/MilkClouds/avdec.git
cd avdec
pip install -e ".[dev]"
pytest
```

## License

[MIT](https://github.com/MilkClouds/avdec/blob/main/LICENSE)
