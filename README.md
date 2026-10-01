<div align="center">

# TensorCodec

CPU video/audio decoding with TorchCodec-style APIs and NumPy output.

<p align="center">
<a href="https://github.com/MilkClouds/tensorcodec/actions/workflows/ci.yml"><img src="https://github.com/MilkClouds/tensorcodec/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
<a href="https://pypi.org/project/tensorcodec/"><img src="https://img.shields.io/pypi/v/tensorcodec" alt="PyPI"></a>
<a href="https://pypi.org/project/tensorcodec/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"></a>
<!-- wheel-size-badge:start -->
<a href="#package-size"><img src="https://img.shields.io/badge/wheel-10.9%20MB-blue" alt="Wheel download"></a>
<!-- wheel-size-badge:end -->
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License: MIT"></a>
</p>

[Quick start](#quick-start) · [Features](#features) · [Package size](#package-size) · [Compatibility](docs/compatibility.md)

</div>

- **API.** CPU video/audio decoder interfaces follow TorchCodec, returning NumPy
  arrays instead of PyTorch tensors. PyTorch is not a dependency.
- **Playback semantics.** Frame selection, ordering, timestamps and audio ranges
  are validated against TorchCodec 0.17.0 and independently generated media.
- **Implementation.** Rust/PyO3 bindings to FFmpeg decode each frame batch in one
  native call, releasing the GIL during decoding.
- **Distribution.** Linux wheels are 10.7–10.9 MB (v0.1.1), including FFmpeg shared
  libraries. NumPy is the only Python dependency.

## Quick start

```sh
uv pip install tensorcodec
```

Use an existing virtual environment, or create one with `uv venv` first.
No separate FFmpeg installation is needed for the published Linux wheels.

```python
from tensorcodec.decoders import VideoDecoder, AudioDecoder

with VideoDecoder("video.mp4") as video:
    frame = video[0]                                  # RGB array: (C, H, W)
    batch = video.get_frames_at([4, 0, 4])              # requested order, including duplicates
    clip = video.get_frames_played_in_range(0, 1, fps=8)

with AudioDecoder("audio.wav", sample_rate=16000, num_channels=1) as audio:
    samples = audio.get_samples_played_in_range(0, 1)
    waveform = samples.data                           # float32: (channels, samples)
```

Arrays keep their storage after the decoder closes. Paths, URLs, encoded bytes,
1-D uint8 arrays and seekable file objects are supported.

## Features

| Capability | Supported |
| --- | --- |
| **Frame selection** | Indexing, slicing, batches, playback timestamps and FPS sampling |
| **Seeking** | Exact by default; approximate mode and custom frame mappings |
| **Video output** | NCHW / NHWC RGB; uint8, uint16 and float32 |
| **Video fidelity** | High-bit-depth output, PQ/HLG signals and right-angle display rotation |
| **Audio** | Time ranges, resampling and channel mixing; float32 output |
| **Interoperability** | NumPy arrays, array interface and DLPack |

For high-bit-depth video, use `VideoDecoder(path, output_dtype="auto")` to select
float32 above 8 bits, or `output_dtype="uint16"` for full-range 16-bit RGB.
HDR output retains PQ/HLG encoding without SDR tone mapping. Rotation is applied
automatically, and metadata dimensions match the output.

## Package size

The runtime dependencies for CPU video/audio decoding are:

| Runtime footprint | TensorCodec | TorchCodec 0.17.0 |
| --- | --- | --- |
| Python packages | TensorCodec + NumPy | TorchCodec + PyTorch and its dependencies |
| FFmpeg shared libraries | Bundled in the wheel | Separate installation |
| Decoder output | NumPy arrays | PyTorch tensors |

TensorCodec's wheel size includes its FFmpeg runtime. TorchCodec's excludes
PyTorch and FFmpeg, so the standalone wheel sizes below are not complete
installation sizes. Existing dependencies can be reused by either package.

<details>
<summary>Published wheel sizes and measurement details</summary>

<!-- wheel-size:start -->
Published PyPI Linux wheels, CPython 3.10: TensorCodec 0.1.1 / TorchCodec 0.17.0.

| Package | Architecture | Download | Unpacked |
| --- | --- | ---: | ---: |
| TensorCodec | x86_64 | 10.7 MB | 25.9 MB |
| TensorCodec | aarch64 | 10.9 MB | 24.0 MB |
| TorchCodec | x86_64 | 10.0 MB | 23.4 MB |
| TorchCodec | aarch64 | 8.8 MB | 21.9 MB |
<!-- wheel-size:end -->

MB = 1,000,000 bytes. Download is the wheel archive; unpacked is the sum of its
entries. Both exclude external dependencies. The badge shows the largest
published TensorCodec Linux wheel. These are PyPI builds; PyTorch's CPU/CUDA
indexes can provide different artifacts.

[Measurement and release policy](docs/package_size.md) · [Use an existing FFmpeg installation](docs/system_ffmpeg.md)

</details>

## Scope and compatibility

TensorCodec targets TorchCodec 0.17.0's **CPU video/audio playback semantics**:
frame selection, ordering, timestamps and audio ranges. Tests compare these
against the pinned reference and independently generated media.

- **Wheels:** Linux x86_64 and ARM64, glibc 2.17+, CPython 3.10+; requires a compatible NumPy wheel.
- **Scope:** CPU decoding. CUDA, images, encoders, clip samplers and decoder transforms are not implemented.
- **Seeking cost:** exact mode scans packet timestamps on open; audio ranges decode from the beginning.

See the [compatibility contract](docs/compatibility.md) for pixel tolerances and
API differences, and [container behavior](docs/container_robustness.md) for seek
limitations. Measure performance on your workload with the [benchmark tools](benchmarks/README.md).

## Development and verification

<details>
<summary>Build from source and run tests</summary>

Source builds require Rust, Clang/libclang, pkg-config and FFmpeg 7 development
headers/libraries. Python handles API and playback selection; Rust + PyO3 handles
FFmpeg. Each batch crosses the native boundary once, releasing the GIL during decoding.

```sh
uv sync --group dev --group oracle

uv run --group oracle pytest tests/test_video_contract.py tests/test_audio_contract.py --backend torchcodec
uv run --group oracle pytest --compare

# Rebuild after changing Rust code.
uv run --group oracle maturin develop --locked --uv
```

Tests generate media with FFmpeg/ffprobe and Python's `wave` module.
`--compare` requires the pinned oracle; differential tests otherwise skip.

[Playback rules](docs/playback_semantics.md) · [Release guide](docs/releasing.md) · [Dependency licenses](licenses/README.md)

</details>

TensorCodec's own code is [MIT licensed](LICENSE).
