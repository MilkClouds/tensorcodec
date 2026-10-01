<div align="center">

# TensorCodec

CPU video/audio decoding with TorchCodec-style APIs and NumPy output.

<p align="center">
<a href="https://github.com/MilkClouds/tensorcodec/actions/workflows/ci.yml"><img src="https://github.com/MilkClouds/tensorcodec/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
<a href="https://pypi.org/project/tensorcodec/"><img src="https://img.shields.io/pypi/v/tensorcodec" alt="PyPI"></a>
<a href="https://pypi.org/project/tensorcodec/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"></a>
<!-- wheel-size-badge:start -->
<a href="#package-size"><img src="https://img.shields.io/badge/wheel-10.4%20MiB-blue" alt="Wheel download"></a>
<!-- wheel-size-badge:end -->
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License: MIT"></a>
</p>

[Quick start](#quick-start) · [Features](#features) · [Package size](#package-size) · [Compatibility](docs/compatibility.md)

</div>

- **TorchCodec API without PyTorch.** CPU video/audio decoder interfaces follow
  TorchCodec and return NumPy arrays.
- **Validated playback semantics.** Frame selection, ordering, timestamps and audio
  ranges are checked against TorchCodec 0.17.0 and independently generated media.
- **Efficient batch decoding.** Rust/PyO3 bindings to FFmpeg process frame batches
  in a single native call, avoiding per-frame Python calls.
- **Lightweight installation.** Linux wheels are 10.2–10.4 MiB (v0.1.2), including
  FFmpeg shared libraries. NumPy is the only Python dependency.

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

Current source relative to TorchCodec 0.17.0. Native output and macOS wheels
are not in the published 0.1.2 release.
✓ supported · △ partial support · — not implemented.

| Component | TensorCodec | TorchCodec 0.17.0 |
| --- | --- | --- |
| Video decoder | △ CPU, SDR/HDR RGB | ✓ CPU / CUDA |
| Audio decoder | ✓ CPU | ✓ CPU |
| Image decoders | — | ✓ |
| Video / audio / image encoders | — | ✓ |
| Clip samplers | — | ✓ |
| Decoder transforms | — | ✓ |

FPS-based frame queries are supported; clip samplers are a separate API.

### Decoder compatibility

| Capability | TensorCodec | TorchCodec 0.17.0 |
| --- | --- | --- |
| Index / slice / batch selection | ✓ | ✓ |
| Playback timestamp / range queries | ✓ | ✓ |
| Request order and duplicate frames | Preserved | Preserved |
| Exact / approximate seeking | ✓ Default: exact | ✓ |
| FPS queries / custom frame mappings | ✓ | ✓ |
| CFR / VFR / offset PTS / B-frames | ✓ Tested | ✓ |
| NCHW / NHWC RGB output | ✓ | ✓ |
| uint8 / float32 / automatic dtype | ✓ SDR and high-bit-depth video | ✓ |
| uint16 RGB output | ✓ Full-range RGB48 | — |
| Native grayscale/depth and packed RGB(A) | ✓ Values preserved | — |
| PQ / HLG decoding | ✓ Transfer-encoded RGB | ✓ |
| Right-angle display rotation | ✓ | ✓ |
| Audio ranges / resampling / channel mixing | ✓ float32 | ✓ |
| Paths / URLs / bytes / seekable file objects | ✓ | ✓ |
| Encoded array input | 1-D uint8 NumPy array | PyTorch tensor |
| Decoded output | NumPy array; array interface / DLPack | PyTorch tensor |
| CUDA decoding | — | ✓ |

For high-bit-depth video, use `VideoDecoder(path, output_dtype="auto")` to select
float32 above 8 bits, or `output_dtype="uint16"` for full-range 16-bit RGB.
HDR output retains PQ/HLG encoding without SDR tone mapping. Rotation is applied
automatically, and metadata dimensions match the output.

For unmodified samples, use `VideoDecoder(path, output_format="native")`.
Supported formats: `gray`, `gray12le`, `gray16le/be`, `rgb24`, `rgba`.
Native output preserves channel count, integer values and pixel coordinates;
`expected_pixel_format` optionally asserts the source format.

## Package size

<!-- wheel-size:start -->
Linux CPU wheels, Python 3.12. Download / unpacked size in MiB.

| Package | x86_64 | ARM64 |
| --- | ---: | ---: |
| TensorCodec | 10.2 / 24.7 | 10.4 / 22.9 |
| PyAV | 33.4 / 125.5 | 31.2 / 90.4 |
| TorchCodec + PyTorch (CPU) | 196.7 / 704.7 | 160.3 / 585.7 |
<!-- wheel-size:end -->

TensorCodec and PyAV bundle FFmpeg; TorchCodec needs it separately.
Other dependencies are excluded. [Measurements](docs/package_size.md).

## Scope and compatibility

The supported CPU API is checked for frame selection, ordering, timestamps,
durations, stream selection and metadata, both against TorchCodec 0.17.0 and
independently generated media.

- Pixel comparisons allow color-conversion rounding of at most 1 uint8 unit or
  1/65535 for float32 in the tested cases.
- Empty index lists are supported, including the case affected by the reference's
  empty-list dtype inference bug.
- NumPy output preserves the decoder API structure; callers expecting
  `torch.Tensor` must adapt their array handling.

See the [compatibility contract](docs/compatibility.md) and
[playback rules](docs/playback_semantics.md) for the tested behavior.

### Current limits

- **Wheels:** Linux x86_64 and ARM64 (aarch64), glibc 2.17+, CPython 3.10+.
  NumPy must also provide a compatible wheel; newer Python versions may require
  a newer glibc. macOS, Windows, musl/Alpine and free-threaded Python wheels are
  not release targets yet.
- **Exact seeking:** scans packet timestamps when opening the decoder. Incorrect
  container keyframe flags can produce corrupt frames; repaired input or corrected
  frame mappings are needed in that case.
- **Audio ranges:** decode from the beginning, so late ranges can be expensive.
- **Video conversion:** no HDR-to-SDR tone mapping or native YUV-plane output.
  Reflected and non-right-angle display matrices are unsupported.

See [container behavior](docs/container_robustness.md) for seek limitations and
[benchmark tools](benchmarks/README.md) for workload measurements.

## Development and verification

<details>
<summary>Build from source and run tests</summary>

Source builds require Rust, Clang/libclang, pkg-config and FFmpeg 7 development
headers/libraries. Python handles API and playback selection; Rust + PyO3 handles
FFmpeg. Native decoding releases the GIL, allowing separate decoder instances to
run concurrently across Python threads. Calls on the same instance are serialized.

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
