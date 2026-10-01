# TensorCodec

[![CI](https://github.com/MilkClouds/tensorcodec/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MilkClouds/tensorcodec/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tensorcodec)](https://pypi.org/project/tensorcodec/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://pypi.org/project/tensorcodec/)
<!-- wheel-size-badge:start -->
[![Wheel download](https://img.shields.io/badge/wheel-10.9%20MB-blue)](#package-size)
<!-- wheel-size-badge:end -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

**TorchCodec-style video and audio decoding, without PyTorch.**

- Use the CPU decoder API and playback rules of TorchCodec 0.17.0.
- Get NumPy arrays instead of `torch.Tensor`.
- Install NumPy + TensorCodec. No Torch, PyAV, or FFmpeg CLI at runtime.

The goal is predictable frame selection, timestamps and audio ranges with a small
runtime dependency set. This is a **CPU decoding subset**, not the entire
TorchCodec package. It does not promise a speedup over PyAV or TorchCodec.

## Scope compared with TorchCodec

TorchCodec includes decoders, encoders, sampling and transforms.

Legend for both tables: ✓ supported · △ limited support · — unavailable.

| Module family | Component | TorchCodec 0.17.0 | TensorCodec 0.1.1 |
| --- | --- | :---: | --- |
| Decoders | Video · `VideoDecoder` | ✓ | △ CPU, SDR/HDR RGB |
| | Audio · `AudioDecoder` | ✓ | △ CPU |
| | Images | ✓ | — |
| Encoders | Video / audio / JPEG / PNG | ✓ | — |
| Samplers | Clip sampling | ✓ | — |
| Transforms | Decoder transforms | ✓ | — |

FPS-based decoder queries are available; clip samplers are not implemented.

### Decoder compatibility

| Area | Capability | TorchCodec 0.17.0 | TensorCodec 0.1.1 |
| --- | --- | :---: | --- |
| Video · selection | Index / slice / batch | ✓ | ✓ |
| | Playback time / range | ✓ | ✓ |
| | Order / duplicates preserved | ✓ | ✓ |
| | Exact / approximate seek | ✓ | ✓ Default: exact |
| | FPS queries / custom frame mappings | ✓ | ✓ |
| Video · formats | CFR / VFR / offset PTS / B-frames | ✓ | ✓ Tested |
| | NCHW / NHWC RGB | ✓ | ✓ |
| | uint8 / float32 | ✓ | ✓ SDR/HDR; `auto` preserves high-depth precision |
| | uint16 RGB | — | ✓ Full-range RGB48 |
| | HDR transfer / display rotation | ✓ | △ PQ/HLG signal preservation; right-angle rotations |
| Audio | Ranges / resampling / channel mixing | ✓ | ✓ float32 |
| Input / output | Paths / URLs / bytes / seekable files | ✓ | ✓ |
| | Encoded array input | `torch.Tensor` | 1-D uint8 NumPy arrays |
| | Decoded arrays | `torch.Tensor` | `numpy.ndarray` + array interface / DLPack |
| Execution | CPU | ✓ | ✓ |
| | CUDA | ✓ | — |
| | Python runtime dependency | PyTorch | NumPy |

### Compatibility means

- Match the supported CPU API's frame selection, ordering, timing and metadata.
- Check behavior independently and against pinned TorchCodec 0.17.0.
- Allow color-conversion rounding: at most 1 uint8 unit or 1/65535 for float32
  in the tested cases. Do not claim identical pixels across every FFmpeg build.
- Accept empty index lists, including the case affected by the reference's
  empty-list dtype inference bug.

Details and the tested scope: [compatibility contract](docs/compatibility.md).

### Current limits

- Binary wheels: **Linux x86_64 / ARM64 (aarch64), glibc 2.17+, CPython 3.10+**.
- A compatible NumPy wheel is also required. On older glibc, the installer may
  select an older NumPy; newer Python versions may require a newer glibc.
- No macOS, Windows or musl/Alpine wheels yet; free-threaded Python is not a release target.
- Exact video seeking scans packet timestamps when opening the decoder.
- Seeking trusts container keyframe flags; incorrect flags can corrupt decoded
  frames. Corrected frame mappings or a repaired input are needed in that case.
- Audio range queries currently decode from the beginning; late ranges can be
  expensive.
- NumPy return types require caller changes where code expects Torch tensors.

## Install

```sh
uv pip install tensorcodec
```

Use an existing virtual environment, or create one with `uv venv` first.
Linux wheels bundle shared FFmpeg libraries. Source builds need Rust, libclang
and FFmpeg 7 development headers/libraries.

## Package size

<!-- wheel-size:start -->
Published PyPI Linux wheels, CPython 3.10: TensorCodec 0.1.1 / TorchCodec 0.17.0.

| Package | Architecture | Download | Unpacked |
| --- | --- | ---: | ---: |
| TensorCodec | x86_64 | 10.7 MB | 25.9 MB |
| TensorCodec | aarch64 | 10.9 MB | 24.0 MB |
| TorchCodec | x86_64 | 10.0 MB | 23.4 MB |
| TorchCodec | aarch64 | 8.8 MB | 21.9 MB |
<!-- wheel-size:end -->

Sizes include everything inside each wheel; external dependencies are excluded.
MB = 1,000,000 bytes. Unpacked size is the sum of archive entries, not filesystem
usage or total environment size. The badge shows the largest published TensorCodec
Linux wheel download.

| Runtime requirement | TensorCodec | TorchCodec 0.17.0 |
| --- | --- | --- |
| Python dependency | NumPy | PyTorch (install separately) |
| FFmpeg shared libraries | Included, minimal FFmpeg 7 | Install separately |
| Image codec libraries | No image API | Included |

TorchCodec's wheel alone is smaller in this comparison. TensorCodec's installation
advantage is avoiding PyTorch and a separate FFmpeg setup, rather than the smallest
standalone wheel. These are PyPI artifacts; PyTorch's separate CPU/CUDA indexes can
provide different builds. This is a size comparison, not a feature or speed comparison.

Release limits per TensorCodec wheel: 15 MiB download / 35 MiB unpacked
(1 MiB = 1,048,576 bytes). [Measurement and release policy](docs/package_size.md).
For an existing FFmpeg installation, see the [source-build guide](docs/system_ffmpeg.md).

## Use

```python
from tensorcodec.decoders import VideoDecoder, AudioDecoder

with VideoDecoder("video.mp4") as video:
    frame = video.get_frame_played_at(1.25)  # frame playing at this time
    print(frame.data.shape)                 # CHW NumPy array
    batch = video.get_frames_at([4, 0, 4])   # order and duplicates preserved
    clip = video.get_frames_played_in_range(0, 1, fps=8)

with AudioDecoder("audio.wav", sample_rate=16000, num_channels=1) as audio:
    samples = audio.get_samples_played_in_range(0, 1)
    print(samples.data.shape)               # channels × samples, float32
```

Decoded arrays keep their storage after the decoder closes. Input file objects
remain caller-owned.

Use `output_dtype="auto"` to preserve high-depth video as float32, or
`output_dtype=np.uint16` for full-range 16-bit RGB. PQ/HLG output preserves the
encoded signal; it is not tone mapped to SDR. Right-angle display rotation is
applied automatically, and metadata dimensions match the output.

## Implementation

| Layer | Responsibility |
| --- | --- |
| Python | Public API, frame/time selection, validation, result objects |
| Rust + PyO3 | FFmpeg handles, seeking/decoding, color conversion, resampling |
| FFmpeg | Codec and container implementations |

A batch crosses the Python/Rust boundary once. Native decoding releases the GIL.

## Development and verification

```sh
# Requires uv, Rust, Clang/libclang, pkg-config and FFmpeg 7 development libraries.
# Build the editable package and install development + pinned CPU oracle groups.
uv sync --group dev --group oracle

uv run --group oracle pytest tests/test_video_contract.py tests/test_audio_contract.py --backend torchcodec
uv run --group oracle pytest --compare

# Rebuild after changing Rust code.
uv run --group oracle maturin develop --locked --uv
```

Tests generate fixtures with FFmpeg/ffprobe and Python's `wave` module.
`--compare` requires the exact oracle version; otherwise differential tests skip.

- [Benchmark guide](benchmarks/README.md)
- [Container and seek behavior](docs/container_robustness.md)
- [Playback rules](docs/playback_semantics.md)
- [Release builds and PyPI publishing](docs/releasing.md)
- [Native dependency licenses and source/build notices](licenses/README.md)

TensorCodec's own code is MIT licensed.
