# TensorCodec

[![CI](https://github.com/MilkClouds/tensorcodec/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MilkClouds/tensorcodec/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tensorcodec)](https://pypi.org/project/tensorcodec/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://pypi.org/project/tensorcodec/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

**TorchCodec-style video and audio decoding, without PyTorch.**

- Use the CPU decoder API and playback rules of **TorchCodec 0.17.0**.
- Get **NumPy arrays** instead of `torch.Tensor`.
- Install **NumPy + TensorCodec**. No Torch, PyAV, or FFmpeg CLI at runtime.

The goal is predictable frame selection, timestamps and audio ranges with a small
runtime dependency set. This is a **CPU decoding subset**, not the entire
TorchCodec package. It does not promise a speedup over PyAV or TorchCodec.

## Scope compared with TorchCodec

TorchCodec includes **decoders and encoders**, plus sampling and transforms.
TensorCodec currently implements the **CPU video/audio decoder subset**.

| Module family | TorchCodec 0.17.0 | TensorCodec 0.1.0 |
| --- | --- | --- |
| **Decoders** · video | `VideoDecoder` | **Implemented** for CPU SDR video |
| **Decoders** · audio | `AudioDecoder` | **Implemented** for CPU audio |
| **Decoders** · images | Image decoding APIs | **Not implemented** |
| **Encoders** · video / audio / images | Video, audio, JPEG and PNG encoders | **Not implemented** |
| **Samplers** | Clip sampling APIs | **Not implemented**; decoder FPS queries are available |
| **Transforms** | Decoder transforms | **Not implemented** |

### Decoder compatibility

| Area | Capability | TorchCodec 0.17.0 | TensorCodec 0.1.0 |
| --- | --- | --- | --- |
| **Video · selection** | Index, slice and batch access | Supported | Supported |
| | Playback time and range access | Supported | Supported |
| | Ordering and duplicates | Preserved | Preserved |
| | Exact / approximate seeking | Supported | Supported; exact by default |
| | FPS queries, custom frame mappings | Supported | Supported |
| **Video · formats** | CFR, VFR, offset PTS, B-frames | Supported | Tested |
| | NCHW / NHWC RGB | Supported | Supported |
| | uint8 / float32 output | Supported | Supported for SDR |
| | HDR transfer / display rotation | Supported | **Rejected explicitly** |
| **Audio** | Ranges, resampling, channel mixing | Supported | Supported; float32 output |
| **Input / output** | Paths, URLs, bytes, seekable files | Supported | Supported |
| | Encoded array input | `torch.Tensor` | 1-D uint8 NumPy arrays |
| | Decoded arrays | `torch.Tensor` | `numpy.ndarray`; array interface + DLPack |
| **Execution** | CPU | Supported | Supported |
| | CUDA | Supported | **Not implemented** |
| | Python runtime dependency | PyTorch | NumPy |

### Compatibility means

- Match the supported CPU API's frame selection, ordering, timing and metadata.
- Check behavior independently **and** against pinned TorchCodec 0.17.0.
- Allow color-conversion rounding: at most 1 uint8 unit or 1/65535 for float32
  in the tested cases. Do not claim identical pixels across every FFmpeg build.
- Accept empty index lists, including the case affected by the reference's
  empty-list dtype inference bug.

Details and the tested scope: [compatibility contract](docs/compatibility.md).

### Current limits

- Binary wheels: **Linux x86_64, glibc 2.28+, CPython 3.10+**.
- No macOS or Windows wheels yet; free-threaded Python is not a release target.
- Exact video seeking scans packet timestamps when opening the decoder.
- Audio range queries currently decode from the beginning; late ranges can be
  expensive.
- NumPy return types require caller changes where code expects Torch tensors.
- Historical avdec benchmarks are not TensorCodec performance results.

## Install

```sh
uv pip install tensorcodec
```

Use an existing virtual environment, or create one with `uv venv` first.
Linux wheels bundle shared FFmpeg libraries. Source builds need Rust, libclang
and FFmpeg 7 development headers/libraries.

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
The old avdec decoder and tests are never executed.

- [Playback rules](docs/playback_semantics.md)
- [Release builds and PyPI publishing](docs/releasing.md)
- [Native dependency licenses and source/build notices](licenses/README.md)

TensorCodec's own code is MIT licensed.
