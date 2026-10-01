# TensorCodec

**TorchCodec-style video and audio decoding, without PyTorch.**

- Use the CPU decoder API and playback rules of **TorchCodec 0.17.0**.
- Get **NumPy arrays** instead of `torch.Tensor`.
- Install **NumPy + TensorCodec**. No Torch, PyAV, or FFmpeg CLI at runtime.

The goal is predictable frame selection, timestamps and audio ranges with a small
runtime dependency set. This is a **CPU decoding subset**, not the entire
TorchCodec package. It does not promise a speedup over PyAV or TorchCodec.

## What is available?

| Capability | TorchCodec 0.17.0 | TensorCodec 0.1.0 |
| --- | --- | --- |
| Python runtime dependency | PyTorch | NumPy |
| Output arrays | `torch.Tensor` | `numpy.ndarray`; array interface + DLPack |
| Video index/slice/batch access | Supported | Supported |
| Playback time/range access | Supported | Supported |
| CFR, VFR, offset PTS, B-frames | Supported | Tested |
| Request ordering and duplicates | Preserved | Preserved |
| Exact / approximate seeking | Supported | Supported; exact is the default |
| NCHW / NHWC RGB | Supported | Supported |
| uint8 / float32 video | Supported | Supported for SDR |
| FPS sampling, custom frame mappings | Supported | Supported |
| Audio ranges, resampling, channel mixing | Supported | Supported; float32 output |
| Paths, URLs, bytes, seekable file objects | Supported | Supported |
| Encoded tensor input | `torch.Tensor` | 1-D uint8 NumPy arrays |
| CUDA decoding | Supported | **Not implemented** |
| Decoder transforms | Supported | **Not implemented** |
| HDR inputs / display rotation | Supported | **Rejected explicitly** |
| Other modules, including samplers/encoders | Available | **Outside the initial scope** |

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
python -m pip install tensorcodec
```

Linux wheels bundle shared FFmpeg libraries. Source builds need Rust, libclang
and FFmpeg 7 development headers/libraries.

Before the first PyPI upload, install a local wheel:

```sh
python -m pip install dist/tensorcodec-*.whl
```

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
# Requires Rust, Clang/libclang, pkg-config and FFmpeg 7 development libraries.
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pytest ruff 'maturin>=1.8,<2'
maturin develop --locked

# Reference dependencies are for tests only.
python -m pip install torch==2.14.1 torchcodec==0.17.0 \
  --index-url https://download.pytorch.org/whl/cpu
pytest tests/test_video_contract.py tests/test_audio_contract.py --backend torchcodec
pytest --compare
```

Tests generate fixtures with FFmpeg/ffprobe and Python's `wave` module.
`--compare` requires the exact oracle version; otherwise differential tests skip.
The old avdec decoder and tests are never executed.

- [Playback rules](docs/playback_semantics.md)
- [Release builds and PyPI publishing](docs/releasing.md)
- [Native dependency licenses and source/build notices](licenses/README.md)

TensorCodec's own code is MIT licensed.
