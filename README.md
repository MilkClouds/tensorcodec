# TensorCodec

NumPy audio/video decoding with the CPU playback semantics of **TorchCodec
0.17.0**, without a PyTorch runtime dependency.

Python implements the public API and frame-selection rules. A small Rust/PyO3
extension owns FFmpeg decoding, seeking, color conversion and resampling.
The sole Python runtime dependency is NumPy. Linux wheels bundle shared FFmpeg
libraries; source builds need FFmpeg development libraries and Rust.

```python
from tensorcodec.decoders import VideoDecoder, AudioDecoder

with VideoDecoder("video.mp4") as video:
    frame = video.get_frame_played_at(1.25)
    print(frame.data.shape, frame.pts_seconds)  # CHW NumPy array
    batch = video.get_frames_at([4, 0, 4])      # preserves order and duplicates
    clip = video.get_frames_played_in_range(0, 1, fps=8)

with AudioDecoder("audio.wav", sample_rate=16000, num_channels=1) as audio:
    samples = audio.get_samples_played_in_range(0, 1)
    print(samples.data.shape)                 # channels × samples, float32
```

Video supports uint8 and high-precision float32 RGB, NCHW/NHWC layout, exact or
approximate seeking, custom frame mappings and stream metadata. Inputs can be
paths, URLs, encoded bytes, 1-D uint8 arrays or seekable binary file objects.
NumPy arrays provide the array interface and DLPack interoperability, and keep
their storage after a decoder closes. File objects remain caller-owned.

The initial scope is CPU video/audio decoding. CUDA, transforms, HDR tone
mapping, rotated video, encoders and samplers are unsupported. Output types are
NumPy arrays; this is not full TorchCodec package compatibility. Audio range
queries currently decode from the beginning, so late queries can be expensive.
See [the compatibility contract](docs/compatibility.md) for playback rules and
verification boundaries. No historical avdec benchmark is a TensorCodec result.

## Installation

Install from PyPI:

```sh
python -m pip install tensorcodec
```

For a checkout before publication, install a local wheel with
`python -m pip install dist/tensorcodec-*.whl`.

Release wheels target Linux x86_64, glibc 2.28+, and the CPython stable ABI
starting at Python 3.10. macOS and Windows wheels are not yet provided. Wheels require neither a system FFmpeg executable nor Torch/PyAV.

## Development

Install Rust, Python 3.10+, Clang/libclang and pkg-config. For a system-library
build, install **FFmpeg 7 development headers/libraries**, then:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install numpy pytest ruff 'maturin>=1.8,<2'
maturin develop
```

Alternatively build the shared FFmpeg libraries bundled in Linux wheels. This
needs curl, make, a C compiler, NASM on x86 and OpenSSL development headers:

```sh
scripts/build_ffmpeg.sh "$PWD/.ffmpeg"
export FFMPEG_DIR="$PWD/.ffmpeg"
export LD_LIBRARY_PATH="$FFMPEG_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
maturin develop
maturin build --release --auditwheel repair --out dist
```

`maturin` needs `patchelf` for wheel repair. Use `--locked` when invoking Cargo
checks; `native/Cargo.lock` records the Rust dependency graph.

## Verification

Tests generate their own fixtures with the FFmpeg/ffprobe command-line tools;
those tools are development dependencies only. Fixtures cover CFR/VFR, offset
PTS, B-frames, multiple streams, color matrices, odd frame widths, PCM, AAC,
MP3, FLAC, resampling, ownership and concurrent decoders. The previous avdec
implementation and tests are never executed.

Install the pinned reference only in the test environment:

```sh
python -m pip install torch==2.14.1 torchcodec==0.17.0 \
  --index-url https://download.pytorch.org/whl/cpu
pytest tests/test_video_contract.py tests/test_audio_contract.py --backend torchcodec
pytest --compare
ruff check src/tensorcodec tests
cargo fmt --manifest-path native/Cargo.toml --check
cargo clippy --manifest-path native/Cargo.toml --locked -- -D warnings
```

`--compare` fails if the reference is unavailable or the wrong version. Without
it, differential tests explicitly skip. Timing and pixels are checked separately;
RGB comparisons allow documented conversion-rounding tolerances.

TensorCodec is MIT licensed. Bundled native dependencies have their own
[licenses and source/build notices](licenses/README.md).

[Release setup and publishing](docs/releasing.md) use PyPI Trusted Publishing.
