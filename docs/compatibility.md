# TensorCodec compatibility contract

Reference: **TorchCodec 0.17.0**, CPU audio/video decoding. Python operations,
frame selection, ordering, timestamps, durations, stream selection and metadata
are tested independently and against this pinned version. Native decoding uses
FFmpeg; arrays are returned as NumPy instead of torch.Tensor.

## Public surface

- `tensorcodec.Frame`, `FrameBatch`, `AudioSamples`
- `tensorcodec.decoders.VideoDecoder`: constructor, `len`, integer/slice indexing,
  `get_frame_at`, `get_frames_at`, `get_frames_in_range`, `get_frame_played_at`,
  `get_frames_played_at`, `get_frames_played_in_range`, metadata, stream_index.
  Also `get_all_frames`, FPS resampling and custom JSON frame mappings.
- `tensorcodec.decoders.AudioDecoder`: constructor, metadata, stream_index,
  `get_all_samples`, `get_samples_played_in_range`, resampling and channel mixing.
- NumPy uint8 or float32 video and float32 audio; float64 batch timestamps/durations.
  Float32 RGB uses 16-bit color conversion rather than scaling uint8 output.
- NCHW/NHWC, paths/URLs, encoded bytes, uint8 arrays and seekable file-like input.
- Exact and approximate video seeking; exact is the default.

CUDA, torch inputs, torchvision transforms, HDR tone mapping, rotated video,
encoders and samplers are outside the
initial CPU decoding contract. Unsupported device/transform options fail explicitly.
Do not advertise full-package or torch.Tensor type compatibility.

## Playback rules

Timestamp retrieval selects frame i for `pts[i] <= t < pts[i+1]`; the last frame
ends at the content-derived stream end. This does not imply that returned frame
duration equals the next PTS gap. In 0.17.0 the implementation of time ranges
includes the frame playing at start, even when its PTS is before start. It excludes
the frame starting exactly at stop. This differs from the reference docstring;
we follow the tested implementation, with empty output when start equals stop.
Indices preserve input order and duplicates. Shape/layout, bounds, defaults and
exception classes follow the reference, including its restrictions on slice steps.

## Tests first

Fixtures are generated with FFmpeg from known grayscale frame identities and
explicit timestamp schedules, and with Python's wave module from known PCM.
ffprobe validates encoded packet metadata independently. The original avdec
decoder and tests are not executed. Run the same contract tests against the
reference before adding implementation:

```sh
pytest tests/test_video_contract.py tests/test_audio_contract.py --backend torchcodec
pytest --compare
```

Comparisons check timing separately from pixels. One uint8 RGB unit or 1/65535
for float32 RGB is allowed for conversion rounding; frame identities have independent
expectations. Tests requiring the oracle must fail on a missing/wrong reference
when `--compare` is requested. Production installation does not require torch.

Intentional fix: TensorCodec accepts empty index lists. TorchCodec 0.17.0 infers
float for empty index lists; oracle tests use explicitly typed input tensors to
isolate playback semantics from that conversion bug.
