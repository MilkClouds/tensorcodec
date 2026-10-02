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
- NumPy uint8, uint16 or float32 video and float32 audio; float64 batch timestamps/durations.
  Float32 RGB uses 16-bit color conversion rather than scaling uint8 output.
- NCHW/NHWC, paths/URLs, encoded bytes, uint8 arrays and seekable file-like input.
- Exact and approximate video seeking; exact is the default.
- Decoder transforms (`tensorcodec.transforms`): `Resize` (bilinear, antialiased), `CenterCrop`
  and `RandomCrop`, applied in order to RGB frames after display rotation, as TorchCodec does
  (frames match TorchCodec 0.17.0 within one level). The TorchCodec and TorchVision v2
  counterparts are accepted and converted; other transforms fail explicitly. `RandomCrop`
  draws its position once per decoder from NumPy's global random state. Native output
  takes no transforms.

CUDA, torch inputs, HDR tone mapping, arbitrary-angle rotation,
video/audio encoders and samplers are outside the
CPU decoding contract. Unsupported device/transform options fail explicitly.
Do not advertise full-package or torch.Tensor type compatibility.

Image decoding and JPEG/PNG encoding have a separate [image API contract](images.md),
including supported formats, array layouts and compatibility limits.

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
ffprobe validates encoded packet metadata independently. Run the same contract
tests against the reference before adding implementation:

```sh
uv run --group oracle pytest tests/test_video_contract.py tests/test_audio_contract.py --backend torchcodec
uv run --group oracle pytest --compare
```

Comparisons check timing separately from pixels. One uint8 RGB unit or 1/65535
for float32 RGB is allowed for conversion rounding; frame identities have independent
expectations. Tests requiring the oracle must fail on a missing/wrong reference
when `--compare` is requested. Production installation does not require torch.

Intentional fix: TensorCodec accepts empty index lists. TorchCodec 0.17.0 infers
float for empty index lists; oracle tests use explicitly typed input tensors to
isolate playback semantics from that conversion bug.

## Video fidelity extensions

PQ/HLG inputs decode as transfer-encoded RGB without tone mapping. `auto` uses
float32 for source component depths above 8 bits, including high-depth SDR.
Explicit uint16 returns full-range RGB48 and is a TensorCodec extension; it is
not native YUV output. Source `bit_depth` and `color_range` metadata are also
TensorCodec extensions. Right-angle display rotations are applied automatically;
metadata dimensions describe the rotated output. Reflected and non-right-angle
display matrices remain unsupported. Color metadata and pixel aspect ratio
describe the source; HDR output is not linear light or sRGB.

## Native video output

`output_format="native"` bypasses color conversion and display transforms for
`gray`, `gray12le`, `gray16le`, `gray16be`, `rgb24` and `rgba`. NCHW/NHWC keeps
1, 3 or 4 channels. Output uses host-endian uint8/uint16 without range scaling;
`output_dtype` may be omitted, `"auto"`, or the matching integer dtype.
`expected_pixel_format` asserts the source layout. Unsupported formats, dtype
conversions and changes of pixel format or dimensions fail explicitly.

Frame/FrameBatch `pixel_format` records the native source format and survives
indexing and FPS resampling. RGB results retain their existing behavior. Native
mode preserves encoded pixel coordinates, including inputs with display matrices.
Playback selection follows the same TorchCodec contract as RGB, including the
frame overlapping a range's start; it does not copy PyAV's legacy PTS-only range rule.
## Timestamp mode

`seek_mode="timestamp"` is an opt-in TensorCodec extension. Existing `exact` and
`approximate` modes are unchanged. It skips the initial full packet scan and
selects frames by `PTS[i] <= t < PTS[i+1]`, without average-FPS conversion.
The final frame requires a positive decoded duration; requests beyond that
duration, before the first frame, or with nonfinite times fail explicitly.
Missing/non-increasing decoded timestamps also fail rather than guessing.

Queries are sorted and deduplicated, then returned in the caller's order. Nearby
queries share decoding and one-frame lookahead; known later container keyframes
allow seeking across gaps. Overshooting seeks retry at exponentially earlier
positions, down to the stream start. If no frame at/before the request can be
recovered, decoding fails. Bad or absent indexes can still require substantial I/O.

Frame indices, slices, `len`, `get_all_frames`, custom mappings and ranges without
explicit `fps` are unsupported. FPS ranges retain the regular mode's resampling
timestamps and durations. Header timing remains advisory; `metadata.num_frames`
and content-derived metadata are `None`. No complete frame index is implied.
RGB/native formats, dtype, rotation, array ownership and file-like input follow
the existing output contracts. Decoder construction may still probe media data.
