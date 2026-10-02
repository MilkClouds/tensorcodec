# TensorCodec compatibility contract

Reference: **TorchCodec 0.17.0**, CPU audio/video/image decoding. Python operations,
frame selection, ordering, timestamps, durations, stream selection and metadata
are tested independently and against this pinned version. Audio/video decoding uses FFmpeg; images use dedicated codec libraries.
Arrays are returned as NumPy instead of torch.Tensor.

## Images

`tensorcodec.decoders` exports `decode_image`, `decode_jpeg`, `decode_png`,
`decode_webp`, `decode_gif`, `decode_avif`, `decode_heic` and `ImageReadMode`.
They return arrays directly, not Frame objects. The defaults are `mode="RGB"`
and `output_dtype=np.uint8`. One image is CHW; animated/multi-image output is NCHW.

- Inputs: local `str`/`Path` paths, encoded `bytes`/`bytearray`, or 1-D uint8 NumPy
  arrays. Format detection uses encoded content, not the extension. File-like
  inputs and URLs are not part of the image API.
- Modes: `UNCHANGED`, `GRAY`, `GRAY_ALPHA`, `RGB`, `RGB_ALPHA` (also `RGBA`).
  Case-insensitive strings or `ImageReadMode` are accepted. Missing alpha becomes
  fully opaque; existing alpha is retained when requested.
- JPEG/PNG/WebP EXIF orientation and AVIF rotation/mirroring are applied.
  HEIC transformations are handled by libheif.
- Dtypes: NumPy uint8/uint16 or `"auto"`. PNG `UNCHANGED`/`auto` preserves native
  8/16-bit samples. Integer conversion scales the range, not merely the dtype:
  uint8 to uint16 multiplies by 257. High-bit AVIF/HEIC yields full-range uint16,
  not unscaled sensor codes.
- JPEG accepts a list/tuple and returns a list, preserving order and independent
  storage. `device="cpu"` is the only supported device; CUDA fails explicitly.
- AVIF accepts `num_threads` (positive integer, default 1), passed to libavif.
- CMYK JPEG `UNCHANGED` returns CMYK samples; RGB/gray use the same conversion
  as TorchCodec/Pillow. The fourth CMYK component is not alpha.

Image decoding releases the GIL and does not use FFmpeg:

| Format | Backend | Distribution |
| --- | --- | --- |
| JPEG | libjpeg-turbo 3.2.0 | Bundled shared library |
| PNG | Rust png 0.18.1 | Compiled into extension |
| WebP, including animation | libwebp/libwebpdemux 1.6.0 | Bundled shared libraries |
| GIF | giflib | Vendored from TorchCodec v0.17.0 |
| AVIF, including alpha and sequences | libavif 1.4.2, dav1d, libyuv | Bundled shared library |
| HEIC | System libheif | Optional; missing library raises ImportError |

HEIC handles high depth, alpha and multiple images of equal shape, channel count
and bit depth, subject to the decoders compiled into libheif. Animated PNG is
unsupported and fails explicitly. No first-frame fallback is used for unsupported
animations. PNG UNCHANGED expands palette indices; non-palette tRNS keys retain
native channels, matching TorchCodec. Alpha modes expand those keys.

PNG modes/dtypes, JPEG subsampling/progressive/CMYK, still/animated WebP, GIF
compositing, AVIF alpha/high depth/orientation and HEIC fixtures are tested against
TorchCodec. Textured JPEG/PNG/WebP fixtures compare exactly. This is sampled
compatibility, not a guarantee of bit-exact output for every codec version or
file. GIF RGB background composition intentionally follows TorchCodec; Pillow
uses different colors in some transparent regions. Color profiles are not
converted to sRGB and HDR is not tone-mapped.

Image fixtures use known PNG sample values, independent FFmpeg/Pillow encodings,
and pinned upstream HEIC assets with their license. Pillow is test-only.
See [backend measurements](image-backends.md) for the selection evidence.

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
