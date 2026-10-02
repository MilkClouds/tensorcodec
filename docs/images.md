# Optional CPU image codecs

Image decoding and JPEG/PNG encoding use the user's OpenCV installation through
Python. NumPy remains the only required Python dependency; importing TensorCodec
does not import OpenCV or Pillow. No native image libraries or build steps are
added to TensorCodec. Pillow is used only to generate independent test inputs.

Use an existing `cv2 >= 4.13` installation, or install one explicitly:

```sh
uv pip install 'tensorcodec[images]'
```

The extra selects `opencv-python-headless`. Do not install multiple OpenCV wheel
variants in the same environment. OpenCV adds its own wheel size and dependencies;
it is not part of TensorCodec's advertised base wheel size.

```python
from tensorcodec.decoders import decode_image, decode_jpeg
from tensorcodec.encoders import JpegEncoder, PngEncoder

rgb = decode_image('input.webp')             # uint8 CHW, RGB
frames = decode_image('animation.gif')      # NCHW for multiple frames
batch = decode_jpeg(['one.jpg', 'two.jpg'])  # list, possibly different sizes
PngEncoder(rgb).to_file('output.png', compression_level=6)
encoded = JpegEncoder(rgb).to_tensor(quality=90)  # 1-D uint8 NumPy array
```

## Contract and limits

- `decode_image`, `decode_jpeg`, `decode_png`, `decode_webp`, `decode_gif`,
  `decode_avif`, `decode_heic` accept paths, bytes, bytearray or 1-D uint8 arrays.
- `mode` accepts case-insensitive `UNCHANGED`, `GRAY`, `GRAY_ALPHA`, `RGB`
  (default), `RGB_ALPHA`/`RGBA`, or `ImageReadMode` values.
- `output_dtype` accepts uint8 (default), uint16 or `"auto"`. PNG preserves native
  8/16-bit precision with `"auto"`; explicit conversions scale the integer range.
- Multiple frames return NCHW; still images return CHW. Animated WebP keeps NCHW
  even with one frame. Frame timings and loop counts are not returned.
- JPEG/PNG/WebP EXIF orientation and AVIF primary-item rotation/mirror are applied.
  AVIF track-specific transforms are outside this adapter's contract.
- APNG, high-bit-depth AVIF, CMYK JPEG `UNCHANGED`, GPU decoding and nondefault
  AVIF `num_threads` raise explicit errors. OpenCV has no per-call AVIF thread
  setting; the adapter never modifies global OpenCV thread settings.
- AVIF color conversion follows OpenCV. Dropping alpha preserves straight RGB;
  TorchCodec 0.17.0 premultiplies AVIF RGB in that case. Pixel identity with
  TorchCodec is not promised across formats, builds or codec versions.
- HEIC uses a small optional system `libheif` binding because the tested OpenCV
  wheel cannot decode HEIC. It supports 8/10/12/16-bit images and multiple top-level
  images, not timed HEIF sequence tracks. libheif and its decoder plugins must be
  available separately; it is selected by format, never as a fallback on failure.
- Other formats depend on the installed OpenCV build. Missing dependencies,
  unsupported codecs and decode failures raise; no alternate decoder is tried.
- Encoders accept nonempty CHW uint8 arrays with 1 or 3 channels. Both provide
  `to_file`, `to_file_like` and `to_tensor`; JPEG quality is 1–100 (default 75),
  PNG compression level is 0–9 (default 6). Encoded bytes need not match TorchCodec.

## Measured selection rationale

An exploratory Linux CPU comparison used OpenCV 4.13.0.92, TorchCodec 0.17.0,
three contents (photograph, graphics, seeded noise), three sizes (224 square,
640×480, 1920×1080), and six encodings: 54 cases total. Decoding from memory to
RGB CHW matched TorchCodec exactly in all 54 cases. Median timings used five
batches after three warmups, one pinned CPU, and one OpenCV/Torch thread.

| Encoding | OpenCV / TorchCodec latency, geometric mean |
| --- | ---: |
| JPEG 4:2:0 | 0.98 |
| JPEG 4:4:4 | 1.00 |
| Progressive JPEG | 0.97 |
| PNG | 1.16 |
| Lossy WebP | 1.00 |
| Lossless WebP | 1.10 |

These are exploratory direct-OpenCV results, not universal performance claims or
measurements of the final adapter. PNG was approximately 1.95× slower than the
abandoned specialized native implementation on that corpus. Avoiding native build
and vendoring complexity is the tradeoff, not an assertion that OpenCV is fastest.
The small `benchmarks/image_codecs.py` script measures the final adapter and checks
RGB pixel agreement on a supplied corpus; it prints versions and per-file results.

The final adapter was rerun on the same 54 inputs with CPU affinity pinned and
one OpenCV/Torch thread. All RGB pixels still matched. Adapter / TorchCodec
geometric-mean latency ratios were 1.05 (JPEG 4:2:0), 1.03 (JPEG 4:4:4), 1.02
(progressive JPEG), 1.28 (PNG), 1.07 (lossy WebP), and 1.11 (lossless WebP).
Encoding speed has not been benchmarked; tests cover independent Pillow decoding,
PNG lossless round trips, options, file output and partial stream writes.
