# Image backend selection

The image API uses format-specific decoders rather than FFmpeg. Selection includes
pixel compatibility, decoded features, packaging and measured latency; neither C
nor Rust is assumed faster for every format.

| Format | Selected backend | Evidence |
| --- | --- | --- |
| JPEG | libjpeg-turbo 3.2.0 | Exact sampled TorchCodec pixels; baseline JPEG faster than both Rust candidates |
| PNG | Rust png 0.18.1 | Exact native samples/modes after matching libpng rounding; faster than libpng on this corpus |
| WebP | libwebp/libwebpdemux 1.6.0 | Exact sampled pixels; faster than image-webp, including full animation compositing |
| GIF | giflib | TorchCodec-compatible palette/background/disposal rules |
| AVIF | libavif 1.4.2 + dav1d + libyuv | Same conversion path as TorchCodec; alpha, sequences, high depth and orientation |
| HEIC | Optional system libheif | Same library as TorchCodec; not bundled |

The last three selections are primarily compatibility/feature decisions: this
benchmark does not establish their performance advantage over other libraries.

## Reproducible comparison

Measured on 2026-10-02, Linux x86-64, Intel Xeon Gold 6336Y, one CPU affinity.
Release builds use portable compiler defaults and runtime SIMD dispatch, not
`-march=native`. Python 3.12, NumPy 2.5.3, Pillow 12.3.0, TorchCodec 0.17.0 CPU.
The C-only prototype used libpng 1.6.59; the selected implementation uses Rust png.

54 inputs: a frame from the repository's NASA video, deterministic colored
rectangles, and seeded noise, each at 224x224, 640x480 and 1920x1080. Each is
encoded as JPEG 4:2:0, JPEG 4:4:4, progressive JPEG, PNG, lossy WebP and lossless
WebP. These are three content patterns, not 54 independent photographs or a
representative production dataset. The previous FFmpeg experiment used only the
NASA frame; its results should not be compared as if the corpus were identical.

The timed operation accepts identical encoded bytes in memory and returns a
fully decoded RGB uint8 CHW NumPy array. Pillow's lazy open alone is not timed:
RGB conversion and materialization are included. Torch's `.numpy()` is a view.
CHW does not require contiguous storage. Encoding, file I/O, imports and warmup
are excluded. Each case uses three warmups followed by five batches with rotated
backend order; the per-case result is the median milliseconds per image. Summary
ratios are geometric means of matched case latencies. Shared-machine scheduling
introduces noise; small differences are not conclusive.

| Format (9 cases each) | Selected / TorchCodec latency | Selected / Pillow latency | Largest sampled difference vs TorchCodec |
| --- | ---: | ---: | ---: |
| JPEG 4:2:0 | 1.04x | 0.81x | 0 |
| JPEG 4:4:4 | 1.02x | 0.85x | 0 |
| Progressive JPEG | 0.99x | 0.87x | 0 |
| PNG | 0.59x | 0.30x | 0 |
| Lossy WebP | 0.99x | 0.78x | 0 |
| Lossless WebP | 1.05x | 0.54x | 0 |

Lower latency ratios are better. This supports roughly TorchCodec-level JPEG/WebP
latency and about 1.7x faster PNG on these inputs, not universal speed claims.
No ARM performance claim follows from this x86 measurement.

Candidates are image 0.25.10 (png 0.18.1, zune-jpeg 0.5.15, image-webp 0.2.4)
and libjpeg-turbo-rs 0.8.0, built as a separate PyO3/NumPy extension. In the final
run, image-rs JPEG was 22-23% slower for baseline JPEG and within 2% for progressive,
with maximum differences of 11/255 (subsampled) or 3/255 (4:4:4). Its WebP was
78% slower for lossy and 12% slower for lossless, with exact pixels. Its PNG
latency matched the selected Rust PNG implementation. libjpeg-turbo-rs was exact
on these 27 JPEGs, 7-8% slower for baseline JPEG and 11% faster for progressive.
It remains a credible alternative, but did not establish an overall advantage
for replacing the existing libjpeg-turbo choice.

The initial C-vs-Rust run is `benchmarks/image_candidates/baseline.json`; the
separate JPEG-port check is `turbo-baseline.json`; the final rotated comparison
is `selected.json`. Differences between runs illustrate why small ratios should
not be treated as stable rankings.

```sh
# Use an existing uv environment with the project dev/oracle groups installed.
uv run --no-sync maturin develop --release --locked --uv
uv run --no-sync maturin develop --release --locked \
  --manifest-path benchmarks/image_candidates/Cargo.toml --uv
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python benchmarks/image_decode.py --candidates --output results.json
```

The experimental crates are not dependencies of TensorCodec and are not required
to use its image API. Tests separately cover native 16-bit PNG, all modes/dtypes,
Adam7, palette/transparency, orientations, malformed input, CMYK, animation,
AVIF alpha/high depth and licensed HEIC samples. Passing these tests is sampled
coverage, not certification of all images, versions or color profiles.

## Reading external performance claims

The [image-rs PNG report](https://blog.image-rs.org/2026/06/18/png-adoption.html)
uses 2,848 images from an independent corpus and compares against libpng with
zlib-ng. It also reports Chromium default adoption since M139, rather than only
an experiment. These are substantial signals for evaluating Rust PNG, but do not
rank JPEG/WebP or establish TensorCodec's end-to-end throughput. The report is
maintainer-authored and its hardware, workload and options still matter.

The [libjpeg-turbo-rs report](https://github.com/developer0hye/libjpeg-turbo-rs)
is likewise a useful candidate source, not a substitute for matching pixel
semantics and measuring our wrapper. Encoder and resize benchmarks do not decide
this decoder-only API. Debug-build results are not deployment measurements.
