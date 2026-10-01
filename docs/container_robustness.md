> Historical avdec/PyAV document. These results do not describe TensorCodec.

# Container & Codec Robustness

avdec delegates all FFmpeg interaction to [PyAV](https://github.com/PyAV-Org/PyAV),
the most widely used Python FFmpeg binding.  This page documents how that
architectural choice affects real-world performance across different video
containers and codecs.

## TL;DR

TorchCodec's **approximate seek** is optimised for MP4.  On **Matroska
containers (MKV / WebM)** it is **5–12× slower** than on MP4 for most
codecs (H.264, H.265, VP9, MPEG-4, MJPEG).  AV1 is an exception — it
performs consistently across all containers.  avdec is unaffected; the
same codec in MP4 or MKV yields the same throughput.

avdec shows **consistent performance regardless of container format**.

## Methodology

* 30-second synthetic videos, 640 × 480, 30 fps, GOP 10 (unless noted).
* 100 random-seek window queries (11 frames each, VLA-style workload).
* Each decoder re-opens the file per query (simulates training data loader).
* Single run, single thread.  Numbers are FPS (frames / second).

## Codec × Container Matrix

| Codec | Container | avdec | TC (approx) | TC (exact) | approx / avdec |
|-------|-----------|------:|------------:|-----------:|:--------------:|
| H.264 | mp4       |   227 |         250 |        415 | 1.10×          |
| H.264 | **mkv**   |   246 |      **44** |        409 | **0.18×** 🔴   |
| H.264 | avi       |   232 |         360 |        344 | 1.55×          |
| H.264 | ts        |   170 |         191 |        223 | 1.12×          |
| H.265 | mp4       |   126 |         305 |        399 | 2.42×          |
| H.265 | **mkv**   |   140 |      **52** |        503 | **0.37×** 🔴   |
| H.265 | ts        |   102 |         285 |        425 | 2.79×          |
| VP9   | mp4       |   239 |         496 |        650 | 2.08×          |
| VP9   | **mkv**   |   174 |      **40** |        442 | **0.23×** 🔴   |
| VP9   | **webm**  |   183 |      **42** |        456 | **0.23×** 🔴   |
| VP9   | avi       |   303 |         470 |        642 | 1.55×          |
| AV1   | mp4       |    19 |          35 |         36 | 1.84×          |
| AV1   | mkv       |    19 |          35 |         37 | 1.84×          |
| AV1   | webm      |    19 |          35 |         36 | 1.84×          |
| MPEG-4| **mkv**   |   518 |     **108** |       1029 | **0.21×** 🔴   |
| MPEG-4| avi       |   553 |        1144 |       1029 | 2.07×          |
| MJPEG | **mkv**   |   458 |      **67** |       1314 | **0.15×** 🔴   |
| MJPEG | avi       |   476 |        1833 |       1417 | 3.85×          |

🔴 = TorchCodec approximate seek **5× or more slower** than avdec.

## B-frame Interaction

B-frames worsen the MKV penalty.  Same H.264 stream, varying `bf`:

| B-frames | Container | avdec | TC (approx) | TC (exact) | approx / avdec |
|:--------:|-----------|------:|------------:|-----------:|:--------------:|
| 0        | mp4       |   228 |         413 |        422 | 1.81×          |
| 0        | mkv       |   218 |         349 |        468 | 1.60×          |
| 3        | mp4       |   251 |         227 |        408 | 0.90×          |
| 3        | **mkv**   |   271 |      **46** |        377 | **0.17×** 🔴   |
| 8        | mp4       |   265 |         275 |        453 | 1.04×          |
| 8        | **mkv**   |   269 |      **54** |        441 | **0.20×** 🔴   |

## GOP Size (MP4 only)

On MP4, both decoders scale similarly with GOP size:

| GOP | avdec | TC (approx) | TC (exact) |
|----:|------:|------------:|-----------:|
|   1 |   221 |         502 |        499 |
|  10 |   223 |         236 |        374 |
|  30 |   230 |         198 |        301 |
| 120 |   179 |         184 |        252 |
| 250 |   169 |         155 |        237 |

## What Is Not Affected

The following showed **no meaningful difference** between decoders:

* **Edge-case timestamps** — `t = 0`, near-end, past-end, negative,
  duplicates, reverse order: both raise equivalent errors or return
  identical results.
* **Resolution** — 160 × 120 through 1920 × 1080, including non-standard
  sizes (322 × 242): both work correctly.
* **Short videos** — 0.5 s and 1 s videos: both decode without issue.

## Why This Happens

All three decoders (avdec, TorchCodec, decord) ultimately call FFmpeg for
demuxing and decoding.  The difference is the **binding layer**:

| Decoder    | FFmpeg binding                |
|------------|-------------------------------|
| avdec      | [PyAV](https://github.com/PyAV-Org/PyAV) (independent project, est. 2012) |
| TorchCodec | Custom C++ (part of TorchCodec) |
| decord     | Custom C++ (unmaintained since 2021) |

TorchCodec's approximate seek appears to rely on container-specific index
structures (e.g. MP4's `stss` sync-sample table) that are well-optimised
for MP4 but not equivalently implemented for Matroska/WebM Cue indexes.
AV1 is the one exception — it performs identically across containers,
suggesting TorchCodec's AV1 code path handles Matroska seeks correctly.
PyAV delegates all container handling to FFmpeg's battle-tested
`libavformat`, which handles every container uniformly.

TorchCodec's **exact seek** does not exhibit this problem — it performs
consistently across containers.  However, exact seek reads
[~47× more data per frame](../benchmarks/RESULTS.md) than approximate
seek, making it unsuitable for I/O-constrained training pipelines.

## Implications for ML Training

Real-world training datasets are not exclusively MP4:

* **Robotics cameras** often record MJPEG in AVI or MKV.
* **Web-scraped datasets** contain MKV, WebM, and MPEG-TS.
* **Research datasets** (e.g. Ego4D) use varied containers.

If your pipeline encounters non-MP4 files, TorchCodec's headline
benchmark numbers (measured on MP4) will not hold.  avdec's performance
is container-agnostic.

## Reproducing

The benchmark script is at [`benchmarks/container_bench.py`](../benchmarks/container_bench.py).

```bash
python benchmarks/container_bench.py          # run all combos
python benchmarks/container_bench.py --help   # options
```

## Environment

* Python 3.11, Linux x86_64
* avdec 0.1.0 (PyAV 16.1.0)
* torchcodec 0.10.0+cu126

