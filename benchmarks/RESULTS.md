> Historical avdec/PyAV document. These results do not describe TensorCodec.

<!--
=== CLUSTER CONFIGURATION (internal) ===

Environment setup required before running benchmarks on this cluster:

  . /sw/apps/init-lmod.sh
  ml gcc/11 cuda/12.8 cudnn

Then set library paths so torchcodec can find CUDA 12 NPP and PyAV's bundled FFmpeg:

  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$PWD/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH
  export CUDA_VISIBLE_DEVICES=1   # GPU 0 is often occupied; GPU 1 is usually free

Why this is needed:
- System has CUDA 13.0 toolkit (nvcc 13.0.88), but torchcodec cu126 links against
  libnppicc.so.12 (CUDA 12 NPP). The cuda/12.8 module provides this.
- PyAV bundles its own FFmpeg 8 shared libs in av.libs/. torchcodec's
  libtorchcodec_core8.so needs libavutil.so.60 etc. on LD_LIBRARY_PATH.
- cudnn module provides libcudnn for potential future use.

Full benchmark commands:

  # Speed (all decoders, ~5 min)
  python -m benchmarks --no-io --runs 3 --timeout 20 --output json --save benchmarks/results/full_speed.json

  # I/O (only chart decoders, ~1 min)
  python -m benchmarks --runs 3 --timeout 20 --scenarios temporal_window \
      --decoders avdec 'torchcodec(seek=approximate,thr=1)' 'torchcodec(seek=exact,thr=1)' \
      --output json --save benchmarks/results/full_io.json

  # Regenerate chart
  python -m benchmarks.plot_results
-->

# Video Decoder Benchmark Results

## Test Environment

| Component | Value |
|---|---|
| OS | Linux 5.15.0 (Ubuntu), x86_64 |
| CPU | 2× Intel Xeon Gold 6336Y @ 2.40 GHz (96 logical CPUs) |
| GPU | 2× NVIDIA A100 80GB PCIe |
| Memory | 503 GB DDR4 |
| Python | 3.11.11 |
| avdec | 0.1.0 (PyAV 16.1.0, NumPy 2.4.2) |
| torchcodec | 0.10.0+cu126 (PyTorch 2.10.0+cu126) |
| decord | 0.6.0 |
| OpenCV | 4.13.0 |
| torchvision | 0.25.0+cu126 |

## Video Corpus

64 synthetic videos (640×480, H.264, 30 fps, GOP 10, 300 s each, ~648 MB total).

## Scenarios

**temporal_window** — VLA-style random access: 64 videos × 5 random timestamps,
11-frame window per query (320 queries, 3,520 frames).

**sequential_range** — Full sequential decode: 64 videos × 9,000 frames each.

## Decode Speed

3 runs, 20 s timeout, no FUSE. Higher FPS is better.

| Decoder | temporal_window | sequential_range |
|---|---:|---:|
| torchcodec (approximate, thr=1) | **276** | 1,198 ⏱ |
| torchcodec (exact, thr=1) | 239 | 1,214 ⏱ |
| **avdec** | **188** | 654 ⏱ |
| torchcodec (gpu, approximate) | 174 ⏱ | 892 ⏱ |
| torchcodec (gpu, exact) | 171 ⏱ | 1,039 ⏱ |
| torchcodec (approximate, thr=0) | 143 ⏱ | 2,188 ⏱ |
| torchcodec (exact, thr=0) | 117 ⏱ | **2,325** ⏱ |
| decord | 88 ⏱ | 1,094 ⏱ |
| opencv | 58 ⏱ | 1,047 ⏱ |
| torchvision-pyav | 28 ⏱ | 734 ⏱ |

⏱ = timed out (FPS from frames decoded before timeout).

## Disk I/O (temporal_window)

Measured via FUSE passthrough (`CountingFS`). 3 runs, 20 s timeout.

| Decoder | Bytes / Frame | Relative |
|---|---:|---:|
| **avdec** | **20.1 KB** | 1.0× |
| torchcodec (approximate) | 21.4 KB | 1.1× |
| torchcodec (exact) | 952.7 KB | **47.4×** |

## GPU Decoding

torchcodec supports CUDA-accelerated decoding via `device="cuda"`. We use
`set_cuda_backend("beta")` which is significantly faster than the default
FFmpeg-based CUDA path. On the A100:

- **Random seek**: 174 FPS — competitive with avdec (188 FPS CPU). The beta
  CUDA backend eliminates the massive per-call overhead present in the default
  backend (which scored only ~9 FPS in the same scenario).
- **Sequential decode**: 892–1,039 FPS — lower than CPU torchcodec (1,198 FPS)
  due to GPU↔CPU transfer overhead (benchmark protocol requires NumPy output).

GPU decoding is most useful when decoded frames stay on the GPU for downstream
model inference, avoiding the CPU round-trip that this benchmark measures.

## Analysis

### VLA-Style Access (temporal_window)

1. **torchcodec (approximate) is fastest** at 276 FPS — 47% faster than avdec.
2. **avdec is second** at 188 FPS with zero configuration. It outperforms
   decord (88), OpenCV (58), and torchvision (28) despite being pure Python.
3. **GPU (beta backend) is viable** at 174 FPS — close to avdec; requires
   `set_cuda_backend("beta")` (default CUDA backend is ~20× slower).
4. **Multi-threading hurts seek workloads**: `thr=0` drops approximate from
   276 → 143 FPS (−48%) due to thread coordination overhead on short bursts.
5. **Exact seek reads 47× more I/O** (953 KB vs 20 KB per frame).

### Sequential Decode (sequential_range)

1. **torchcodec (auto-threads) dominates** at 2,325 FPS.
2. **avdec** reaches 654 FPS — Python overhead accumulates but remains
   competitive with decord (1,094) and OpenCV (1,047).

### Recommendations

| Use Case | Decoder | Config |
|---|---|---|
| VLA training (seek-heavy) | torchcodec | `seek_mode=approximate`, `thr=1` |
| VLA training (no C++ deps) | **avdec** | Default |
| VLA training (GPU pipeline) | torchcodec (gpu) | `set_cuda_backend("beta")` |
| Bulk sequential decode | torchcodec | `seek_mode=approximate`, `thr=0` |
| I/O-constrained storage | avdec or torchcodec (approximate) | Avoid exact seek |

## Keyframe Interval (GOP) Ablation

The main benchmark uses GOP 10. Shorter keyframe intervals reduce seek
cost (fewer frames to decode from the nearest keyframe) but increase file
size. All numbers below use the same temporal_window scenario (64 × 5 min,
640×480, H.264, MP4, 3 runs, no FUSE).

| GOP | File size (per 5 min) | avdec (FPS) | TC approx (FPS) | TC exact (FPS) | approx / avdec |
|----:|----------------------:|------------:|-----------------:|---------------:|:--------------:|
| 1 (all-intra) | 28.1 MB | 186 | **496** | 315 | 2.67× |
| 2 · [LeRobot](https://github.com/huggingface/lerobot) default | 17.3 MB | 203 | **365** | 254 | 1.80× |
| 10 | 10.1 MB | 167 | 227 | 193 | 1.36× |
| 30 | 9.1 MB | 170 | 233 | 195 | 1.37× |
| 120 | 7.9 MB | 148 | 209 | 193 | 1.41× |
| 250 | 7.0 MB | 139 | 193 | 181 | 1.38× |

At GOP ≤ 2, every frame is at most 1 frame from a keyframe, so seek
overhead is near zero and pure decode throughput dominates — TorchCodec's
C++ pipeline is 1.8–2.7× faster. At GOP ≥ 10 the gap narrows to ~1.4×,
while file sizes drop 1.7–4× (10 MB vs 17–28 MB per 5 min clip).

## Methodology

- **Timeout**: 20 s per scenario. ⏱ marks partial results.
- **Runs**: 3 per measurement.
- **I/O**: FUSE passthrough counts every `read()` syscall. FUSE adds ~14%
  overhead to avdec, ~68% to torchcodec-exact (many small reads).
- **Seed**: 42 (deterministic queries across decoders).

### Reproducing

```bash
python -m benchmarks --prepare
python -m benchmarks --no-io --runs 3 --timeout 20
python -m benchmarks --runs 3 --timeout 20 --scenarios temporal_window \
    --decoders avdec 'torchcodec(seek=approximate,thr=1)' 'torchcodec(seek=exact,thr=1)'
python -m benchmarks.plot_results
```