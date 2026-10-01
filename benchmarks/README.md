# TensorCodec benchmarks

These tools measure the current TensorCodec implementation and optional comparison
backends. Performance depends on the input, seek mode, thread count, storage and
cache state. Measure your workload before choosing a decoder.

## Setup

Use the [development environment](../README.md#development-and-verification).
TensorCodec-only runs need NumPy and TensorCodec; TorchCodec comparisons also need
the pinned oracle group. FFmpeg CLI with the requested encoders is required to
generate fixtures. Run from the repository root.

```sh
uv run --no-sync python -m benchmarks --list-decoders
uv run --no-sync python -m benchmarks --prepare --num-videos 4 --video-duration 10
uv run --no-sync python -m benchmarks --no-io --decoders tensorcodec \
  --runs 3 --output json --save benchmarks/results/speed.json
```

For a CPU comparison, explicitly select the same thread count and seek mode:

```sh
uv run --no-sync python -m benchmarks --no-io \
  --decoders tensorcodec 'torchcodec(seek=exact,thr=1)' \
  --runs 3 --output json --save benchmarks/results/speed.json
```

## Measurements

| Scenario | Workload | Included costs |
| --- | --- | --- |
| `temporal_window` | Random playback queries, 11 samples over a 1-second window | Decoder open, exact-mode packet scan, seeking, decoding, NumPy output |
| `sequential_range` | Decode the full video range | Decoder open, range selection, decoding, NumPy output |

The TensorCodec adapter opens and closes a decoder for each request. Results do
not represent a persistent decoder cache such as MediaRef's. Compare exact and
approximate modes separately: approximate mode does not guarantee exact VFR frame
selection. GPU comparison adapters include transfer back to CPU NumPy arrays.

For codec/container throughput experiments:

```sh
uv run --no-sync python -m benchmarks.container_bench --help
uv run --no-sync python -m benchmarks.container_bench
```

This tool generates fresh inputs with FFmpeg. Its codec matrix measures speed;
it does not assert pixel or timestamp correctness. See
[container and seek behavior](../docs/container_robustness.md) for tested guarantees.

## Optional I/O and plotting

FUSE measurements require Linux FUSE access, `pyfuse3` and `trio`. Install these
only in the benchmark environment, then omit `--no-io`. The runner can fall back
to speed-only results when FUSE is unavailable; check that `io_bytes` is populated
before reporting I/O figures. FUSE timing includes its own filesystem overhead.

```sh
uv run --no-sync python -m benchmarks --decoders tensorcodec \
  'torchcodec(seek=exact,thr=1)' --scenarios temporal_window \
  --runs 3 --output json --save benchmarks/results/io.json
```

Install `matplotlib` in the benchmark environment to plot measurements:

```sh
uv run --no-sync python -m benchmarks.plot_results \
  --speed benchmarks/results/speed.json --io benchmarks/results/io.json
```

## Reporting results

Generated JSON and charts stay in `benchmarks/results/`, which is gitignored.
Publish results only with the TensorCodec/TorchCodec/FFmpeg/NumPy versions,
hardware, corpus, seek/thread settings, cache conditions and commands. Identify
timeouts as partial results. Check frame selection and pixel agreement separately
using the [compatibility contract](../docs/compatibility.md); throughput alone is
not a correctness check.
