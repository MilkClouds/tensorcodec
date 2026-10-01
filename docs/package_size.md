# Package size policy

TensorCodec keeps its NumPy-only Python dependency set and bundles a minimal
FFmpeg/OpenSSL runtime in its default Linux wheels. Size limits prevent additions
from silently increasing the distributed binary footprint.

## What is measured

| Metric | Definition | Per-wheel limit |
| --- | --- | ---: |
| Download | Final `.whl` file size, in bytes | 15 MiB |
| Unpacked | Sum of ZIP entry file sizes, including bundled libraries | 35 MiB |

The sole policy file is [`packaging/size-policy.json`](../packaging/size-policy.json).
The checker uses only Python's standard library and never extracts the archive.
Unpacked size excludes filesystem allocation overhead. Both metrics exclude NumPy,
Python, package caches and other external dependencies; they are not total
installation sizes. Reports use MiB (2^20 bytes); the README uses MB (10^6 bytes).

Check final, repaired wheels locally:

```sh
uv run --no-project python scripts/check_wheel_size.py dist/*.whl --output reports/wheel-size.json
```

Each architecture is checked independently. Exactly reaching a limit passes;
exceeding either limit fails. The release workflow runs this after `auditwheel
repair`, before uploading distributions. JSON reports are separate artifacts, not
files in `dist/`. Actions summaries include changes from the committed published
baseline. Ordinary CI tests the checker without building FFmpeg from source.

Before changing a limit, explain the feature, the measured byte increase on both
architectures and why a smaller configuration would not provide the same behavior.
A limit change should be reviewed with the change that needs it.

## TorchCodec comparison and badge

[`packaging/size-baseline.json`](../packaging/size-baseline.json) records published
PyPI URLs, SHA-256 hashes and exact sizes for TensorCodec and the pinned TorchCodec
playback reference. Comparison uses Linux x86_64 and ARM64 CPython 3.10 artifacts.
TensorCodec's ABI3 wheel also serves later supported Python versions. Other
Python, platform and CPU/CUDA builds may differ; the TorchCodec PyPI artifacts
are not claimed to be the same binaries as the CPU-index playback oracle.

TorchCodec's standalone wheel is currently smaller. Its external PyTorch and
FFmpeg requirements are listed separately in the README, not added to its wheel
size. TorchCodec supports substantially more features, including image APIs,
encoding and CUDA. This comparison makes no feature-equivalence, performance
or total-environment-size claim.
The runtime requirements follow the
[TorchCodec 0.17.0 installation guide](https://github.com/pytorch/torchcodec/blob/v0.17.0/README.md#installing-torchcodec).

The badge reports the largest published TensorCodec Linux wheel download. After
a successful PyPI publication, a separate job downloads both projects' pinned
artifacts, verifies their PyPI SHA-256 hashes and sizes, checks TensorCodec's limits,
and updates the README and baseline in one commit. Git history retains previous
snapshots. Candidate builds never update the published badge. Repository visibility
does not matter: Shields renders a static size value in the committed README.

To reproduce or recover a documentation update after publication:

```sh
uv run --no-project python scripts/update_size_comparison.py --version 0.1.1
```

Review and commit `README.md` and `packaging/size-baseline.json` together. The script
requires both architectures to have been published and fails on ambiguous wheels,
yanked wheels, missing files or hash mismatches. If publication succeeded but the
documentation job failed, repair the documentation separately; do not republish
the same version. A concurrent change to `main` can reject the documentation push;
the workflow never force-pushes.

## Why FFmpeg stays bundled by default

Bundling the selected FFmpeg libraries provides one-step installation and fixes the
runtime ABI and codec configuration used by the release tests. A full conda-forge
FFmpeg environment can include many additional codec, graphics and system packages;
moving these outside the wheel does not necessarily reduce total installation size.

Reusing an existing shared FFmpeg 7 installation is an advanced source-build option:
[system FFmpeg guide](system_ffmpeg.md). FFmpeg CLI availability alone does not
satisfy the native library requirement.
