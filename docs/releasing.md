# Publishing TensorCodec

Release version: `0.1.1`. Distribution and import name: `tensorcodec`.
Binary wheels target Linux x86_64 and ARM64 (aarch64), glibc 2.17+, CPython 3.10+.
NumPy must also provide a compatible wheel for the selected Python/glibc pair.
The wheel bundles shared FFmpeg 7.1.5 and OpenSSL 3.5.9 LTS; its only Python
runtime dependency is NumPy. macOS/Windows wheels are not yet provided.

## One-time account setup

1. Rename `MilkClouds/avdec` to `tensorcodec` in GitHub repository Settings.
   Preserve the current visibility; publishing does not require making it public.
2. Open https://pypi.org/manage/account/publishing/ and add a **pending GitHub
   publisher** for a new project with these values:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `tensorcodec` |
   | GitHub owner | `MilkClouds` |
   | Repository | `tensorcodec` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` |

A pending publisher creates the project on its first successful upload. If the
project already exists, register the publisher in that project's Publishing
settings instead. Do not put an API token in the repository or chat.

## Release

Run the **Publish to PyPI** workflow on `main`. It builds the portable Linux wheels
and source distribution, checks package metadata, validates the pinned oracle
and compares playback before uploading through PyPI Trusted Publishing. It uses
the existing GitHub `pypi` environment. Publication fails if authorization is
missing, tests fail, or the version has already been uploaded.

```sh
gh workflow run publish.yml --repo MilkClouds/tensorcodec --ref main
```

For a build and full validation without uploading, pass `--field publish=false`.

Check the workflow and https://pypi.org/project/tensorcodec/0.1.1/ before reporting
success. Verify a fresh `uv pip install tensorcodec==0.1.1` and a decode without
Torch/PyAV on both architectures. Update the version before subsequent releases;
PyPI versions cannot be overwritten.

The local Linux build is reproducible using `scripts/build_linux_wheel.sh` inside
`quay.io/pypa/manylinux2014_x86_64` or
`quay.io/pypa/manylinux2014_aarch64` with Rust, maturin, libclang, NASM and Perl.
Both native source archives are version- and checksum-pinned. Their licensing
and source links are recorded in `licenses/README.md`.

## CI versus release builds

- Ordinary CI uses prebuilt conda-forge FFmpeg 7.1.1 through Pixi, including its
  headers and shared libraries. It builds only the TensorCodec extension.
- PyPI wheels use the smaller LGPL FFmpeg 7.1.5 build plus OpenSSL 3.5.9.
  Their native prefix is cached by architecture, glibc baseline and build-script
  checksums. This preserves the wheel's codec set, dependency size and licensing rather than bundling the full
  conda-forge dependency graph.
- Release validation installs each repaired wheel on glibc 2.17 with Python 3.10
  and 3.13 and decodes video/audio without Torch, PyAV or a system FFmpeg. Python
  3.10 also checks the minimum NumPy line (1.26.4). Native
  x86_64 and ARM64 runners also run the full pinned playback oracle comparison.
- Release validation still tests the installed repaired wheel. The fixture CLI
  can be FFmpeg 6 or 7; fixtures explicitly remove auxiliary sentinel packets.
