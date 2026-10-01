# Publishing TensorCodec

Release version: `0.1.0`. Distribution and import name: `tensorcodec`.
The first binary release targets Linux x86_64, glibc 2.28+, CPython 3.10+.
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

Run the **Publish to PyPI** workflow on `main`. It builds the portable Linux wheel
and source distribution, checks package metadata, validates the pinned oracle
and compares playback before uploading through PyPI Trusted Publishing. It uses
the existing GitHub `pypi` environment. Publication fails if authorization is
missing, tests fail, or the version has already been uploaded.

```sh
gh workflow run publish.yml --repo MilkClouds/tensorcodec --ref main
```

For a build and full validation without uploading, pass `--field publish=false`.

Check the workflow and https://pypi.org/project/tensorcodec/0.1.0/ before reporting
success. Verify a fresh `pip install tensorcodec==0.1.0` and a decode without
Torch/PyAV. Update the version before subsequent releases; PyPI versions cannot
be overwritten.

The local Linux build is reproducible using `scripts/build_linux_wheel.sh` inside
`quay.io/pypa/manylinux_2_28_x86_64` with Rust, maturin, libclang, NASM and Perl.
Both native source archives are version- and checksum-pinned. Their licensing
and source links are recorded in `licenses/README.md`.
