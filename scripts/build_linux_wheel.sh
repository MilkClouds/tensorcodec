#!/usr/bin/env bash
# Run inside manylinux2014 with Rust, maturin, libclang, NASM and Perl installed.
set -euo pipefail
build_prefix="${TENSORCODEC_NATIVE_PREFIX:-/opt/tensorcodec}"
scripts/build_openssl.sh "$build_prefix/openssl"
export PKG_CONFIG_PATH="$build_prefix/openssl/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$build_prefix/openssl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
scripts/build_ffmpeg.sh "$build_prefix/ffmpeg"
export FFMPEG_DIR="$build_prefix/ffmpeg"
export LD_LIBRARY_PATH="$FFMPEG_DIR/lib:$LD_LIBRARY_PATH"
maturin build --release --locked --auditwheel repair --compatibility manylinux2014 --out dist
