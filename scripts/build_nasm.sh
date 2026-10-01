#!/usr/bin/env bash
# Build-only assembler; manylinux2014's system NASM is too old for FFmpeg.
set -euo pipefail
prefix="${1:?usage: build_nasm.sh ABSOLUTE_INSTALL_PREFIX}"
version=2.16.03
build_root="$(mktemp -d)"
curl -fsSL "https://www.nasm.us/pub/nasm/releasebuilds/${version}/nasm-${version}.tar.xz" -o "$build_root/nasm.tar.xz"
echo "1412a1c760bbd05db026b6c0d1657affd6631cd0a63cddb6f73cc6d4aa616148  $build_root/nasm.tar.xz" | sha256sum --check
tar -xf "$build_root/nasm.tar.xz" -C "$build_root"
cd "$build_root/nasm-${version}"
./configure --prefix="$prefix"
make -j "${TENSORCODEC_BUILD_JOBS:-4}"
make install
