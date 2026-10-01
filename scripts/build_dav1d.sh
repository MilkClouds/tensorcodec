#!/usr/bin/env bash
# dav1d (BSD-2-Clause) for FFmpeg's software AV1 decoder; FFmpeg's own `av1` decoder
# needs a hardware accelerator. Static and position-independent: it links into
# libavcodec, so wheels bundle no extra library. Needs Meson and Ninja; NASM on x86-64.
set -euo pipefail
prefix="${1:?usage: build_dav1d.sh ABSOLUTE_INSTALL_PREFIX}"
version=1.5.4
build_root="$(mktemp -d)"
curl -fsSL "https://downloads.videolan.org/pub/videolan/dav1d/${version}/dav1d-${version}.tar.xz" -o "$build_root/dav1d.tar.xz"
echo "686616b7c69eb88d44459391ab25cac13b6647a3b288835c5784e71c1514a5c5  $build_root/dav1d.tar.xz" | sha256sum --check
tar -xf "$build_root/dav1d.tar.xz" -C "$build_root"
cd "$build_root/dav1d-${version}"
meson setup build --prefix="$prefix" --libdir=lib --buildtype=release \
  --default-library=static -Db_staticpic=true -Denable_tools=false -Denable_tests=false -Denable_docs=false
ninja -C build -j "${TENSORCODEC_BUILD_JOBS:-4}"
ninja -C build install
mkdir -p "$prefix/share/licenses/dav1d"
cp COPYING "$prefix/share/licenses/dav1d/"
