#!/usr/bin/env bash
# Small, shared LGPL FFmpeg build: no CLI programs, encoders, muxers, or optional
# third-party codec libraries except dav1d (AV1, linked statically into libavcodec).
# OpenSSL provides HTTPS input support.
set -euo pipefail
prefix="${1:?usage: build_ffmpeg.sh ABSOLUTE_INSTALL_PREFIX}"
ffmpeg_version=7.1.5
build_root="${TENSORCODEC_BUILD_ROOT:-$(mktemp -d)}"
mkdir -p "$build_root" "$prefix"
bash "$(dirname "$0")/build_dav1d.sh" "$build_root/dav1d"
export PKG_CONFIG_PATH="$build_root/dav1d/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${ffmpeg_version}.tar.xz" -o "$build_root/ffmpeg.tar.xz"
echo "de668509caf9e35e3cd162473441fdb29538c6d96ed080292b3cf9e6fc5d558f  $build_root/ffmpeg.tar.xz" | sha256sum --check
tar -xf "$build_root/ffmpeg.tar.xz" -C "$build_root"
cd "$build_root/ffmpeg-${ffmpeg_version}"
./configure \
  --prefix="$prefix" \
  --enable-shared --disable-static --disable-autodetect \
  --disable-programs --disable-doc --disable-avdevice --disable-avfilter \
  --disable-postproc --disable-encoders --disable-muxers \
  --enable-openssl --enable-version3 --enable-zlib --enable-libdav1d \
  --pkg-config-flags=--static
make -j "${TENSORCODEC_BUILD_JOBS:-4}"
make install
mkdir -p "$prefix/share/licenses/ffmpeg"
cp COPYING.LGPLv3 LICENSE.md "$prefix/share/licenses/ffmpeg/"
cp -r "$build_root/dav1d/share/licenses/dav1d" "$prefix/share/licenses/"
