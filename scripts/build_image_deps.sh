#!/usr/bin/env bash
# Dedicated image decoders, with no encoder tools. PNG and GIF are built by Cargo.
set -euo pipefail
prefix="${1:?usage: build_image_deps.sh ABSOLUTE_INSTALL_PREFIX}"
root="$(mktemp -d)"
jobs="${TENSORCODEC_BUILD_JOBS:-4}"
fetch() {
  curl -fLsS "$2" -o "$root/$1.tar.gz"
  echo "$3  $root/$1.tar.gz" | sha256sum --check
  mkdir "$root/$1"
  tar -xf "$root/$1.tar.gz" -C "$root/$1" --strip-components=1
}
build() {
  local name="$1"
  shift
  cmake -S "$root/$name" -B "$root/$name-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_POSITION_INDEPENDENT_CODE=ON "$@"
  cmake --build "$root/$name-build" --parallel "$jobs"
  cmake --install "$root/$name-build"
}
fetch jpeg https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.2.0.tar.gz \
  980dd81f425082aa6d7c9e47fef27554ce7a9ffc8e2f6e863b97d263c5c50858
build jpeg -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=OFF -DWITH_TOOLS=OFF -DWITH_TESTS=OFF
fetch webp https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz \
  e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
build webp -DBUILD_SHARED_LIBS=ON -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF \
  -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF \
  -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF \
  -DWEBP_BUILD_EXTRAS=OFF
fetch avif https://github.com/AOMediaCodec/libavif/archive/refs/tags/v1.4.2.tar.gz \
  2b645287340ba5a631d268b551dc2d72bd73ac33335962dd36dcdb6d8366921d
# libavif's pinned SIMD color converter, also used by TorchCodec 0.17.0.
git init -q "$root/avif/ext/libyuv"
git -C "$root/avif/ext/libyuv" remote add origin https://chromium.googlesource.com/libyuv/libyuv
git -C "$root/avif/ext/libyuv" fetch --depth=1 origin 644251f252a84bf8ce91ff0aca86a9b16b069ab8
git -C "$root/avif/ext/libyuv" checkout --detach FETCH_HEAD
test "$(git -C "$root/avif/ext/libyuv" rev-parse HEAD)" = 644251f252a84bf8ce91ff0aca86a9b16b069ab8
bash "$(dirname "$0")/build_dav1d.sh" "$root/dav1d"
export PKG_CONFIG_PATH="$root/dav1d/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
build avif -DBUILD_SHARED_LIBS=ON -DAVIF_CODEC_DAV1D=SYSTEM -DAVIF_LIBYUV=LOCAL \
  -DAVIF_CODEC_AOM=OFF -DAVIF_CODEC_RAV1E=OFF -DAVIF_CODEC_SVT=OFF -DAVIF_CODEC_LIBGAV1=OFF \
  -DAVIF_CODEC_AVM=OFF -DAVIF_LIBSHARPYUV=OFF -DAVIF_JPEG=OFF -DAVIF_ZLIBPNG=OFF \
  -DAVIF_LIBXML2=OFF -DAVIF_BUILD_APPS=OFF -DAVIF_BUILD_TESTS=OFF -DAVIF_BUILD_EXAMPLES=OFF
# This prefix ships only shared libavif, with dav1d/libyuv statically embedded.
# Its consumers need neither dependency's development package or static archive.
sed '/^Requires.private:/d; /^Libs.private:/d' "$prefix/lib/pkgconfig/libavif.pc" > "$root/libavif.pc"
mv "$root/libavif.pc" "$prefix/lib/pkgconfig/libavif.pc"
