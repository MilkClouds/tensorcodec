# Bundled native libraries

TensorCodec's own code is Apache-2.0 licensed. Linux and macOS wheels bundle shared FFmpeg
7.1.5 libraries, built without GPL codec libraries using
`scripts/build_ffmpeg.sh`. This configuration is LGPL-3.0-or-later. Its notices
and both the LGPLv3 and incorporated GPLv3 texts are included here. The exact
upstream source is https://ffmpeg.org/releases/ffmpeg-7.1.5.tar.xz; the build
script records the configuration. FFmpeg libraries remain dynamically linked
and can be rebuilt/replaced with an ABI-compatible build.

libavcodec statically includes dav1d 1.5.4 (AV1 decoding), licensed under the
BSD 2-Clause license (`dav1d.txt`). Its exact source is
https://downloads.videolan.org/pub/videolan/dav1d/1.5.4/dav1d-1.5.4.tar.xz;
`scripts/build_dav1d.sh` records the checksum and build configuration.

Release wheels also bundle shared OpenSSL 3.5.9 LTS, licensed
under Apache-2.0. Its exact source is
https://github.com/openssl/openssl/releases/download/openssl-3.5.9/openssl-3.5.9.tar.gz;
`scripts/build_openssl.sh` records the checksum and build configuration.
System-library builds can additionally depend on Zstandard; its notices are
retained here. Inspect repaired wheels when changing the native build.

Image decoding bundles libjpeg-turbo 3.2.0 (IJG / BSD-3-Clause / zlib), libwebp
1.6.0 (BSD-3-Clause), and libavif 1.4.2 (BSD-2-Clause). libavif includes dav1d
1.5.4 and libyuv commit 644251f252a84bf8ce91ff0aca86a9b16b069ab8 (BSD-3-Clause).
`scripts/build_image_deps.sh` pins their sources and build configuration.
PNG uses the Rust png 0.18.1 crate (MIT OR Apache-2.0), with Cargo.lock pinning its
transitive dependencies. GIF uses the MIT-licensed giflib sources vendored from
TorchCodec v0.17.0, whose provenance and notices are in native/vendor/giflib/README.
Image bindings follow TorchCodec's mode/conversion semantics; its BSD notice is
included in torchcodec.txt.

HEIC optionally loads the user's system libheif; it is not bundled. The source distribution's
HEIC test assets retain TorchCodec's BSD license in `tests/resources/images/`.
