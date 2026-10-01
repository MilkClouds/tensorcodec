# Bundled native libraries

TensorCodec's own code is MIT licensed. Linux wheels bundle shared FFmpeg
7.1.5 libraries, built without GPL codec libraries using
`scripts/build_ffmpeg.sh`. This configuration is LGPL-3.0-or-later. Its notices
and both the LGPLv3 and incorporated GPLv3 texts are included here. The exact
upstream source is https://ffmpeg.org/releases/ffmpeg-7.1.5.tar.xz; the build
script records the configuration. FFmpeg libraries remain dynamically linked
and can be rebuilt/replaced with an ABI-compatible build.

Portable Linux release wheels also bundle shared OpenSSL 3.5.9 LTS, licensed
under Apache-2.0. Its exact source is
https://github.com/openssl/openssl/releases/download/openssl-3.5.9/openssl-3.5.9.tar.gz;
`scripts/build_openssl.sh` records the checksum and build configuration.
System-library builds can additionally depend on Zstandard; its notices are
retained here. Inspect repaired wheels when changing the native build.
