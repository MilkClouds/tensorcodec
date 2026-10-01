#!/usr/bin/env bash
# OpenSSL 3.5 LTS, built against the release wheel's glibc baseline.
set -euo pipefail
prefix="${1:?usage: build_openssl.sh ABSOLUTE_INSTALL_PREFIX}"
version=3.5.9
build_root="$(mktemp -d)"
curl -fsSL "https://github.com/openssl/openssl/releases/download/openssl-${version}/openssl-${version}.tar.gz" -o "$build_root/openssl.tar.gz"
echo "603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a  $build_root/openssl.tar.gz" | sha256sum --check
tar -xf "$build_root/openssl.tar.gz" -C "$build_root"
cd "$build_root/openssl-${version}"
./Configure --prefix="$prefix" --openssldir=/etc/ssl --libdir=lib shared no-tests
make -j "${TENSORCODEC_BUILD_JOBS:-4}"
make install_sw
mkdir -p "$prefix/share/licenses/openssl"
cp LICENSE.txt "$prefix/share/licenses/openssl/"
