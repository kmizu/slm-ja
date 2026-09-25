#!/bin/bash
# 密な層の C カーネル（native/libslmkern.so）を作る。AVX-512 の無い CPU や C コンパイラの無い環境では何もしない
# （Model は Scala のカーネルで動く）。-ffp-contract=off は、Scala 側と同じく a * b + c を融合させないため。
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=native/slmkern.c
LIB=native/libslmkern.so
CC="${CC:-gcc}"
if ! command -v "$CC" > /dev/null; then
  echo "build-native: $CC が無いので C のカーネルは作らない" >&2
  exit 0
fi
if ! grep -qw avx512f /proc/cpuinfo 2> /dev/null; then
  echo "build-native: AVX-512 が無いので C のカーネルは作らない" >&2
  exit 0
fi
if [ -f "$LIB" ] && [ "$LIB" -nt "$SRC" ] && [ "$LIB" -nt "$0" ]; then
  exit 0
fi
"$CC" -O3 -mavx512f -mavx512vl -mfma -ffp-contract=off -fno-fast-math -fPIC -shared -Wall -o "$LIB.tmp" "$SRC"
mv "$LIB.tmp" "$LIB"
echo "build-native: $LIB を作った" >&2
