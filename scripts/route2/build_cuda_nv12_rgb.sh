#!/usr/bin/env bash
# NV12 ER → NCHW RGB CUDA kernel（路线 2 EGL 直绑 TRT）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
SRC="$ROOT/platform/jetson/cuda/cuda_nv12_to_rgb.cu"
OUT_DIR="$ROOT/build/route2"
OUT="$OUT_DIR/route2_nv12_rgb.so"

mkdir -p "$OUT_DIR"

ARCH="${CUDA_ARCH:-}"
if [[ -z "$ARCH" ]]; then
  if [[ "$(uname -m)" == "aarch64" ]]; then
    ARCH="sm_87"
  else
    ARCH="sm_75"
  fi
fi

NVCC="${NVCC:-nvcc}"
echo "Building $OUT (arch=$ARCH)"
"$NVCC" -shared -O3 -Xcompiler -fPIC \
  -gencode "arch=compute_${ARCH#sm_},code=${ARCH}" \
  -o "$OUT" "$SRC" -lcudart

ls -la "$OUT"
