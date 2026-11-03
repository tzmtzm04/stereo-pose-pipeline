#!/usr/bin/env bash
# Route2 EGL+GL 硬件预览（NV12 fd → GL_TEXTURE_EXTERNAL_OES）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
SRC="$ROOT/platform/jetson/gl/route2_gl_display.cpp"
OUT_DIR="$ROOT/build/route2"
OUT="$OUT_DIR/route2_gl_display.so"

mkdir -p "$OUT_DIR"

JMAPI="${JETSON_MMAPI:-/usr/src/jetson_multimedia_api/include}"
CXX="${CXX:-g++}"

NVLIB="${NVLIB:-/usr/lib/aarch64-linux-gnu/nvidia}"

echo "Building $OUT"
"$CXX" -shared -O3 -fPIC -std=c++17 \
  -I"$JMAPI" \
  -L"$NVLIB" -Wl,-rpath,"$NVLIB" \
  -o "$OUT" "$SRC" \
  -lnvbufsurface -lEGL -lGLESv2 -lX11 -ldl

ls -la "$OUT"
