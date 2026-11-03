#!/usr/bin/env bash
# 实时相机预览（无 infer / pose）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/build/route2/route2_gl_display.so" ]]; then
  bash scripts/route2/build_gl_display.sh
fi

if [[ ! -f "$ROOT/build/route2/route2_nv12_rgb.so" ]]; then
  bash scripts/route2/build_cuda_nv12_rgb.sh
fi

export PYTHONUNBUFFERED=1
export DISPLAY="${DISPLAY:-:0}"

CAMERA_EXPOSURE_PRESET="${ROUTE2_CAMERA_EXPOSURE:-auto}"
if [[ "$CAMERA_EXPOSURE_PRESET" != "off" && "$CAMERA_EXPOSURE_PRESET" != "none" ]]; then
  bash "$ROOT/scripts/route2/set_camera_exposure.sh" "$CAMERA_EXPOSURE_PRESET" --device /dev/video0
fi

python3 scripts/route2/run_camera_preview.py "$@"
