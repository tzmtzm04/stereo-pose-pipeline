#!/usr/bin/env bash
# Route2 实时黑盒 + CPU 叠图预览（OpenCV imshow，无 EGL/GL、无 HTML）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ENGINE="${ROOT}/models/yolo26_pose/yolo26n-pose_fp16.trt"
B2_ENGINE="${ROOT}/models/yolo26_pose/yolo26n-pose_fp16_b2.trt"
CALIB="${ROOT}/calib/stereo_sbs.json"

if [[ ! -f "$ENGINE" && ! -f "$B2_ENGINE" ]]; then
  echo "缺少 yolo26n TRT engine，先运行:"
  echo "  bash scripts/route2/download_yolo26_pose_onnx.sh"
  echo "  bash scripts/route2/build_yolo26_pose_trt.sh"
  echo "  bash scripts/route2/build_yolo26_pose_trt_b2.sh"
  exit 1
fi
if [[ ! -f "$CALIB" ]]; then
  echo "缺少标定文件: $CALIB"
  exit 1
fi
if [[ ! -f "$ROOT/build/route2/route2_nv12_rgb.so" ]]; then
  bash "$ROOT/scripts/route2/build_cuda_nv12_rgb.sh"
fi

export PYTHONUNBUFFERED=1
export DISPLAY="${DISPLAY:-:0}"

CAMERA_EXPOSURE_PRESET="${ROUTE2_CAMERA_EXPOSURE:-auto}"
if [[ "$CAMERA_EXPOSURE_PRESET" != "off" && "$CAMERA_EXPOSURE_PRESET" != "none" ]]; then
  bash "$ROOT/scripts/route2/set_camera_exposure.sh" "$CAMERA_EXPOSURE_PRESET" --device /dev/video0
fi

echo "预览: OpenCV route2_pose3d（CPU NV12→BGR + live_compose 叠骨架）"
echo "策略: 无骨架原画直显，有骨架 pose→video exact 叠图"
echo "退出: 预览窗焦点按 q（无时限，勿加 --max-frames）"

exec python3 -m app.route2.main \
  --calib "$CALIB" \
  --engine "$ENGINE" \
  --device /dev/video0 \
  --width 3840 --height 1080 --fps 60 \
  --log-every 30 \
  --no-display-egl-gl \
  --cpu-preview \
  "$@"
