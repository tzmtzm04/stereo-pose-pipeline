#!/usr/bin/env bash
# 路线 2 实时 3D：形态 B 默认（mjpeg ingress + split display + PoseStore）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
cd "$ROOT"

ENGINE="${ROOT}/models/yolo26_pose/yolo26n-pose_fp16.trt"
B2_ENGINE="${ROOT}/models/yolo26_pose/yolo26n-pose_fp16_b2.trt"
CALIB="${ROOT}/calib/stereo_sbs.json"

if [[ ! -f "$ENGINE" && ! -f "$B2_ENGINE" ]]; then
  echo "缺少 yolo26n TRT engine，先运行:"
  echo "  bash scripts/route2/download_yolo26_pose_onnx.sh"
  echo "  bash scripts/route2/build_yolo26_pose_trt.sh"
  echo "  bash scripts/route2/build_yolo26_pose_trt_b2.sh   # GPU letterbox 双目 batch=2"
  exit 1
fi
if [[ ! -f "$CALIB" ]]; then
  echo "缺少标定文件: $CALIB"
  exit 1
fi
if [[ ! -f "$ROOT/build/route2/route2_nv12_rgb.so" ]]; then
  bash scripts/route2/build_cuda_nv12_rgb.sh
fi

# display 默认 OpenCV + GPU CUDA 裁眼（不与 infer 争 VIC）；GL：ROUTE2_DISPLAY_GL=1
DISPLAY_GL="${ROUTE2_DISPLAY_GL:-0}"
DISPLAY_ARGS=()
if [[ "$DISPLAY_GL" == "1" ]]; then
  if [[ ! -f "$ROOT/build/route2/route2_gl_display.so" ]]; then
    bash scripts/route2/build_gl_display.sh
  fi
else
  DISPLAY_ARGS=(--no-display-egl-gl)
fi

export PYTHONUNBUFFERED=1
export DISPLAY="${DISPLAY:-:0}"

CAMERA_EXPOSURE_PRESET="${ROUTE2_CAMERA_EXPOSURE:-auto}"
if [[ "$CAMERA_EXPOSURE_PRESET" != "off" && "$CAMERA_EXPOSURE_PRESET" != "none" ]]; then
  bash "$ROOT/scripts/route2/set_camera_exposure.sh" "$CAMERA_EXPOSURE_PRESET" --device /dev/video0
fi

python3 -m app.route2.main \
  --calib "$CALIB" \
  --engine "$ENGINE" \
  --device /dev/video0 \
  --width 3840 --height 1080 --fps 60 \
  --log-every 30 \
  "${DISPLAY_ARGS[@]}" \
  "$@"

