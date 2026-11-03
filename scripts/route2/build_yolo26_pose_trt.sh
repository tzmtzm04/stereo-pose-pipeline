#!/usr/bin/env bash
# yolo26n-pose.onnx → TensorRT engine（FP16 默认）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
MODEL_DIR="$ROOT/models/yolo26_pose"
ONNX="$MODEL_DIR/yolo26n-pose.onnx"
PRECISION="${1:-fp16}"

case "$PRECISION" in
  fp16) TRT="$MODEL_DIR/yolo26n-pose_fp16.trt"; TRTEXEC_EXTRA=(--fp16) ;;
  fp32) TRT="$MODEL_DIR/yolo26n-pose_fp32.trt"; TRTEXEC_EXTRA=() ;;
  int8) TRT="$MODEL_DIR/yolo26n-pose_int8.trt"; TRTEXEC_EXTRA=(--int8) ;;
  *)
    echo "用法: $0 [fp16|fp32|int8]"
    exit 1
    ;;
esac

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [[ ! -x "$TRTEXEC" ]]; then
  TRTEXEC="$(command -v trtexec || true)"
fi
if [[ -z "$TRTEXEC" || ! -x "$TRTEXEC" ]]; then
  echo "未找到 trtexec，请设置 TRTEXEC 或安装 TensorRT"
  exit 1
fi

if [[ ! -f "$ONNX" ]]; then
  echo "缺少 ONNX: $ONNX"
  echo "先运行: bash scripts/route2/download_yolo26_pose_onnx.sh"
  exit 1
fi

if [[ -f "$TRT" ]]; then
  echo "Engine 已存在: $TRT（删除后重建: rm $TRT）"
  ls -la "$TRT"
  exit 0
fi

echo "构建 TensorRT engine ($PRECISION) ..."
echo "  ONNX:   $ONNX"
echo "  Engine: $TRT"
echo "  预计 Jetson 上需数分钟，请耐心等待"

"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$TRT" \
  "${TRTEXEC_EXTRA[@]}" \
  --skipInference

echo "完成:"
ls -la "$TRT"
