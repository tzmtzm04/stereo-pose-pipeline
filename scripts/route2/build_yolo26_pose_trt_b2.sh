#!/usr/bin/env bash
# yolo26n-pose.onnx → TensorRT batch=2 engine（FP16）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
MODEL_DIR="$ROOT/models/yolo26_pose"
ONNX="$MODEL_DIR/yolo26n-pose_b2.onnx"
TRT="$MODEL_DIR/yolo26n-pose_fp16_b2.trt"

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [[ ! -x "$TRTEXEC" ]]; then
  TRTEXEC="$(command -v trtexec || true)"
fi
if [[ -z "$TRTEXEC" || ! -x "$TRTEXEC" ]]; then
  echo "未找到 trtexec，请设置 TRTEXEC 或安装 TensorRT"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/yolo26n-pose.onnx" ]]; then
  echo "缺少 ONNX: $MODEL_DIR/yolo26n-pose.onnx"
  echo "先运行: bash scripts/route2/download_yolo26_pose_onnx.sh"
  exit 1
fi

if [[ ! -f "$ONNX" ]]; then
  echo "缺少 batch=2 ONNX，先导出 ..."
  bash "$ROOT/scripts/route2/export_yolo26_pose_onnx_b2.sh"
fi

if [[ -f "$TRT" ]]; then
  echo "Batch=2 engine 已存在: $TRT（删除后重建: rm $TRT）"
  ls -la "$TRT"
  exit 0
fi

# 带进度条构建（日志: outputs/build_b2_trt.log）
exec python3 "$ROOT/scripts/route2/build_yolo26_pose_trt_b2_progress.py"
