#!/usr/bin/env bash
# 下载 YOLO26n-pose 权重；导出 ONNX（供 TensorRT 转换）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
MODEL_DIR="$ROOT/models/yolo26_pose"
EXPORT_PYTHON="${EXPORT_PYTHON:-python3}"
PT_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt"

mkdir -p "$MODEL_DIR"
PT="$MODEL_DIR/yolo26n-pose.pt"
ONNX="$MODEL_DIR/yolo26n-pose.onnx"

if [[ ! -f "$PT" ]]; then
  echo "下载 yolo26n-pose.pt ..."
  curl -L -o "$PT" "$PT_URL"
fi

if [[ -f "$ONNX" ]]; then
  echo "ONNX 已存在: $ONNX"
else
  echo "导出 ONNX（$EXPORT_PYTHON + ultralytics）..."
  if ! "$EXPORT_PYTHON" -c "import torch; from ultralytics import YOLO" 2>/dev/null; then
    echo "错误: PyTorch/ultralytics 不可用。"
    echo "  安装 ultralytics 后重试，或设置 EXPORT_PYTHON=/path/to/venv/bin/python"
    exit 1
  fi
  "$EXPORT_PYTHON" - "$PT" "$ONNX" <<'PY'
import sys
from pathlib import Path
from ultralytics import YOLO

pt, onnx = Path(sys.argv[1]), Path(sys.argv[2])
out = YOLO(str(pt)).export(format="onnx", imgsz=640, simplify=True)
out_path = Path(out)
if out_path.resolve() != onnx.resolve():
    onnx.write_bytes(out_path.read_bytes())
print("exported:", onnx)
PY
fi

echo "完成:"
ls -la "$ONNX"
echo "下一步: bash scripts/route2/build_yolo26_pose_trt.sh"
