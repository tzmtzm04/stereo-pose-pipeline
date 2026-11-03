#!/usr/bin/env python3
"""将 yolo26n-pose.onnx（batch=1）复制为 batch=2 静态 ONNX。"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "yolo26_pose"
SRC = MODEL_DIR / "yolo26n-pose.onnx"
DST = MODEL_DIR / "yolo26n-pose_b2.onnx"


def _patch_batch(proto: onnx.ValueInfoProto, batch: int = 2) -> None:
    tt = proto.type.tensor_type
    if len(tt.shape.dim) >= 1 and tt.shape.dim[0].dim_value == 1:
        tt.shape.dim[0].dim_value = batch


def main() -> int:
    if not SRC.is_file():
        print(f"缺少 ONNX: {SRC}", file=sys.stderr)
        return 1
    if DST.is_file() and DST.stat().st_mtime >= SRC.stat().st_mtime:
        print(f"已是最新: {DST}")
        return 0
    m = onnx.load(str(SRC))
    for t in m.graph.input:
        _patch_batch(t)
    for t in m.graph.output:
        _patch_batch(t)
    for vi in m.graph.value_info:
        _patch_batch(vi)
    onnx.checker.check_model(m)
    onnx.save(m, str(DST))
    print(f"写入 {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
