#!/usr/bin/env python3
"""trtexec batch=2 构建，终端显示阶段进度条。"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "yolo26_pose"

PHASE_WEIGHTS = (0.08, 0.87, 0.05)
DEFAULT_BUILD_EST = {"n": 360.0, "m": 720.0, "l": 1200.0}
BAR_W = 40


def _paths(size: str) -> tuple[Path, Path, Path]:
    size = size.lower()
    if size not in ("n", "m", "l"):
        raise SystemExit(f"不支持的 --size: {size}（n | m | l）")
    onnx = MODEL_DIR / f"yolo26{size}-pose_b2.onnx"
    trt = MODEL_DIR / f"yolo26{size}-pose_fp16_b2.trt"
    log = ROOT / "outputs" / f"build_yolo26{size}_b2_trt.log"
    return onnx, trt, log


def _bar(pct: float, width: int = BAR_W) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100.0)
    return "█" * filled + "░" * (width - filled)


def _fmt_elapsed(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def _find_trtexec() -> str:
    for cand in (
        os.environ.get("TRTEXEC", ""),
        "/usr/src/tensorrt/bin/trtexec",
    ):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    from shutil import which

    p = which("trtexec")
    if p:
        return p
    raise SystemExit("未找到 trtexec")


def _phase_progress(phase: int, phase_elapsed: float, build_est_sec: float) -> float:
    w0, w1, w2 = PHASE_WEIGHTS
    if phase == 0:
        inner = min(1.0, phase_elapsed / 15.0)
        return 100.0 * w0 * inner
    if phase == 1:
        inner = min(0.98, phase_elapsed / max(build_est_sec, 30.0))
        return 100.0 * (w0 + w1 * inner)
    if phase == 2:
        inner = min(1.0, phase_elapsed / 20.0)
        return 100.0 * (w0 + w1 + w2 * inner)
    return 100.0


def _detect_phase(line: str, phase: int) -> tuple[int, str]:
    lo = line.lower()
    if "start parsing network model" in lo or "loading onnx" in lo:
        return 0, "解析 ONNX"
    if "finished parsing network model" in lo or "parse time:" in lo:
        return 1, "构建 TensorRT engine（选 tactic，较慢）"
    if "building engine" in lo or "builder kernel library" in lo:
        return 1, "构建 TensorRT engine（选 tactic，较慢）"
    if "completed creating engine" in lo or "engine built" in lo:
        return 2, "写入 engine 文件"
    if "save engine" in lo and phase >= 1:
        return 2, "写入 engine 文件"
    if phase == 0 and ("onnx ir version" in lo or "opset version" in lo):
        return 0, "解析 ONNX"
    return phase, ""


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 YOLO26-pose batch=2 TRT engine")
    ap.add_argument("--size", choices=("n", "m", "l"), default="m", help="模型尺寸")
    ap.add_argument("--force", action="store_true", help="已存在 engine 时也重建")
    args = ap.parse_args()

    onnx, trt, log = _paths(args.size)
    build_est = float(
        os.environ.get(
            "TRT_B2_BUILD_EST_SEC",
            str(DEFAULT_BUILD_EST.get(args.size, 600.0)),
        )
    )

    if trt.is_file() and not args.force:
        print(f"Batch=2 engine 已存在: {trt}")
        print(f"  大小: {trt.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  删除后重建: rm {trt}")
        return 0

    if not onnx.is_file():
        print(f"缺少 ONNX: {onnx}")
        if args.size == "l":
            print("先运行: bash scripts/route2/export_yolo26l_pose_onnx_b2.sh")
        elif args.size == "m":
            print("先运行: bash scripts/route2/export_yolo26m_pose_onnx_b2.sh")
        else:
            print("先运行: bash scripts/route2/export_yolo26_pose_onnx_b2.sh")
        return 1

    trtexec = _find_trtexec()
    log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        trtexec,
        f"--onnx={onnx}",
        f"--saveEngine={trt}",
        "--fp16",
        "--skipInference",
    ]

    print(f"TensorRT yolo26{args.size}-pose batch=2 构建")
    print(f"  ONNX:   {onnx}")
    print(f"  Engine: {trt}")
    print(f"  Log:    {log}")
    print(f"  预估构建 ~{int(build_est // 60)} 分钟（TRT_B2_BUILD_EST_SEC 可调）")
    print()

    t0 = time.monotonic()
    phase = 0
    phase_t0 = t0
    status = "启动 trtexec"
    last_line = ""
    pct = 0.0
    done = threading.Event()
    log_lines: list[str] = []

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _reader() -> None:
        nonlocal phase, phase_t0, status, last_line
        assert proc.stdout is not None
        with log.open("w", encoding="utf-8") as lf:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                log_lines.append(line)
                lf.write(line + "\n")
                lf.flush()
                last_line = line
                new_phase, new_status = _detect_phase(line, phase)
                if new_phase != phase:
                    phase = new_phase
                    phase_t0 = time.monotonic()
                if new_status:
                    status = new_status
        done.set()

    threading.Thread(target=_reader, daemon=True).start()

    try:
        while not done.is_set():
            elapsed = time.monotonic() - t0
            phase_elapsed = time.monotonic() - phase_t0
            pct = _phase_progress(phase, phase_elapsed, build_est)
            short = (last_line[:72] + "…") if len(last_line) > 72 else last_line
            sys.stdout.write(
                f"\r[{_bar(pct)}] {pct:5.1f}%  {_fmt_elapsed(elapsed)}  {status}"
                + (f"  | {short}" if short else "")
                + " " * 8
            )
            sys.stdout.flush()
            time.sleep(0.25)
            if proc.poll() is not None and done.is_set():
                break
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n\n已中断。可重新运行本脚本继续构建。")
        return 130

    elapsed = time.monotonic() - t0
    rc = proc.returncode or 0

    if rc == 0 and trt.is_file():
        mb = trt.stat().st_size / 1024 / 1024
        sys.stdout.write(
            f"\r[{_bar(100.0)}] 100.0%  {_fmt_elapsed(elapsed)}  完成  "
            f"engine {mb:.1f} MB\n"
        )
        sys.stdout.flush()
        return 0

    sys.stdout.write(f"\r[{_bar(pct)}] 失败  {_fmt_elapsed(elapsed)}\n")
    sys.stdout.flush()
    print(f"trtexec 退出码: {rc}")
    tail = [ln for ln in log_lines if re.search(r"\[E\]|ERROR|FAILED", ln, re.I)]
    if tail:
        print("--- 错误摘要 ---")
        for ln in tail[-8:]:
            print(ln)
    print(f"完整日志: {log}")
    return rc or 1


if __name__ == "__main__":
    raise SystemExit(main())
