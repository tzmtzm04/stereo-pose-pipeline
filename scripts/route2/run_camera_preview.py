#!/usr/bin/env python3
"""实时相机预览（无 infer / pose / 录像）。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from core.display.gpu_display_preview import (
    gpu_display_preview_available,
    nv12_display_frame_to_preview_bgr,
)
from orchestration.display_ui import check_display_env, teardown_display_window
from orchestration.metrics import ThroughputTracker
from platform.jetson.capture import JetsonGstNvmmCapture, gst_nvmm_available
from platform.jetson.nv12_gl_display import Nv12GlDisplay, nv12_gl_display_available

DisplayEye = Literal["left", "right"]
PREVIEW_WINDOW = "route2_camera_preview"
DEFAULT_EYE_W = 1920
DEFAULT_EYE_H = 1080
MIN_ROI_SPAN = 0.02
ROI_ZOOM_IN = 0.85
ROI_ZOOM_OUT = 1.18
HELP_TEXT = "左键框选局部放大，+/- 缩放，0/r 复位，q 退出"


def normalize_display_eye(eye: str) -> DisplayEye:
    key = str(eye).strip().lower()
    if key in ("right", "r", "1"):
        return "right"
    return "left"


class PreviewViewport:
    """局部 ROI 放大（归一化 uv 0~1）。"""

    def __init__(self) -> None:
        self.u0 = 0.0
        self.v0 = 0.0
        self.u1 = 1.0
        self.v1 = 1.0
        self._drag_start: tuple[int, int] | None = None

    def reset(self) -> None:
        self.u0 = 0.0
        self.v0 = 0.0
        self.u1 = 1.0
        self.v1 = 1.0

    def _clamp(self) -> None:
        self.u0 = max(0.0, min(1.0, self.u0))
        self.v0 = max(0.0, min(1.0, self.v0))
        self.u1 = max(0.0, min(1.0, self.u1))
        self.v1 = max(0.0, min(1.0, self.v1))
        if self.u1 < self.u0:
            self.u0, self.u1 = self.u1, self.u0
        if self.v1 < self.v0:
            self.v0, self.v1 = self.v1, self.v0
        if self.u1 - self.u0 < MIN_ROI_SPAN:
            c = 0.5 * (self.u0 + self.u1)
            self.u0 = max(0.0, c - MIN_ROI_SPAN * 0.5)
            self.u1 = min(1.0, self.u0 + MIN_ROI_SPAN)
            self.u0 = self.u1 - MIN_ROI_SPAN
        if self.v1 - self.v0 < MIN_ROI_SPAN:
            c = 0.5 * (self.v0 + self.v1)
            self.v0 = max(0.0, c - MIN_ROI_SPAN * 0.5)
            self.v1 = min(1.0, self.v0 + MIN_ROI_SPAN)
            self.v0 = self.v1 - MIN_ROI_SPAN

    def _set_crop(self, u0: float, v0: float, u1: float, v1: float) -> None:
        self.u0, self.v0, self.u1, self.v1 = u0, v0, u1, v1
        self._clamp()

    def zoom_in(self) -> None:
        self._zoom(ROI_ZOOM_IN)

    def zoom_out(self) -> None:
        self._zoom(ROI_ZOOM_OUT)

    def _zoom(self, factor: float) -> None:
        cu = 0.5 * (self.u0 + self.u1)
        cv = 0.5 * (self.v0 + self.v1)
        hw = 0.5 * (self.u1 - self.u0) * factor
        hh = 0.5 * (self.v1 - self.v0) * factor
        self._set_crop(cu - hw, cv - hh, cu + hw, cv + hh)

    def apply_window_selection(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        win_w: int,
        win_h: int,
    ) -> None:
        if win_w <= 0 or win_h <= 0:
            return
        sx0, sx1 = sorted((int(x0), int(x1)))
        sy0, sy1 = sorted((int(y0), int(y1)))
        if sx1 - sx0 < 8 or sy1 - sy0 < 8:
            return
        rel_u0 = sx0 / float(win_w)
        rel_v0 = sy0 / float(win_h)
        rel_u1 = sx1 / float(win_w)
        rel_v1 = sy1 / float(win_h)
        span_u = self.u1 - self.u0
        span_v = self.v1 - self.v0
        self._set_crop(
            self.u0 + rel_u0 * span_u,
            self.v0 + rel_v0 * span_v,
            self.u0 + rel_u1 * span_u,
            self.v0 + rel_v1 * span_v,
        )

    def crop_bgr(self, bgr):
        h, w = bgr.shape[:2]
        x0 = int(self.u0 * w)
        y0 = int(self.v0 * h)
        x1 = max(x0 + 1, int(self.u1 * w))
        y1 = max(y0 + 1, int(self.v1 * h))
        return bgr[y0:y1, x0:x1]

    def status_tag(self) -> str:
        zoom_u = 1.0 / max(MIN_ROI_SPAN, self.u1 - self.u0)
        zoom_v = 1.0 / max(MIN_ROI_SPAN, self.v1 - self.v0)
        return f"roi={max(zoom_u, zoom_v):.1f}x"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Route2 实时相机预览（纯画面，无 pose）")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=3840, help="SBS 全幅宽")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument(
        "--eye",
        choices=("left", "right"),
        default="left",
        help="单眼预览（默认左眼）",
    )
    ap.add_argument(
        "--sbs",
        action="store_true",
        help="显示全幅 SBS（3840×1080），不裁单眼",
    )
    ap.add_argument(
        "--display-width",
        type=int,
        default=0,
        help="窗口宽（0=按 scale×原图；与 --scale 二选一）",
    )
    ap.add_argument("--display-height", type=int, default=0)
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="初始窗口相对原图缩放（仅窗口大小，非局部放大）",
    )
    ap.add_argument(
        "--display-egl-gl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="P4 EGL+GL NV12 fd 直显（默认开）",
    )
    ap.add_argument(
        "--cpu-preview",
        action="store_true",
        help="OpenCV imshow + CPU NV12→BGR（回退路径）",
    )
    ap.add_argument("--log-every", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=0, help="0=不限，按 q 退出")
    ap.add_argument("--decoder-extra-surfaces", type=int, default=8)
    return ap.parse_args()


def _native_display_size(args: argparse.Namespace) -> tuple[int, int]:
    if args.sbs:
        return int(args.width), int(args.height)
    return DEFAULT_EYE_W, DEFAULT_EYE_H


def _initial_window_size(args: argparse.Namespace) -> tuple[int, int]:
    native_w, native_h = _native_display_size(args)
    if args.display_width > 0:
        w = int(args.display_width)
        h = int(args.display_height) if args.display_height > 0 else w
        return w, h
    scale = max(0.15, min(2.0, float(args.scale)))
    return (
        max(160, int(round(native_w * scale))),
        max(120, int(round(native_h * scale))),
    )


def _gl_sbs_mode(args: argparse.Namespace) -> int:
    """0=全幅 SBS，1=左眼，2=右眼。"""
    if args.sbs:
        return 0
    return 2 if normalize_display_eye(args.eye) == "right" else 1


def _check_device(device: str) -> None:
    p = Path(device)
    if not p.exists():
        print(f"[!] 采集设备不存在: {device}", flush=True)


def _open_capture(args: argparse.Namespace) -> JetsonGstNvmmCapture:
    use_gl = bool(args.display_egl_gl) and not args.cpu_preview
    cap = JetsonGstNvmmCapture(
        camera=args.device,
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        sbs_mode="both",
        ring_depth=2,
        mjpeg_split=False,
        display_sbs_branch=True,
        display_full_sbs=use_gl or normalize_display_eye(args.eye) == "right",
        decoder_extra_surfaces=max(0, int(args.decoder_extra_surfaces)),
    )
    if cap.sink_display_left is None:
        raise RuntimeError("display 支路未就绪（需要 sink_display_left）")
    cap.ensure_playing()
    return cap


def _poll_imshow_keys(viewport: PreviewViewport) -> bool:
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return True
    if key in (ord("+"), ord("=")):
        viewport.zoom_in()
    elif key in (ord("-"), ord("_")):
        viewport.zoom_out()
    elif key in (ord("0"), ord("r"), ord("R")):
        viewport.reset()
    return False


def _make_imshow_mouse_cb(viewport: PreviewViewport, win_size: list[int]):
    def _on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            viewport._drag_start = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONUP and viewport._drag_start is not None:
            x0, y0 = viewport._drag_start
            viewport._drag_start = None
            viewport.apply_window_selection(
                x0,
                y0,
                int(x),
                int(y),
                win_w=win_size[0],
                win_h=win_size[1],
            )
        elif event == cv2.EVENT_MOUSEMOVE and viewport._drag_start is not None:
            pass

    return _on_mouse


def _sync_imshow_window(width: int, height: int) -> tuple[int, int]:
    _ensure_imshow_window(width, height)
    try:
        rect = cv2.getWindowImageRect(PREVIEW_WINDOW)
        if rect[2] > 0 and rect[3] > 0:
            return int(rect[2]), int(rect[3])
    except Exception:
        pass
    return width, height


def _ensure_imshow_window(width: int, height: int) -> None:
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WINDOW, width, height)


def _show_imshow_frame(bgr, status: str) -> None:
    canvas = bgr.copy()
    cv2.putText(
        canvas,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(PREVIEW_WINDOW, canvas)


def _run_gl_preview(args: argparse.Namespace, cap: JetsonGstNvmmCapture) -> int:
    dw, dh = _initial_window_size(args)
    gl = Nv12GlDisplay(title=PREVIEW_WINDOW, width=dw, height=dh)
    gl.set_sbs_eye(_gl_sbs_mode(args))
    gl.enable_roi_zoom(True)
    throughput = ThroughputTracker()
    shown = 0
    fail_streak = 0
    print(
        f"camera preview EGL+GL eye={args.eye} sbs={args.sbs} "
        f"window={dw}x{dh}（{HELP_TEXT}）",
        flush=True,
    )
    try:
        while True:
            if args.max_frames > 0 and shown >= args.max_frames:
                break
            t0 = time.perf_counter()
            ok, frame, capture_seq = cap.read_display_nv12()
            pull_ms = (time.perf_counter() - t0) * 1000.0
            if not ok or frame is None:
                fail_streak += 1
                if fail_streak == 1 or fail_streak % 30 == 0:
                    print(f"preview pull_fail streak={fail_streak}", flush=True)
                if gl.poll_quit():
                    break
                time.sleep(0.001)
                continue
            fail_streak = 0
            try:
                if frame.dmabuf_fd is None:
                    raise RuntimeError("display NV12 缺少 dmabuf_fd")
                throughput.mark()
                ui_fps = throughput.rolling_fps()
                win_w, win_h = gl.window_size()
                u0, v0, u1, v1 = gl.view_crop()
                zoom = max(1.0 / max(1e-6, u1 - u0), 1.0 / max(1e-6, v1 - v0))
                gl.set_status(
                    f"seq={capture_seq} ui_fps={ui_fps:.1f} pull={pull_ms:.1f}ms "
                    f"win={win_w}x{win_h} roi={zoom:.1f}x"
                )
                gl.clear_pose()
                render_ms = gl.render_fd(int(frame.dmabuf_fd))
            finally:
                try:
                    frame.release()
                except Exception:
                    pass
            shown += 1
            if shown == 1:
                print(
                    f"preview 首帧 seq={capture_seq} pull={pull_ms:.1f}ms "
                    f"gl={render_ms:.1f}ms",
                    flush=True,
                )
            elif shown % max(1, int(args.log_every)) == 0:
                print(
                    f"preview n={shown} seq={capture_seq} ui_fps={ui_fps:.1f} "
                    f"pull={pull_ms:.1f} gl={render_ms:.1f}",
                    flush=True,
                )
            if gl.poll_quit():
                break
    finally:
        gl.close()
    print(f"preview 结束 frames={shown}", flush=True)
    return 0


def _run_imshow_preview(args: argparse.Namespace, cap: JetsonGstNvmmCapture) -> int:
    init_w, init_h = _initial_window_size(args)
    win_size = [init_w, init_h]
    viewport = PreviewViewport()
    dw, dh = _sync_imshow_window(init_w, init_h)
    win_size[0], win_size[1] = dw, dh
    cv2.setMouseCallback(PREVIEW_WINDOW, _make_imshow_mouse_cb(viewport, win_size))
    use_gpu = gpu_display_preview_available() and not args.cpu_preview
    throughput = ThroughputTracker()
    shown = 0
    fail_streak = 0
    eye = normalize_display_eye(args.eye)
    print(
        f"camera preview imshow eye={eye} sbs={args.sbs} "
        f"path={'GPU' if use_gpu else 'CPU'} window={dw}x{dh} "
        f"（{HELP_TEXT}）",
        flush=True,
    )
    try:
        while True:
            if args.max_frames > 0 and shown >= args.max_frames:
                break
            t0 = time.perf_counter()
            ok, frame, capture_seq = cap.read_display_nv12()
            pull_ms = (time.perf_counter() - t0) * 1000.0
            if not ok or frame is None:
                fail_streak += 1
                if fail_streak == 1 or fail_streak % 30 == 0:
                    print(f"preview pull_fail streak={fail_streak}", flush=True)
                if _poll_imshow_keys(viewport):
                    break
                time.sleep(0.001)
                continue
            fail_streak = 0
            preview_ms = 0.0
            try:
                dw, dh = _sync_imshow_window(init_w, init_h)
                win_size[0], win_size[1] = dw, dh
                if args.sbs:
                    bgr, preview_ms = nv12_display_frame_to_preview_bgr(
                        frame,
                        display_eye="left",
                    )
                else:
                    bgr, preview_ms = nv12_display_frame_to_preview_bgr(
                        frame,
                        display_eye=eye,
                    )
                throughput.mark()
                ui_fps = throughput.rolling_fps()
                cropped = viewport.crop_bgr(bgr)
                status = (
                    f"seq={capture_seq} ui_fps={ui_fps:.1f} "
                    f"pull={pull_ms:.1f} prev={preview_ms:.1f}ms "
                    f"{viewport.status_tag()}"
                )
                if cropped.shape[1] != dw or cropped.shape[0] != dh:
                    cropped = cv2.resize(
                        cropped,
                        (dw, dh),
                        interpolation=cv2.INTER_LINEAR,
                    )
                _show_imshow_frame(cropped, status)
            finally:
                try:
                    frame.release()
                except Exception:
                    pass
            shown += 1
            if shown == 1:
                print(
                    f"preview 首帧 seq={capture_seq} pull={pull_ms:.1f}ms "
                    f"prev={preview_ms:.1f}ms",
                    flush=True,
                )
            elif shown % max(1, int(args.log_every)) == 0:
                print(
                    f"preview n={shown} seq={capture_seq} ui_fps={ui_fps:.1f} "
                    f"pull={pull_ms:.1f} prev={preview_ms:.1f}",
                    flush=True,
                )
            if _poll_imshow_keys(viewport):
                break
    finally:
        cv2.destroyWindow(PREVIEW_WINDOW)
        teardown_display_window()
    print(f"preview 结束 frames={shown}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if not gst_nvmm_available():
        print("ERROR: NVMM 不可用（需要 GStreamer gi + libnvbufsurface）", flush=True)
        return 1

    check_display_env(no_display=False)
    if not os.environ.get("DISPLAY"):
        print("提示: 本机桌面请先 export DISPLAY=:0", flush=True)

    _check_device(args.device)

    use_gl = bool(args.display_egl_gl) and not args.cpu_preview
    if use_gl and not nv12_gl_display_available():
        print(
            "WARN: EGL+GL 不可用，回退 OpenCV imshow（可先 build_gl_display.sh）",
            flush=True,
        )
        use_gl = False

    cap: JetsonGstNvmmCapture | None = None
    try:
        print("打开 NVMM 采集（仅 display 支路）...", flush=True)
        cap = _open_capture(args)
        if use_gl:
            return _run_gl_preview(args, cap)
        return _run_imshow_preview(args, cap)
    except KeyboardInterrupt:
        print("\npreview 中断", flush=True)
        return 0
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
