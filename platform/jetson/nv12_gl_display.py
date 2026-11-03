"""ctypes 绑定 route2_gl_display.so（EGL+GL NV12 预览 + 2D 骨架）。"""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Any, Sequence

from app.paths import ROOT

_lib: ctypes.CDLL | None = None
_lib_path: Path | None = None

# 嵌入预览屏外预热坐标（eglSwap 需要 map，但不能出现在可见桌面）
STAGING_ORIGIN_X = -8192
STAGING_ORIGIN_Y = -8192


class Route2GlKpt(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("conf", ctypes.c_float),
        ("has_depth", ctypes.c_uint8),
    ]


def _default_so_path() -> Path:
    return ROOT / "build" / "route2" / "route2_gl_display.so"


def _load() -> ctypes.CDLL:
    global _lib, _lib_path
    if _lib is not None:
        return _lib
    env = os.environ.get("ROUTE2_GL_DISPLAY_SO")
    path = Path(env) if env else _default_so_path()
    _lib_path = path
    _lib = ctypes.CDLL(str(path))
    _lib.route2_gl_display_available.restype = ctypes.c_int
    _lib.route2_gl_display_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.route2_gl_display_create.restype = ctypes.c_void_p
    _lib.route2_gl_display_create_ex.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.route2_gl_display_create_ex.restype = ctypes.c_void_p
    _lib.route2_gl_display_destroy.argtypes = [ctypes.c_void_p]
    _lib.route2_gl_display_set_status.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    _lib.route2_gl_display_clear_pose.argtypes = [ctypes.c_void_p]
    _lib.route2_gl_display_set_sbs_left_eye.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.route2_gl_display_set_sbs_eye.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.route2_gl_display_set_pose.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Route2GlKpt),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.route2_gl_display_render_fd.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.route2_gl_display_render_fd.restype = ctypes.c_double
    _lib.route2_gl_display_poll_quit.argtypes = [ctypes.c_void_p]
    _lib.route2_gl_display_poll_quit.restype = ctypes.c_int
    _lib.route2_gl_display_set_window_size.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.route2_gl_display_set_window_position.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.route2_gl_display_set_window_geometry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.route2_gl_display_get_window_geometry.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    _lib.route2_gl_display_set_window_mapped.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    _lib.route2_gl_display_set_override_redirect.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    _lib.route2_gl_display_set_window_above.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    _lib.route2_gl_display_raise.argtypes = [ctypes.c_void_p]
    _lib.route2_gl_display_get_window_size.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    _lib.route2_gl_display_enable_roi_zoom.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    _lib.route2_gl_display_reset_view.argtypes = [ctypes.c_void_p]
    _lib.route2_gl_display_set_view_crop.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    _lib.route2_gl_display_get_view_crop.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    return _lib


def gl_display_so_path() -> Path | None:
    return _lib_path


def nv12_gl_display_available() -> bool:
    try:
        path = _default_so_path()
        if not path.is_file():
            return False
        return bool(_load().route2_gl_display_available())
    except OSError:
        return False


class Nv12GlDisplay:
    """NVMM fd → EGLImage → GL 全屏预览 + 可选 2D 骨架 overlay。"""

    def __init__(
        self,
        *,
        title: str,
        width: int,
        height: int,
        start_mapped: bool = True,
        override_redirect: bool = False,
        origin_x: int = 0,
        origin_y: int = 0,
        offscreen_stage: bool = False,
    ) -> None:
        lib = _load()
        handle = lib.route2_gl_display_create_ex(
            title.encode("utf-8"),
            int(width),
            int(height),
            1 if start_mapped else 0,
            1 if override_redirect else 0,
            int(origin_x),
            int(origin_y),
        )
        if not handle:
            raise RuntimeError(
                f"route2_gl_display_create_ex 失败（so={_lib_path}，需 DISPLAY 与 X11）"
            )
        self._handle: Any = handle
        self._offscreen_stage = bool(offscreen_stage)
        self._embed_lock = threading.Lock()
        # (x, y, w, h, mode, crop) mode: "live"|"staging"|"hidden"
        self._pending_embed: (
            tuple[int, int, int, int, str, tuple[float, float, float, float] | None] | None
        ) = None
        self._embed_seq = 0
        self._applied_embed_seq = 0
        self._last_applied_geom: tuple[int, int, int, int, str] | None = None
        self._render_ok_count = 0
        self._renders_since_geom = 0
        self._render_lock = threading.Lock()
        self._embed_locked = False
        if self._offscreen_stage:
            # 创建后即处于屏外 staging，首帧 swap 也不会闪到桌面
            self._last_applied_geom = (
                STAGING_ORIGIN_X,
                STAGING_ORIGIN_Y,
                int(width),
                int(height),
                "staging",
            )

    @property
    def applied_embed_seq(self) -> int:
        return self._applied_embed_seq

    @property
    def embed_mapped(self) -> bool:
        last = self._last_applied_geom
        return bool(last and last[4] == "live")

    @property
    def embed_mode(self) -> str:
        last = self._last_applied_geom
        return str(last[4]) if last else "hidden"

    @property
    def render_ok_count(self) -> int:
        with self._render_lock:
            return self._render_ok_count

    @property
    def renders_since_geom(self) -> int:
        with self._render_lock:
            return self._renders_since_geom

    def note_render_ok(self) -> None:
        """UI 线程：成功 render_fd 后计数（后台预热判定）。"""
        with self._render_lock:
            self._render_ok_count += 1
            self._renders_since_geom += 1

    def _reset_renders_since_geom(self) -> None:
        with self._render_lock:
            self._renders_since_geom = 0

    def lock_embed(self) -> None:
        """呈现后锁死几何：忽略后续 live 微调（phase 重对齐前需 unlock）。"""
        with self._embed_lock:
            self._embed_locked = True

    def unlock_embed(self) -> None:
        with self._embed_lock:
            self._embed_locked = False

    @property
    def embed_locked(self) -> bool:
        with self._embed_lock:
            return self._embed_locked

    def request_embed(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        mapped: bool = True,
        crop_uv: tuple[float, float, float, float] | None = None,
    ) -> int:
        """由 Host 线程排队嵌套几何；必须在 UI 线程 apply_pending_embed。"""
        mode = "live" if mapped else "staging"
        with self._embed_lock:
            # 锁定后只允许 staging/hidden；live 重对齐需先 unlock
            if self._embed_locked and mode == "live":
                return self._embed_seq
            self._embed_seq += 1
            self._pending_embed = (
                int(x),
                int(y),
                int(width),
                int(height),
                mode,
                crop_uv,
            )
            return self._embed_seq

    def request_hide(self) -> int:
        with self._embed_lock:
            self._embed_locked = False
            self._embed_seq += 1
            self._pending_embed = (0, 0, 64, 64, "hidden", None)
            return self._embed_seq

    def apply_pending_embed(self) -> bool:
        """UI/GL 线程：staging=屏外 map 预热；live=移入黑框；hidden=卸下。

        仅在几何/模式真正变化时 MoveResize/Raise，避免每帧 X11 抖动卡顿。
        """
        with self._embed_lock:
            pending = self._pending_embed
            if pending is None:
                return False
            self._pending_embed = None
            seq = self._embed_seq
        x, y, w, h, mode, crop = pending
        key = (x, y, w, h, mode)
        prev = self._last_applied_geom
        size_changed = prev is None or prev[2:4] != (w, h)
        mode_changed = prev is None or prev[4] != mode
        pos_changed = prev is None or prev[:2] != (x, y)

        if mode == "live":
            if size_changed or pos_changed or mode_changed:
                self.set_window_geometry(x, y, w, h)
                if crop is not None and (size_changed or mode_changed):
                    self.set_view_crop(*crop)
            if mode_changed:
                self.set_window_mapped(True)
                self.raise_window()
            # 已 live 且仅微调位置：只 MoveResize，不再 Raise（Raise 会拖垮合成器）
        elif mode == "staging":
            need_place = size_changed or mode_changed
            if prev is not None and prev[4] == "staging":
                # 已在 staging：只在尺寸变时改窗（坐标固定屏外）
                need_place = size_changed
            if need_place:
                self.set_window_geometry(STAGING_ORIGIN_X, STAGING_ORIGIN_Y, w, h)
                if crop is not None:
                    self.set_view_crop(*crop)
                if size_changed:
                    self._reset_renders_since_geom()
            if mode_changed:
                self.set_window_mapped(True)
        else:
            if mode_changed or prev is None:
                self.set_window_geometry(STAGING_ORIGIN_X, STAGING_ORIGIN_Y, 64, 64)
                self.set_window_mapped(False)

        self._last_applied_geom = key
        self._applied_embed_seq = seq
        return size_changed or mode_changed or pos_changed

    def set_status(self, text: str) -> None:
        _load().route2_gl_display_set_status(
            self._handle,
            text.encode("utf-8") if text else None,
        )

    def clear_pose(self) -> None:
        _load().route2_gl_display_clear_pose(self._handle)

    def set_sbs_left_eye(self, enabled: bool) -> None:
        """全幅 SBS fd 时在 GL 采样 u∈[0,0.5] 显示左眼。"""
        self.set_sbs_eye(1 if enabled else 0)

    def set_sbs_eye(self, mode: int) -> None:
        """0=全幅，1=左眼，2=右眼。"""
        _load().route2_gl_display_set_sbs_eye(self._handle, int(mode))

    def set_pose(
        self,
        kpts: Sequence[Route2GlKpt],
        *,
        conf_thresh: float,
        video_w: int,
        video_h: int,
    ) -> None:
        if not kpts:
            self.clear_pose()
            return
        arr = (Route2GlKpt * len(kpts))(*kpts)
        _load().route2_gl_display_set_pose(
            self._handle,
            arr,
            len(kpts),
            float(conf_thresh),
            int(video_w),
            int(video_h),
        )

    def render_fd(self, fd: int) -> float:
        """渲染 fd，返回毫秒；失败抛 RuntimeError。"""
        ms = float(_load().route2_gl_display_render_fd(self._handle, int(fd)))
        if ms < 0:
            raise RuntimeError("route2_gl_display_render_fd 失败")
        return ms

    def set_window_size(self, width: int, height: int) -> None:
        """调整 X11 窗口尺寸（支持缩放预览）。"""
        _load().route2_gl_display_set_window_size(
            self._handle,
            int(width),
            int(height),
        )

    def set_window_position(self, x: int, y: int) -> None:
        """移动 X11 窗口（对齐嵌入区域）。"""
        _load().route2_gl_display_set_window_position(
            self._handle,
            int(x),
            int(y),
        )

    def set_window_geometry(self, x: int, y: int, width: int, height: int) -> None:
        """原子 move+resize（嵌入区域）。"""
        _load().route2_gl_display_set_window_geometry(
            self._handle,
            int(x),
            int(y),
            int(width),
            int(height),
        )

    def window_geometry(self) -> tuple[int, int, int, int]:
        x = ctypes.c_int()
        y = ctypes.c_int()
        w = ctypes.c_int()
        h = ctypes.c_int()
        _load().route2_gl_display_get_window_geometry(
            self._handle,
            ctypes.byref(x),
            ctypes.byref(y),
            ctypes.byref(w),
            ctypes.byref(h),
        )
        return int(x.value), int(y.value), int(w.value), int(h.value)

    def set_window_mapped(self, mapped: bool) -> None:
        """显示/隐藏 X11 窗口（不销毁 GL 上下文）。"""
        _load().route2_gl_display_set_window_mapped(
            self._handle,
            1 if mapped else 0,
        )

    def set_override_redirect(self, enabled: bool) -> None:
        """绕过 WM：盖过全屏宿主窗（会先 unmap）。"""
        _load().route2_gl_display_set_override_redirect(
            self._handle,
            1 if enabled else 0,
        )

    def set_window_above(self, above: bool) -> None:
        """置顶 / 取消置顶。"""
        _load().route2_gl_display_set_window_above(
            self._handle,
            1 if above else 0,
        )

    def raise_window(self) -> None:
        _load().route2_gl_display_raise(self._handle)

    def window_size(self) -> tuple[int, int]:
        w = ctypes.c_int()
        h = ctypes.c_int()
        _load().route2_gl_display_get_window_size(
            self._handle,
            ctypes.byref(w),
            ctypes.byref(h),
        )
        return int(w.value), int(h.value)

    def enable_roi_zoom(self, enabled: bool = True) -> None:
        """开启局部 ROI 放大：鼠标框选、+/- 缩放、0/r 复位。"""
        _load().route2_gl_display_enable_roi_zoom(
            self._handle,
            1 if enabled else 0,
        )

    def reset_view(self) -> None:
        _load().route2_gl_display_reset_view(self._handle)

    def set_view_crop(self, u0: float, v0: float, u1: float, v1: float) -> None:
        """设置 UV 裁切（相对当前眼；用于 phone 框内 object-fit: cover）。"""
        _load().route2_gl_display_set_view_crop(
            self._handle,
            float(u0),
            float(v0),
            float(u1),
            float(v1),
        )

    def view_crop(self) -> tuple[float, float, float, float]:
        u0 = ctypes.c_float()
        v0 = ctypes.c_float()
        u1 = ctypes.c_float()
        v1 = ctypes.c_float()
        _load().route2_gl_display_get_view_crop(
            self._handle,
            ctypes.byref(u0),
            ctypes.byref(v0),
            ctypes.byref(u1),
            ctypes.byref(v1),
        )
        return float(u0.value), float(v0.value), float(u1.value), float(v1.value)

    def poll_quit(self) -> bool:
        return bool(_load().route2_gl_display_poll_quit(self._handle))

    def close(self) -> None:
        if self._handle:
            _load().route2_gl_display_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> Nv12GlDisplay:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
