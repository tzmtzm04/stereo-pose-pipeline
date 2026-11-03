"""把已有 X11 窗（如 OpenCV Qt）改成 override_redirect 并贴到指定矩形。

全屏宿主窗可能压住普通 TOPMOST；嵌入 GL 预览用 override_redirect。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
from functools import lru_cache

CWOverrideRedirect = 1 << 9

# title → 已做过 override_redirect 的 wid
_prepared: dict[str, int] = {}
_prep_lock = threading.Lock()


class XSetWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("background_pixmap", ctypes.c_ulong),
        ("background_pixel", ctypes.c_ulong),
        ("border_pixmap", ctypes.c_ulong),
        ("border_pixel", ctypes.c_ulong),
        ("bit_gravity", ctypes.c_int),
        ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int),
        ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong),
        ("save_under", ctypes.c_int),
        ("event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int),
        ("colormap", ctypes.c_ulong),
        ("cursor", ctypes.c_ulong),
    ]


@lru_cache(maxsize=1)
def _libx11():
    path = ctypes.util.find_library("X11")
    if not path:
        raise RuntimeError("libX11 not found")
    lib = ctypes.CDLL(path)
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    lib.XDefaultRootWindow.restype = ctypes.c_ulong
    lib.XQueryTree.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.XQueryTree.restype = ctypes.c_int
    lib.XFetchName.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.XFetchName.restype = ctypes.c_int
    lib.XFree.argtypes = [ctypes.c_void_p]
    lib.XFree.restype = None
    lib.XUnmapWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    lib.XMapRaised.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    lib.XMoveResizeWindow.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
    ]
    lib.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    lib.XReparentWindow.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.XChangeWindowAttributes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.POINTER(XSetWindowAttributes),
    ]
    lib.XFlush.argtypes = [ctypes.c_void_p]
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    lib.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.XGetGeometry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.XGetGeometry.restype = ctypes.c_int
    lib.XSetWindowBorderWidth.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_uint,
    ]
    return lib


def _walk_find_by_name(lib, dpy, root: int, title: str, out: list[int], depth: int = 0) -> None:
    if depth > 10:
        return
    name_p = ctypes.c_char_p()
    if lib.XFetchName(dpy, root, ctypes.byref(name_p)) and name_p.value:
        try:
            if name_p.value.decode("utf-8", "replace") == title:
                out.append(int(root))
        finally:
            lib.XFree(name_p)
    root_ret = ctypes.c_ulong()
    parent = ctypes.c_ulong()
    children = ctypes.POINTER(ctypes.c_ulong)()
    n = ctypes.c_uint()
    if not lib.XQueryTree(
        dpy,
        root,
        ctypes.byref(root_ret),
        ctypes.byref(parent),
        ctypes.byref(children),
        ctypes.byref(n),
    ):
        return
    try:
        for i in range(int(n.value)):
            _walk_find_by_name(lib, dpy, int(children[i]), title, out, depth + 1)
    finally:
        if children:
            lib.XFree(children)


def _toplevel_under_root(lib, dpy, wid: int, root: int) -> int:
    """爬到紧贴 root 的祖先（WM frame 或已是 override 窗）。"""
    cur = int(wid)
    for _ in range(16):
        root_ret = ctypes.c_ulong()
        parent = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        n = ctypes.c_uint()
        if not lib.XQueryTree(
            dpy,
            cur,
            ctypes.byref(root_ret),
            ctypes.byref(parent),
            ctypes.byref(children),
            ctypes.byref(n),
        ):
            break
        if children:
            lib.XFree(children)
        p = int(parent.value)
        if p == 0 or p == root:
            return cur
        cur = p
    return int(wid)


def find_window_ids_by_name(title: str) -> list[int]:
    lib = _libx11()
    disp_name = os.environ.get("DISPLAY", ":0").encode("utf-8")
    dpy = lib.XOpenDisplay(disp_name)
    if not dpy:
        return []
    try:
        root = int(lib.XDefaultRootWindow(dpy))
        found: list[int] = []
        _walk_find_by_name(lib, dpy, root, title, found)
        return found
    finally:
        lib.XCloseDisplay(dpy)


def clear_prepared(title: str | None = None) -> None:
    with _prep_lock:
        if title is None:
            _prepared.clear()
        else:
            _prepared.pop(title, None)


def place_override_redirect(
    title: str,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    mapped: bool = True,
) -> bool:
    """按 WM_NAME 找到窗，reparent+override_redirect 并 MoveResize。"""
    lib = _libx11()
    disp_name = os.environ.get("DISPLAY", ":0").encode("utf-8")
    dpy = lib.XOpenDisplay(disp_name)
    if not dpy:
        return False
    try:
        root = int(lib.XDefaultRootWindow(dpy))
        found: list[int] = []
        _walk_find_by_name(lib, dpy, root, title, found)
        if not found:
            return False

        # 必须用带 WM_NAME 的 OpenCV/Qt 客户窗本身。
        # 若爬到 WM frame 再 reparent，画面会停在第一帧（后续 imshow 打到客户区）。
        with _prep_lock:
            prepared = _prepared.get(title)

        if prepared is not None and prepared in found:
            wid = prepared
        else:
            best = found[0]
            best_area = 0
            for cand in found:
                root_r = ctypes.c_ulong()
                gx = ctypes.c_int()
                gy = ctypes.c_int()
                gw = ctypes.c_uint()
                gh = ctypes.c_uint()
                bw = ctypes.c_uint()
                depth = ctypes.c_uint()
                if lib.XGetGeometry(
                    dpy,
                    cand,
                    ctypes.byref(root_r),
                    ctypes.byref(gx),
                    ctypes.byref(gy),
                    ctypes.byref(gw),
                    ctypes.byref(gh),
                    ctypes.byref(bw),
                    ctypes.byref(depth),
                ):
                    area = int(gw.value) * int(gh.value)
                    if area >= best_area:
                        best_area = area
                        best = cand
            wid = best

        w = max(64, int(width))
        h = max(64, int(height))
        need_prep = prepared != wid
        if need_prep:
            # 脱离 WM：unmap → override_redirect → reparent 到 root
            lib.XUnmapWindow(dpy, wid)
            lib.XSync(dpy, 0)
            attrs = XSetWindowAttributes()
            attrs.override_redirect = 1
            lib.XChangeWindowAttributes(
                dpy, wid, CWOverrideRedirect, ctypes.byref(attrs)
            )
            lib.XSetWindowBorderWidth(dpy, wid, 0)
            lib.XReparentWindow(dpy, wid, root, int(x), int(y))
            with _prep_lock:
                _prepared[title] = wid

        lib.XSetWindowBorderWidth(dpy, wid, 0)
        lib.XMoveResizeWindow(dpy, wid, int(x), int(y), w, h)
        if mapped:
            lib.XMapRaised(dpy, wid)
            lib.XRaiseWindow(dpy, wid)
        else:
            lib.XUnmapWindow(dpy, wid)
        lib.XSync(dpy, 0)
        lib.XFlush(dpy)
        return True
    finally:
        lib.XCloseDisplay(dpy)
