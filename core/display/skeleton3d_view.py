"""纯 3D 骨架渲染（OpenCV，可鼠标拖拽旋转）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

ProjectionMode = Literal["ortho_fit", "perspective", "volume"]

# COCO 17 骨架连线
COCO_SKELETON_17 = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

# 测距用稳定躯干关节（肩/髋/膝），避免随检测点数量变化而缩放
_SPAN_JOINTS = (5, 6, 11, 12, 13, 14)
_DEFAULT_SPAN_M = 0.42
_MIN_Z_M = 0.05
_MAX_Z_M = 4.5
_MIN_CORE_JOINTS = 4
_HOLD_MAX_FRAMES = 45

_last_good_kpts: np.ndarray | None = None
_hold_frames = 0
_render_tag = "none"


@dataclass
class PerspectiveIntrinsics:
    """左相机针孔内参（像素），用于 u=fx·X/Z + cx。"""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def for_canvas(
        cls,
        canvas_w: int,
        canvas_h: int,
        *,
        ref_fx: float = 1151.0,
        ref_fy: float = 1152.0,
        ref_image_height: int = 1080,
    ) -> PerspectiveIntrinsics:
        scale = float(canvas_h) / float(ref_image_height)
        return cls(
            fx=ref_fx * scale,
            fy=ref_fy * scale,
            cx=float(canvas_w) * 0.5,
            cy=float(canvas_h) * 0.5,
        )

    @classmethod
    def from_stereo_calib(
        cls,
        calib: object,
        canvas_w: int,
        canvas_h: int,
    ) -> PerspectiveIntrinsics:
        k1 = np.asarray(getattr(calib, "K1"), dtype=np.float64)
        ref_h = int(getattr(calib, "image_height", 1080) or 1080)
        scale = float(canvas_h) / float(ref_h)
        return cls(
            fx=float(k1[0, 0]) * scale,
            fy=float(k1[1, 1]) * scale,
            cx=float(canvas_w) * 0.5,
            cy=float(canvas_h) * 0.5,
        )


@dataclass(frozen=True)
class Volume3DBox:
    """左相机系固定立方体空间（米）：相机在原点，+X 右 +Y 下 +Z 前。"""

    size_m: float = 5.0
    origin: tuple[float, float, float] = (-2.5, -2.5, 0.0)

    @classmethod
    def cube(cls, size_m: float = 5.0) -> Volume3DBox:
        half = float(size_m) * 0.5
        return cls(size_m=float(size_m), origin=(-half, -half, 0.0))

    @property
    def center(self) -> np.ndarray:
        o = np.asarray(self.origin, dtype=np.float64)
        return o + 0.5 * float(self.size_m)

    def contains(self, pt: np.ndarray) -> bool:
        """点是否在立方体观测范围内（含边界）。"""
        p = np.asarray(pt, dtype=np.float64).reshape(3)
        ox, oy, oz = self.origin
        s = float(self.size_m)
        return bool(
            ox <= p[0] <= ox + s
            and oy <= p[1] <= oy + s
            and oz <= p[2] <= oz + s
        )

    def corners(self) -> np.ndarray:
        ox, oy, oz = self.origin
        s = float(self.size_m)
        return np.array(
            [
                [ox, oy, oz],
                [ox + s, oy, oz],
                [ox, oy + s, oz],
                [ox + s, oy + s, oz],
                [ox, oy, oz + s],
                [ox + s, oy, oz + s],
                [ox, oy + s, oz + s],
                [ox + s, oy + s, oz + s],
            ],
            dtype=np.float64,
        )


DEFAULT_VOLUME_5M = Volume3DBox.cube(5.0)

_VOLUME_WIRE_COLOR = (78, 98, 118)
_VOLUME_GRID_COLOR = (58, 82, 58)
_VOLUME_CAMERA_COLOR = (60, 60, 220)
_VOLUME_AXIS_COLORS = ((80, 80, 220), (80, 220, 80), (220, 80, 80))  # X,Y,Z BGR

_CUBE_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@dataclass
class Skeleton3DRotation:
    """绕骨架中心旋转（弧度）。y_up_locked 时仅 yaw（Y 轴竖直 turntable）。"""

    yaw: float = 0.35
    pitch: float = 0.0
    y_up_locked: bool = True
    _drag: bool = False
    _last_x: int = 0
    _last_y: int = 0

    def nudge(self, dyaw: float = 0.0, dpitch: float = 0.0) -> None:
        self.yaw += dyaw
        if not self.y_up_locked:
            self.pitch += dpitch

    def apply_drag(self, dx: int, dy: int, *, scale: float = 0.008) -> None:
        self.yaw += int(dx) * scale
        if not self.y_up_locked:
            self.pitch += int(dy) * scale


@dataclass
class Skeleton3DViewport:
    """居中慢跟、缩放限速，避免三角化抖动导致骨架忽大忽小。"""

    cx: float = 0.0
    cy: float = 0.0
    span: float = _DEFAULT_SPAN_M
    ready: bool = False
    center_alpha: float = 0.10
    span_alpha: float = 0.05
    max_span_step_ratio: float = 0.03

    def smooth(self, cx: float, cy: float, span_raw: float) -> tuple[float, float, float]:
        span_raw = float(max(span_raw, 0.28))
        if not self.ready:
            self.cx, self.cy, self.span = float(cx), float(cy), span_raw
            self.ready = True
            return self.cx, self.cy, self.span
        self.cx += self.center_alpha * (cx - self.cx)
        self.cy += self.center_alpha * (cy - self.cy)
        # 全身 span 可达 ~2.5m；放大立即跟上，缩小慢跟（防裁切四肢）
        if span_raw > self.span:
            self.span = span_raw
        else:
            target = self.span + self.span_alpha * (span_raw - self.span)
            lo = self.span * (1.0 - self.max_span_step_ratio)
            hi = self.span * (1.0 + self.max_span_step_ratio)
            self.span = float(np.clip(target, lo, hi))
        return self.cx, self.cy, self.span

    def reset(self) -> None:
        self.ready = False
        self.span = _DEFAULT_SPAN_M


_viewport = Skeleton3DViewport()

_GRID_STEP = 40
_GRID_COLOR = (48, 48, 56)
_VIEW_MARGIN = 0.05
_SPAN_PAD = 1.22


@dataclass
class _CanvasBuffers:
    static_bg: np.ndarray
    canvas: np.ndarray
    size: tuple[int, int]
    bg: tuple[int, int, int]


_buffers: _CanvasBuffers | None = None


def reset_canvas_buffers() -> None:
    """分辨率或 teardown 时释放缓存画布。"""
    global _buffers, _last_good_kpts, _hold_frames, _render_tag
    _buffers = None
    _last_good_kpts = None
    _hold_frames = 0
    _render_tag = "none"


def reset_skeleton_render_state(*, reset_viewport: bool = False) -> None:
    """双窗/多 pass 渲染前清 hold 缓存，避免左窗误用右窗骨架。"""
    global _last_good_kpts, _hold_frames, _render_tag
    _last_good_kpts = None
    _hold_frames = 0
    _render_tag = "none"
    if reset_viewport:
        _viewport.reset()


def last_skeleton_render_tag() -> str:
    """最近一次渲染模式：live / hold / none / low。"""
    return _render_tag


def _build_static_bg(
    w: int,
    h: int,
    bg: tuple[int, int, int],
) -> np.ndarray:
    canvas = np.full((h, w, 3), bg, np.uint8)
    for x in range(0, w, _GRID_STEP):
        cv2.line(canvas, (x, 0), (x, h - 1), _GRID_COLOR, 1, cv2.LINE_AA)
    for y in range(0, h, _GRID_STEP):
        cv2.line(canvas, (0, y), (w - 1, y), _GRID_COLOR, 1, cv2.LINE_AA)
    return canvas


def _canvas(w: int, h: int, bg: tuple[int, int, int]) -> np.ndarray:
    """每帧整幅恢复静态底图再画骨架，避免 dirty rect 残影闪烁。"""
    global _buffers
    if (
        _buffers is None
        or _buffers.size != (w, h)
        or _buffers.bg != bg
    ):
        static_bg = _build_static_bg(w, h, bg)
        _buffers = _CanvasBuffers(
            static_bg=static_bg,
            canvas=static_bg.copy(),
            size=(w, h),
            bg=bg,
        )
    np.copyto(_buffers.canvas, _buffers.static_bg)
    return _buffers.canvas


def _rotation_matrix(yaw: float, pitch: float, *, y_up_locked: bool = True) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    ry = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    if y_up_locked:
        return ry
    cp, sp = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    return ry @ rx


def _skeleton_center(
    kpts3d: np.ndarray,
    valid: np.ndarray,
    *,
    prefer_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    k = np.asarray(kpts3d, dtype=np.float64)
    if prefer_indices:
        pts = [k[i, :3] for i in prefer_indices if i < k.shape[0] and valid[i]]
        if pts:
            return np.mean(pts, axis=0)
    hips = []
    if valid[11]:
        hips.append(k[11, :3])
    if valid[12]:
        hips.append(k[12, :3])
    if hips:
        return np.mean(hips, axis=0)
    if valid.any():
        return k[valid, :3].mean(axis=0)
    return np.zeros(3, dtype=np.float64)


def _valid_mask(
    kpts3d: np.ndarray,
    conf: float,
    *,
    ignore_conf: bool = False,
) -> np.ndarray:
    """有效关节 mask；ignore_conf 时仅排除 NaN/全零。"""
    k = np.asarray(kpts3d, dtype=np.float64)
    finite = np.all(np.isfinite(k[:, :3]), axis=1)
    nonzero = np.linalg.norm(k[:, :3], axis=1) > 1e-4
    if ignore_conf:
        return finite & nonzero

    conf_ok = (k[:, 3] >= conf) & finite
    if not conf_ok.any():
        return conf_ok

    in_band = conf_ok & (k[:, 2] >= _MIN_Z_M) & (k[:, 2] <= _MAX_Z_M)
    if in_band.sum() >= _MIN_CORE_JOINTS:
        return in_band

    positive = conf_ok & (k[:, 2] > 0.02)
    if positive.sum() >= _MIN_CORE_JOINTS:
        med_z = float(np.median(k[positive, 2]))
        z_band = max(0.45, med_z * 0.55)
        robust = conf_ok & (k[:, 2] > 0.02) & (np.abs(k[:, 2] - med_z) <= z_band)
        if robust.sum() >= _MIN_CORE_JOINTS:
            return robust

    finite_ok = conf_ok & (np.abs(k[:, 0]) < 2.5) & (np.abs(k[:, 1]) < 2.5)
    finite_ok &= np.abs(k[:, 2]) < 8.0
    return finite_ok


def _body_span_m(uv: np.ndarray, valid: np.ndarray) -> float:
    idx = [i for i in _SPAN_JOINTS if valid[i]]
    if len(idx) < 2:
        idx = [i for i in range(uv.shape[0]) if valid[i]]
    if len(idx) < 2:
        return _viewport.span if _viewport.ready else _DEFAULT_SPAN_M
    pts = uv[idx]
    du = float(pts[:, 0].max() - pts[:, 0].min())
    dv = float(pts[:, 1].max() - pts[:, 1].min())
    return max(du, dv, 0.28)


def _skeleton_fit_uv(
    uv: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float, float] | None:
    """全身有效关节包围盒 → 居中与 span（含 padding，避免裁切四肢）。"""
    idx = [i for i in range(uv.shape[0]) if valid[i]]
    if len(idx) < 2:
        return None
    xs = uv[idx, 0]
    ys = uv[idx, 1]
    du = float(xs.max() - xs.min())
    dv = float(ys.max() - ys.min())
    span_raw = max(du, dv, 0.28) * float(_SPAN_PAD)
    cx_raw = float((xs.max() + xs.min()) * 0.5)
    cy_raw = float((ys.max() + ys.min()) * 0.5)
    return cx_raw, cy_raw, span_raw


def project_skeleton3d(
    kpts3d: np.ndarray,
    *,
    yaw: float,
    pitch: float,
    conf: float,
    y_up_locked: bool = True,
    ignore_conf: bool = False,
    center_indices: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    左相机坐标 (米) → 旋转后正交投影到画布平面 (u, v)，单位米。

    返回 (uv, valid_mask)，uv shape (17, 2)。
    """
    k = np.asarray(kpts3d, dtype=np.float64)
    valid = _valid_mask(k, conf, ignore_conf=ignore_conf)
    center = _skeleton_center(k, valid, prefer_indices=center_indices)
    pts = k[:, :3] - center
    rot = _rotation_matrix(yaw, pitch, y_up_locked=y_up_locked)
    rotated = pts @ rot.T
    uv = np.zeros((k.shape[0], 2), dtype=np.float64)
    uv[:, 0] = rotated[:, 0]
    # 左相机系 Y 向下为正；画布 v 亦向下 → 不取反，头（y 小）在上
    uv[:, 1] = rotated[:, 1]
    return uv, valid


def project_skeleton3d_perspective(
    kpts3d: np.ndarray,
    *,
    yaw: float,
    pitch: float,
    conf: float,
    intrinsics: PerspectiveIntrinsics,
    y_up_locked: bool = True,
    ignore_conf: bool = False,
    center_indices: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    左相机坐标 (米) → 绕中心旋转 → 针孔透视 u=fx·X/Z+cx, v=fy·Y/Z+cy（像素）。

    旋转改变视角；各关节按各自视空间 Z 缩放（策略 B）。
    """
    k = np.asarray(kpts3d, dtype=np.float64)
    valid = _valid_mask(k, conf, ignore_conf=ignore_conf)
    center = _skeleton_center(k, valid, prefer_indices=center_indices)
    # 透视除法用左相机绝对深度 Z；仅 XY 平移到躯干中心。
    pts = k[:, :3].copy()
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    rot = _rotation_matrix(yaw, pitch, y_up_locked=y_up_locked)
    rotated = pts @ rot.T

    z = rotated[:, 2]
    valid = valid & (z > _MIN_Z_M)
    z_safe = np.where(valid, np.maximum(z, _MIN_Z_M), 1.0)
    u = intrinsics.fx * rotated[:, 0] / z_safe + intrinsics.cx
    v = intrinsics.fy * rotated[:, 1] / z_safe + intrinsics.cy
    uv = np.stack([u, v], axis=1)
    return uv, valid


def _volume_pixel_scale(canvas_w: int, canvas_h: int, size_m: float) -> float:
    usable = min(canvas_w, canvas_h) * (1.0 - 2.0 * _VIEW_MARGIN)
    return usable / max(float(size_m), 1e-6)


def _project_volume_points(
    pts3: np.ndarray,
    *,
    volume: Volume3DBox,
    yaw: float,
    pitch: float,
    canvas_w: int,
    canvas_h: int,
    y_up_locked: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """绝对坐标 (米) → 画布像素；绕立方体中心旋转，固定立方体边长映射。"""
    pts = np.asarray(pts3, dtype=np.float64).reshape(-1, 3)
    center = volume.center
    rotated = (pts - center) @ _rotation_matrix(
        yaw, pitch, y_up_locked=y_up_locked
    ).T
    scale = _volume_pixel_scale(canvas_w, canvas_h, volume.size_m)
    u = rotated[:, 0] * scale + canvas_w * 0.5
    v = rotated[:, 1] * scale + canvas_h * 0.5
    uv = np.stack([u, v], axis=1)
    return uv, rotated[:, 2], rotated


def project_volume3d(
    kpts3d: np.ndarray,
    *,
    volume: Volume3DBox,
    yaw: float,
    pitch: float,
    canvas_w: int,
    canvas_h: int,
    conf: float,
    y_up_locked: bool = True,
    ignore_conf: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """骨架左相机坐标 → 固定立方体空间内的画布投影。"""
    k = np.asarray(kpts3d, dtype=np.float64)
    valid = _valid_mask(k, conf, ignore_conf=ignore_conf)
    ox, oy, oz = volume.origin
    s = float(volume.size_m)
    inside = (
        (k[:, 0] >= ox)
        & (k[:, 0] <= ox + s)
        & (k[:, 1] >= oy)
        & (k[:, 1] <= oy + s)
        & (k[:, 2] >= oz)
        & (k[:, 2] <= oz + s)
    )
    valid = valid & inside
    uv, _, _ = _project_volume_points(
        k[:, :3],
        volume=volume,
        yaw=yaw,
        pitch=pitch,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        y_up_locked=y_up_locked,
    )
    return uv, valid


def _draw_volume_line(
    canvas: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_volume_scene(
    canvas: np.ndarray,
    volume: Volume3DBox,
    *,
    rotation: Skeleton3DRotation,
    canvas_w: int,
    canvas_h: int,
) -> None:
    """绘制立方体线框、地面网格、相机原点。"""
    corners = volume.corners()
    uv, depth, _ = _project_volume_points(
        corners,
        volume=volume,
        yaw=rotation.yaw,
        pitch=rotation.pitch,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        y_up_locked=rotation.y_up_locked,
    )
    px_corners = [
        (int(round(uv[i, 0])), int(round(uv[i, 1]))) for i in range(len(corners))
    ]

    edge_order = sorted(
        _CUBE_EDGES,
        key=lambda e: float(depth[e[0]] + depth[e[1]]) * 0.5,
        reverse=True,
    )
    for i, j in edge_order:
        _draw_volume_line(
            canvas, px_corners[i], px_corners[j], _VOLUME_WIRE_COLOR, 1
        )

    ox, oy, oz = volume.origin
    s = float(volume.size_m)
    grid_step = max(1.0, s / 10.0)
    floor_y = oy + s
    z_vals = np.arange(oz, oz + s + 1e-6, grid_step)
    x_vals = np.arange(ox, ox + s + 1e-6, grid_step)
    grid_pts: list[np.ndarray] = []
    for z in z_vals:
        grid_pts.append([ox, floor_y, z])
        grid_pts.append([ox + s, floor_y, z])
    for x in x_vals:
        grid_pts.append([x, floor_y, oz])
        grid_pts.append([x, floor_y, oz + s])
    if grid_pts:
        g = np.asarray(grid_pts, dtype=np.float64)
        guv, _, _ = _project_volume_points(
            g,
            volume=volume,
            yaw=rotation.yaw,
            pitch=rotation.pitch,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            y_up_locked=rotation.y_up_locked,
        )
        for i in range(0, len(guv) - 1, 2):
            p1 = (int(round(guv[i, 0])), int(round(guv[i, 1])))
            p2 = (int(round(guv[i + 1, 0])), int(round(guv[i + 1, 1])))
            _draw_volume_line(canvas, p1, p2, _VOLUME_GRID_COLOR, 1)

    cam = np.zeros((1, 3), dtype=np.float64)
    cuv, _, _ = _project_volume_points(
        cam,
        volume=volume,
        yaw=rotation.yaw,
        pitch=rotation.pitch,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        y_up_locked=rotation.y_up_locked,
    )
    cx, cy = int(round(cuv[0, 0])), int(round(cuv[0, 1]))
    cv2.circle(canvas, (cx, cy), 5, _VOLUME_CAMERA_COLOR, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "cam",
        (cx + 8, cy - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        _VOLUME_CAMERA_COLOR,
        1,
        cv2.LINE_AA,
    )

    axis_len = min(1.2, s * 0.12)
    axis_orig = volume.center
    axis_pts = np.array(
        [
            axis_orig,
            axis_orig + [axis_len, 0.0, 0.0],
            axis_orig,
            axis_orig + [0.0, axis_len, 0.0],
            axis_orig,
            axis_orig + [0.0, 0.0, axis_len],
        ],
        dtype=np.float64,
    )
    auv, _, _ = _project_volume_points(
        axis_pts,
        volume=volume,
        yaw=rotation.yaw,
        pitch=rotation.pitch,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        y_up_locked=rotation.y_up_locked,
    )
    axis_labels = ("X", "Y", "Z")
    for i, label in enumerate(axis_labels):
        p0 = (int(round(auv[i * 2, 0])), int(round(auv[i * 2, 1])))
        p1 = (int(round(auv[i * 2 + 1, 0])), int(round(auv[i * 2 + 1, 1])))
        color = _VOLUME_AXIS_COLORS[i]
        _draw_volume_line(canvas, p0, p1, color, 2)
        cv2.putText(
            canvas,
            label,
            (p1[0] + 4, p1[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    ox_i, oy_i, oz_i = volume.origin
    hint = (
        f"vol {s:.0f}m cube  X[{ox_i:.0f},{ox_i + s:.0f}] "
        f"Y[{oy_i:.0f},{oy_i + s:.0f}] Z[{oz_i:.0f},{oz_i + s:.0f}]"
    )
    cv2.putText(
        canvas,
        hint,
        (10, canvas_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (120, 130, 150),
        1,
        cv2.LINE_AA,
    )


def _proj_label_suffix(
    mode: ProjectionMode,
    volume_box: Volume3DBox | None = None,
) -> str:
    if mode == "perspective":
        return " persp"
    if mode == "volume":
        size_m = float((volume_box or DEFAULT_VOLUME_5M).size_m)
        tag = f"{int(size_m)}" if size_m == int(size_m) else f"{size_m:g}"
        return f" vol{tag}m"
    return ""


def _core_joint_count(valid: np.ndarray) -> int:
    return sum(1 for i in _SPAN_JOINTS if valid[i])


def _text_has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


_CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)

_cjk_font = None
_cjk_font_failed = False


def _get_cjk_font(size: int = 16):
    global _cjk_font, _cjk_font_failed
    if _cjk_font_failed:
        return None
    if _cjk_font is not None:
        return _cjk_font
    try:
        from PIL import ImageFont
    except ImportError:
        _cjk_font_failed = True
        return None
    for path in _CJK_FONT_CANDIDATES:
        try:
            _cjk_font = ImageFont.truetype(path, size=size)
            return _cjk_font
        except OSError:
            continue
    _cjk_font_failed = True
    return None


def _draw_joint_labels(
    canvas: np.ndarray,
    kpts3d: np.ndarray,
    labels: list[str] | tuple[str, ...],
    *,
    w: int,
    h: int,
    conf: float,
    rotation: Skeleton3DRotation,
    ignore_conf: bool,
    label_color: tuple[int, int, int] = (220, 240, 255),
    projection_mode: ProjectionMode = "ortho_fit",
    perspective_intrinsics: PerspectiveIntrinsics | None = None,
    volume_box: Volume3DBox | None = None,
    center_indices: tuple[int, ...] | None = None,
) -> None:
    if not labels:
        return
    if projection_mode == "volume":
        vol = volume_box or DEFAULT_VOLUME_5M
        uv, valid = project_volume3d(
            kpts3d,
            volume=vol,
            yaw=rotation.yaw,
            pitch=rotation.pitch,
            canvas_w=w,
            canvas_h=h,
            conf=conf,
            y_up_locked=rotation.y_up_locked,
            ignore_conf=ignore_conf,
        )
    elif projection_mode == "perspective":
        if perspective_intrinsics is None:
            perspective_intrinsics = PerspectiveIntrinsics.for_canvas(w, h)
        uv, valid = project_skeleton3d_perspective(
            kpts3d,
            yaw=rotation.yaw,
            pitch=rotation.pitch,
            conf=conf,
            intrinsics=perspective_intrinsics,
            y_up_locked=rotation.y_up_locked,
            ignore_conf=ignore_conf,
            center_indices=center_indices,
        )
    else:
        uv, valid = project_skeleton3d(
            kpts3d,
            yaw=rotation.yaw,
            pitch=rotation.pitch,
            conf=conf,
            y_up_locked=rotation.y_up_locked,
            ignore_conf=ignore_conf,
            center_indices=center_indices,
        )

    if projection_mode in ("perspective", "volume"):

        def to_px(i: int) -> tuple[int, int] | None:
            if not valid[i]:
                return None
            return int(uv[i, 0]), int(uv[i, 1])
    else:
        fit = _skeleton_fit_uv(uv, valid)
        if fit is None:
            return
        cx_raw, cy_raw, span_raw = fit
        cx, cy, span = _viewport.smooth(cx_raw, cy_raw, span_raw)
        scale = min(w * (1 - 2 * _VIEW_MARGIN), h * (1 - 2 * _VIEW_MARGIN)) / span

        def to_px(i: int) -> tuple[int, int] | None:
            if not valid[i]:
                return None
            px = int((uv[i, 0] - cx) * scale + w * 0.5)
            py = int((uv[i, 1] - cy) * scale + h * 0.5)
            return px, py

    placements: list[tuple[str, tuple[int, int]]] = []
    for i in range(min(kpts3d.shape[0], len(labels))):
        p = to_px(i)
        if p is None:
            continue
        text = str(labels[i])
        tx = min(max(p[0] + 7, 2), w - 96)
        ty = min(max(p[1] - 16, 2), h - 20)
        placements.append((text, (tx, ty)))

    if not placements:
        return

    use_cjk = any(_text_has_non_ascii(t) for t, _ in placements)
    font = _get_cjk_font(16) if use_cjk else None
    if font is not None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            font = None
        else:
            rgb = (int(label_color[2]), int(label_color[1]), int(label_color[0]))
            img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            for text, (tx, ty) in placements:
                draw.text((tx, ty), text, font=font, fill=rgb)
            canvas[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return

    for text, (tx, ty) in placements:
        cv2.putText(
            canvas,
            text,
            (tx, ty + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            label_color,
            1,
            cv2.LINE_AA,
        )


def _draw_skeleton(
    canvas: np.ndarray,
    kpts3d: np.ndarray,
    *,
    w: int,
    h: int,
    conf: float,
    rotation: Skeleton3DRotation,
    dimmed: bool = False,
    ignore_conf: bool = False,
    skeleton_edges: list[tuple[int, int]] | None = None,
    bone_color: tuple[int, int, int] | None = None,
    joint_color: tuple[int, int, int] | None = None,
    joint_labels: list[str] | tuple[str, ...] | None = None,
    layout: str = "coco17",
    marker_names: list[str] | tuple[str, ...] | None = None,
    center_indices: tuple[int, ...] | None = None,
    exclude_joint_indices: frozenset[int] | None = None,
    projection_mode: ProjectionMode = "ortho_fit",
    perspective_intrinsics: PerspectiveIntrinsics | None = None,
    volume_box: Volume3DBox | None = None,
) -> bool:
    """绘制骨架；返回是否成功画出至少一条骨段。"""
    hidden = exclude_joint_indices or frozenset()
    if projection_mode == "volume":
        vol = volume_box or DEFAULT_VOLUME_5M
        uv, valid = project_volume3d(
            kpts3d,
            volume=vol,
            yaw=rotation.yaw,
            pitch=rotation.pitch,
            canvas_w=w,
            canvas_h=h,
            conf=conf,
            y_up_locked=rotation.y_up_locked,
            ignore_conf=ignore_conf,
        )
    elif projection_mode == "perspective":
        if perspective_intrinsics is None:
            perspective_intrinsics = PerspectiveIntrinsics.for_canvas(w, h)
        uv, valid = project_skeleton3d_perspective(
            kpts3d,
            yaw=rotation.yaw,
            pitch=rotation.pitch,
            conf=conf,
            intrinsics=perspective_intrinsics,
            y_up_locked=rotation.y_up_locked,
            ignore_conf=ignore_conf,
            center_indices=center_indices,
        )
    else:
        uv, valid = project_skeleton3d(
            kpts3d,
            yaw=rotation.yaw,
            pitch=rotation.pitch,
            conf=conf,
            y_up_locked=rotation.y_up_locked,
            ignore_conf=ignore_conf,
            center_indices=center_indices,
        )
    if valid.sum() < 2:
        return False

    if projection_mode in ("perspective", "volume"):

        def to_px(i: int) -> tuple[int, int] | None:
            if not valid[i]:
                return None
            return int(uv[i, 0]), int(uv[i, 1])
    else:
        fit = _skeleton_fit_uv(uv, valid)
        if fit is None:
            return False
        cx_raw, cy_raw, span_raw = fit
        cx, cy, span = _viewport.smooth(cx_raw, cy_raw, span_raw)
        scale = min(w * (1 - 2 * _VIEW_MARGIN), h * (1 - 2 * _VIEW_MARGIN)) / span

        def to_px(i: int) -> tuple[int, int] | None:
            if not valid[i]:
                return None
            px = int((uv[i, 0] - cx) * scale + w * 0.5)
            py = int((uv[i, 1] - cy) * scale + h * 0.5)
            return px, py

    edges = skeleton_edges if skeleton_edges is not None else COCO_SKELETON_17
    bone = bone_color or ((0, 180, 100) if dimmed else (0, 255, 140))
    default_joint = joint_color or ((0, 150, 200) if dimmed else (0, 200, 255))
    drew = False
    for i, j in edges:
        if i in hidden or j in hidden:
            continue
        if i >= kpts3d.shape[0] or j >= kpts3d.shape[0]:
            continue
        p1, p2 = to_px(i), to_px(j)
        if p1 is None or p2 is None:
            continue
        cv2.line(canvas, p1, p2, bone, 2, cv2.LINE_AA)
        drew = True

    from core.display.skeleton3d_layout import joint_style_for_layout

    for i in range(kpts3d.shape[0]):
        if i in hidden:
            continue
        p = to_px(i)
        if p is None:
            continue
        if layout == "opencap_aug" and marker_names is not None and joint_color is None:
            from core.opencap.opencap_skeleton_layout import joint_style_opencap_aug

            radius, color = joint_style_opencap_aug(
                i, list(marker_names), dimmed=dimmed
            )
        elif layout in ("wholebody133", "body_feet", "soma78") and joint_color is None:
            radius, color = joint_style_for_layout(i, layout, dimmed=dimmed)
        else:
            radius, color = 5, default_joint
        cv2.circle(canvas, p, radius, color, -1, cv2.LINE_AA)

    if joint_labels:
        _draw_joint_labels(
            canvas,
            kpts3d,
            joint_labels,
            w=w,
            h=h,
            conf=conf,
            rotation=rotation,
            ignore_conf=ignore_conf,
            label_color=(220, 240, 255) if not dimmed else (140, 160, 180),
            projection_mode=projection_mode,
            perspective_intrinsics=perspective_intrinsics,
            volume_box=volume_box,
            center_indices=center_indices,
        )
    return drew


def render_skeleton3d_bgr(
    kpts3d: np.ndarray | None,
    *,
    width: int,
    height: int,
    conf: float,
    rotation: Skeleton3DRotation,
    bg: tuple[int, int, int] = (28, 28, 32),
    ignore_conf: bool = False,
    skeleton_edges: list[tuple[int, int]] | None = None,
    label: str | None = None,
    bone_color: tuple[int, int, int] | None = None,
    joint_color: tuple[int, int, int] | None = None,
    joint_labels: list[str] | tuple[str, ...] | None = None,
    layout: str = "coco17",
    marker_names: list[str] | tuple[str, ...] | None = None,
    center_indices: tuple[int, ...] | None = None,
    exclude_joint_indices: frozenset[int] | None = None,
    projection_mode: ProjectionMode = "ortho_fit",
    perspective_intrinsics: PerspectiveIntrinsics | None = None,
    volume_box: Volume3DBox | None = None,
) -> np.ndarray:
    """黑底纯骨架 BGR 画布（每帧全幅重绘）。"""
    global _last_good_kpts, _hold_frames, _render_tag
    w, h = max(64, int(width)), max(64, int(height))
    if projection_mode == "volume":
        vol = volume_box or DEFAULT_VOLUME_5M
        canvas = np.full((h, w, 3), bg, np.uint8)
        _draw_volume_scene(
            canvas, vol, rotation=rotation, canvas_w=w, canvas_h=h
        )
    else:
        canvas = _canvas(w, h, bg)

    if kpts3d is None:
        _viewport.reset()
        _last_good_kpts = None
        _hold_frames = 0
        _render_tag = "none"
        cv2.putText(
            canvas,
            "no pose",
            (w // 2 - 50, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (120, 120, 140),
            2,
            cv2.LINE_AA,
        )
        return canvas.copy()

    def _draw(**kwargs: object) -> bool:
        return _draw_skeleton(
            canvas,
            kpts3d,
            w=w,
            h=h,
            conf=conf,
            rotation=rotation,
            ignore_conf=ignore_conf,
            skeleton_edges=skeleton_edges,
            bone_color=bone_color,
            joint_color=joint_color,
            joint_labels=joint_labels,
            layout=layout,
            marker_names=marker_names,
            center_indices=center_indices,
            exclude_joint_indices=exclude_joint_indices,
            projection_mode=projection_mode,
            perspective_intrinsics=perspective_intrinsics,
            volume_box=volume_box,
            **kwargs,
        )

    drew = _draw(dimmed=False)
    if drew:
        _last_good_kpts = np.asarray(kpts3d, dtype=np.float32).copy()
        _hold_frames = 0
        _render_tag = "live"
        if label:
            cv2.putText(
                canvas,
                f"{label}{_proj_label_suffix(projection_mode, volume_box)}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 220, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas.copy()

    if _last_good_kpts is not None and _hold_frames < _HOLD_MAX_FRAMES:
        _hold_frames += 1
        kpts3d = _last_good_kpts
        if _draw(dimmed=True):
            _render_tag = "hold"
            if label:
                cv2.putText(
                    canvas,
                    f"{label}{_proj_label_suffix(projection_mode, volume_box)}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (140, 160, 180),
                    1,
                    cv2.LINE_AA,
                )
            return canvas.copy()

    _render_tag = "low"
    cv2.putText(
        canvas,
        label or "low conf",
        (w // 2 - 55, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (120, 120, 140),
        2,
        cv2.LINE_AA,
    )
    return canvas.copy()


def register_skeleton_mouse(window: str, rotation: Skeleton3DRotation) -> None:
    """在 OpenCV 窗口上注册拖拽旋转。"""

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            rotation._drag = True
            rotation._last_x, rotation._last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            rotation._drag = False
        elif event == cv2.EVENT_MOUSEMOVE and rotation._drag:
            dx = x - rotation._last_x
            rotation.apply_drag(dx, 0)
            rotation._last_x, rotation._last_y = x, y

    cv2.setMouseCallback(window, _on_mouse)
