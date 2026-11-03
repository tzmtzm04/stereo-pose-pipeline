"""离线 session 预览：左眼原画 + 2D 骨架（与实时 live_compose 同色/拓扑）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.display.pose2d_overlay import overlay_pose2d_on_preview
from core.geometry.calib_model import StereoCalibration
from core.geometry.project import project_kpts3d_to_left2d
from core.geometry.stereo_rectify import (
    StereoRectifier,
    remap_det_left_rectified_to_distorted,
)


def fit_preview(preview: np.ndarray, cw: int, ch: int) -> np.ndarray:
    ih, iw = preview.shape[:2]
    if ih == ch and iw == cw:
        return preview
    scale = min(cw / float(iw), ch / float(ih))
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    resized = cv2.resize(preview, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((ch, cw, 3), (24, 24, 28), np.uint8)
    ox = (cw - nw) // 2
    oy = (ch - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


@dataclass(frozen=True)
class OfflinePoseRow:
    capture_seq: int
    infer_seq: int
    tri_ok: bool
    kpts3d: np.ndarray | None
    det_left: dict | None
    det_right: dict | None = None


def _det_from_row_keys(
    row: dict,
    *,
    kpts_key: str,
    box_key: str,
    box_conf_key: str,
) -> dict | None:
    raw = row.get(kpts_key)
    if raw is None:
        return None
    kp = np.asarray(raw, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[1] < 3:
        return None
    det: dict = {"keypoints": kp}
    box = row.get(box_key)
    if box is not None and len(box) >= 4:
        det["box_xyxy"] = tuple(int(v) for v in box[:4])
    det_conf = row.get(box_conf_key)
    if det_conf is not None:
        det["conf"] = float(det_conf)
    return det


def _det_from_row(row: dict) -> dict | None:
    return _det_from_row_keys(
        row,
        kpts_key="det_left_kpts",
        box_key="det_left_box",
        box_conf_key="det_left_box_conf",
    )


def _det_right_from_row(row: dict) -> dict | None:
    return _det_from_row_keys(
        row,
        kpts_key="det_right_kpts",
        box_key="det_right_box",
        box_conf_key="det_right_box_conf",
    )


def _parse_kpts3d(raw: object) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def parse_offline_pose_row(row: dict) -> OfflinePoseRow:
    kpts3d = _parse_kpts3d(row.get("kpts3d"))
    det_left = _det_from_row(row)
    det_right = _det_right_from_row(row)
    return OfflinePoseRow(
        capture_seq=int(row["capture_seq"]),
        infer_seq=int(row.get("infer_seq", -1)),
        tri_ok=bool(row.get("tri_ok", False)),
        kpts3d=kpts3d,
        det_left=det_left,
        det_right=det_right,
    )


def load_offline_pose_index(path: Path) -> dict[int, OfflinePoseRow]:
    """capture_seq → pose 行。"""
    out: dict[int, OfflinePoseRow] = {}
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parsed = parse_offline_pose_row(json.loads(line))
            out[int(parsed.capture_seq)] = parsed
    return out


def load_offline_pose_sequence(
    path: Path,
    *,
    start_capture_seq: int | None = None,
    end_capture_seq: int | None = None,
    stride: int = 1,
) -> list[OfflinePoseRow]:
    """按 capture_seq 升序的 pose 列表（3D preview 用，无需解码视频）。"""
    rows = list(load_offline_pose_index(path).values())
    rows.sort(key=lambda r: int(r.capture_seq))
    if start_capture_seq is not None:
        rows = [r for r in rows if int(r.capture_seq) >= int(start_capture_seq)]
    if end_capture_seq is not None:
        rows = [r for r in rows if int(r.capture_seq) <= int(end_capture_seq)]
    if int(stride) > 1:
        rows = rows[:: int(stride)]
    return rows


def resolve_det_left(
    row: OfflinePoseRow | None,
    calib: StereoCalibration | None,
    *,
    allow_proj3d: bool = True,
) -> tuple[dict | None, str]:
    """返回 (det_left, source)；source: none | det2d | proj3d。"""
    if row is None:
        return None, "none"
    if row.det_left is not None:
        return row.det_left, "det2d"
    if allow_proj3d and calib is not None and row.kpts3d is not None:
        return project_kpts3d_to_left2d(row.kpts3d, calib), "proj3d"
    return None, "none"


def pose_index_has_det2d(poses: dict[int, OfflinePoseRow], *, eye: str = "left") -> bool:
    if eye == "right":
        return any(r.det_right is not None for r in poses.values())
    return any(r.det_left is not None for r in poses.values())


def resolve_det_for_eye(
    row: OfflinePoseRow | None,
    calib: StereoCalibration | None,
    *,
    eye: str = "left",
    allow_proj3d: bool = True,
    prefer_proj3d: bool = False,
    coords_rectified: bool = False,
    rectifier: StereoRectifier | None = None,
) -> tuple[dict | None, str]:
    """返回 (det, source)；source: none | det2d | det2d(distort) | proj3d。"""
    if row is None:
        return None, "none"
    if eye == "right":
        if row.det_right is not None:
            return row.det_right, "det2d"
        return None, "none"
    if prefer_proj3d and allow_proj3d and calib is not None and row.kpts3d is not None:
        return project_kpts3d_to_left2d(row.kpts3d, calib), "proj3d"
    if row.det_left is not None:
        det = row.det_left
        if coords_rectified and rectifier is not None:
            return remap_det_left_rectified_to_distorted(det, rectifier), "det2d(distort)"
        return det, "det2d"
    if allow_proj3d and calib is not None and row.kpts3d is not None:
        return project_kpts3d_to_left2d(row.kpts3d, calib), "proj3d"
    return None, "none"


def resolve_det_left(
    row: OfflinePoseRow | None,
    calib: StereoCalibration | None,
    *,
    allow_proj3d: bool = True,
) -> tuple[dict | None, str]:
    return resolve_det_for_eye(
        row, calib, eye="left", allow_proj3d=allow_proj3d
    )


def compose_offline_eye_frame(
    eye_bgr: np.ndarray,
    row: OfflinePoseRow | None,
    *,
    eye: str = "left",
    calib: StereoCalibration | None = None,
    conf: float = 0.25,
    overlay: bool = True,
    allow_proj3d: bool = True,
    prefer_proj3d: bool = False,
    coords_rectified: bool = False,
    rectifier: StereoRectifier | None = None,
    display_width: int = 1280,
    display_height: int = 720,
) -> tuple[np.ndarray, str]:
    """单眼 BGR + 2D 骨架 → composite；返回 (bgr, overlay_source)。"""
    vis = eye_bgr
    source = "none"
    if overlay:
        det, source = resolve_det_for_eye(
            row,
            calib,
            eye=eye,
            allow_proj3d=allow_proj3d,
            prefer_proj3d=prefer_proj3d,
            coords_rectified=coords_rectified,
            rectifier=rectifier,
        )
        kpts3d = row.kpts3d if row is not None else None
        if det is not None:
            vis = overlay_pose2d_on_preview(
                eye_bgr,
                det,
                kpts3d=kpts3d if source == "det2d" else None,
                conf=conf,
            )
    return fit_preview(vis, int(display_width), int(display_height)), source


def compose_offline_left_eye_frame(
    left_bgr: np.ndarray,
    row: OfflinePoseRow | None,
    *,
    calib: StereoCalibration | None = None,
    conf: float = 0.25,
    overlay: bool = True,
    allow_proj3d: bool = True,
    prefer_proj3d: bool = False,
    coords_rectified: bool = False,
    rectifier: StereoRectifier | None = None,
    display_width: int = 1280,
    display_height: int = 720,
) -> tuple[np.ndarray, str]:
    """左眼 BGR + 2D 骨架 → composite；返回 (bgr, overlay_source)。"""
    return compose_offline_eye_frame(
        left_bgr,
        row,
        eye="left",
        calib=calib,
        conf=conf,
        overlay=overlay,
        allow_proj3d=allow_proj3d,
        prefer_proj3d=prefer_proj3d,
        coords_rectified=coords_rectified,
        rectifier=rectifier,
        display_width=display_width,
        display_height=display_height,
    )


def compose_offline_right_eye_frame(
    right_bgr: np.ndarray,
    row: OfflinePoseRow | None,
    *,
    conf: float = 0.25,
    overlay: bool = True,
    display_width: int = 1280,
    display_height: int = 720,
) -> tuple[np.ndarray, str]:
    """右眼 BGR + 2D 骨架 → composite；返回 (bgr, overlay_source)。"""
    return compose_offline_eye_frame(
        right_bgr,
        row,
        eye="right",
        calib=None,
        conf=conf,
        overlay=overlay,
        allow_proj3d=False,
        display_width=display_width,
        display_height=display_height,
    )


def build_offline_status_line(
    *,
    capture_seq: int,
    infer_seq: int,
    tri_ok: bool,
    frame_idx: int,
    fps: float,
    overlay: bool,
    overlay_source: str = "none",
    eye: str = "left",
) -> str:
    tri_s = "tri_ok" if tri_ok else "tri_fail"
    eye_s = "stereo" if eye == "both" else ("right" if eye == "right" else "left")
    if not overlay:
        ov = "raw"
    elif overlay_source == "det2d":
        ov = "+2D"
    elif "|" in overlay_source:
        ov = f"+2D({overlay_source})"
    elif overlay_source == "proj3d":
        ov = "+2D(proj lag)"
    else:
        ov = "no_pose"
    return (
        f"offline {eye_s} preview n={frame_idx} seq={capture_seq} "
        f"infer={infer_seq} {tri_s} fps={fps:.1f} {ov} [q=quit space=pause]"
    )


def show_offline_frame(window: str, bgr: np.ndarray, status: str) -> bool:
    """显示一帧；返回 True 表示用户请求退出。"""
    h, w = bgr.shape[:2]
    bar_h = 28
    canvas = np.full((h + bar_h, w, 3), (24, 24, 28), np.uint8)
    canvas[bar_h:, :] = bgr
    cv2.putText(
        canvas,
        status,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(window, canvas)
    return False
