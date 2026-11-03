"""左眼 preview 上叠 2D COCO-17 骨架（与 postprocess 原图坐标一致）。"""

from __future__ import annotations

import cv2
import numpy as np

from core.display.skeleton3d_view import COCO_SKELETON_17


def _kpt_visible(kpts: np.ndarray, idx: int, *, conf: float) -> bool:
    if idx < 0 or idx >= kpts.shape[0]:
        return False
    if float(kpts[idx, 2]) < conf:
        return False
    return bool(np.isfinite(kpts[idx, 0]) and np.isfinite(kpts[idx, 1]))


def _draw_det_bbox(bgr: np.ndarray, det: dict, *, conf: float = 0.25) -> None:
    box = det.get("box_xyxy")
    if box is None or len(box) < 4:
        return
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 128, 0), 2, cv2.LINE_AA)
    det_conf = det.get("conf")
    if det_conf is not None and float(det_conf) >= conf:
        label = f"person {float(det_conf):.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        ty = max(th + 4, y1 - 4)
        cv2.rectangle(
            bgr,
            (x1, ty - th - 4),
            (x1 + tw + 4, ty + baseline),
            (255, 128, 0),
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            label,
            (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (16, 16, 16),
            1,
            cv2.LINE_AA,
        )


def draw_pose2d_overlay(
    bgr: np.ndarray,
    det: dict | None,
    *,
    kpts3d: np.ndarray | None = None,
    conf: float = 0.25,
    draw_bbox: bool = True,
) -> None:
    """in-place 绘制；调用方应先 copy preview。"""
    if det is None:
        return
    if draw_bbox:
        _draw_det_bbox(bgr, det, conf=conf)
    kpts = det.get("keypoints")
    if kpts is None or len(kpts) == 0:
        return
    kpts = np.asarray(kpts, dtype=np.float32)

    for i, j in COCO_SKELETON_17:
        if not _kpt_visible(kpts, i, conf=conf) or not _kpt_visible(kpts, j, conf=conf):
            continue
        p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
        p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
        cv2.line(bgr, p1, p2, (0, 255, 128), 2, cv2.LINE_AA)

    for ki in range(kpts.shape[0]):
        if not _kpt_visible(kpts, ki, conf=conf):
            continue
        x, y = int(kpts[ki, 0]), int(kpts[ki, 1])
        if (
            kpts3d is not None
            and ki < kpts3d.shape[0]
            and kpts3d[ki, 3] >= conf
            and kpts3d[ki, 2] > 0
        ):
            z_cm = int(kpts3d[ki, 2] * 100)
            cv2.circle(bgr, (x, y), 4, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(
                bgr,
                f"{z_cm}",
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(bgr, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)


def transform_det_keypoints(
    det: dict | None,
    *,
    scale: float = 1.0,
    ox: float = 0.0,
    oy: float = 0.0,
) -> dict | None:
    """将 det 关键点与 bbox 从原图坐标映射到画布坐标（scale + letterbox offset）。"""
    if det is None:
        return None
    out = dict(det)
    kp = det.get("keypoints")
    if kp is not None:
        scaled = np.asarray(kp, dtype=np.float32).copy()
        scaled[:, 0] = scaled[:, 0] * float(scale) + float(ox)
        scaled[:, 1] = scaled[:, 1] * float(scale) + float(oy)
        out["keypoints"] = scaled
    box = det.get("box_xyxy")
    if box is not None and len(box) >= 4:
        x1, y1, x2, y2 = box[:4]
        out["box_xyxy"] = (
            int(x1 * float(scale) + float(ox)),
            int(y1 * float(scale) + float(oy)),
            int(x2 * float(scale) + float(ox)),
            int(y2 * float(scale) + float(oy)),
        )
    return out


def overlay_pose2d_on_preview(
    preview_bgr: np.ndarray,
    det: dict | None,
    *,
    kpts3d: np.ndarray | None = None,
    conf: float = 0.25,
    draw_bbox: bool = True,
) -> np.ndarray:
    """copy + 叠 YOLO 框与骨架，返回可显示的 BGR。"""
    vis = preview_bgr.copy()
    draw_pose2d_overlay(
        vis, det, kpts3d=kpts3d, conf=conf, draw_bbox=draw_bbox
    )
    return vis
