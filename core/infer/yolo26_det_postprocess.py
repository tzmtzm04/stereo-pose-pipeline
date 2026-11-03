"""YOLO26 detect V26 后处理（person class only）。"""

from __future__ import annotations

import numpy as np

COCO_PERSON_CLASS = 0


def postprocess_v26_detect(
    raw_output: np.ndarray,
    original_size: tuple[int, int],
    letterbox_size: tuple[int, int] = (640, 640),
    conf_threshold: float = 0.25,
    letterbox_geometry=None,
    *,
    person_class: int = COCO_PERSON_CLASS,
) -> list[dict]:
    from core.infer.postprocess import _resolve_geometry
    from platform.jetson.letterbox_util import letterbox_tensor_to_orig

    out = np.asarray(raw_output, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2 or out.shape[1] < 6:
        raise ValueError(f"期望 detect [N, 6+] 输出，收到 shape {out.shape}")

    ow, oh, geo = _resolve_geometry(
        original_size, letterbox_size, letterbox_geometry
    )

    results: list[dict] = []
    for row in out:
        x1, y1, x2, y2, conf, cls_id = (
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
        )
        if int(round(cls_id)) != int(person_class):
            continue
        if float(conf) < conf_threshold:
            continue
        ox1, oy1 = letterbox_tensor_to_orig(x1, y1, geo)
        ox2, oy2 = letterbox_tensor_to_orig(x2, y2, geo)
        bx1 = int(np.clip(min(ox1, ox2), 0, ow - 1))
        by1 = int(np.clip(min(oy1, oy2), 0, oh - 1))
        bx2 = int(np.clip(max(ox1, ox2), 0, ow - 1))
        by2 = int(np.clip(max(oy1, oy2), 0, oh - 1))
        if bx2 <= bx1 or by2 <= by1:
            continue
        results.append(
            {
                "box_xyxy": (bx1, by1, bx2, by2),
                "conf": float(conf),
            }
        )
    return results


def best_person_bbox(
    detections: list[dict],
) -> tuple[tuple[int, int, int, int], float] | None:
    if not detections:
        return None
    best = max(detections, key=lambda d: float(d.get("conf", 0.0)))
    box = best["box_xyxy"]
    return (int(box[0]), int(box[1]), int(box[2]), int(box[3])), float(best["conf"])
