"""YOLO26-pose V26 后处理。"""

from __future__ import annotations

import numpy as np

NUM_KEYPOINTS = 17
FEATURES_PER_KEYPOINT = 3


def _resolve_geometry(
    original_size: tuple[int, int],
    letterbox_size: tuple[int, int],
    letterbox_geometry=None,
):
    from platform.jetson.letterbox_util import (
        LetterboxGeometry,
        ultralytics_letterbox_geometry,
    )

    if letterbox_geometry is not None:
        geo: LetterboxGeometry = letterbox_geometry
        return geo.src_w, geo.src_h, geo

    ow, oh = int(original_size[0]), int(original_size[1])
    lw, lh = int(letterbox_size[0]), int(letterbox_size[1])
    geo = ultralytics_letterbox_geometry(ow, oh, lw, lh)
    return ow, oh, geo


def postprocess_v26(
    raw_output: np.ndarray,
    original_size: tuple[int, int],
    letterbox_size: tuple[int, int] = (640, 640),
    conf_threshold: float = 0.25,
    letterbox_geometry=None,
    orig_map_mode: str = "float_pad",
) -> list[dict]:
    from platform.jetson.letterbox_util import letterbox_tensor_to_orig

    out = np.asarray(raw_output, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"期望 [N, F] 输出，收到 shape {out.shape}")

    ow, oh, geo = _resolve_geometry(
        original_size, letterbox_size, letterbox_geometry
    )

    results: list[dict] = []
    for d in range(out.shape[0]):
        row = out[d]
        x1, y1, x2, y2, conf = row[0], row[1], row[2], row[3], row[4]
        if conf < conf_threshold:
            continue
        ox1, oy1 = letterbox_tensor_to_orig(x1, y1, geo, orig_map_mode=orig_map_mode)
        ox2, oy2 = letterbox_tensor_to_orig(x2, y2, geo, orig_map_mode=orig_map_mode)
        bx1 = int(np.clip(min(ox1, ox2), 0, ow - 1))
        by1 = int(np.clip(min(oy1, oy2), 0, oh - 1))
        bx2 = int(np.clip(max(ox1, ox2), 0, ow - 1))
        by2 = int(np.clip(max(oy1, oy2), 0, oh - 1))
        kpts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        for k in range(NUM_KEYPOINTS):
            kb = 6 + k * FEATURES_PER_KEYPOINT
            kx, ky = letterbox_tensor_to_orig(
                row[kb + 0], row[kb + 1], geo, orig_map_mode=orig_map_mode
            )
            kpts[k, 0] = np.clip(kx, 0, ow - 1)
            kpts[k, 1] = np.clip(ky, 0, oh - 1)
            kpts[k, 2] = row[kb + 2]
        results.append(
            {
                "box_xyxy": (bx1, by1, bx2, by2),
                "conf": float(conf),
                "keypoints": kpts,
            }
        )
    return results
