"""MMPose 官方 tracking + One Euro（YOLO/ViTPose det 之后的 2D 时序滤波）。

镜像 ``mmpose/apis/inference_tracking.py`` 与 ``mmpose/core/post_processing/one_euro_filter.py``，
不依赖 mmpose 包。det 仍由 YOLO 提供，此处只做 IoU track_id 匹配与 One Euro 平滑。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


def _compute_iou(bbox_a: list | np.ndarray, bbox_b: list | np.ndarray) -> float:
    a = list(bbox_a)[:4]
    b = list(bbox_b)[:4]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = float(area_a + area_b - inter)
    if union <= 0.0:
        union = 1e-5
        warnings.warn("union_area=0 is unexpected", stacklevel=2)
    return inter / union


def _smoothing_factor(t_e: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(a: np.ndarray, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
    return a * x + (1 - a) * x_prev


class MMPoseOneEuroFilter:
    """MMPose 向量 One Euro：对 ``[K, 2]`` 整帧关键点一次滤波。"""

    def __init__(
        self,
        x0: np.ndarray,
        *,
        min_cutoff: float = 1.7,
        beta: float = 0.3,
        d_cutoff: float = 30.0,
        fps: float | None = None,
    ) -> None:
        x0 = np.asarray(x0, dtype=np.float32)
        self.data_shape = x0.shape
        self.min_cutoff = np.full(x0.shape, float(min_cutoff), dtype=np.float32)
        self.beta = np.full(x0.shape, float(beta), dtype=np.float32)
        self.d_cutoff = np.full(x0.shape, float(d_cutoff), dtype=np.float32)
        self.x_prev = x0.astype(np.float32)
        self.dx_prev = np.zeros(x0.shape, dtype=np.float32)
        self.realtime = fps is None
        if fps is None:
            self.t_e = None
            self.skip_frame_factor = float(d_cutoff)
        else:
            self.d_cutoff = np.full(x0.shape, float(fps), dtype=np.float32)
        self.t_prev = 0.0

    def __call__(self, x: np.ndarray, t_e: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape != self.data_shape:
            raise ValueError(f"期望 shape {self.data_shape}，收到 {x.shape}")
        t_e_arr = np.full(x.shape, float(t_e), dtype=np.float32)
        mask = np.ma.masked_where(x <= 0, x)
        a_d = _smoothing_factor(t_e_arr, self.d_cutoff)
        dx = (x - self.x_prev) / np.maximum(t_e_arr, 1e-5)
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _smoothing_factor(t_e_arr, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)
        np.copyto(x_hat, -10.0, where=mask.mask)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def _track_by_iou(
    res: dict,
    results_last: list[dict],
    thr: float,
) -> tuple[int, list[dict], dict]:
    bbox = list(res["bbox"])
    max_iou = -1.0
    max_index = -1
    match_result: dict = {}
    for index, res_last in enumerate(results_last):
        iou = _compute_iou(bbox, list(res_last["bbox"]))
        if iou > max_iou:
            max_iou = iou
            max_index = index
    if max_iou > thr and max_index >= 0:
        track_id = int(results_last[max_index]["track_id"])
        match_result = results_last[max_index]
        del results_last[max_index]
    else:
        track_id = -1
    return track_id, results_last, match_result


def _get_area(results: list[dict]) -> list[dict]:
    for result in results:
        if "bbox" in result:
            bb = result["bbox"]
            result["area"] = float((bb[2] - bb[0]) * (bb[3] - bb[1]))
        else:
            kpts = np.asarray(result["keypoints"])
            xs = kpts[:, 0][kpts[:, 0] > 0]
            ys = kpts[:, 1][kpts[:, 1] > 0]
            xmin = float(np.min(xs)) if xs.size else 1e10
            xmax = float(np.max(kpts[:, 0]))
            ymin = float(np.min(ys)) if ys.size else 1e10
            ymax = float(np.max(kpts[:, 1]))
            result["area"] = (xmax - xmin) * (ymax - ymin)
            result["bbox"] = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
    return results


def _temporal_refine(result: dict, match_result: dict, *, fps: float | None) -> np.ndarray:
    xy = np.asarray(result["keypoints"][:, :2], dtype=np.float32)
    if "one_euro" in match_result:
        xy_f = match_result["one_euro"](xy)
        result["one_euro"] = match_result["one_euro"]
    else:
        filt = MMPoseOneEuroFilter(xy, fps=fps)
        xy_f = filt(xy)
        result["one_euro"] = filt
    out = np.asarray(result["keypoints"], dtype=np.float32).copy()
    out[:, :2] = xy_f
    return out


def get_track_id(
    results: list[dict],
    results_last: list[dict],
    next_id: int,
    *,
    min_keypoints: int = 3,
    tracking_thr: float = 0.3,
    use_one_euro: bool = False,
    fps: float | None = None,
) -> tuple[list[dict], int]:
    """MMPose 官方 IoU tracking + 可选 One Euro。"""
    results = _get_area(list(results))
    for result in results:
        track_id, results_last, match_result = _track_by_iou(
            result, results_last, tracking_thr
        )
        if track_id == -1:
            vis_y = np.count_nonzero(result["keypoints"][:, 1] > 0)
            if vis_y > min_keypoints:
                result["track_id"] = next_id
                next_id += 1
            else:
                result["keypoints"][:, 1] = -10
                result["bbox"] *= 0
                result["track_id"] = -1
        else:
            result["track_id"] = track_id
        if use_one_euro:
            result["keypoints"] = _temporal_refine(result, match_result, fps=fps)
    return results, next_id


def _det_to_track_result(det: dict) -> dict:
    box = det["box_xyxy"]
    kpts = np.asarray(det["keypoints"], dtype=np.float32).copy()
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return {
        "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
        "keypoints": kpts,
    }


def _track_result_to_det(res: dict, orig: dict) -> dict | None:
    if int(res.get("track_id", -1)) < 0:
        return None
    kpts_f = np.asarray(res["keypoints"], dtype=np.float32)
    orig_k = np.asarray(orig["keypoints"], dtype=np.float32)
    out_k = orig_k.copy()
    out_k[:, :2] = kpts_f[:, :2]
    bad = out_k[:, 1] < -5
    if np.any(bad):
        out_k[bad, 2] = 0.0
    out = dict(orig)
    out["keypoints"] = out_k
    bb = res.get("bbox")
    if bb is not None:
        out["box_xyxy"] = (
            int(round(float(bb[0]))),
            int(round(float(bb[1]))),
            int(round(float(bb[2]))),
            int(round(float(bb[3]))),
        )
    out["track_id"] = int(res["track_id"])
    return out


@dataclass(frozen=True)
class MMPoseTracking2DConfig:
    tracking_thr: float = 0.3
    min_keypoints: int = 3
    min_cutoff: float = 1.7
    beta: float = 0.3
    fps: float = 60.0
    use_one_euro: bool = True


class EyeMMPoseTracking2DFilter:
    """单目 det dict → IoU tracking + One Euro。"""

    def __init__(self, cfg: MMPoseTracking2DConfig) -> None:
        self._cfg = cfg
        self._results_last: list[dict] = []
        self._next_id = 0

    def reset(self) -> None:
        self._results_last = []
        self._next_id = 0

    def filter(self, det: dict | None) -> dict | None:
        if det is None:
            return None
        results = [_det_to_track_result(det)]
        results, self._next_id = get_track_id(
            results,
            list(self._results_last),
            self._next_id,
            min_keypoints=int(self._cfg.min_keypoints),
            tracking_thr=float(self._cfg.tracking_thr),
            use_one_euro=bool(self._cfg.use_one_euro),
            fps=float(self._cfg.fps) if self._cfg.use_one_euro else None,
        )
        self._results_last = results
        if not results:
            return None
        return _track_result_to_det(results[0], det)


class StereoMMPoseTracking2DFilter:
    """左右眼独立 IoU tracking + One Euro（YOLO 框 + ViTPose 点）。"""

    FILTER_TAG = "yolo+mmpose_tracking_one_euro"

    def __init__(self, cfg: MMPoseTracking2DConfig) -> None:
        self._left = EyeMMPoseTracking2DFilter(cfg)
        self._right = EyeMMPoseTracking2DFilter(cfg)

    def reset(self) -> None:
        self._left.reset()
        self._right.reset()

    def filter_pair(
        self,
        det_left: dict | None,
        det_right: dict | None,
    ) -> tuple[dict | None, dict | None]:
        return self._left.filter(det_left), self._right.filter(det_right)
