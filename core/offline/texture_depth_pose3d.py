"""clean_2d + VPI 纹理 depth → 左相机 COCO-17 3D（X/Y 来自反投影，Z 来自 stereodisp）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.geometry.calib_model import StereoCalibration
from core.geometry.stereo_rectify import StereoRectifier
from core.geometry.stereo_texture_depth import distorted_pixels_to_rectified
from core.geometry.vpi_stereo_texture_depth import (
    VpiStereoConfig,
    compute_vpi_stereo_texture_depth,
)
from core.offline.offline_preview import OfflinePoseRow, load_offline_pose_index
from core.offline.session_decoder import load_index_rows, split_sbs_bgr

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


@dataclass(frozen=True)
class TextureDepthPose3DStats:
    frames_in: int
    frames_out: int
    frames_skip_decode: int
    joints_valid_mean: float


def _row_conf(kp2d: np.ndarray, i: int) -> float:
    if kp2d.shape[0] <= i or kp2d.shape[1] < 3:
        return 0.0
    return float(kp2d[i, 2])


def build_texture_kpts3d_row(
    pose2d_row: OfflinePoseRow,
    texture_kpts3d: np.ndarray,
    *,
    min_conf: float = 0.25,
) -> dict:
    """OfflinePoseRow + N×3 左相机 XYZ → jsonl 行（COCO-17 + conf）。"""
    kp2d = np.asarray(pose2d_row.det_left["keypoints"], dtype=np.float64)
    n = min(17, int(texture_kpts3d.shape[0]), int(kp2d.shape[0]))
    out = np.full((17, 4), np.nan, dtype=np.float64)
    valid = 0
    for j in range(n):
        conf = _row_conf(kp2d, j)
        xyz = np.asarray(texture_kpts3d[j], dtype=np.float64)
        if conf >= float(min_conf) and np.all(np.isfinite(xyz)) and xyz[2] > 0.05:
            out[j, :3] = xyz
            out[j, 3] = conf
            valid += 1
        elif conf >= float(min_conf):
            out[j, 3] = conf
    return {
        "capture_seq": int(pose2d_row.capture_seq),
        "infer_seq": int(pose2d_row.infer_seq),
        "tri_ok": valid >= 5,
        "kpts3d": out.tolist(),
        "source": "texture_depth_vpi",
    }


def run_texture_depth_pose3d_session(
    *,
    mjpeg_path: Path,
    index_rows: list,
    pose2d_by_seq: dict[int, OfflinePoseRow],
    calib: StereoCalibration,
    rectifier: StereoRectifier,
    vpi_cfg: VpiStereoConfig,
    scale: float = 0.5,
    min_conf: float = 0.25,
) -> tuple[list[dict], TextureDepthPose3DStats]:
    if cv2 is None:
        raise ImportError("需要 opencv-python")

    out_rows: list[dict] = []
    skip_decode = 0
    valid_counts: list[int] = []

    with Path(mjpeg_path).open("rb") as fp:
        for row in index_rows:
            seq = int(row.capture_seq)
            pose2d_row = pose2d_by_seq.get(seq)
            if pose2d_row is None or pose2d_row.det_left is None:
                continue

            fp.seek(int(row.offset))
            data = fp.read(int(row.size))
            sbs = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if sbs is None:
                skip_decode += 1
                continue

            left, right = split_sbs_bgr(sbs)
            kp2d = np.asarray(pose2d_row.det_left["keypoints"], dtype=np.float64)
            if kp2d.shape[0] < 17:
                continue
            uv = kp2d[:17, :2].copy()
            bad = kp2d[:17, 2] < float(min_conf)
            uv[bad] = np.nan
            uv_rect = distorted_pixels_to_rectified(uv, calib)

            result = compute_vpi_stereo_texture_depth(
                left,
                right,
                calib,
                rectifier=rectifier,
                cfg=vpi_cfg,
                scale=float(scale),
                sample_uv_rect=uv_rect,
                full_depth=False,
            )
            if result.sample_kpts3d_left is None:
                continue

            json_row = build_texture_kpts3d_row(
                pose2d_row,
                result.sample_kpts3d_left,
                min_conf=min_conf,
            )
            out_rows.append(json_row)
            valid_counts.append(
                int(np.sum(np.isfinite(np.asarray(json_row["kpts3d"])[:, 2])))
            )

    stats = TextureDepthPose3DStats(
        frames_in=len(index_rows),
        frames_out=len(out_rows),
        frames_skip_decode=skip_decode,
        joints_valid_mean=float(np.mean(valid_counts)) if valid_counts else 0.0,
    )
    return out_rows, stats


def write_texture_depth_pose3d_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_texture_depth_jsonl_with_pose2d(
    texture_path: Path,
    pose2d_path: Path,
    out_path: Path,
) -> int:
    """texture_depth_3d + clean_2d → 含 det_* 的 jsonl（供 Stage 3 filter3d）。"""
    _2d_keys = (
        "det_left_kpts",
        "det_left_kpts_hypo",
        "det_left_box",
        "det_left_box_conf",
        "det_right_kpts",
        "det_right_kpts_hypo",
        "det_right_box",
        "det_right_box_conf",
        "clean_2d",
    )
    raw_pose2d: dict[int, dict] = {}
    with Path(pose2d_path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_pose2d[int(row["capture_seq"])] = row

    out_rows: list[dict] = []
    with Path(texture_path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            tex = json.loads(line)
            seq = int(tex["capture_seq"])
            p2 = raw_pose2d.get(seq)
            if p2 is None:
                continue
            merged = dict(tex)
            for key in _2d_keys:
                if key in p2:
                    merged[key] = p2[key]
            out_rows.append(merged)

    write_texture_depth_pose3d_jsonl(out_rows, out_path)
    return len(out_rows)


def load_texture_depth_pose3d_index(path: Path) -> dict[int, OfflinePoseRow]:
    """texture_depth_3d.jsonl → OfflinePoseRow（供 3D preview）。"""
    out: dict[int, OfflinePoseRow] = {}
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kpts3d = np.asarray(row.get("kpts3d"), dtype=np.float32)
            if kpts3d.size == 0:
                continue
            parsed = OfflinePoseRow(
                capture_seq=int(row["capture_seq"]),
                infer_seq=int(row.get("infer_seq", -1)),
                tri_ok=bool(row.get("tri_ok", False)),
                kpts3d=kpts3d,
                det_left=None,
            )
            out[int(parsed.capture_seq)] = parsed
    return out
