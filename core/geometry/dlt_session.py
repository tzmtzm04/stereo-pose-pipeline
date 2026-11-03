"""Session 2D → 逐帧 OpenCV DLT 三角化（无 optim / 时序平滑 / 骨长约束）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.geometry.calib_loader import load_stereo_calibration
from core.geometry.triangulate import triangulate_matched_keypoints


@dataclass
class DltTriStats:
    frames_in: int = 0
    frames_out: int = 0
    tri_ok_count: int = 0


def _load_pose2d_rows(
    path: Path,
    *,
    start_seq: int | None,
    end_seq: int | None,
    max_frames: int,
) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seq = int(row.get("capture_seq", -1))
            if start_seq is not None and seq < int(start_seq):
                continue
            if end_seq is not None and seq > int(end_seq):
                continue
            rows.append(row)
            if max_frames > 0 and len(rows) >= int(max_frames):
                break
    return rows


def run_session_dlt_triangulation(
    *,
    pose2d_path: Path,
    calib_path: Path,
    out_path: Path,
    min_conf: float = 0.0,
    stereo_rectified: bool = False,
    start_seq: int | None = None,
    end_seq: int | None = None,
    max_frames: int = 0,
) -> DltTriStats:
    """逐帧 cv2.triangulatePoints；帧间无耦合。"""
    pose2d_path = Path(pose2d_path).expanduser().resolve()
    calib_path = Path(calib_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    calib = load_stereo_calibration(calib_path)
    rows = _load_pose2d_rows(
        pose2d_path,
        start_seq=start_seq,
        end_seq=end_seq,
        max_frames=int(max_frames),
    )
    stats = DltTriStats(frames_in=len(rows))
    tri_ok_count = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for row in rows:
            rec = dict(row)
            kl_raw = row.get("det_left_kpts")
            kr_raw = row.get("det_right_kpts")
            kpts3d = None
            tri_ok = False
            if kl_raw is not None and kr_raw is not None:
                kl = np.asarray(kl_raw, dtype=np.float64)
                kr = np.asarray(kr_raw, dtype=np.float64)
                if kl.ndim == 2 and kr.ndim == 2 and kl.shape == kr.shape:
                    kpts3d = triangulate_matched_keypoints(
                        kl[:, :2],
                        kr[:, :2],
                        calib,
                        min_conf=float(min_conf),
                        conf_left=kl[:, 2],
                        conf_right=kr[:, 2],
                        mode="opencv",
                        stereo_rectified=bool(stereo_rectified),
                    )
                    tri_ok = bool(
                        kpts3d is not None
                        and float(np.asarray(kpts3d)[:, 3].sum()) > 0
                    )
            if kpts3d is not None:
                rec["kpts3d"] = np.asarray(kpts3d, dtype=np.float32).tolist()
            else:
                rec.pop("kpts3d", None)
            rec["tri_ok"] = tri_ok
            rec["tri_mode"] = "opencv_dlt"
            rec["anipose_optim"] = False
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if tri_ok:
                tri_ok_count += 1

    stats.frames_out = len(rows)
    stats.tri_ok_count = tri_ok_count
    return stats
