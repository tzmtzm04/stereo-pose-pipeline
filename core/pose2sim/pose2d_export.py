"""full_session_pose2d.jsonl → Pose2Sim pose/cam*_json（OpenPose COCO-17）。"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.pose2sim.frame_range import load_jsonl_rows

_COCO17_N = 17
_FLAT_LEN = _COCO17_N * 3


@dataclass(frozen=True)
class Pose2DExportStats:
    frames_in: int = 0
    frames_written: int = 0
    pose_dir: Path | None = None


def _parse_kpts17(raw: object) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < _COCO17_N or arr.shape[1] < 3:
        return None
    return arr[:_COCO17_N, :3]


def kpts_to_openpose_flat(kpts: np.ndarray) -> list[float]:
    """COCO-17 (N×3) → OpenPose pose_keypoints_2d（按 keypoint id 索引）。"""
    flat = [0.0] * _FLAT_LEN
    n = min(int(kpts.shape[0]), _COCO17_N)
    for ji in range(n):
        x, y, c = float(kpts[ji, 0]), float(kpts[ji, 1]), float(kpts[ji, 2])
        if not np.isfinite(x):
            x = 0.0
        if not np.isfinite(y):
            y = 0.0
        if not np.isfinite(c):
            c = 0.0
        base = ji * 3
        flat[base] = x
        flat[base + 1] = y
        flat[base + 2] = c
    return flat


def export_pose2d_json(
    rows: list[dict],
    pose_dir: Path,
    *,
    cam_left: str = "cam01_json",
    cam_right: str = "cam02_json",
) -> tuple[int, list[int]]:
    """
    写 Pose2Sim 2D json（单帧单文件，帧号 0..N-1）。

    返回 (写入帧数, 每帧 capture_seq)。
    """
    pose_dir = Path(pose_dir).expanduser().resolve()
    left_dir = pose_dir / cam_left
    right_dir = pose_dir / cam_right
    if pose_dir.exists():
        shutil.rmtree(pose_dir)
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    capture_seqs: list[int] = []
    written = 0
    for fi, row in enumerate(rows):
        seq = int(row.get("capture_seq", fi))
        capture_seqs.append(seq)
        for ci, key in enumerate(("det_left_kpts", "det_right_kpts")):
            kpts = _parse_kpts17(row.get(key))
            person = {"pose_keypoints_2d": [0.0] * _FLAT_LEN}
            if kpts is not None:
                person["pose_keypoints_2d"] = kpts_to_openpose_flat(kpts)
            payload = {"version": 1.3, "people": [person]}
            out_dir = left_dir if ci == 0 else right_dir
            (out_dir / f"frame_{fi:06d}.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        written += 1

    return written, capture_seqs


def export_pose2d_from_jsonl(
    pose2d_path: Path,
    trial_dir: Path,
    *,
    start_seq: int | None = None,
    end_seq: int | None = None,
    max_frames: int = 0,
) -> Pose2DExportStats:
    rows = load_jsonl_rows(
        pose2d_path,
        start_seq=start_seq,
        end_seq=end_seq,
        max_frames=int(max_frames),
    )
    pose_dir = Path(trial_dir).expanduser().resolve() / "pose"
    n, _seqs = export_pose2d_json(rows, pose_dir)
    return Pose2DExportStats(
        frames_in=len(rows),
        frames_written=n,
        pose_dir=pose_dir,
    )
