"""离线 session：tri 3D + 双目 2D → SOMA StereoBody 矫正。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.geometry.calib_loader import load_stereo_calibration
from core.geometry.calib_model import StereoCalibration
from core.geometry.stereo_body_pipeline import (
    StereoBodyConfig,
    StereoBodyPipeline,
    tri_usable_for_body_ik,
)
from core.geometry.smpl_soma_pipeline import SmplSomaPipeline
from core.offline.offline_preview import (
    OfflinePoseRow,
    load_offline_pose_index,
    load_offline_pose_sequence,
)


def resolve_session_stereo_calib(session_dir: Path) -> Path:
    session_dir = Path(session_dir).expanduser().resolve()
    meta_path = session_dir / "full_session_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cp = meta.get("calib_path")
        if cp:
            p = Path(str(cp)).expanduser()
            if p.is_file():
                return p.resolve()
    for rel in ("calibration/stereo_sbs.json", "calib/stereo_sbs.json"):
        p = session_dir / rel
        if p.is_file():
            return p.resolve()
    from app.paths import DEFAULT_STEREO_CALIB

    if DEFAULT_STEREO_CALIB.is_file():
        return DEFAULT_STEREO_CALIB.resolve()
    raise FileNotFoundError(f"找不到 stereo 标定: {session_dir}")


def _coco17_from_kpts3d(kpts3d: np.ndarray) -> np.ndarray:
    k = np.asarray(kpts3d, dtype=np.float32).reshape(-1, 4)
    if k.shape[0] >= 17:
        return k[:17].copy()
    out = np.zeros((17, 4), dtype=np.float32)
    n = min(17, k.shape[0])
    out[:n] = k[:n]
    return out


def _merge_coco17_into_kpts3d(
    template: np.ndarray,
    coco17: np.ndarray,
) -> np.ndarray:
    k = np.asarray(template, dtype=np.float32).reshape(-1, 4).copy()
    c = np.asarray(coco17, dtype=np.float32).reshape(17, 4)
    n = min(17, k.shape[0])
    k[:n, :3] = c[:n, :3]
    k[:n, 3] = np.maximum(k[:n, 3], c[:n, 3])
    return k


@dataclass
class SomaStereoBodyStats:
    frames_in_raw: int = 0
    frames_in: int = 0
    frames_out: int = 0
    frames_skip_tri: int = 0
    frames_skip_det: int = 0
    frames_skip_body: int = 0
    mean_shift_m: float = 0.0
    max_shift_m: float = 0.0
    mean_loss: float = 0.0
    ik_only: bool = False


def run_session_soma_stereo_body(
    *,
    tri_path: Path,
    pose2d_path: Path,
    out_path: Path,
    calib: StereoCalibration | None = None,
    calib_path: Path | None = None,
    cfg: StereoBodyConfig | None = None,
    ik_only: bool = False,
    start_seq: int | None = None,
    end_seq: int | None = None,
    max_frames: int = 0,
) -> SomaStereoBodyStats:
    tri_path = Path(tri_path).expanduser().resolve()
    pose2d_path = Path(pose2d_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    if calib is None:
        if calib_path is None:
            raise ValueError("calib 或 calib_path 必须提供其一")
        calib = load_stereo_calibration(calib_path)

    tri_rows = load_offline_pose_sequence(
        tri_path,
        start_capture_seq=start_seq,
        end_capture_seq=end_seq,
    )
    det_index = load_offline_pose_index(pose2d_path)
    if max_frames > 0:
        tri_rows = tri_rows[: int(max_frames)]

    pipeline = StereoBodyPipeline(calib, cfg=cfg)
    stats = SomaStereoBodyStats(
        frames_in_raw=len(load_offline_pose_index(tri_path)),
        frames_in=len(tri_rows),
        ik_only=bool(ik_only),
    )

    shifts: list[float] = []
    losses: list[float] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for tri_row in tri_rows:
            seq = int(tri_row.capture_seq)
            if tri_row.kpts3d is None or not tri_usable_for_body_ik(tri_row.kpts3d):
                stats.frames_skip_tri += 1
                continue

            det_row: OfflinePoseRow | None = det_index.get(seq)
            if (
                det_row is None
                or det_row.det_left is None
                or det_row.det_right is None
            ):
                stats.frames_skip_det += 1
                continue

            coco_in = _coco17_from_kpts3d(tri_row.kpts3d)
            if ik_only:
                out = pipeline.process_frame_ik_only(
                    det_row.det_left,
                    det_row.det_right,
                    coco_in,
                )
            else:
                out = pipeline.process_frame(
                    det_row.det_left,
                    det_row.det_right,
                    coco_in,
                )
            if out is None:
                stats.frames_skip_body += 1
                continue

            kpts_new = _merge_coco17_into_kpts3d(tri_row.kpts3d, out.kpts3d)
            delta = np.linalg.norm(
                kpts_new[:17, :3].astype(np.float64)
                - coco_in[:17, :3].astype(np.float64),
                axis=1,
            )
            mask = np.isfinite(coco_in[:17, :3]).all(axis=1)
            if np.any(mask):
                shifts.append(float(np.max(delta[mask])))
            if out.final_loss > 0:
                losses.append(float(out.final_loss))

            rec = {
                "capture_seq": seq,
                "infer_seq": int(tri_row.infer_seq),
                "tri_ok": True,
                "kpts3d": kpts_new.tolist(),
                "soma_joints78": out.joints78.tolist(),
                "soma_pose77": out.pose77.tolist(),
                "soma_trans": out.trans.tolist(),
                "soma_final_loss": float(out.final_loss),
                "soma_ik_only": bool(ik_only),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.frames_out += 1

    if shifts:
        stats.mean_shift_m = float(np.mean(shifts))
        stats.max_shift_m = float(np.max(shifts))
    if losses:
        stats.mean_loss = float(np.mean(losses))
    return stats


@dataclass
class SomaSmplMapStats:
    frames_in_raw: int = 0
    frames_in: int = 0
    frames_out: int = 0
    frames_skip_tri: int = 0
    mean_shift_m: float = 0.0
    max_shift_m: float = 0.0
    mean_smpl_map_error_m: float = 0.0


def run_session_soma_via_smpl_map(
    *,
    tri_path: Path,
    out_path: Path,
    start_seq: int | None = None,
    end_seq: int | None = None,
    max_frames: int = 0,
    min_conf: float = 0.25,
) -> SomaSmplMapStats:
    """COCO tri → SMPL-24 → SOMA（无需 pose2d / 标定）。"""
    tri_path = Path(tri_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    tri_rows = load_offline_pose_sequence(
        tri_path,
        start_capture_seq=start_seq,
        end_capture_seq=end_seq,
    )
    if max_frames > 0:
        tri_rows = tri_rows[: int(max_frames)]

    pipeline = SmplSomaPipeline(min_conf=float(min_conf))
    stats = SomaSmplMapStats(
        frames_in_raw=len(load_offline_pose_index(tri_path)),
        frames_in=len(tri_rows),
    )

    shifts: list[float] = []
    map_errs: list[float] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for tri_row in tri_rows:
            seq = int(tri_row.capture_seq)
            if tri_row.kpts3d is None or not tri_usable_for_body_ik(
                tri_row.kpts3d, min_conf=min_conf
            ):
                stats.frames_skip_tri += 1
                continue

            coco_in = _coco17_from_kpts3d(tri_row.kpts3d)
            out = pipeline.process_coco17(coco_in)
            if out is None:
                stats.frames_skip_tri += 1
                continue

            kpts_new = _merge_coco17_into_kpts3d(tri_row.kpts3d, out.kpts3d)
            delta = np.linalg.norm(
                kpts_new[:17, :3].astype(np.float64)
                - coco_in[:17, :3].astype(np.float64),
                axis=1,
            )
            mask = np.isfinite(coco_in[:17, :3]).all(axis=1)
            if np.any(mask):
                shifts.append(float(np.max(delta[mask])))
            map_errs.append(float(out.smpl_map_error_m))

            rec = {
                "capture_seq": seq,
                "infer_seq": int(tri_row.infer_seq),
                "tri_ok": True,
                "kpts3d": kpts_new.tolist(),
                "smpl24_joints": out.smpl24.tolist(),
                "soma_joints78": out.joints78.tolist(),
                "soma_pose77": out.pose77.tolist(),
                "soma_trans": out.trans.tolist(),
                "smpl_map_error_m": float(out.smpl_map_error_m),
                "pipeline": "coco_smpl24_soma",
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.frames_out += 1

    if shifts:
        stats.mean_shift_m = float(np.mean(shifts))
        stats.max_shift_m = float(np.max(shifts))
    if map_errs:
        stats.mean_smpl_map_error_m = float(np.mean(map_errs))
    return stats
