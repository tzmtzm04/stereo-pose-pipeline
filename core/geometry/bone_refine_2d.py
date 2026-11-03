"""多峰 + 骨长 refine：默认 8 核心骨 argmin SSE，tie ε=1e-8 比热图分。"""

from __future__ import annotations

from collections.abc import Callable
import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from core.geometry.anipose_constraints import BODY_LIMBS_BONE_CONSTRAINTS
from core.geometry.bone_length_calib import (
    BONE_NAMES,
    BODY_LIMBS_JOINT_INDICES,
    CORE_BONE_INDICES,
    FIXED_JOINT_LOCAL_INDICES,
    N_BODY_LIMBS_JOINTS,
    SEARCH_JOINT_LOCAL_INDICES,
)
from core.geometry.calib_loader import load_stereo_calibration
from core.geometry.calib_model import StereoCalibration

SEARCH_JOINT_INDICES: tuple[int, ...] = BODY_LIMBS_JOINT_INDICES
N_BODY_JOINTS = int(N_BODY_LIMBS_JOINTS)
N_SEARCH_JOINTS = len(SEARCH_JOINT_LOCAL_INDICES)
N_PEAKS_DEFAULT = 3
DEFAULT_BONE_TOL_M = 0.03
DEFAULT_TIE_EPS = 1e-8
DEFAULT_SELECTION = "min_core_bone_err"
DEFAULT_CONF_REFINE_TH_OTHER = 0.90
DEFAULT_CONF_REFINE_TH_HIP = 0.85
# body_limbs 局部下标：L_hip=6, R_hip=7
CONF_REFINE_HIP_BODY_LOCAL: tuple[int, ...] = (6, 7)

_CORE_BONE_INDICES_ARR = np.asarray(CORE_BONE_INDICES, dtype=np.int32)
_SEARCH_JOINT_LOCAL_ARR = np.asarray(SEARCH_JOINT_LOCAL_INDICES, dtype=np.int32)
_FIXED_JOINT_LOCAL_ARR = np.asarray(FIXED_JOINT_LOCAL_INDICES, dtype=np.int32)

_BONE_PAIRS_LOCAL: tuple[tuple[int, int], ...] = BODY_LIMBS_BONE_CONSTRAINTS
_BONE_PAIRS_LOCAL_ARR = np.asarray(_BONE_PAIRS_LOCAL, dtype=np.int32)
_COMBO_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _get_peak_combos(n_joints: int, n_peaks: int) -> np.ndarray:
    key = (int(n_joints), int(n_peaks))
    if key not in _COMBO_CACHE:
        _COMBO_CACHE[key] = np.array(
            list(product(range(n_peaks), repeat=n_joints)), dtype=np.int32
        )
    return _COMBO_CACHE[key]


def _search_joint_lock_mask(
    body_l: np.ndarray,
    body_r: np.ndarray,
    *,
    th_other: float,
    th_hip: float,
) -> np.ndarray:
    """(8,) True = min(L,R) top-1 conf 已达阈值，锁 peak 0。"""
    mask = np.zeros(N_SEARCH_JOINTS, dtype=bool)
    hip_set = set(CONF_REFINE_HIP_BODY_LOCAL)
    for si, local in enumerate(_SEARCH_JOINT_LOCAL_ARR):
        c = min(float(body_l[local, 0, 2]), float(body_r[local, 0, 2]))
        th = float(th_hip) if int(local) in hip_set else float(th_other)
        mask[si] = c >= th
    return mask


def _get_gated_peak_combos(lock_mask: np.ndarray, n_peaks: int) -> np.ndarray:
    """锁定的搜索关节固定 peak 0，仅对未锁关节做笛卡尔积。"""
    lock_mask = np.asarray(lock_mask, dtype=bool).reshape(-1)
    if lock_mask.shape[0] != N_SEARCH_JOINTS:
        raise ValueError(f"lock_mask 长度应为 {N_SEARCH_JOINTS}")
    unlocked = np.where(~lock_mask)[0]
    if unlocked.size == 0:
        return np.zeros((1, N_SEARCH_JOINTS), dtype=np.int32)
    sub = _get_peak_combos(int(unlocked.size), int(n_peaks))
    full = np.zeros((sub.shape[0], N_SEARCH_JOINTS), dtype=np.int32)
    full[:, unlocked] = sub
    return full


def _resolve_search_combos(
    body_l: np.ndarray,
    body_r: np.ndarray,
    *,
    n_peaks: int,
    conf_refine_th: float | None,
    conf_refine_th_hip: float | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """→ (combos, lock_mask, conf_gated)。"""
    if conf_refine_th is None or conf_refine_th_hip is None:
        lock = np.zeros(N_SEARCH_JOINTS, dtype=bool)
        combos = _get_peak_combos(N_SEARCH_JOINTS, int(n_peaks))
        return combos, lock, False
    lock = _search_joint_lock_mask(
        body_l,
        body_r,
        th_other=float(conf_refine_th),
        th_hip=float(conf_refine_th_hip),
    )
    combos = _get_gated_peak_combos(lock, int(n_peaks))
    return combos, lock, True


def _expand_search_combos(search_combos: np.ndarray) -> np.ndarray:
    """(B, 8) 搜索峰索引 → (B, 12) 全关节峰索引（腕/踝固定 0）。"""
    sc = np.asarray(search_combos, dtype=np.int32)
    if sc.ndim == 1:
        sc = sc.reshape(1, -1)
    b = sc.shape[0]
    full = np.zeros((b, N_BODY_JOINTS), dtype=np.int32)
    full[:, _SEARCH_JOINT_LOCAL_ARR] = sc
    return full


def _search_score_sum(left_conf: np.ndarray, right_conf: np.ndarray) -> np.ndarray:
    """仅 8 个搜索关节的热图分之和（log conf）。"""
    conf = np.minimum(left_conf, right_conf)
    c = conf[:, _SEARCH_JOINT_LOCAL_ARR]
    return np.sum(np.log(np.clip(c, 1e-6, None)), axis=1)


def _projection_matrices(
    calib: StereoCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k1, d1 = calib.K1, calib.D1
    k2, d2 = calib.K2, calib.D2
    r, t = calib.R, calib.T
    p1 = k1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = k2 @ np.hstack([r, t])
    return k1, d1, k2, d2, p1, p2


def _undistort_batch(xy: np.ndarray, k: np.ndarray, d: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, k, d, None, k).reshape(-1, 2)


def _precompute_body_hypo_undistort(
    body_l: np.ndarray,
    body_r: np.ndarray,
    k1: np.ndarray,
    d1: np.ndarray,
    k2: np.ndarray,
    d2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(J,P,3) → undistorted xy (J,P,2)。"""
    j, p, _ = body_l.shape
    flat_l = body_l[..., :2].reshape(j * p, 2)
    flat_r = body_r[..., :2].reshape(j * p, 2)
    ud_l = _undistort_batch(flat_l, k1, d1).reshape(j, p, 2)
    ud_r = _undistort_batch(flat_r, k2, d2).reshape(j, p, 2)
    return ud_l, ud_r


def _build_dlt_ata_batch(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    *,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """(N,2) 已 undistort → (N,4,4) 对称 DLT 正规方程 A^T A。"""
    x1 = left_xy[:, 0:1]
    y1 = left_xy[:, 1:2]
    x2 = right_xy[:, 0:1]
    y2 = right_xy[:, 1:2]
    rows = (
        x1 * p1[2] - p1[0],
        y1 * p1[2] - p1[1],
        x2 * p2[2] - p2[0],
        y2 * p2[2] - p2[1],
    )
    n = left_xy.shape[0]
    ata = np.zeros((n, 4, 4), dtype=np.float32)
    for r in rows:
        ata += r[:, :, None] * r[:, None, :]
    return ata


def _xyz_from_dlt_ata_batch(ata: np.ndarray) -> np.ndarray:
    """(N,4,4) A^T A → (N,3)；最小特征值特征向量 = DLT 闭式 null。"""
    _, evecs = np.linalg.eigh(ata)
    hom = evecs[:, :, 0]
    w = hom[:, 3:4]
    valid = np.abs(w[:, 0]) > 1e-9
    xyz = np.full((ata.shape[0], 3), np.nan, dtype=np.float32)
    xyz[valid] = (hom[valid, :3] / w[valid]).astype(np.float32)
    return xyz


def triangulate_stereo_batch(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    *,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """(N, 2) 已 undistort → (N, 3)。闭式 DLT：A^T A 最小特征向量（等价 SVD null）。"""
    p1f = np.asarray(p1, dtype=np.float32)
    p2f = np.asarray(p2, dtype=np.float32)
    lxy = np.asarray(left_xy, dtype=np.float32)
    rxy = np.asarray(right_xy, dtype=np.float32)
    ata = _build_dlt_ata_batch(lxy, rxy, p1=p1f, p2=p2f)
    return _xyz_from_dlt_ata_batch(ata)


def load_bone_length_targets(path: Path) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    bl = data.get("bone_lengths_m_median") or {}
    out = np.array([float(bl[name]) for name in BONE_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"骨长表含 NaN: {path}")
    return out


def _gather_hypo(body_hypo: np.ndarray, combos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    j_idx = np.arange(body_hypo.shape[0], dtype=np.int32)
    picked = body_hypo[j_idx[None, :], combos]
    return picked[..., :2].astype(np.float64), picked[..., 2].astype(np.float64)


def _bone_deltas_batch(pts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """(B,J,3) → (B,12) 有符号 ΔL。"""
    pa = pts[:, _BONE_PAIRS_LOCAL_ARR[:, 0], :]
    pb = pts[:, _BONE_PAIRS_LOCAL_ARR[:, 1], :]
    lengths = np.linalg.norm(pb - pa, axis=2)
    return lengths - targets[np.newaxis, :]


def _core_bone_err_sse(bone_deltas: np.ndarray) -> np.ndarray:
    """(B,12) → (B,) 4 肢段（上臂+大腿）SSE。"""
    d = bone_deltas[:, _CORE_BONE_INDICES_ARR]
    return np.sum(d * d, axis=1)


def _select_best_combo_min_core_err(
    bone_deltas: np.ndarray,
    scores: np.ndarray,
    *,
    tie_eps: float,
) -> tuple[int, str]:
    """8 核心骨 argmin SSE；并列时 tie_eps 内 argmax 热图分。"""
    err = _core_bone_err_sse(bone_deltas)
    min_err = float(np.min(err))
    tied = np.where(err <= min_err + float(tie_eps))[0]
    pick = int(tied[np.argmax(scores[tied])])
    return pick, "min_core_bone_err"


def _select_best_combo(
    bone_deltas: np.ndarray,
    scores: np.ndarray,
    *,
    tol_m: float,
) -> tuple[int, str]:
    """合格池内 argmax score；空池 fallback combo 0（ViTPose top-1）。"""
    pick, score = _best_feasible_in_batch(bone_deltas, scores, tol_m=tol_m)
    if pick is not None:
        return pick, "score_in_feasible"
    return 0, "fallback_top1"


def _pick_combo_from_batch(
    bone_deltas: np.ndarray,
    scores: np.ndarray,
    *,
    selection: str,
    tol_m: float,
    tie_eps: float,
) -> tuple[int, float, float, str]:
    sel = str(selection).strip().lower()
    if sel == "feasible_score":
        pick, mode = _select_best_combo(bone_deltas, scores, tol_m=float(tol_m))
        bone_err = float(np.sum(bone_deltas[pick] ** 2))
    else:
        pick, mode = _select_best_combo_min_core_err(
            bone_deltas, scores, tie_eps=float(tie_eps)
        )
        bone_err = float(_core_bone_err_sse(bone_deltas)[pick])
    return pick, bone_err, float(scores[pick]), mode


def _merge_min_core_err_chunks(
    *,
    best_pick: int,
    best_err: float,
    best_score: float,
    offset: int,
    bone_deltas: np.ndarray,
    scores: np.ndarray,
    tie_eps: float,
) -> tuple[int, float, float]:
    err = _core_bone_err_sse(bone_deltas)
    chunk_min = float(np.min(err))
    tied = np.where(err <= chunk_min + float(tie_eps))[0]
    for local in tied:
        e = float(err[local])
        sc = float(scores[local])
        if e < best_err - float(tie_eps) or (
            abs(e - best_err) <= float(tie_eps) and sc > best_score
        ):
            best_err = e
            best_score = sc
            best_pick = offset + int(local)
    return best_pick, best_err, best_score


def _best_feasible_in_batch(
    bone_deltas: np.ndarray,
    scores: np.ndarray,
    *,
    tol_m: float,
) -> tuple[int | None, float]:
    """单批合格池内 argmax score；无合格返回 (None, -inf)。"""
    abs_d = np.abs(bone_deltas)
    feasible = np.all(abs_d <= float(tol_m), axis=1)
    if not np.any(feasible):
        return None, float("-inf")
    idx = np.where(feasible)[0]
    pick = int(idx[np.argmax(scores[idx])])
    return pick, float(scores[pick])


def _evaluate_combos_cpu(
    body_l_sc: np.ndarray,
    body_r_sc: np.ndarray,
    combos: np.ndarray,
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    bone_targets: np.ndarray,
    selection: str = DEFAULT_SELECTION,
    tol_m: float = DEFAULT_BONE_TOL_M,
    tie_eps: float = DEFAULT_TIE_EPS,
) -> tuple[int, float, float, str]:
    b = combos.shape[0]
    full_combos = _expand_search_combos(combos)
    lx, ls = _gather_hypo(body_l_sc, full_combos)
    rx, rs = _gather_hypo(body_r_sc, full_combos)
    n_pts = b * N_BODY_JOINTS
    xyz = triangulate_stereo_batch(
        lx.reshape(n_pts, 2),
        rx.reshape(n_pts, 2),
        p1=p1,
        p2=p2,
    ).reshape(b, N_BODY_JOINTS, 3)
    deltas = _bone_deltas_batch(xyz.astype(np.float64), bone_targets)
    scores = _search_score_sum(ls, rs)
    return _pick_combo_from_batch(
        deltas,
        scores,
        selection=selection,
        tol_m=tol_m,
        tie_eps=tie_eps,
    )


@dataclass
class FrameRefineResult:
    capture_seq: int
    combo_index: int
    bone_err: float
    score_sum: float
    peak_indices: tuple[int, ...]
    changed_joints: int
    refine_mode: str = ""
    n_combos: int = 0
    n_locked_joints: int = 0


def _apply_combo_to_row(
    row: dict,
    *,
    combos: np.ndarray,
    combo_index: int,
    body_l: np.ndarray,
    body_r: np.ndarray,
    bone_err: float,
    score_sum: float,
    refine_mode: str,
) -> tuple[dict, FrameRefineResult]:
    search_sel = np.asarray(combos[int(combo_index)], dtype=np.int32).reshape(-1)
    full_sel = np.zeros(N_BODY_JOINTS, dtype=np.int32)
    full_sel[_SEARCH_JOINT_LOCAL_ARR] = search_sel
    out = dict(row)
    top_l = np.asarray(row["det_left_kpts"], dtype=np.float32).copy()
    top_r = np.asarray(row["det_right_kpts"], dtype=np.float32).copy()
    changed = 0
    for ji in range(N_BODY_JOINTS):
        peak_ix = int(full_sel[ji])
        g = 5 + ji
        top_l[g] = body_l[ji, peak_ix].astype(np.float32)
        top_r[g] = body_r[ji, peak_ix].astype(np.float32)
        if peak_ix != 0:
            changed += 1
    out["det_left_kpts"] = top_l.tolist()
    out["det_right_kpts"] = top_r.tolist()
    out["bone_refine_peak_idx"] = [int(x) for x in full_sel]
    out["bone_refine_search_peak_idx"] = [int(x) for x in search_sel]
    out["bone_refine_bone_err"] = float(bone_err)
    out["bone_refine_score_sum"] = float(score_sum)
    out["bone_refine_mode"] = str(refine_mode)
    meta = FrameRefineResult(
        capture_seq=int(row["capture_seq"]),
        combo_index=int(combo_index),
        bone_err=float(bone_err),
        score_sum=float(score_sum),
        peak_indices=tuple(int(x) for x in full_sel),
        changed_joints=int(changed),
        refine_mode=str(refine_mode),
    )
    return out, meta


def _attach_conf_gate_meta(
    out: dict,
    meta: FrameRefineResult,
    *,
    lock_mask: np.ndarray,
    n_combos: int,
    conf_gated: bool,
) -> tuple[dict, FrameRefineResult]:
    if conf_gated:
        out["bone_refine_conf_locked"] = [bool(x) for x in lock_mask]
        meta = FrameRefineResult(
            capture_seq=meta.capture_seq,
            combo_index=meta.combo_index,
            bone_err=meta.bone_err,
            score_sum=meta.score_sum,
            peak_indices=meta.peak_indices,
            changed_joints=meta.changed_joints,
            refine_mode=meta.refine_mode,
            n_combos=int(n_combos),
            n_locked_joints=int(np.sum(lock_mask)),
        )
    else:
        meta = FrameRefineResult(
            capture_seq=meta.capture_seq,
            combo_index=meta.combo_index,
            bone_err=meta.bone_err,
            score_sum=meta.score_sum,
            peak_indices=meta.peak_indices,
            changed_joints=meta.changed_joints,
            refine_mode=meta.refine_mode,
            n_combos=int(n_combos),
            n_locked_joints=0,
        )
    return out, meta


def refine_frame_cpu_batch(
    row: dict,
    *,
    calib: StereoCalibration,
    bone_targets: np.ndarray,
    n_peaks: int = N_PEAKS_DEFAULT,
    selection: str = DEFAULT_SELECTION,
    tol_m: float = DEFAULT_BONE_TOL_M,
    tie_eps: float = DEFAULT_TIE_EPS,
    conf_refine_th: float | None = DEFAULT_CONF_REFINE_TH_OTHER,
    conf_refine_th_hip: float | None = DEFAULT_CONF_REFINE_TH_HIP,
    fast_path_top1: bool = False,
    chunk_size: int = 0,
) -> tuple[dict, FrameRefineResult]:
    _ = chunk_size  # 全批；GPU 路径为主
    k1, d1, k2, d2, p1, p2 = _projection_matrices(calib)
    lh = np.asarray(row["det_left_kpts_hypo"], dtype=np.float64)
    rh = np.asarray(row["det_right_kpts_hypo"], dtype=np.float64)
    body_l = lh[5:17]
    body_r = rh[5:17]
    p = min(int(n_peaks), int(body_l.shape[1]))
    body_l_ud, body_r_ud = _precompute_body_hypo_undistort(
        body_l[:, :p], body_r[:, :p], k1, d1, k2, d2
    )
    body_l_sc = body_l[:, :p].copy()
    body_r_sc = body_r[:, :p].copy()
    body_l_sc[..., :2] = body_l_ud
    body_r_sc[..., :2] = body_r_ud

    combos, lock_mask, conf_gated = _resolve_search_combos(
        body_l[:, :p],
        body_r[:, :p],
        n_peaks=p,
        conf_refine_th=conf_refine_th,
        conf_refine_th_hip=conf_refine_th_hip,
    )
    eval_kw = {
        "p1": p1,
        "p2": p2,
        "bone_targets": bone_targets,
        "selection": selection,
        "tol_m": tol_m,
        "tie_eps": tie_eps,
    }
    if conf_gated and bool(np.all(lock_mask)):
        _, err, sc, _ = _evaluate_combos_cpu(
            body_l_sc,
            body_r_sc,
            combos[:1],
            **eval_kw,
        )
        out, meta = _apply_combo_to_row(
            row,
            combos=combos,
            combo_index=0,
            body_l=body_l[:, :p],
            body_r=body_r[:, :p],
            bone_err=err,
            score_sum=sc,
            refine_mode="conf_skip",
        )
        return _attach_conf_gate_meta(
            out, meta, lock_mask=lock_mask, n_combos=1, conf_gated=True
        )

    if fast_path_top1 and str(selection).strip().lower() == "feasible_score":
        pick0, err0, sc0, mode0 = _evaluate_combos_cpu(
            body_l_sc,
            body_r_sc,
            combos[:1],
            **eval_kw,
        )
        if mode0 == "score_in_feasible":
            out, meta = _apply_combo_to_row(
                row,
                combos=combos,
                combo_index=0,
                body_l=body_l[:, :p],
                body_r=body_r[:, :p],
                bone_err=err0,
                score_sum=sc0,
                refine_mode="fast_top1",
            )
            return _attach_conf_gate_meta(
                out,
                meta,
                lock_mask=lock_mask,
                n_combos=int(combos.shape[0]),
                conf_gated=conf_gated,
            )

    pick, err, sc, mode = _evaluate_combos_cpu(
        body_l_sc,
        body_r_sc,
        combos,
        **eval_kw,
    )
    out, meta = _apply_combo_to_row(
        row,
        combos=combos,
        combo_index=pick,
        body_l=body_l[:, :p],
        body_r=body_r[:, :p],
        bone_err=err,
        score_sum=sc,
        refine_mode=mode,
    )
    return _attach_conf_gate_meta(
        out, meta, lock_mask=lock_mask, n_combos=int(combos.shape[0]), conf_gated=conf_gated
    )


def refine_frame_gpu_batch_row(
    row: dict,
    *,
    calib: StereoCalibration,
    bone_targets: np.ndarray,
    n_peaks: int = N_PEAKS_DEFAULT,
    selection: str = DEFAULT_SELECTION,
    tol_m: float = DEFAULT_BONE_TOL_M,
    tie_eps: float = DEFAULT_TIE_EPS,
    conf_refine_th: float | None = DEFAULT_CONF_REFINE_TH_OTHER,
    conf_refine_th_hip: float | None = DEFAULT_CONF_REFINE_TH_HIP,
    fast_path_top1: bool = False,
    chunk_size: int = 32768,
) -> tuple[dict, FrameRefineResult]:
    from core.geometry.bone_refine_2d_gpu import refine_frame_gpu_batch

    lh = np.asarray(row["det_left_kpts_hypo"], dtype=np.float64)
    rh = np.asarray(row["det_right_kpts_hypo"], dtype=np.float64)
    body_l = lh[5:17]
    body_r = rh[5:17]
    p = min(int(n_peaks), int(body_l.shape[1]))
    combos, lock_mask, conf_gated = _resolve_search_combos(
        body_l[:, :p],
        body_r[:, :p],
        n_peaks=p,
        conf_refine_th=conf_refine_th,
        conf_refine_th_hip=conf_refine_th_hip,
    )
    pick, err, sc, mode = refine_frame_gpu_batch(
        row,
        calib=calib,
        bone_targets=bone_targets,
        n_peaks=p,
        selection=selection,
        tol_m=tol_m,
        tie_eps=tie_eps,
        fast_path_top1=fast_path_top1,
        chunk_size=int(chunk_size),
        combos=combos,
    )
    if conf_gated and bool(np.all(lock_mask)):
        mode = "conf_skip"
    out, meta = _apply_combo_to_row(
        row,
        combos=combos,
        combo_index=pick,
        body_l=body_l[:, :p],
        body_r=body_r[:, :p],
        bone_err=err,
        score_sum=sc,
        refine_mode=mode,
    )
    return _attach_conf_gate_meta(
        out, meta, lock_mask=lock_mask, n_combos=int(combos.shape[0]), conf_gated=conf_gated
    )


@dataclass
class RefineBatchStats:
    frames_in: int = 0
    frames_out: int = 0
    mean_bone_err: float = 0.0
    mean_changed_joints: float = 0.0
    n_combos_per_frame: int = 0
    mean_n_combos: float = 0.0
    backend: str = "cpu"


def refine_pose2d_session_batch(
    *,
    pose2d_path: Path,
    bone_lengths_path: Path,
    calib_path: Path,
    out_path: Path,
    start_seq: int | None = None,
    end_seq: int | None = None,
    max_frames: int = 0,
    n_peaks: int = N_PEAKS_DEFAULT,
    selection: str = DEFAULT_SELECTION,
    tol_m: float = DEFAULT_BONE_TOL_M,
    tie_eps: float = DEFAULT_TIE_EPS,
    conf_refine_th: float | None = DEFAULT_CONF_REFINE_TH_OTHER,
    conf_refine_th_hip: float | None = DEFAULT_CONF_REFINE_TH_HIP,
    backend: str = "cpu",
    chunk_size: int = 32768,
    on_frame_start: Callable[[int, int, int], None] | None = None,
    on_frame_done: Callable[[int, int, FrameRefineResult, float], None] | None = None,
) -> tuple[RefineBatchStats, list[FrameRefineResult]]:
    calib = load_stereo_calibration(calib_path)
    targets = load_bone_length_targets(bone_lengths_path)
    p = int(n_peaks)
    n_combo = p ** N_SEARCH_JOINTS
    _ = _get_peak_combos(N_SEARCH_JOINTS, p)

    use_gpu = str(backend).strip().lower() == "gpu"
    if use_gpu:
        try:
            from core.geometry.bone_refine_2d_gpu import warmup_gpu

            warmup_gpu(calib=calib)
        except Exception as exc:
            raise RuntimeError(
                f"GPU backend 不可用: {exc}\n请 pip install cupy-cuda12x 或 --backend cpu"
            ) from exc

    refine_fn = refine_frame_gpu_batch_row if use_gpu else refine_frame_cpu_batch

    metas: list[FrameRefineResult] = []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    target_n = int(max_frames) if max_frames > 0 else 0
    with Path(pose2d_path).open(encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seq = int(row.get("capture_seq", -1))
            if start_seq is not None and seq < int(start_seq):
                continue
            if end_seq is not None and seq > int(end_seq):
                continue
            if max_frames > 0 and n_written >= int(max_frames):
                break
            if "det_left_kpts_hypo" not in row or "det_right_kpts_hypo" not in row:
                continue
            frame_ix = n_written + 1
            total = target_n if target_n > 0 else frame_ix
            if on_frame_start is not None:
                on_frame_start(frame_ix, total, seq)
            t_frame = time.perf_counter()
            refine_kw: dict = {
                "calib": calib,
                "bone_targets": targets,
                "n_peaks": p,
                "selection": str(selection),
                "tol_m": float(tol_m),
                "tie_eps": float(tie_eps),
                "conf_refine_th": conf_refine_th,
                "conf_refine_th_hip": conf_refine_th_hip,
                "chunk_size": int(chunk_size),
            }
            refined, meta = refine_fn(row, **refine_kw)
            frame_wall = time.perf_counter() - t_frame
            fout.write(json.dumps(refined, ensure_ascii=False) + "\n")
            fout.flush()
            metas.append(meta)
            n_written += 1
            if on_frame_done is not None:
                total = target_n if target_n > 0 else n_written
                on_frame_done(n_written, total, meta, frame_wall)

    stats = RefineBatchStats(
        frames_in=n_written,
        frames_out=n_written,
        n_combos_per_frame=n_combo,
        backend="gpu" if use_gpu else "cpu",
    )
    if metas:
        stats.mean_bone_err = float(np.mean([m.bone_err for m in metas]))
        stats.mean_changed_joints = float(np.mean([m.changed_joints for m in metas]))
        stats.mean_n_combos = float(np.mean([m.n_combos for m in metas if m.n_combos > 0]))
    return stats, metas


# 兼容旧名
refine_pose2d_session_cpu_batch = refine_pose2d_session_batch
