"""骨长多峰 refine · GPU batch（CuPy 闭式 DLT + eigh）。"""

from __future__ import annotations

import numpy as np

from core.geometry.bone_refine_2d import (
    DEFAULT_BONE_TOL_M,
    DEFAULT_SELECTION,
    DEFAULT_TIE_EPS,
    N_BODY_JOINTS,
    N_SEARCH_JOINTS,
    _BONE_PAIRS_LOCAL_ARR,
    _best_feasible_in_batch,
    _expand_search_combos,
    _gather_hypo,
    _get_peak_combos,
    _merge_min_core_err_chunks,
    _pick_combo_from_batch,
    _precompute_body_hypo_undistort,
    _projection_matrices,
    _search_score_sum,
)
from core.geometry.calib_model import StereoCalibration


def _triangulate_batch_cupy(
    left_xy: "cp.ndarray",
    right_xy: "cp.ndarray",
    p1: "cp.ndarray",
    p2: "cp.ndarray",
) -> "cp.ndarray":
    """(N, 2) undistorted → (N, 3)；闭式 DLT（A^T A + eigh）。"""
    import cupy as cp

    x1 = left_xy[:, 0:1]
    y1 = left_xy[:, 1:2]
    x2 = right_xy[:, 0:1]
    y2 = right_xy[:, 1:2]
    n = left_xy.shape[0]
    rows = (
        x1 * p1[2] - p1[0],
        y1 * p1[2] - p1[1],
        x2 * p2[2] - p2[0],
        y2 * p2[2] - p2[1],
    )
    ata = cp.zeros((n, 4, 4), dtype=cp.float32)
    for r in rows:
        ata += r[:, :, None] * r[:, None, :]
    _, evecs = cp.linalg.eigh(ata)
    hom = evecs[:, :, 0]
    w = hom[:, 3:4]
    xyz = cp.empty((n, 3), dtype=cp.float32)
    valid = cp.abs(w[:, 0]) > 1e-9
    xyz[valid] = (hom[valid, :3] / w[valid]).astype(cp.float32)
    xyz[~valid] = cp.nan
    return xyz


def _bone_deltas_batch_cupy(
    pts: "cp.ndarray",
    targets: "cp.ndarray",
) -> "cp.ndarray":
    """(B, J, 3) → (B, 12) 有符号 ΔL。"""
    import cupy as cp

    pa = pts[:, _BONE_PAIRS_LOCAL_ARR[:, 0], :]
    pb = pts[:, _BONE_PAIRS_LOCAL_ARR[:, 1], :]
    lengths = cp.linalg.norm(pb - pa, axis=2)
    return lengths - targets[None, :]


def warmup_gpu(*, calib: StereoCalibration, n: int = 512) -> None:
    """首次 CuPy kernel 编译；session 启动时调用一次。"""
    import cupy as cp

    _ = cp.cuda.runtime.getDeviceCount()
    _, _, _, _, p1, p2 = _projection_matrices(calib)
    lx = cp.random.randn(n, 2, dtype=cp.float32)
    rx = cp.random.randn(n, 2, dtype=cp.float32)
    p1_g = cp.asarray(p1, dtype=cp.float32)
    p2_g = cp.asarray(p2, dtype=cp.float32)
    _triangulate_batch_cupy(lx, rx, p1_g, p2_g)
    cp.cuda.Stream.null.synchronize()
    print("GPU warmup OK", flush=True)


def _evaluate_chunk_gpu(
    body_l_sc: np.ndarray,
    body_r_sc: np.ndarray,
    combos: np.ndarray,
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    bone_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """单 chunk → (deltas, scores)。"""
    import cupy as cp

    b = combos.shape[0]
    full_combos = _expand_search_combos(combos)
    lx, ls = _gather_hypo(body_l_sc, full_combos)
    rx, rs = _gather_hypo(body_r_sc, full_combos)
    scores_np = _search_score_sum(ls, rs).astype(np.float64)

    lx_g = cp.asarray(lx.reshape(b * N_BODY_JOINTS, 2), dtype=cp.float32)
    rx_g = cp.asarray(rx.reshape(b * N_BODY_JOINTS, 2), dtype=cp.float32)
    p1_g = cp.asarray(p1, dtype=cp.float32)
    p2_g = cp.asarray(p2, dtype=cp.float32)
    targets_g = cp.asarray(bone_targets, dtype=cp.float32)

    xyz = _triangulate_batch_cupy(lx_g, rx_g, p1_g, p2_g).reshape(
        b, N_BODY_JOINTS, 3
    )
    deltas = _bone_deltas_batch_cupy(xyz, targets_g)
    deltas_np = cp.asnumpy(deltas).astype(np.float64)
    return deltas_np, scores_np


def evaluate_combos_gpu(
    body_l_ud: np.ndarray,
    body_r_ud: np.ndarray,
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
    chunk_size: int = 32768,
) -> tuple[int, float, float, str]:
    """全组合 GPU 评估 → (best_combo_idx, bone_err_sse, score, mode)。"""
    b = combos.shape[0]
    sel = str(selection).strip().lower()
    use_feasible = sel == "feasible_score"

    def _finish(deltas_np: np.ndarray, scores_np: np.ndarray) -> tuple[int, float, float, str]:
        return _pick_combo_from_batch(
            deltas_np,
            scores_np,
            selection=selection,
            tol_m=float(tol_m),
            tie_eps=float(tie_eps),
        )

    if b <= 1 or int(chunk_size) <= 0 or b <= int(chunk_size):
        deltas_np, scores_np = _evaluate_chunk_gpu(
            body_l_sc,
            body_r_sc,
            combos,
            p1=p1,
            p2=p2,
            bone_targets=bone_targets,
        )
        return _finish(deltas_np, scores_np)

    cs = int(chunk_size)
    if use_feasible:
        best_pick: int | None = None
        best_score = float("-inf")
        best_bone_err = 0.0
        fallback_deltas: np.ndarray | None = None
        fallback_score = 0.0
        for off in range(0, b, cs):
            chunk = combos[off : off + cs]
            deltas_np, scores_np = _evaluate_chunk_gpu(
                body_l_sc,
                body_r_sc,
                chunk,
                p1=p1,
                p2=p2,
                bone_targets=bone_targets,
            )
            if off == 0:
                fallback_deltas = deltas_np
                fallback_score = float(scores_np[0])
            local_pick, local_score = _best_feasible_in_batch(
                deltas_np, scores_np, tol_m=float(tol_m)
            )
            if local_pick is not None and local_score > best_score:
                best_pick = off + local_pick
                best_score = local_score
                best_bone_err = float(np.sum(deltas_np[local_pick] ** 2))
        if best_pick is not None:
            return best_pick, best_bone_err, best_score, "score_in_feasible"
        assert fallback_deltas is not None
        bone_err = float(np.sum(fallback_deltas[0] ** 2))
        return 0, bone_err, fallback_score, "fallback_top1"

    best_pick = 0
    best_err = float("inf")
    best_score = float("-inf")
    for off in range(0, b, cs):
        chunk = combos[off : off + cs]
        deltas_np, scores_np = _evaluate_chunk_gpu(
            body_l_sc,
            body_r_sc,
            chunk,
            p1=p1,
            p2=p2,
            bone_targets=bone_targets,
        )
        best_pick, best_err, best_score = _merge_min_core_err_chunks(
            best_pick=best_pick,
            best_err=best_err,
            best_score=best_score,
            offset=off,
            bone_deltas=deltas_np,
            scores=scores_np,
            tie_eps=float(tie_eps),
        )
    return best_pick, best_err, best_score, "min_core_bone_err"


def refine_frame_gpu_batch(
    row: dict,
    *,
    calib: StereoCalibration,
    bone_targets: np.ndarray,
    n_peaks: int,
    selection: str = DEFAULT_SELECTION,
    tol_m: float = DEFAULT_BONE_TOL_M,
    tie_eps: float = DEFAULT_TIE_EPS,
    fast_path_top1: bool = False,
    chunk_size: int = 32768,
    combos: np.ndarray | None = None,
) -> tuple[int, float, float, str]:
    """返回 (combo_index, bone_err, score, mode)。"""
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

    eval_kw = {
        "p1": p1,
        "p2": p2,
        "bone_targets": bone_targets,
        "selection": selection,
        "tol_m": tol_m,
        "tie_eps": tie_eps,
        "chunk_size": int(chunk_size),
    }

    if fast_path_top1 and str(selection).strip().lower() == "feasible_score" and p > 0:
        combo0 = np.zeros((1, N_SEARCH_JOINTS), dtype=np.int32)
        pick0, err0, sc0, mode0 = evaluate_combos_gpu(
            body_l_ud,
            body_r_ud,
            body_l_sc,
            body_r_sc,
            combo0,
            **eval_kw,
        )
        if mode0 == "score_in_feasible":
            return 0, err0, sc0, "fast_top1"

    if combos is None:
        combos = _get_peak_combos(N_SEARCH_JOINTS, p)
    pick, err, sc, mode = evaluate_combos_gpu(
        body_l_ud,
        body_r_ud,
        body_l_sc,
        body_r_sc,
        combos,
        **eval_kw,
    )
    return pick, err, sc, mode
