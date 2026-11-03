"""Pose2Sim 三角化核心（vendored，避免 Pose2Sim.common 的 c3d 等重依赖）。

算法来源: Pose2Sim/triangulation.py + common.py (BSD-3-Clause, David Pagnon)
"""

from __future__ import annotations

import itertools as it

import cv2
import numpy as np


def weighted_triangulation(
    p_all: list[np.ndarray],
    x_all: np.ndarray,
    y_all: np.ndarray,
    likelihood_all: np.ndarray,
) -> np.ndarray:
    a = np.empty((0, 4), dtype=np.float64)
    for c in range(len(x_all)):
        p_cam = p_all[c]
        w = float(likelihood_all[c])
        a = np.vstack((a, (p_cam[0] - x_all[c] * p_cam[2]) * w))
        a = np.vstack((a, (p_cam[1] - y_all[c] * p_cam[2]) * w))
    if a.shape[0] >= 4:
        _s, _u, vt = cv2.SVDecomp(a)
        v = vt.T
        return np.array(
            [v[0][3] / v[3][3], v[1][3] / v[3][3], v[2][3] / v[3][3], 1.0],
            dtype=np.float64,
        )
    return np.array([np.nan, np.nan, np.nan, 1.0], dtype=np.float64)


def reprojection(p_all: list[np.ndarray], q: np.ndarray) -> tuple[list[float], list[float]]:
    x_calc: list[float] = []
    y_calc: list[float] = []
    for p_cam in p_all:
        x_calc.append(float(p_cam[0] @ q / (p_cam[2] @ q)))
        y_calc.append(float(p_cam[1] @ q / (p_cam[2] @ q)))
    return x_calc, y_calc


def euclidean_distance(q1, q2) -> float:
    d = np.asarray(q2, dtype=np.float64) - np.asarray(q1, dtype=np.float64)
    if np.isnan(d).all():
        return float("inf")
    return float(np.sqrt(np.nansum(d**2)))


def triangulation_from_best_cameras(
    config_dict: dict,
    coords_2d_kpt: tuple[np.ndarray, np.ndarray, np.ndarray],
    coords_2d_kpt_swapped: tuple[np.ndarray, np.ndarray, np.ndarray],
    projection_matrices: list[np.ndarray],
    calib_params: dict | None = None,
) -> tuple[np.ndarray, float, int, list[int]]:
    """Pose2Sim 逐点鲁棒三角化（加权 DLT + 重投影阈值 + 可选左右互换）。"""
    tri_cfg = config_dict.get("triangulation", {})
    error_threshold = float(tri_cfg.get("reproj_error_threshold_triangulation", 15))
    min_cameras = int(tri_cfg.get("min_cameras_for_triangulation", 2))
    handle_lr_swap = bool(tri_cfg.get("handle_LR_swap", False))
    undistort_points = bool(tri_cfg.get("undistort_points", False))

    x_files, y_files, likelihood_files = coords_2d_kpt
    x_files_swapped, y_files_swapped, likelihood_files_swapped = coords_2d_kpt_swapped
    n_cams = len(x_files)
    error_min = float("inf")
    nb_cams_off = 0
    best_cams: int | None = None
    q_best = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    nb_cams_excluded = n_cams
    id_excluded_cams = list(range(n_cams))

    if undistort_points and calib_params:
        calib_params_k = calib_params["K"]
        calib_params_dist = calib_params["dist"]
        calib_params_r = calib_params["R"]
        calib_params_t = calib_params["T"]

    while error_min > error_threshold and n_cams - nb_cams_off >= min_cameras:
        id_cams_off = np.array(list(it.combinations(range(n_cams), nb_cams_off)))
        projection_matrices_filt = [projection_matrices] * len(id_cams_off)

        x_files_filt = np.vstack([x_files.copy()] * len(id_cams_off))
        y_files_filt = np.vstack([y_files.copy()] * len(id_cams_off))
        x_files_swapped_filt = np.vstack([x_files_swapped.copy()] * len(id_cams_off))
        y_files_swapped_filt = np.vstack([y_files_swapped.copy()] * len(id_cams_off))
        likelihood_files_filt = np.vstack([likelihood_files.copy()] * len(id_cams_off))

        if nb_cams_off > 0:
            for i in range(len(id_cams_off)):
                for idx in id_cams_off[i]:
                    x_files_filt[i][idx] = np.nan
                    y_files_filt[i][idx] = np.nan
                    x_files_swapped_filt[i][idx] = np.nan
                    y_files_swapped_filt[i][idx] = np.nan
                    likelihood_files_filt[i][idx] = np.nan

        id_cams_off_tot = [
            np.argwhere(np.isnan(x)).ravel() for x in likelihood_files_filt
        ]
        nb_cams_excluded_filt = [
            int(np.count_nonzero(np.nan_to_num(x) == 0)) for x in likelihood_files_filt
        ]
        nb_cams_off_tot = max(nb_cams_excluded_filt) if nb_cams_excluded_filt else 0
        if nb_cams_off_tot > n_cams - min_cameras:
            break

        if undistort_points and calib_params:
            calib_params_k_filt = [[calib_params_k] * len(id_cams_off)][0]
            calib_params_dist_filt = [[calib_params_dist] * len(id_cams_off)][0]
            calib_params_r_filt = [[calib_params_r] * len(id_cams_off)][0]
            calib_params_t_filt = [[calib_params_t] * len(id_cams_off)][0]
            calib_params_k_filt = [
                [
                    c[i]
                    for i in range(n_cams)
                    if not np.isnan(likelihood_files_filt[j][i])
                    and likelihood_files_filt[j][i] != 0.0
                ]
                for j, c in enumerate(calib_params_k_filt)
            ]
            calib_params_dist_filt = [
                [
                    c[i]
                    for i in range(n_cams)
                    if not np.isnan(likelihood_files_filt[j][i])
                    and likelihood_files_filt[j][i] != 0.0
                ]
                for j, c in enumerate(calib_params_dist_filt)
            ]
            calib_params_r_filt = [
                [
                    c[i]
                    for i in range(n_cams)
                    if not np.isnan(likelihood_files_filt[j][i])
                    and likelihood_files_filt[j][i] != 0.0
                ]
                for j, c in enumerate(calib_params_r_filt)
            ]
            calib_params_t_filt = [
                [
                    c[i]
                    for i in range(n_cams)
                    if not np.isnan(likelihood_files_filt[j][i])
                    and likelihood_files_filt[j][i] != 0.0
                ]
                for j, c in enumerate(calib_params_t_filt)
            ]

        projection_matrices_filt = [
            [
                p[i]
                for i in range(n_cams)
                if not np.isnan(likelihood_files_filt[j][i])
                and likelihood_files_filt[j][i] != 0.0
            ]
            for j, p in enumerate(projection_matrices_filt)
        ]

        x_files_filt = [
            np.array(
                [
                    xx
                    for ii, xx in enumerate(x)
                    if not np.isnan(likelihood_files_filt[i][ii])
                    and likelihood_files_filt[i][ii] != 0.0
                ]
            )
            for i, x in enumerate(x_files_filt)
        ]
        y_files_filt = [
            np.array(
                [
                    xx
                    for ii, xx in enumerate(x)
                    if not np.isnan(likelihood_files_filt[i][ii])
                    and likelihood_files_filt[i][ii] != 0.0
                ]
            )
            for i, x in enumerate(y_files_filt)
        ]
        x_files_swapped_filt = [
            np.array(
                [
                    xx
                    for ii, xx in enumerate(x)
                    if not np.isnan(likelihood_files_filt[i][ii])
                    and likelihood_files_filt[i][ii] != 0.0
                ]
            )
            for i, x in enumerate(x_files_swapped_filt)
        ]
        y_files_swapped_filt = [
            np.array(
                [
                    xx
                    for ii, xx in enumerate(x)
                    if not np.isnan(likelihood_files_filt[i][ii])
                    and likelihood_files_filt[i][ii] != 0.0
                ]
            )
            for i, x in enumerate(y_files_swapped_filt)
        ]
        likelihood_files_filt = [
            np.array([xx for xx in x if not np.isnan(xx) and xx != 0.0])
            for x in likelihood_files_filt
        ]

        q_filt = [
            weighted_triangulation(
                projection_matrices_filt[i],
                x_files_filt[i],
                y_files_filt[i],
                likelihood_files_filt[i],
            )
            for i in range(len(id_cams_off))
        ]

        if undistort_points and calib_params:
            coords_calc = [
                np.array(
                    [
                        cv2.projectPoints(
                            np.array(q_filt[i][:-1]),
                            calib_params_r_filt[i][j],
                            calib_params_t_filt[i][j],
                            calib_params_k_filt[i][j],
                            calib_params_dist_filt[i][j],
                        )[0].ravel()
                        for j in range(n_cams - nb_cams_excluded_filt[i])
                    ]
                )
                for i in range(len(id_cams_off))
            ]
            coords_calc = [
                [coords_calc[i][:, 0], coords_calc[i][:, 1]] for i in range(len(id_cams_off))
            ]
        else:
            coords_calc = [
                reprojection(projection_matrices_filt[i], q_filt[i])
                for i in range(len(id_cams_off))
            ]
        coords_calc = np.array(coords_calc, dtype=object)
        x_calc_filt = coords_calc[:, 0]
        y_calc_filt = coords_calc[:, 1]

        errors: list[float] = []
        for config_off_id in range(len(x_calc_filt)):
            q_file = [
                (x_files_filt[config_off_id][i], y_files_filt[config_off_id][i])
                for i in range(len(x_files_filt[config_off_id]))
            ]
            q_calc = [
                (x_calc_filt[config_off_id][i], y_calc_filt[config_off_id][i])
                for i in range(len(x_calc_filt[config_off_id]))
            ]
            errors.append(
                float(
                    np.mean(
                        [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))]
                    )
                )
            )

        error_min = float(np.nanmin(errors))
        best_cams = int(np.nanargmin(errors))
        nb_cams_excluded = nb_cams_excluded_filt[best_cams]
        q_best = q_filt[best_cams][:-1]

        if handle_lr_swap and error_min > error_threshold:
            n_cams_swapped = 1
            error_off_swap_min = error_min
            while (
                error_off_swap_min > error_threshold
                and n_cams_swapped < (n_cams - nb_cams_off_tot) / 2
            ):
                id_cams_swapped = np.array(
                    list(it.combinations(range(n_cams - nb_cams_off_tot), n_cams_swapped))
                )
                x_off_swap = [[x] * len(id_cams_swapped) for x in x_files_filt]
                y_off_swap = [[y] * len(id_cams_swapped) for y in y_files_filt]
                for id_off in range(len(id_cams_off)):
                    for id_swapped, config_swapped in enumerate(id_cams_swapped):
                        x_off_swap[id_off][id_swapped][config_swapped] = (
                            x_files_swapped_filt[id_off][config_swapped]
                        )
                        y_off_swap[id_off][id_swapped][config_swapped] = (
                            y_files_swapped_filt[id_off][config_swapped]
                        )

                q_off_swap = np.array(
                    [
                        [
                            weighted_triangulation(
                                projection_matrices_filt[id_off],
                                x_off_swap[id_off][id_swapped],
                                y_off_swap[id_off][id_swapped],
                                likelihood_files_filt[id_off],
                            )
                            for id_swapped in range(len(id_cams_swapped))
                        ]
                        for id_off in range(len(id_cams_off))
                    ]
                )

                if undistort_points and calib_params:
                    coords_off_swap = [
                        np.array(
                            [
                                [
                                    cv2.projectPoints(
                                        np.array(q_off_swap[id_off][id_swapped][:-1]),
                                        calib_params_r_filt[id_off][j],
                                        calib_params_t_filt[id_off][j],
                                        calib_params_k_filt[id_off][j],
                                        calib_params_dist_filt[id_off][j],
                                    )[0].ravel()
                                    for j in range(n_cams - nb_cams_off_tot)
                                ]
                                for id_swapped in range(len(id_cams_swapped))
                            ]
                        )
                        for id_off in range(len(id_cams_off))
                    ]
                    coords_off_swap = np.array(
                        [
                            [
                                [
                                    coords_off_swap[id_off][id_swapped][:, 0],
                                    coords_off_swap[id_off][id_swapped][:, 1],
                                ]
                                for id_swapped in range(len(id_cams_swapped))
                            ]
                            for id_off in range(len(id_cams_off))
                        ]
                    )
                else:
                    coords_off_swap = [
                        np.array(
                            [
                                reprojection(
                                    projection_matrices_filt[id_off],
                                    q_off_swap[id_off][id_swapped],
                                )
                                for id_swapped in range(len(id_cams_swapped))
                            ]
                        )
                        for id_off in range(len(id_cams_off))
                    ]

                x_calc_off_swap = [c[:, 0] for c in coords_off_swap]
                y_calc_off_swap = [c[:, 1] for c in coords_off_swap]
                error_off_swap: list[list[float]] = []
                for id_off in range(len(id_cams_off)):
                    per_cam: list[float] = []
                    for id_swapped, _config_swapped in enumerate(id_cams_swapped):
                        q_file = [
                            (
                                x_off_swap[id_off][id_swapped][i],
                                y_off_swap[id_off][id_swapped][i],
                            )
                            for i in range(n_cams - nb_cams_off_tot)
                        ]
                        q_calc = [
                            (
                                x_calc_off_swap[id_off][id_swapped][i],
                                y_calc_off_swap[id_off][id_swapped][i],
                            )
                            for i in range(n_cams - nb_cams_off_tot)
                        ]
                        per_cam.append(
                            float(
                                np.mean(
                                    [
                                        euclidean_distance(q_file[i], q_calc[i])
                                        for i in range(len(q_file))
                                    ]
                                )
                            )
                        )
                    error_off_swap.append(per_cam)
                error_off_swap_arr = np.array(error_off_swap)
                error_off_swap_min = float(np.min(error_off_swap_arr))
                best_off_swap = np.unravel_index(
                    error_off_swap_arr.argmin(), error_off_swap_arr.shape
                )
                if error_off_swap_min < error_min:
                    error_min = error_off_swap_min
                    best_cams = int(best_off_swap[0])
                    q_best = q_off_swap[best_off_swap][:-1]
                n_cams_swapped += 1

        nb_cams_off += 1

    if best_cams is not None:
        id_excluded_cams = list(id_cams_off_tot[best_cams])
    else:
        id_excluded_cams = list(range(n_cams))
        nb_cams_excluded = n_cams

    if error_min > error_threshold:
        return (
            np.array([np.nan, np.nan, np.nan], dtype=np.float64),
            float("nan"),
            nb_cams_excluded,
            id_excluded_cams,
        )
    return q_best, error_min, nb_cams_excluded, id_excluded_cams
