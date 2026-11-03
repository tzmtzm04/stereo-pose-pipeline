"""L2 离线 pose3d 帧类型（权威 3D 归档）。"""

from __future__ import annotations

from typing import Any, TypedDict


class Pose3DFrame(TypedDict, total=False):
    """clip_*_pose3d.jsonl 单行。"""

    capture_seq: int
    infer_seq: int
    tri_ok: bool
    kpts3d: list[list[float]]  # (17, 4) x,y,z,conf
    source: str  # 建议 "l2_offline"
    model_id: str
    calib_version: str


def pose3d_row(
    *,
    capture_seq: int,
    infer_seq: int,
    tri_ok: bool,
    kpts3d: list[list[float]] | None = None,
    source: str = "l2_offline",
    model_id: str = "",
    calib_version: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "capture_seq": int(capture_seq),
        "infer_seq": int(infer_seq),
        "tri_ok": bool(tri_ok),
        "source": source,
    }
    if model_id:
        row["model_id"] = model_id
    if calib_version:
        row["calib_version"] = calib_version
    if kpts3d is not None:
        row["kpts3d"] = kpts3d
    return row
