"""实机叠图落盘：原画 / UI 合成 / 关键点 meta（ROUTE2_OVERLAY_DUMP=1）。

用法:
  ROUTE2_OVERLAY_DUMP=1 bash scripts/route2/run_camera_pose3d.sh

输出目录默认 outputs/overlay_dump/<ts>/：
  seq<N>_ring.png           UI 从 VideoRing take 后立刻快照（叠图前）
  seq<N>_compose.png        compose 入口单眼 BGR（ensure_display_eye 后）
  seq<N>_video.png          同 compose（兼容旧名）
  seq<N>_ring_vs_compose.png  ring | compose | abs diff×4 并排
  seq<N>_ui.png             UI 合成（fit/cover 前全分辨率叠图）
  seq<N>_ui_window.png      窗口尺寸 composite（实际 imshow 内容）
  seq<N>_meta.json          seq、ring↔compose diff、关键点等
  seq<N>_side.png           左 compose | 右 ui

ring≈compose → 问题在 VideoRing 之前（prod pull / catch-up）；
ring≠compose → 问题在 Ring take 与叠图之间。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.paths import OUTPUTS
from app.route2.session_config import Route2SessionConfig
from core.display.display_eye import ensure_display_eye_bgr, select_pose_det
from core.display.live_compose import OverlayFn, compose_display_frame
from core.display.pose_store import PoseDisplaySample


def overlay_dump_enabled() -> bool:
    return os.environ.get("ROUTE2_OVERLAY_DUMP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


@dataclass
class OverlayDumpConfig:
    every: int = 15
    max_dumps: int = 24
    dir: Path | None = None

    @classmethod
    def from_env(cls) -> OverlayDumpConfig:
        every = max(1, int(os.environ.get("ROUTE2_OVERLAY_DUMP_EVERY", "15")))
        max_dumps = max(1, int(os.environ.get("ROUTE2_OVERLAY_DUMP_MAX", "24")))
        raw = os.environ.get("ROUTE2_OVERLAY_DUMP_DIR", "").strip()
        out = Path(raw) if raw else None
        return cls(every=every, max_dumps=max_dumps, dir=out)


class OverlayDumpSession:
    def __init__(self, cfg: OverlayDumpConfig) -> None:
        self._cfg = cfg
        self._count = 0
        self._ring_by_seq: dict[int, np.ndarray] = {}
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = cfg.dir if cfg.dir is not None else OUTPUTS / "overlay_dump" / ts
        self.dir = base
        self.dir.mkdir(parents=True, exist_ok=True)
        print(
            f"overlay_dump 开 dir={self.dir} every={cfg.every} max={cfg.max_dumps}",
            flush=True,
        )

    def should_dump(self, ui_idx: int) -> bool:
        if self._count >= self._cfg.max_dumps:
            return False
        return ui_idx == 0 or ui_idx % self._cfg.every == 0

    def dump_ring_take(
        self,
        *,
        ui_idx: int,
        video_seq: int,
        ring_bgr: np.ndarray,
        pose_capture_seq: int = -1,
    ) -> None:
        """VideoRing take 后立即深拷贝；与随后 compose 入口对比。"""
        if not self.should_dump(ui_idx):
            return
        seq = int(video_seq)
        if seq < 0:
            return
        snap = np.ascontiguousarray(ring_bgr).copy()
        self._ring_by_seq[seq] = snap
        stem = self.dir / f"seq{seq:05d}"
        cv2.imwrite(str(stem.with_name(stem.name + "_ring.png")), snap)
        print(
            f"overlay_dump ring take seq={seq} ui={ui_idx} "
            f"pose_seq={int(pose_capture_seq)} → {stem.name}_ring.png",
            flush=True,
        )

    def dump_render(
        self,
        *,
        ui_idx: int,
        video_seq: int,
        cfg: Route2SessionConfig,
        video_bgr: np.ndarray,
        composite_window: np.ndarray,
        pose: PoseDisplaySample | None,
        overlay_pose: PoseDisplaySample | None,
        overlay_fn: OverlayFn | None,
        live_capture_seq: int,
        pose_held: bool,
        hole: tuple[int, int] | None,
    ) -> None:
        if not self.should_dump(ui_idx):
            return
        seq = int(video_seq)
        if seq < 0:
            return

        eye = np.ascontiguousarray(ensure_display_eye_bgr(video_bgr, cfg)).copy()
        ring_snap = self._ring_by_seq.pop(seq, None)
        ring_diff = _bgr_diff(ring_snap, eye) if ring_snap is not None else None

        fill = "cover" if hole is not None else "fit"
        ui_full = compose_display_frame(
            cfg,
            eye,
            overlay_pose,
            overlay_fn=overlay_fn,
            out_size=None,
            fill=fill,
        )

        stem = self.dir / f"seq{seq:05d}"
        cv2.imwrite(str(stem.with_name(stem.name + "_compose.png")), eye)
        cv2.imwrite(str(stem.with_name(stem.name + "_video.png")), eye)
        if ring_snap is not None and ring_diff is not None:
            _write_ring_vs_compose(stem, ring_snap, eye, ring_diff)
        cv2.imwrite(str(stem.with_name(stem.name + "_ui.png")), ui_full)
        cv2.imwrite(
            str(stem.with_name(stem.name + "_ui_window.png")),
            composite_window,
        )

        h = max(eye.shape[0], ui_full.shape[0])
        w = eye.shape[1] + ui_full.shape[1]
        side = np.zeros((h, w, 3), dtype=np.uint8)
        side[: eye.shape[0], : eye.shape[1]] = eye
        side[: ui_full.shape[0], eye.shape[1] : eye.shape[1] + ui_full.shape[1]] = (
            ui_full
        )
        cv2.imwrite(str(stem.with_name(stem.name + "_side.png")), side)

        meta = _build_meta(
            ui_idx=ui_idx,
            seq=seq,
            live_capture_seq=int(live_capture_seq),
            pose=pose,
            overlay_pose=overlay_pose,
            pose_held=bool(pose_held),
            cfg=cfg,
            eye_shape=tuple(int(x) for x in eye.shape),
            ui_full_shape=tuple(int(x) for x in ui_full.shape),
            window_shape=tuple(int(x) for x in composite_window.shape),
            ring_diff=ring_diff,
        )
        meta_path = stem.with_name(stem.name + "_meta.json")
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._count += 1
        diff_tag = ""
        if ring_diff is not None:
            diff_tag = (
                f" ring↔compose mean={ring_diff['mean_px']:.3f}px"
                f" max={ring_diff['max_px']:.0f}"
            )
        print(
            f"overlay_dump #{self._count} seq={seq} ui={ui_idx}{diff_tag} "
            f"→ {stem.name}_*.png",
            flush=True,
        )


_dump_session: OverlayDumpSession | None = None


def get_overlay_dump_session() -> OverlayDumpSession | None:
    global _dump_session
    if not overlay_dump_enabled():
        return None
    if _dump_session is None:
        _dump_session = OverlayDumpSession(OverlayDumpConfig.from_env())
    return _dump_session


def _bgr_diff(a: np.ndarray | None, b: np.ndarray) -> dict[str, float | bool] | None:
    if a is None:
        return None
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mean_px = float(np.mean(d))
    max_px = float(np.max(d))
    return {
        "mean_px": mean_px,
        "max_px": max_px,
        "identical": bool(max_px == 0),
    }


def _write_ring_vs_compose(
    stem: Path,
    ring: np.ndarray,
    compose: np.ndarray,
    diff: dict[str, float | bool],
) -> None:
    mean_px = float(diff["mean_px"])
    left = _label_panel(ring, f"ring take seq={stem.name[3:8]}")
    right = _label_panel(
        compose,
        f"compose mean={mean_px:.3f}px max={int(diff['max_px'])}",
    )
    d = cv2.absdiff(ring, cv2.resize(compose, (ring.shape[1], ring.shape[0])))
    diff_vis = np.clip(d.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
    mid = _label_panel(diff_vis, "abs diff x4")
    combo = np.hstack([left, mid, right])
    cv2.imwrite(str(stem.with_name(stem.name + "_ring_vs_compose.png")), combo)


def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (16, 16, 20), -1)
    cv2.putText(
        out,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _wrist_xy(det: dict | None) -> tuple[float, float] | None:
    if det is None:
        return None
    kpts = det.get("keypoints")
    if kpts is None:
        return None
    arr = np.asarray(kpts, dtype=np.float32)
    if arr.shape[0] < 10:
        return None
    x, y, c = float(arr[9, 0]), float(arr[9, 1]), float(arr[9, 2])
    if c < 0.2 or not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def _build_meta(
    *,
    ui_idx: int,
    seq: int,
    live_capture_seq: int,
    pose: PoseDisplaySample | None,
    overlay_pose: PoseDisplaySample | None,
    pose_held: bool,
    cfg: Route2SessionConfig,
    eye_shape: tuple[int, ...],
    ui_full_shape: tuple[int, ...],
    window_shape: tuple[int, ...],
    ring_diff: dict[str, float | bool] | None = None,
) -> dict[str, Any]:
    det = select_pose_det(overlay_pose or pose, cfg)
    wx, wy = (None, None)
    w = _wrist_xy(det)
    if w is not None:
        wx, wy = w
    return {
        "ts": time.time(),
        "ui_idx": int(ui_idx),
        "capture_seq": int(seq),
        "live_capture_seq": int(live_capture_seq),
        "live_minus_video": int(live_capture_seq) - int(seq),
        "pose_capture_seq": int(pose.capture_seq) if pose is not None else -1,
        "overlay_capture_seq": (
            int(overlay_pose.capture_seq) if overlay_pose is not None else -1
        ),
        "infer_seq": int(pose.infer_seq) if pose is not None else -1,
        "pose_held": bool(pose_held),
        "overlay_applied": overlay_pose is not None,
        "display_eye": str(cfg.display_eye),
        "eye_shape": list(eye_shape),
        "ui_full_shape": list(ui_full_shape),
        "window_shape": list(window_shape),
        "wrist_xy": [wx, wy] if wx is not None else None,
        "det_keypoints": (
            np.asarray(det["keypoints"], dtype=float).tolist()
            if det is not None and det.get("keypoints") is not None
            else None
        ),
        "ring_compose_mean_px": (
            float(ring_diff["mean_px"]) if ring_diff is not None else None
        ),
        "ring_compose_max_px": (
            float(ring_diff["max_px"]) if ring_diff is not None else None
        ),
        "ring_compose_identical": (
            bool(ring_diff["identical"]) if ring_diff is not None else None
        ),
        "bisect_hint": (
            "before_video_ring"
            if ring_diff is not None and float(ring_diff["mean_px"]) < 0.01
            else (
                "between_ring_and_compose"
                if ring_diff is not None and float(ring_diff["mean_px"]) >= 0.01
                else "ring_snap_missing"
            )
        ),
    }
