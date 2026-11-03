"""实时结束后：按 segments.json 切分运动段，供离线 SessionBundle 消费。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from app.sdk._segments import load_session_segments, segments_path
from app.sdk.types import FinalizeResult, MotionSegmentRef
from contracts.segment import Segment
from contracts.session_bundle import (
    CLIPS_MANIFEST_FILENAME,
    SESSION_INDEX_FILENAME,
    SESSION_META_FILENAME,
    SESSION_MJPEG_FILENAME,
    SESSION_POSE_FILENAME,
    SessionBundle,
)
from core.ring.mjpeg_segment_extract import extract_mjpeg_segment

MOTION_SEGMENTS_FILENAME = "motion_segments.json"
MOTION_SEGMENTS_DIRNAME = "motion_segments"


def motion_segments_path(session_dir: Path) -> Path:
    return Path(session_dir).expanduser().resolve() / MOTION_SEGMENTS_FILENAME


def _slice_jsonl_by_seq(
    src: Path,
    dst: Path,
    *,
    start_seq: int,
    end_seq: int,
) -> int:
    """按 capture_seq 闭区间切 jsonl；返回写入行数。"""
    if not src.is_file():
        return 0
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            seq = int(obj.get("capture_seq", -1))
            if seq < start_seq or seq > end_seq:
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def _link_or_copy_mjpeg(parent_mjpeg: Path, clip_mjpeg: Path) -> str:
    """运动段目录内挂接父 session.mjpeg（优先 symlink，失败则跳过）。"""
    if not parent_mjpeg.is_file():
        return "missing"
    if clip_mjpeg.exists() or clip_mjpeg.is_symlink():
        clip_mjpeg.unlink()
    try:
        os.symlink(parent_mjpeg.resolve(), clip_mjpeg)
        return "symlink"
    except OSError:
        return "skipped"


def _materialize_segment_mjpeg(
    *,
    parent_mjpeg: Path,
    parent_index: Path,
    clip_mjpeg: Path,
    clip_index: Path,
    start_seq: int,
    end_seq: int,
) -> str:
    """运动段 mjpeg：按 index 裁切；失败则软链父文件。"""
    n = extract_mjpeg_segment(
        src_mjpeg=parent_mjpeg,
        src_index=parent_index,
        dst_mjpeg=clip_mjpeg,
        dst_index=clip_index,
        start_capture_seq=int(start_seq),
        end_capture_seq=int(end_seq),
    )
    if n > 0:
        return f"extracted_{n}"
    return _link_or_copy_mjpeg(parent_mjpeg, clip_mjpeg)


def materialize_motion_segments(
    session_dir: Path,
    segments: list[Segment] | None = None,
) -> FinalizeResult:
    """
    实时结束后切分运动段：
      · clips_manifest.jsonl（post_process）
      · motion_segments.json（离线入口清单）
      · motion_segments/{stem}/ 伪 SessionBundle（index/pose 切片 + mjpeg 裁切）
    """
    session_dir = Path(session_dir).expanduser().resolve()
    bundle = SessionBundle.from_dir(session_dir)
    segs = list(segments) if segments is not None else load_session_segments(session_dir)

    manifest_path = session_dir / CLIPS_MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest_path.unlink()

    clips_root = session_dir / MOTION_SEGMENTS_DIRNAME
    clips_root.mkdir(parents=True, exist_ok=True)

    refs: list[MotionSegmentRef] = []
    parent_meta: dict = {}
    meta_src = session_dir / SESSION_META_FILENAME
    if meta_src.is_file():
        try:
            parent_meta = json.loads(meta_src.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parent_meta = {}

    for i, seg in enumerate(segs):
        stem = seg.stem(i)
        clip_dir = clips_root / stem
        clip_dir.mkdir(parents=True, exist_ok=True)

        start_s = int(seg.start_capture_seq)
        end_s = int(seg.end_capture_seq)
        clip_mjpeg = clip_dir / SESSION_MJPEG_FILENAME
        clip_index = clip_dir / SESSION_INDEX_FILENAME
        link_mode = _materialize_segment_mjpeg(
            parent_mjpeg=bundle.mjpeg_path,
            parent_index=bundle.index_path,
            clip_mjpeg=clip_mjpeg,
            clip_index=clip_index,
            start_seq=start_s,
            end_seq=end_s,
        )
        if link_mode.startswith("extracted_"):
            n_idx = int(link_mode.split("_", 1)[1])
        else:
            n_idx = _slice_jsonl_by_seq(
                bundle.index_path,
                clip_index,
                start_seq=start_s,
                end_seq=end_s,
            )
        n_pose = _slice_jsonl_by_seq(
            bundle.pose_path,
            clip_dir / SESSION_POSE_FILENAME,
            start_seq=start_s,
            end_seq=end_s,
        )

        clip_meta = {
            **parent_meta,
            "session_id": stem,
            "parent_session_id": session_dir.name,
            "parent_session_dir": str(session_dir),
            "segment": {
                "label": seg.label,
                "start_capture_seq": start_s,
                "end_capture_seq": end_s,
                "index": i,
                "stem": stem,
            },
            "frames_indexed": n_idx,
            "frames_pose": n_pose,
            "mjpeg_link": link_mode,
            "ring_mode": parent_meta.get("ring_mode", "session_mjpeg"),
        }
        (clip_dir / SESSION_META_FILENAME).write_text(
            json.dumps(clip_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (clip_dir / "segment.json").write_text(
            json.dumps(clip_meta["segment"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with manifest_path.open("a", encoding="utf-8") as mf:
            mf.write(
                json.dumps(
                    {
                        "type": "persist",
                        "index": i,
                        "start_capture_seq": start_s,
                        "end_capture_seq": end_s,
                        "status": "post_process",
                        "label": seg.label,
                        "stem": stem,
                        "clip_dir": str(clip_dir.relative_to(session_dir)),
                        "ts": time.time(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        refs.append(
            MotionSegmentRef(
                index=i,
                label=str(seg.label) if seg.label else stem,
                start_capture_seq=start_s,
                end_capture_seq=end_s,
                stem=stem,
                clip_dir=clip_dir,
            )
        )
        print(
            f"sdk finalize 运动段[{i}] label={seg.label} "
            f"seq={start_s}..{end_s} index={n_idx} pose={n_pose} → {stem}",
            flush=True,
        )

    catalog = {
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "segments_file": str(segments_path(session_dir)),
        "clips_manifest": str(manifest_path),
        "clips": [asdict(r) | {"clip_dir": str(r.clip_dir)} for r in refs],
    }
    catalog_path = motion_segments_path(session_dir)
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return FinalizeResult(
        session_id=session_dir.name,
        session_dir=session_dir,
        segments=list(segs),
        motion_segments=refs,
        clips_manifest=manifest_path if manifest_path.is_file() else None,
        motion_segments_json=catalog_path,
    )


def load_motion_segment_refs(session_dir: Path) -> list[MotionSegmentRef]:
    path = motion_segments_path(session_dir)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[MotionSegmentRef] = []
    for item in data.get("clips") or []:
        out.append(
            MotionSegmentRef(
                index=int(item["index"]),
                label=str(item.get("label") or ""),
                start_capture_seq=int(item["start_capture_seq"]),
                end_capture_seq=int(item["end_capture_seq"]),
                stem=str(item["stem"]),
                clip_dir=Path(item["clip_dir"]),
            )
        )
    return out
