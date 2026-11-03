"""实时黑盒 SDK：启动 session、嵌套预览、段标记、状态摘要。"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from typing import Literal

from app.paths import CLIP_OUTPUT_ROOT
from app.route2.cli import build_parser
from app.route2.session_config import Route2SessionConfig
from app.sdk._preview import (
    align_gl_to_rect,
    align_opencv_to_rect,
    hide_gl,
    hide_opencv,
    opencv_window_ready,
)
from app.sdk._registry import RealtimeRegistry, get_realtime_registry
from app.sdk._segments import SegmentTracker
from app.sdk.types import (
    FinalizeResult,
    PoseDrainStats,
    PoseFrameDTO,
    PreviewConfig,
    PreviewSyncResult,
    RealtimeStatus,
    ScreenRect,
    SegmentMark,
)
from contracts.segment import Segment
from contracts.session_bundle import SessionBundle


@dataclass
class SessionHandle:
    session_id: str
    _registry: RealtimeRegistry
    _thread: threading.Thread
    _error_holder: dict[str, BaseException | None]
    _preview: PreviewConfig
    _segments: SegmentTracker

    @property
    def running(self) -> bool:
        """本场是否在录（引擎线程可仍存活）。"""
        return self._registry.recording_sid() == self.session_id

    @property
    def error(self) -> BaseException | None:
        return self._error_holder.get("error")

    def bundle(self) -> SessionBundle:
        return SessionBundle.from_dir(CLIP_OUTPUT_ROOT / self.session_id)

    def stop(self) -> None:
        """结束本场录制；不拆相机/引擎。"""
        self._registry.request_stop()

    def wait_camera_released(self, timeout: float = 12.0) -> bool:
        """等本场录制封盘、可开下一场（相机常开时 fd 仍可持有）。"""
        ok = self._registry.wait_camera_released(timeout=timeout)
        if ok:
            return True
        if not self._thread.is_alive() and not self._registry.engine_alive():
            print(
                "sdk wait_camera_released: 引擎已退出但 ready 未 signal",
                flush=True,
            )
            return False
        return False

    def join(self, timeout: float = 60.0) -> bool:
        """等本场停录封盘（非等引擎线程退出）。"""
        return self.wait_camera_released(timeout=timeout)

    def materialize_segments(self) -> FinalizeResult:
        """按 segments 切运动段（调用方须已保证 mjpeg 落盘完成，通常先 join）。"""
        from app.sdk._finalize import materialize_motion_segments

        session_dir = CLIP_OUTPUT_ROOT / self.session_id
        segs = self.segments()
        result = materialize_motion_segments(session_dir, segs)
        print(
            f"sdk RealtimeSession materialize session={self.session_id} "
            f"segments={len(result.motion_segments)} "
            f"catalog={result.motion_segments_json} "
            f"thread_alive={self._thread.is_alive()}",
            flush=True,
        )
        return result

    def finalize(self, *, join_timeout: float = 60.0) -> FinalizeResult:
        """停本场录制 → 等封盘 → 切运动段（不关引擎）。"""
        self.stop()
        released = self.wait_camera_released(timeout=min(45.0, max(1.0, float(join_timeout))))
        print(
            f"sdk RealtimeSession finalize ready_for_next={int(released)} "
            f"session={self.session_id}",
            flush=True,
        )
        return self.materialize_segments()

    def clear_registry(self) -> None:
        """引擎存活时不拆共享资源。"""
        self._registry.clear_session_bindings()

    def _latest_capture_seq(self) -> int:
        store = self._registry.pose_store()
        if store is None:
            return -1
        return int(store.latest_capture_seq)

    def status(self) -> RealtimeStatus:
        store = self._registry.pose_store()
        pose_frames = len(store) if store is not None else 0
        tri_ok = 0
        if store is not None:
            cursor = -1
            while True:
                sample = store.next_pose_after(cursor)
                if sample is None:
                    break
                cursor = int(sample.capture_seq)
                if sample.tri_ok:
                    tri_ok += 1
        rate = float(tri_ok) / float(pose_frames) if pose_frames > 0 else 0.0
        gl = self._registry.gl_display()
        jpeg = self._registry.jpeg_slot()
        err = str(self.error) if self.error is not None else None
        preview_ready = gl is not None or opencv_window_ready()
        if jpeg is not None:
            data, _seq = jpeg.snapshot()
            preview_ready = preview_ready or bool(data)
        return RealtimeStatus(
            session_id=self.session_id,
            running=self.running,
            latest_capture_seq=self._latest_capture_seq(),
            pose_frames=pose_frames,
            tri_ok_rate=rate,
            preview_ready=preview_ready,
            error=err,
        )

    def drain_pose_stats(self, after_capture_seq: int) -> PoseDrainStats:
        drain = self.drain_poses(after_capture_seq)
        if not drain:
            return PoseDrainStats(
                samples=0,
                tri_ok=0,
                last_capture_seq=int(after_capture_seq),
            )
        return PoseDrainStats(
            samples=len(drain),
            tri_ok=sum(1 for p in drain if p.tri_ok),
            last_capture_seq=int(drain[-1].capture_seq),
        )

    def drain_poses(self, after_capture_seq: int) -> list[PoseFrameDTO]:
        """增量拉取 infer 后的 3D pose（流式最新；after 之后的新帧）。"""
        store = self._registry.pose_store()
        if store is None:
            return []
        cursor = int(after_capture_seq)
        out: list[PoseFrameDTO] = []
        while True:
            sample = store.next_pose_after(cursor)
            if sample is None:
                break
            cursor = int(sample.capture_seq)
            out.append(_pose_sample_to_dto(sample))
        return out

    def close_segment(
        self,
        label: str,
        start_capture_seq: int,
        end_capture_seq: int,
    ) -> SegmentMark:
        """算法识别边界：写入 segments.json 闭区间。"""
        seg = self._segments.close_with_seqs(
            label,
            int(start_capture_seq),
            int(end_capture_seq),
        )
        return SegmentMark(
            label=str(label),
            event="end",
            capture_seq=int(seg.end_capture_seq),
        )

    def mark_segment(
        self,
        label: str,
        event: Literal["start", "end"],
    ) -> SegmentMark | None:
        seq = self._latest_capture_seq()
        if seq < 0:
            seq = 0
        seg = self._segments.mark(label, event, seq)
        return SegmentMark(label=str(label), event=event, capture_seq=seq) if seg else SegmentMark(
            label=str(label),
            event=event,
            capture_seq=seq,
        )

    def segments(self) -> list[Segment]:
        return self._segments.segments

    def poll_quit(self) -> bool:
        gl = self._registry.gl_display()
        if gl is None:
            return False
        try:
            return bool(gl.poll_quit())
        except Exception:
            return False

    def gl_display(self):
        """SDK 内部：EmbeddedPreview 使用。"""
        return self._registry.gl_display()


class EmbeddedPreview:
    """嵌套预览同步（GL 或 OpenCV CPU）：屏外 → 移入 Host 量到的矩形。"""

    def __init__(self, handle: SessionHandle, *, preview: PreviewConfig | None = None) -> None:
        self._handle = handle
        self._preview = preview or handle._preview
        self._gl_showing = False
        self._gl_presented = False
        self._gl_geom: tuple[int, int, int, int] | None = None
        self._geom_stable_polls = 0
        self._align_phase = ""
        self._warmup_logged = False
        self._realign_requested = False
        self._backend = ""  # "gl" | "opencv"
        self._opencv_target: tuple[int, int, int, int] | None = None

    def reset_present_gate(self) -> None:
        """离开 vision / 首次进入时用：重新屏外预热后再显示。"""
        self._gl_showing = False
        self._gl_presented = False
        self._gl_geom = None
        self._geom_stable_polls = 0
        self._warmup_logged = False
        self._align_phase = ""
        self._realign_requested = False
        self._opencv_target = None

    def request_realign(self) -> None:
        """换页时保持实况；仅当矩形真变才改窗。"""
        self._realign_requested = True

    def hide(self) -> None:
        gl = self._handle.gl_display()
        if gl is not None:
            gl.unlock_embed()
            hide_gl(gl)
        else:
            hide_opencv()
        self.reset_present_gate()

    def sync(self, rect: ScreenRect, phase: str) -> PreviewSyncResult:
        gl = self._handle.gl_display()
        if gl is not None:
            self._backend = "gl"
            return self._sync_gl(gl, rect, phase)
        self._backend = "opencv"
        return self._sync_opencv(rect, phase)

    def _target_geom(self, rect: ScreenRect) -> tuple[int, int, int, int]:
        inset = max(0, int(self._preview.inset_px))
        return (
            int(rect.x) + inset,
            int(rect.y) + inset,
            max(64, int(rect.width) - inset * 2),
            max(64, int(rect.height) - inset * 2),
        )

    def _geom_same(
        self,
        prev: tuple[int, int, int, int] | None,
        target: tuple[int, int, int, int],
    ) -> bool:
        return (
            prev is not None
            and abs(prev[0] - target[0]) <= 4
            and abs(prev[1] - target[1]) <= 4
            and abs(prev[2] - target[2]) <= 4
            and abs(prev[3] - target[3]) <= 4
        )

    def _sync_opencv(self, rect: ScreenRect, phase: str) -> PreviewSyncResult:
        if not opencv_window_ready():
            # splash/teardown 会拆掉窗；清除 presented，等真正 UI 窗再建后重贴
            if self._gl_presented:
                self._gl_presented = False
                self._gl_showing = False
                self._geom_stable_polls = 0
                self._opencv_target = None
            return PreviewSyncResult(presented=False, warming=True, geom=self._gl_geom)

        if self._gl_presented and not self._realign_requested:
            self._align_phase = phase
            return PreviewSyncResult(
                presented=True,
                warming=False,
                geom=self._gl_geom,
            )

        if rect.width < 64 or rect.height < 64:
            return PreviewSyncResult(
                presented=self._gl_presented,
                warming=not self._gl_presented,
                geom=self._gl_geom,
            )

        # 用「目标矩形」做稳定计数，勿用 staging 应用坐标（否则永远对不上）
        target = self._target_geom(rect)
        prev_target = getattr(self, "_opencv_target", None)
        if self._geom_same(prev_target, target):
            self._geom_stable_polls += 1
        else:
            self._geom_stable_polls = 0
        self._opencv_target = target

        if self._gl_presented and self._realign_requested and self._geom_stable_polls >= 1:
            self._realign_requested = False
            self._align_phase = phase
            return PreviewSyncResult(
                presented=True,
                warming=False,
                geom=self._gl_geom,
            )

        # CPU：窗已开即可移入（稳定 1 次即可；无需 GL 预热）
        keep_live = self._gl_presented
        present_ready = keep_live or self._geom_stable_polls >= 1
        self._gl_geom = align_opencv_to_rect(
            rect,
            inset_px=max(0, int(self._preview.inset_px)),
            last_geom=self._gl_geom if keep_live else None,
            visible=present_ready,
            was_visible=keep_live,
        )
        if not present_ready:
            self._warmup_logged = True
            self._gl_showing = False
            return PreviewSyncResult(presented=False, warming=True, geom=target)

        self._gl_presented = True
        self._gl_showing = True
        self._align_phase = phase
        self._realign_requested = False
        return PreviewSyncResult(presented=True, warming=False, geom=self._gl_geom)

    def _sync_gl(self, gl, rect: ScreenRect, phase: str) -> PreviewSyncResult:
        # 已 live：intro/running/result 切换默认不改窗（避免 MoveResize 拖垮合成器）
        if self._gl_presented and not self._realign_requested:
            self._align_phase = phase
            return PreviewSyncResult(
                presented=True,
                warming=False,
                geom=self._gl_geom,
            )

        if rect.width < 64 or rect.height < 64:
            return PreviewSyncResult(
                presented=self._gl_presented,
                warming=not self._gl_presented,
                geom=self._gl_geom,
            )

        prev = self._gl_geom
        target = self._target_geom(rect)
        geom_same = self._geom_same(prev, target)
        if geom_same:
            self._geom_stable_polls += 1
        else:
            self._geom_stable_polls = 0

        if self._gl_presented and self._realign_requested and geom_same:
            self._realign_requested = False
            self._align_phase = phase
            return PreviewSyncResult(
                presented=True,
                warming=False,
                geom=self._gl_geom,
            )

        if self._gl_presented and not geom_same:
            gl.unlock_embed()

        keep_live = self._gl_presented
        renders = gl.renders_since_geom
        present_ready = keep_live or (
            renders >= int(self._preview.present_min_renders)
            and self._geom_stable_polls >= int(self._preview.present_stable_polls)
        )

        self._gl_geom = align_gl_to_rect(
            gl,
            rect,
            inset_px=max(0, int(self._preview.inset_px)),
            cover=True,
            last_geom=prev,
            visible=present_ready,
            was_visible=keep_live,
        )

        if not present_ready:
            self._warmup_logged = True
            self._gl_showing = False
            return PreviewSyncResult(presented=False, warming=True, geom=self._gl_geom)

        self._gl_presented = True
        self._gl_showing = True
        self._align_phase = phase
        self._realign_requested = False
        gl.lock_embed()
        return PreviewSyncResult(presented=True, warming=False, geom=self._gl_geom)

    @property
    def presented(self) -> bool:
        return self._gl_presented

    @property
    def geom(self) -> tuple[int, int, int, int] | None:
        return self._gl_geom

    @property
    def backend(self) -> str:
        return self._backend


def _pose_sample_to_dto(sample) -> PoseFrameDTO:
    kpts = None
    if sample.kpts3d is not None:
        arr = sample.kpts3d
        kpts = tuple(
            (
                float(arr[i, 0]),
                float(arr[i, 1]),
                float(arr[i, 2]),
                float(arr[i, 3]),
            )
            for i in range(int(arr.shape[0]))
        )
    return PoseFrameDTO(
        capture_seq=int(sample.capture_seq),
        infer_seq=int(sample.infer_seq),
        tri_ok=bool(sample.tri_ok),
        kpts3d=kpts,
    )


class RealtimeSession:
    def start(
        self,
        session_id: str,
        *,
        preview: PreviewConfig | None = None,
    ) -> SessionHandle:
        """开录一场：首次会 boot 相机+推理；再次只切 session 目录。"""
        from orchestration.live_engine import get_live_engine

        preview_cfg = preview or PreviewConfig()
        registry = get_realtime_registry()
        if preview_cfg.preview_mode == "cpu_jpeg":
            from app.sdk._bind import register_jpeg_slot
            from app.sdk.jpeg_slot import LatestJpegSlot

            # 引擎已活则复用 jpeg_slot，避免切录时闪断 /live.jpg
            if registry.jpeg_slot() is None:
                register_jpeg_slot(LatestJpegSlot())

        holder: dict[str, BaseException | None] = {"error": None}
        engine = get_live_engine()
        session_dir = CLIP_OUTPUT_ROOT / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(
                f"sdk RealtimeSession start session_id={session_id} "
                f"preview_mode={preview_cfg.preview_mode} "
                f"engine_alive={engine.is_alive()}",
                flush=True,
            )
            engine.start_recording(
                str(session_id),
                preview=preview_cfg,
                cfg_builder=_build_embedded_config,
            )
        except BaseException as exc:
            holder["error"] = exc
            print(f"sdk RealtimeSession 异常: {exc}", flush=True)
            traceback.print_exc()

        th = engine.thread
        if th is None:
            th = threading.Thread(target=lambda: None, name="sdk_live_engine_missing")
            th.start()
            th.join(timeout=0.01)
        if engine.error is not None and holder["error"] is None:
            holder["error"] = engine.error

        return SessionHandle(
            session_id=str(session_id),
            _registry=registry,
            _thread=th,
            _error_holder=holder,
            _preview=preview_cfg,
            _segments=SegmentTracker(session_dir),
        )

    def close(self, *, join_timeout: float = 60.0) -> None:
        """停引擎并真正释放 /dev/video*（进程退出或主动销毁时调用）。"""
        from orchestration.live_engine import get_live_engine

        print("sdk RealtimeSession.close：停引擎并释放相机…", flush=True)
        get_live_engine().close(join_timeout=join_timeout)


def _build_embedded_config(
    session_id: str,
    preview: PreviewConfig,
) -> Route2SessionConfig:
    from core.geometry.eye_mount_rotate import eye_mount_rotate_from_env

    parser = build_parser()
    args = parser.parse_args([])
    args.clip_ts = str(session_id)
    args.no_display = False
    args.display_width = int(preview.display_width)
    args.display_height = int(preview.display_height)
    if preview.preview_mode == "gl_embed":
        args.display_egl_gl = True
        args.cpu_preview = False
        args.embedded_preview = True
        args.cpu_jpeg_preview = False
    else:
        # 默认 cpu_jpeg：盒外 live_compose → /live.jpg，无 GL 贴窗
        args.display_egl_gl = False
        args.cpu_preview = True
        args.embedded_preview = False
        args.cpu_jpeg_preview = True
    args.no_c2_auto_stop = True
    # parse_args([]) 不会走 _validate_args 的 env 回填；SDK 路径显式读环境变量
    args.eye_rotate = eye_mount_rotate_from_env()
    return Route2SessionConfig.from_namespace(args)
