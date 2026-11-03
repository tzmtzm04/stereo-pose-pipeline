"""Route2 实时会话：启动、预热、shutdown 固定顺序（形态 B）。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TextIO

from app.route2.session_config import Route2SessionConfig
from core.geometry.calib_eye_rotate import adapt_stereo_calib_for_eye_rotate
from core.geometry.calib_loader import load_stereo_calibration
from core.infer.pose3d_pipeline import StereoPose3DPipeline
from core.infer.stereo_pipeline import StereoInferPipeline
from core.ring.session_mjpeg_recorder import MjpegIngressWorker, SessionMjpegRecordSession
from domain.grab import StereoGrabRuntime
from domain.l1.runtime import L1Runtime, SessionMeta
from domain.l3.params import create_l3_runtime
from domain.l3.runtime import L3Runtime
from orchestration.display_ui import (
    check_display_env,
    resolve_display_size,
    show_startup_splash,
    teardown_display_window,
)
from orchestration.metrics import open_metrics_file
from orchestration.pipeline_loop import run_pipeline_loop
from orchestration.shutdown_util import run_with_timeout
from platform.jetson.capture import JetsonGstNvmmCapture, gst_nvmm_available
from platform.jetson.gst_capture import (
    device_held_by_current_process,
    force_close_device_fds,
)


def _tag_session_meta_preview(session_dir: Path, preview_mode: str) -> None:
    """写入 realtime_preview_mode（可视化在盒外，仅 meta 标注）。"""
    import json

    meta_path = Path(session_dir) / "session_meta.json"
    if not meta_path.is_file():
        return
    try:
        body = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        body = {}
    body["realtime_preview_mode"] = str(preview_mode)
    meta_path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Route2LiveSession:
    def __init__(self, cfg: Route2SessionConfig, *, persistent: bool = False) -> None:
        self.cfg = cfg
        self._persistent = bool(persistent)
        self._cap: JetsonGstNvmmCapture | None = None
        self._pose3d: StereoPose3DPipeline | None = None
        self._l1: L1Runtime | None = None
        self._l3: L3Runtime | None = None
        self._metrics_fp: TextIO | None = None
        self._grab_runtime: StereoGrabRuntime | None = None
        self._mjpeg_ingress_worker: MjpegIngressWorker | None = None
        # persistent：表示「可开下一场」；one-shot：表示相机 fd 已释放
        self._camera_released = threading.Event()
        self._pose3d_closed = False
        self._rec_lock = threading.Lock()
        self._recording = False
        self._recording_sid: str | None = None
        self._preview_mode = "cpu_jpeg"
        self._booted = threading.Event()

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def recording_sid(self) -> str | None:
        return self._recording_sid

    def is_recording(self, session_id: str | None = None) -> bool:
        if not self._recording:
            return False
        if session_id is None:
            return True
        return self._recording_sid == str(session_id)

    def run(self) -> None:
        cfg = self.cfg
        if not gst_nvmm_available():
            raise SystemExit("NVMM 不可用（需要 gi + libnvbufsurface）")
        if not cfg.engine.is_file():
            raise SystemExit(f"缺少 engine: {cfg.engine}")

        # Host 快停：等此 event（persistent=可开下一场；否则=相机已释放）
        from app.sdk._registry import get_realtime_registry

        reg = get_realtime_registry()
        self._camera_released.clear()
        reg.bind_camera_released(self._camera_released)
        if self._persistent:
            reg.bind_end_recording(self.end_recording)
            reg.set_engine_alive(True)

        calib = load_stereo_calibration(cfg.calib)
        eye_rotate = str(getattr(cfg, "eye_rotate", "none") or "none")
        if eye_rotate != "none":
            # 横幅标定 → 竖图关键点系（K/R/T；D 暂置零）
            calib = adapt_stereo_calib_for_eye_rotate(calib, eye_rotate)
            print(
                f"[*] eye_rotate={eye_rotate}：infer VIC 裁转竖图；"
                f"标定已变换为 {calib.image_width}x{calib.image_height} "
                f"（K/R/T 适配，D 保留原参数）",
                flush=True,
            )
        expect_w, expect_h = int(cfg.eye_width), int(cfg.height)
        if eye_rotate != "none":
            # upright 单眼：宽=原高，高=原宽
            expect_w, expect_h = int(cfg.height), int(cfg.eye_width)
        if (calib.image_width, calib.image_height) != (expect_w, expect_h):
            print(
                f"[!] 标定分辨率 {calib.image_width}x{calib.image_height} "
                f"与单眼期望 {expect_w}x{expect_h} 不一致",
                flush=True,
            )

        check_display_env(cfg.no_display)
        # embedded / cpu_jpeg：禁止 splash 抢开 OpenCV「route2_pose3d」窗（Kiosk 只走 /live.jpg）
        _splash_off = bool(
            cfg.no_display or cfg.embedded_preview or cfg.cpu_jpeg_preview
        )
        show_startup_splash("打开 NVMM 采集...", no_display=_splash_off)
        _check_capture_device(cfg.device)

        print("打开 NVMM 采集...", flush=True)
        ring_depth = 2
        cap = JetsonGstNvmmCapture(
            camera=cfg.device,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            sbs_mode="both",
            ring_depth=max(2, int(ring_depth)),
            decoder_extra_surfaces=cfg.decoder_extra_surfaces,
            mjpeg_split=True,
            display_sbs_branch=cfg.split_display,
            display_full_sbs=cfg.split_display
            and (cfg.display_cuda_crop or cfg.display_vic_bgr_decode),
            mjpeg_rec_queue_depth=cfg.mjpeg_rec_queue_depth,
        )
        self._cap = cap

        if cfg.split_display and cap.sink_display_left is None:
            raise SystemExit("split_display 需要 GStreamer display tee（sink_display_left）")

        self._grab_runtime = StereoGrabRuntime(cap)

        show_startup_splash("加载 TRT + 预热…", no_display=_splash_off)
        print("加载 TRT + 预热…", flush=True)
        infer = StereoInferPipeline(
            cfg.engine,
            conf_threshold=cfg.conf,
            use_egl_direct=True,
            gpu_letterbox=cfg.gpu_letterbox,
            infer_prep=cfg.infer_prep,
            stereo_letterbox_fused=cfg.infer_stereo_letterbox_fused,
            eye_rotate=str(getattr(cfg, "eye_rotate", "none") or "none"),
        )
        pose3d = StereoPose3DPipeline(
            infer,
            calib,
            min_conf=cfg.tri_conf,
            tri_mode=cfg.tri_mode,  # type: ignore[arg-type]
            use_2d_filter=not cfg.no_2d_filter,
            use_3d_filter=False,
            filter_2d_min_cutoff=cfg.filter_2d_min_cutoff,
            filter_2d_beta=cfg.filter_2d_beta,
            filter_max_miss=cfg.filter_max_miss,
            fps=cfg.fps,
        )
        self._pose3d = pose3d

        mode = (
            "nvdec_vic_bgr_cpu"
            if getattr(infer, "_use_vic_bgr", False)
            else (
                "egl_cuda_gpu_letterbox"
                if infer.gpu_letterbox
                else "egl_cuda_direct"
            )
        )
        if getattr(infer, "_use_vic_bgr", False):
            print(
                "  infer: prep 路 VIC→BGR→letterbox | infer 路 TRT+post（vic_bgr_cpu）",
                flush=True,
            )
            if cfg.infer_pipeline_overlap:
                print(
                    "  infer pipeline: prep 线程 VIC→BGR→LB (latest) → prep_q "
                    "∥ infer 线程 TRT+post (prep 信号唤醒，取最新)",
                    flush=True,
                )
            else:
                print(
                    "  infer pipeline: 同步 VIC→BGR→LB→TRT（--no-infer-pipeline-overlap）",
                    flush=True,
                )
        elif infer.gpu_letterbox and cfg.split_display:
            if cfg.infer_pipeline_overlap:
                print(
                    "infer pipeline overlap: prep(N+1) EGL ∥ TRT(N) letterbox+graph",
                    flush=True,
                )
            else:
                print(
                    "infer pipeline sync: EGL+letterbox+graph @ trt stream（无 overlap）",
                    flush=True,
                )
        if infer.stereo_letterbox_fused and infer.gpu_letterbox:
            print(
                "infer stereo letterbox fused: 单次 CUDA launch → TRT batch[0/1]",
                flush=True,
            )
        metrics_fp = open_metrics_file(
            cfg,
            calib,
            capture=f"nvmm_{mode}",
            mode=mode,
        )
        self._metrics_fp = metrics_fp

        l3 = create_l3_runtime(
            fps=cfg.fps,
            width=cfg.width,
            height=cfg.height,
            clip_start=cfg.clip_start,
            clip_end=cfg.clip_end,
            clips_file=cfg.clips_file,
            clip_dir=cfg.clip_dir,
            project_root=cfg.project_root,
            clip_ts=cfg.clip_ts,
            enable_mock_detect=cfg.enable_mock_detect,
            mock_compute_delay_sec=cfg.mock_compute_delay_sec,
            defer_segment_detect=True,
        )
        self._l3 = l3

        l1 = L1Runtime(l3.params, fps=cfg.fps)
        l1.set_session_meta(
            SessionMeta(
                width=cfg.width,
                height=cfg.height,
                calib_path=cfg.calib,
                device=cfg.device,
                session_id=cfg.clip_ts,
            )
        )
        self._l1 = l1
        l3.bind_l1(l1)

        _print_banner(cfg, calib, l3, l1, mode, metrics_fp)

        self._preview_mode = (
            "cpu_jpeg"
            if cfg.cpu_jpeg_preview
            else ("gl_embed" if cfg.embedded_preview and cfg.display_use_egl_gl else "opencv")
        )
        ingress_worker = MjpegIngressWorker(
            l1.submit_mjpeg_ingress,
            cap.mjpeg_buffer_payload,
            queue_depth=cfg.mjpeg_rec_queue_depth,
        )
        ingress_worker.start()
        self._mjpeg_ingress_worker = ingress_worker

        # 首场录制（SDK persistent 也可随后 begin_recording 切 sid）
        self._arm_recording(str(cfg.clip_ts))

        # ensure_playing / 首帧失败时也必须 release，否则 /dev/video0 一直 busy
        try:
            cap.ensure_playing()
            print(f"  {l1.status_line()}", flush=True)

            show_startup_splash("等待摄像头首帧...", no_display=_splash_off)
            first_pose = _wait_first_pose(cfg, pose3d, self._grab_runtime)
            if first_pose is None:
                raise SystemExit(
                    f"{cfg.startup_timeout:.0f}s 内无法读到帧。"
                    f"检查 {cfg.device}、MJPEG 3840×1080、是否被占用。"
                )

            if cfg.cpu_jpeg_preview:
                teardown_display_window()

            self._booted.set()
            run_pipeline_loop(
                cfg,
                cap,
                pose3d,
                metrics_fp,
                first_pose,
                l1=l1,
                l3=l3,
                grab_runtime=self._grab_runtime,
                session_dir=l3.params.clip_dir,
            )
        finally:
            if self._recording:
                try:
                    self.end_recording()
                except Exception as exc:
                    print(f"WARN: shutdown end_recording: {exc}", flush=True)
            self._shutdown(l1, l3, cap, pose3d, metrics_fp)
            if self._persistent:
                from app.sdk._registry import get_realtime_registry

                get_realtime_registry().set_engine_alive(False)
                get_realtime_registry().bind_end_recording(None)

    def _arm_recording(self, session_id: str) -> None:
        """Boot 时挂上首场录制（L3 clip_dir 已按 cfg 建好）。"""
        l1 = self._l1
        l3 = self._l3
        cap = self._cap
        ingress = self._mjpeg_ingress_worker
        if l1 is None or l3 is None or cap is None or ingress is None:
            raise RuntimeError("recording arm: session 未 boot")
        sid = str(session_id)
        l1.set_session_meta(
            SessionMeta(
                width=self.cfg.width,
                height=self.cfg.height,
                calib_path=self.cfg.calib,
                device=self.cfg.device,
                session_id=sid,
            )
        )
        l1.start_session_mjpeg_recording(
            session_mjpeg_packets=self.cfg.session_enc_packets,
            mjpeg_queue_depth=self.cfg.mjpeg_rec_queue_depth,
        )
        # 勿把共享 ingress 挂到 session：session.stop() 会误停 worker
        if l3.params.clip_dir is not None:
            _tag_session_meta_preview(l3.params.clip_dir, self._preview_mode)
        cap.set_mjpeg_ingress_recorder(ingress.enqueue)
        self._recording = True
        self._recording_sid = sid
        self._camera_released.clear()
        from app.sdk._registry import get_realtime_registry

        get_realtime_registry().set_recording_sid(sid)
        print(f"route2 开录 session={sid}（相机常开切录={int(self._persistent)}）", flush=True)

    def begin_recording(self, session_id: str) -> None:
        """引擎已跑：切到新 session 目录开录（不 reopen 相机）。"""
        with self._rec_lock:
            if self._recording:
                if self._recording_sid == str(session_id):
                    return
                self._end_recording_unlocked()
            l1 = self._l1
            l3 = self._l3
            cap = self._cap
            ingress = self._mjpeg_ingress_worker
            if l1 is None or l3 is None or cap is None or ingress is None:
                raise RuntimeError("begin_recording: 引擎未就绪")
            sid = str(session_id)
            from app.paths import CLIP_OUTPUT_ROOT

            session_dir = CLIP_OUTPUT_ROOT / sid
            l3.switch_session_dir(session_dir, enable_mock_detect=True)
            self.cfg.clip_ts = sid
            self._arm_recording(sid)

    def end_recording(self) -> None:
        """结束本场录制并 signal 可开下一场；不释放相机。"""
        with self._rec_lock:
            self._end_recording_unlocked()

    def _end_recording_unlocked(self) -> None:
        if not self._recording:
            if not self._camera_released.is_set():
                self._camera_released.set()
            return
        l1 = self._l1
        l3 = self._l3
        cap = self._cap
        sid = self._recording_sid
        print(f"route2 停录 session={sid}（保留相机）…", flush=True)
        try:
            if cap is not None:
                cap.set_mjpeg_ingress_recorder(None)
            if l1 is not None:
                try:
                    session = l1._mjpeg_session
                    if isinstance(session, SessionMjpegRecordSession):
                        session.ingress_worker = None
                except Exception:
                    pass
            if l3 is not None and l1 is not None:
                segments = l3.finalize_session(l1)
                if segments:
                    print(
                        f"L3 post_process: 登记 {len(segments)} 段 → manifest",
                        flush=True,
                    )
                    for start, end, label in segments:
                        print(f"  [{start}..{end}] label={label}", flush=True)
            session_summary = {}
            if l1 is not None:
                session_summary = l1.finalize_session_recording() or {}
                l1.release_ring_buffers()
            if session_summary:
                print(
                    f"session 录制 → {session_summary.get('session_dir')} "
                    f"pkts={session_summary.get('mjpeg_packets', 0)} "
                    f"pose={session_summary.get('pose_rows', 0)} "
                    f"duration={session_summary.get('duration_sec', 0)}s",
                    flush=True,
                )
        except Exception as exc:
            print(f"WARN: end_recording 失败: {exc}", flush=True)
        finally:
            self._recording = False
            self._recording_sid = None
            from app.sdk._registry import get_realtime_registry

            get_realtime_registry().set_recording_sid(None)
            self._camera_released.set()
            print("route2 停录完成（可开下一段；相机仍占用）", flush=True)

    def _try_release_camera(self, cap: JetsonGstNvmmCapture) -> bool:
        try:
            if hasattr(cap, "prepare_shutdown"):
                cap.prepare_shutdown()
            return bool(cap.release())
        except Exception as exc:
            print(f"route2 cap.release 失败: {exc}", flush=True)
            return False

    def _signal_camera_released_if_free(self, cap: JetsonGstNvmmCapture) -> bool:
        """one-shot：仅在 V4L2 fd 已关闭时 set event。"""
        if device_held_by_current_process(cap._device):
            return False
        if not self._camera_released.is_set():
            self._camera_released.set()
        return True

    def _release_camera_early(
        self,
        cap: JetsonGstNvmmCapture,
        *,
        pose3d: StereoPose3DPipeline | None = None,
    ) -> bool:
        # 相机释放不能等 pose3d.close（TRT 长录后常 >8s）；与 release 并行
        if pose3d is not None and not self._pose3d_closed:

            def _close_pose3d() -> None:
                try:
                    pose3d.close()
                except Exception as exc:
                    print(f"WARN: pose3d.close 异常: {exc}", flush=True)
                finally:
                    self._pose3d_closed = True

            threading.Thread(
                target=_close_pose3d,
                name="shutdown_pose3d",
                daemon=True,
            ).start()

        for attempt in range(1, 4):
            released = self._try_release_camera(cap)
            if released and self._signal_camera_released_if_free(cap):
                print(
                    "route2 相机已释放（shutdown early，可开下一段）"
                    + (f" attempt={attempt}" if attempt > 1 else ""),
                    flush=True,
                )
                return True
            if attempt < 3:
                time.sleep(0.3)

        if not self._try_release_camera(cap):
            print(
                "route2 相机 release 未完成，camera_released 未 signal",
                flush=True,
            )
        return False

    def _shutdown(
        self,
        l1: L1Runtime,
        l3: L3Runtime,
        cap: JetsonGstNvmmCapture,
        pose3d: StereoPose3DPipeline,
        metrics_fp: TextIO | None,
    ) -> None:
        teardown_display_window()
        ring_stats = None
        released_ok = False
        # 引擎退出：停 ingress + 放相机（录制应已在 end_recording 封盘）
        try:
            try:
                if self._recording:
                    ring_stats = l1.ring_session.stats()
            except RuntimeError:
                pass

            cap.set_mjpeg_ingress_recorder(None)
            ingress = self._mjpeg_ingress_worker
            self._mjpeg_ingress_worker = None
            if ingress is not None:
                try:
                    ingress.stop()
                except Exception as exc:
                    print(f"mjpeg ingress stop 失败: {exc}", flush=True)

            if self._recording:
                try:
                    session = l1._mjpeg_session
                    if isinstance(session, SessionMjpegRecordSession):
                        session.ingress_worker = None
                except Exception:
                    pass
                segments = l3.finalize_session(l1)
                if segments:
                    print(
                        f"L3 post_process: 登记 {len(segments)} 段 → manifest",
                        flush=True,
                    )
                session_summary = l1.finalize_session_recording()
                if session_summary:
                    print(
                        f"session 录制 → {session_summary.get('session_dir', l1.l3_params.clip_dir)}",
                        flush=True,
                    )
                l1.release_ring_buffers()
                self._recording = False

            if self.cfg.c2_auto_stop and l1.l3_params.clip_dir is not None:
                from app.sdk._finalize import materialize_motion_segments
                from app.sdk._segments import load_session_segments

                algo_segs = load_session_segments(l1.l3_params.clip_dir)
                if algo_segs:
                    result = materialize_motion_segments(
                        l1.l3_params.clip_dir,
                        algo_segs,
                    )
                    print(
                        f"C2 auto-stop finalize: segments={len(result.motion_segments)} "
                        f"→ {result.motion_segments_json}",
                        flush=True,
                    )

            print("route2 shutdown: 释放相机（引擎退出）…", flush=True)
            released_ok = self._release_camera_early(cap, pose3d=pose3d)
        except Exception as exc:
            print(f"L1 结束处理失败: {exc}", flush=True)
            released_ok = self._release_camera_early(cap, pose3d=pose3d)

        print(f"L1 结束: {l1.status_line(ring_stats=ring_stats)}", flush=True)
        l1.stop()
        l3.stop()
        if ring_stats is not None:
            print(f"session 结束: {ring_stats}", flush=True)
        if not released_ok or device_held_by_current_process(cap._device):
            force_close_device_fds(cap._device)
            if not device_held_by_current_process(cap._device):
                print("route2 shutdown: 强制关闭 video fd 成功", flush=True)
                if not self._camera_released.is_set():
                    self._camera_released.set()
            else:
                print(
                    "route2 shutdown: 仍持有 video fd（Host 需重启进程）",
                    flush=True,
                )
        if not self._pose3d_closed:
            run_with_timeout("pose3d.close", pose3d.close, timeout_sec=5.0)
            self._pose3d_closed = True
        if metrics_fp is not None:
            metrics_fp.close()
            print(f"metrics → {self.cfg.metrics_log}", flush=True)


def _check_capture_device(device: str) -> None:
    p = Path(device)
    if not p.exists():
        print(f"[!] 采集设备不存在: {device}", flush=True)


def _wait_first_pose(
    cfg: Route2SessionConfig,
    pose3d: StereoPose3DPipeline,
    grab_runtime: StereoGrabRuntime,
):
    print(f"等待摄像头首帧（最多 {cfg.startup_timeout:.0f}s）...", flush=True)
    t0 = time.perf_counter()
    attempt = 0
    while time.perf_counter() - t0 < cfg.startup_timeout:
        attempt += 1
        t_read = time.perf_counter()
        ok, grab_frame = grab_runtime.grab()
        pull_ms = (time.perf_counter() - t_read) * 1000.0
        if not ok or grab_frame is None:
            elapsed = time.perf_counter() - t0
            print(
                f"  读帧失败 #{attempt} ({elapsed:.1f}s)，"
                f"本次等待 {pull_ms:.0f}ms，重试...",
                flush=True,
            )
            continue
        try:
            if grab_frame.sbs is None:
                grab_frame.release()
                continue
            sbs, pull_ms = grab_frame.consume_sbs()
            pose = pose3d.process_sbs(
                sbs,
                pull_ms,
                parallel_eyes=False,
                release_sbs=False,
            )
            try:
                sbs.release()
            except Exception:
                pass
        except Exception as exc:
            grab_frame.release()
            elapsed = time.perf_counter() - t0
            print(
                f"  grab/推理失败 #{attempt} ({elapsed:.1f}s): {exc}",
                flush=True,
            )
            continue
        finally:
            grab_frame.release()

        if pose is not None:
            print(f"首帧 OK（第 {attempt} 次尝试，本次 {pull_ms:.0f}ms）", flush=True)
            return pose
        elapsed = time.perf_counter() - t0
        print(
            f"  推理失败 #{attempt} ({elapsed:.1f}s)，重试...",
            flush=True,
        )
    return None


def _print_banner(cfg, calib, l3, l1, mode, metrics_fp) -> None:
    cw, ch = resolve_display_size(cfg)
    print("路线 2 3D — NVMM NV12 + TRT + 三角化（形态 B）", flush=True)
    print(f"  标定: {cfg.calib}  基线 {calib.baseline_m * 1000:.1f} mm", flush=True)
    print(
        f"  采集: mjpeg_tee ingress + NVDEC fan-out  mode={mode}",
        flush=True,
    )
    if int(cfg.decoder_extra_surfaces) > 0:
        print(
            f"  NVMM 池: nvv4l2decoder num-extra-surfaces={cfg.decoder_extra_surfaces}",
            flush=True,
        )
    sched = (
        "五段流水线（ZED grab + infer|display tee 分路 + PoseStore）"
        if cfg.split_display
        else "三线程（ZED grab + infer，--no-display）"
    )
    print(f"  调度: {sched}", flush=True)
    infer_line = (
        "  推理: VIC 裁切 BGR + CPU letterbox（prep 线程）+ TRT 双眼并行（infer）"
        if str(getattr(cfg, "infer_prep", "vic_bgr")) == "vic_bgr"
        else "  推理: GPU CUDA 全幅 SBS 裁半 + letterbox + TRT 双眼并行"
    )
    print(infer_line, flush=True)
    print("  配对: 左右各取最高置信度（单人）", flush=True)
    if cfg.no_2d_filter:
        print("  2D 滤波: 无（--no-2d-filter）", flush=True)
    else:
        print(
            f"  2D 滤波: One Euro（三角化 + 叠图/session_pose）"
            f" min_cutoff={cfg.filter_2d_min_cutoff} "
            f"beta={cfg.filter_2d_beta} max_miss={cfg.filter_max_miss}",
            flush=True,
        )
    if cfg.no_display:
        print("  预览: 关闭（--no-display）", flush=True)
    elif cfg.split_display:
        overlay_tag = " +2D骨架（live_compose 外置）" if cfg.camera_overlay else ""
        if cfg.display_use_egl_gl:
            prev_tag = "EGL+GL NV12 fd 直显"
            ui_tag = "overlay/EGL swap"
            slot_tag = "prod=NV12 fd / ui=pose→video ring"
        elif cfg.cpu_preview:
            if cfg.display_vic_bgr_decode:
                prev_tag = "VIC ROI→BGR（与 infer 同解码）"
            else:
                prev_tag = "CPU NV12→BGR（--cpu-preview）"
            ui_tag = "overlay/imshow"
            slot_tag = "prod=纯视频 seq+bgr / ui=pose→video ring"
        else:
            prev_tag = "GPU EGL/CUDA 全幅 SBS 裁左眼"
            ui_tag = "overlay/imshow"
            slot_tag = "prod=纯视频 seq+bgr / ui=pose→video ring"
        dsp_src = (
            "display NV12 tee 全幅 SBS"
            if cfg.display_cuda_crop or cfg.display_vic_bgr_decode
            else "display NV12 tee 左眼"
        )
        sync_tag = (
            f" UI pose→video（video_wait={cfg.display_pose_ui_wait_ms * 1000:.0f}ms，"
            f"顺序追帧叠同 seq exact；infer 空窗直显原画；落后>6f catch-up）"
        )
        if cfg.display_ui_lag_frames > 0:
            sync_tag += f" ui_lag={cfg.display_ui_lag_frames}f"
        print(
            f"  预览: {cw}x{ch} {dsp_src} → {prev_tag} → latest_slot"
            f"（{slot_tag}）"
            f" → {ui_tag}{sync_tag}"
            f"{overlay_tag}",
            flush=True,
        )
        if cfg.display_egl_gl and not cfg.display_use_egl_gl:
            print(
                "  [!] --display-egl-gl 已开但 route2_gl_display.so 不可用，"
                "回退 OpenCV imshow",
                flush=True,
            )
    print(
        f"  录像: mjpeg grab ingress → session.mjpeg → {l3.params.clip_dir}",
        flush=True,
    )
    if cfg.c2_auto_stop and cfg.split_display:
        print(
            "  C2: 髋角切段 auto-stop（infer 写 pose 时同步分析 → 停录）",
            flush=True,
        )
    if metrics_fp is not None:
        print(f"  metrics: {cfg.metrics_log}", flush=True)
    tri_gate = (
        f"tri_conf={cfg.tri_conf}"
        if float(cfg.tri_conf) > 0.0
        else "无 conf 门控"
    )
    print(
        f"  L1: NVMM NV12 → GPU/TRT → One Euro 2D → {cfg.tri_mode} 三角化（{tri_gate}）→ 3D",
        flush=True,
    )
    print(
        f"  {l3.params.status_line(width=cfg.width, height=cfg.height, fps=cfg.fps)}",
        flush=True,
    )
    print(f"  {l3.display_status_suffix()}", flush=True)
    if l3.params.clip_dir is not None:
        print(f"  输出目录: {l3.params.clip_dir}", flush=True)
    print(
        "  落盘: session.mjpeg + session_pose.jsonl；L3 段 manifest post_process",
        flush=True,
    )
    print("  按 q 退出\n", flush=True)
