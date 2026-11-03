"""Route2 会话配置（形态 B 默认）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.paths import (
    DEFAULT_STEREO_CALIB,
    DEFAULT_YOLO26_POSE_ENGINE,
    OUTPUTS,
    ROOT,
)


@dataclass
class Route2SessionConfig:
    device: str = "/dev/video0"
    width: int = 3840
    height: int = 1080
    fps: int = 60
    eye_width: int = 1920
    calib: Path = DEFAULT_STEREO_CALIB
    engine: Path = DEFAULT_YOLO26_POSE_ENGINE
    conf: float = 0.25
    tri_conf: float = 0.0
    tri_mode: str = "opencv"
    display_width: int = 0
    display_height: int = 0
    camera_overlay: bool = True
    display_eye: str = "left"
    no_2d_filter: bool = False
    filter_2d_min_cutoff: float = 1.0
    filter_2d_beta: float = 0.007
    filter_min_cutoff: float = 4.0
    filter_beta: float = 1.5
    filter_d_cutoff: float = 1.0
    filter_max_miss: int = 6
    no_display: bool = False
    metrics_log: Path = OUTPUTS / "route2_pose3d_metrics.jsonl"
    no_metrics: bool = False
    pipeline_queue_depth: int = 2
    capture_fail_max: int = 30
    sequential: bool = False
    log_every: int = 30
    startup_timeout: float = 30.0
    session_enc_packets: int = 64
    decoder_extra_surfaces: int = 8
    clips_file: Path | None = None
    clip_start: int | None = None
    clip_end: int | None = None
    clip_dir: Path | None = None
    clip_ts: str = "manual"
    project_root: Path = ROOT
    enable_mock_detect: bool = True
    mock_compute_delay_sec: float = 0.5
    session_mjpeg_rec: bool = True
    mjpeg_rec_queue_depth: int = 64
    max_frames: int = 0
    gpu_letterbox: bool = True
    infer_prep: str = "vic_bgr"
    cpu_preview: bool = False
    display_pose_ui_wait_ms: float = 0.012
    display_ui_lag_frames: int = 0
    display_egl_gl: bool = True
    display_cuda_crop: bool = True
    infer_pipeline_overlap: bool = True
    infer_stereo_letterbox_fused: bool = True
    embedded_preview: bool = False
    cpu_jpeg_preview: bool = False
    c2_auto_stop: bool = False
    # VIC 裁眼后旋转：none|cw90|ccw90（竖装左上右下默认用 cw90，见 VISION_EYE_ROTATE）
    eye_rotate: str = "none"

    @property
    def split_display(self) -> bool:
        """推理/显示 GStreamer 分路 + PoseStore（默认开，--no-display 关）。"""
        return not self.no_display

    @property
    def mjpeg_split_capture(self) -> bool:
        """形态 B：mjpeg_tee + 单 NVDEC → NV12 sbs_tee fan-out。"""
        return self.session_mjpeg_rec

    @property
    def display_use_egl_gl(self) -> bool:
        """P4：EGL+GL NV12 fd 直显（默认开，需 route2_gl_display.so）。"""
        if (
            not self.display_egl_gl
            or self.no_display
            or self.cpu_preview
            or not self.split_display
        ):
            return False
        try:
            from core.display.hw_gl_display import hw_gl_display_available

            return hw_gl_display_available()
        except Exception:
            return False

    @property
    def display_backend_tag(self) -> str:
        if self.display_use_egl_gl:
            return "egl_gl"
        if self.cpu_preview:
            return "opencv_cpu"
        return "opencv_gpu"

    @property
    def display_vic_bgr_decode(self) -> bool:
        """display 不与 infer 争 VIC；OpenCV 预览走 GPU CUDA 裁眼。"""
        return False

    @classmethod
    def from_namespace(cls, args) -> "Route2SessionConfig":
        return cls(
            device=args.device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            eye_width=args.eye_width,
            calib=args.calib,
            engine=args.engine,
            conf=args.conf,
            tri_conf=args.tri_conf,
            tri_mode=str(args.tri_mode),
            display_width=args.display_width,
            display_height=args.display_height,
            camera_overlay=not args.no_camera_overlay,
            display_eye=str(args.display_eye),
            no_2d_filter=not bool(args.filter_2d),
            filter_2d_min_cutoff=args.filter_2d_min_cutoff,
            filter_2d_beta=args.filter_2d_beta,
            filter_min_cutoff=args.filter_min_cutoff,
            filter_beta=args.filter_beta,
            filter_max_miss=args.filter_max_miss,
            no_display=args.no_display,
            metrics_log=args.metrics_log,
            no_metrics=args.no_metrics,
            pipeline_queue_depth=args.pipeline_queue_depth,
            capture_fail_max=args.capture_fail_max,
            sequential=args.sequential,
            log_every=args.log_every,
            startup_timeout=args.startup_timeout,
            session_enc_packets=int(args.session_enc_packets),
            decoder_extra_surfaces=int(args.decoder_extra_surfaces),
            clips_file=args.clips_file,
            clip_start=args.clip_start,
            clip_end=args.clip_end,
            clip_dir=args.clip_dir,
            clip_ts=args.clip_ts,
            project_root=ROOT,
            enable_mock_detect=not args.no_mock_l3_detect,
            mock_compute_delay_sec=args.mock_compute_delay_sec,
            mjpeg_rec_queue_depth=int(args.mjpeg_rec_queue_depth),
            max_frames=int(args.max_frames or 0),
            gpu_letterbox=bool(args.gpu_letterbox),
            infer_prep=str(args.infer_prep),
            cpu_preview=bool(args.cpu_preview),
            display_pose_ui_wait_ms=float(args.display_pose_ui_wait_ms),
            display_ui_lag_frames=int(args.display_ui_lag_frames),
            display_egl_gl=bool(args.display_egl_gl),
            display_cuda_crop=bool(args.display_cuda_crop),
            infer_pipeline_overlap=bool(args.infer_pipeline_overlap),
            infer_stereo_letterbox_fused=bool(args.infer_stereo_letterbox_fused),
            embedded_preview=bool(getattr(args, "embedded_preview", False)),
            cpu_jpeg_preview=bool(getattr(args, "cpu_jpeg_preview", False)),
            c2_auto_stop=not bool(getattr(args, "no_c2_auto_stop", False)),
            eye_rotate=str(getattr(args, "eye_rotate", "none") or "none"),
        )
