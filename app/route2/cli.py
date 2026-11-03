"""Route2 CLI 参数（形态 B 默认）。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.paths import DEFAULT_STEREO_CALIB, DEFAULT_YOLO26_POSE_ENGINE, OUTPUTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route2 YOLO26 pose → stereo 3D")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--eye-width", type=int, default=1920)
    parser.add_argument("--calib", type=Path, default=DEFAULT_STEREO_CALIB)
    parser.add_argument("--engine", type=Path, default=DEFAULT_YOLO26_POSE_ENGINE)
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO det 置信度")
    parser.add_argument(
        "--tri-conf",
        type=float,
        default=0.0,
        help="三角化 min(L,R) 2D 置信度门控（实时默认 0=不门控）",
    )
    parser.add_argument(
        "--tri-mode",
        choices=("opencv", "pose2sim"),
        default="opencv",
        help="实时三角化：opencv=DLT 直算（默认）；pose2sim=重投影质检",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=0,
        help="预览宽度（0=默认 1920，与左眼原图同尺寸）",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=0,
        help="预览高度（0=默认 1080；仅设宽时默认与宽相同）",
    )
    parser.add_argument(
        "--no-camera-overlay",
        action="store_true",
        help="预览不叠 2D 骨架（默认开启叠加）",
    )
    parser.add_argument(
        "--display-eye",
        choices=("left", "right"),
        default="left",
        help="实时预览单眼：left（默认）| right",
    )
    parser.add_argument(
        "--2d-filter",
        dest="filter_2d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="2D One Euro：三角化 + 叠图/session_pose（默认开）；关：--no-2d-filter",
    )
    parser.add_argument(
        "--filter-2d-min-cutoff",
        type=float,
        default=1.0,
        help="2D One Euro min_cutoff Hz（默认 1.0）",
    )
    parser.add_argument(
        "--filter-2d-beta",
        type=float,
        default=0.007,
        help="2D One Euro beta（默认 0.007）",
    )
    parser.add_argument(
        "--filter-min-cutoff",
        type=float,
        default=4.0,
        help="（保留）3D One Euro min_cutoff Hz；实时默认不用 3D 滤波",
    )
    parser.add_argument(
        "--filter-beta",
        type=float,
        default=1.5,
        help="3D One Euro beta（默认 1.5）",
    )
    parser.add_argument(
        "--filter-max-miss",
        type=int,
        default=6,
        help="单眼丢检时保持上一帧滤波结果的帧数",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--metrics-log",
        type=Path,
        default=OUTPUTS / "route2_pose3d_metrics.jsonl",
    )
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument("--pipeline-queue-depth", type=int, default=2)
    parser.add_argument("--capture-fail-max", type=int, default=30)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="TRT+后处理也串行（默认 GPU letterbox + TRT 并行）",
    )
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument(
        "--infer-prep",
        choices=("gpu", "vic_bgr"),
        default="vic_bgr",
        help="infer 预处理：vic_bgr=VIC 裁 BGR+CPU letterbox（默认，对齐离线）；gpu=CUDA letterbox",
    )
    parser.add_argument(
        "--gpu-letterbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="infer_prep=gpu 时 NV12 CUDA letterbox（vic_bgr 时忽略）",
    )
    parser.add_argument(
        "--cpu-preview",
        action="store_true",
        help="display 用 CPU NV12→BGR（默认 GPU CUDA 裁左眼→BGR）",
    )
    parser.add_argument(
        "--display-pose-ui-wait-ms",
        type=float,
        default=0.025,
        help="UI exact match 短等（秒，默认 0.025=25ms；超时则不叠骨架）",
    )
    parser.add_argument(
        "--display-ui-lag-frames",
        type=int,
        default=0,
        help="（可选）UI 播放滞后 N 帧；默认 0=latest+25ms 短等",
    )
    parser.add_argument(
        "--display-cuda-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="display 全幅 SBS NVDEC + CUDA 裁左眼（默认开）",
    )
    parser.add_argument(
        "--display-egl-gl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="P4：EGL+GL NV12 fd 直显（默认开）；需 build_gl_display.sh",
    )
    parser.add_argument(
        "--infer-pipeline-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="infer 双缓冲（默认开）：vic_bgr=prep 线程 VIC→BGR→LB ∥ infer TRT+post；"
        "gpu=EGL∥TRT。关：--no-infer-pipeline-overlap",
    )
    parser.add_argument(
        "--infer-stereo-letterbox-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SBS 双眼 crop+letterbox 单次 CUDA launch 写 batch[0/1]（默认开）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="跑满 N 帧后自动退出（0=不限，验收脚本用）",
    )
    parser.add_argument(
        "--session-enc-packets",
        type=int,
        default=64,
        help="session：内存 MJPEG 包环容量（默认 64）",
    )
    parser.add_argument(
        "--mjpeg-rec-queue-depth",
        type=int,
        default=64,
        help="形态 B mjpeg ingress 队列深度（默认 64）",
    )
    parser.add_argument(
        "--decoder-extra-surfaces",
        type=int,
        default=8,
        help="nvv4l2decoder NVMM 池额外 surface 数（默认 8，0=插件默认）",
    )
    parser.add_argument("--clips-file", type=Path, default=None)
    parser.add_argument("--clip-start", type=int, default=None)
    parser.add_argument("--clip-end", type=int, default=None)
    parser.add_argument("--clip-dir", type=Path, default=None)
    parser.add_argument("--no-mock-l3-detect", action="store_true")
    parser.add_argument("--mock-compute-delay-sec", type=float, default=0.5)
    parser.add_argument("--cpu-jpeg-preview", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-c2-auto-stop",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--eye-rotate",
        choices=("none", "cw90", "ccw90"),
        default=None,
        help="VIC 裁单眼后旋转（竖装：左上右下用 cw90；也可用环境变量 VISION_EYE_ROTATE）",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    args.clip_ts = time.strftime("%Y%m%d_%H%M%S")
    return args


def _validate_args(args: argparse.Namespace) -> None:
    from core.geometry.eye_mount_rotate import eye_mount_rotate_from_env

    if getattr(args, "eye_rotate", None) is None:
        args.eye_rotate = eye_mount_rotate_from_env()
    if args.decoder_extra_surfaces < 0:
        raise SystemExit("--decoder-extra-surfaces 须 >= 0")
    if args.mjpeg_rec_queue_depth < 8:
        raise SystemExit("--mjpeg-rec-queue-depth 须 >= 8")
    if args.session_enc_packets <= 0:
        raise SystemExit("--session-enc-packets 须 > 0")
    if args.clips_file is not None and (
        args.clip_start is not None or args.clip_end is not None
    ):
        raise SystemExit("--clips-file 与 --clip-start/end 二选一")
    if (args.clip_start is not None) or (args.clip_end is not None):
        if args.clip_start is None or args.clip_end is None:
            raise SystemExit("--clip-start 与 --clip-end 须同时指定")
