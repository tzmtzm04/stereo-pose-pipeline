"""项目根路径与默认资源。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS_YOLO26 = ROOT / "models" / "yolo26_pose"
DEFAULT_YOLO26_POSE_ENGINE = MODELS_YOLO26 / "yolo26n-pose_fp16.trt"
DEFAULT_YOLO26_POSE_ENGINE_B2 = MODELS_YOLO26 / "yolo26n-pose_fp16_b2.trt"
DEFAULT_YOLO26_POSE_ONNX = MODELS_YOLO26 / "yolo26n-pose.onnx"

MODELS_YOLO26_DET = ROOT / "models" / "yolo26_det"
DEFAULT_YOLO26_DET_ENGINE = MODELS_YOLO26_DET / "yolo26l-det_fp16.trt"
DEFAULT_YOLO26_DET_DLA_ENGINE = MODELS_YOLO26_DET / "yolo26l-det_fp16_dla.trt"
DEFAULT_YOLO26_DET_ONNX = MODELS_YOLO26_DET / "yolo26l-det.onnx"


def yolo26_det_engine_path(
    size: str = "l",
    *,
    prefer_dla: bool = False,
    precision: str = "fp16",
) -> Path:
    """YOLO26{size}-det TRT 路径（size: n/s/m/l/x）。"""
    tag = f"yolo26{size}-det_{precision}"
    gpu = MODELS_YOLO26_DET / f"{tag}.trt"
    if not prefer_dla:
        if gpu.is_file():
            return gpu
        dla = MODELS_YOLO26_DET / f"{tag}_dla.trt"
        if dla.is_file():
            return dla
        return gpu
    dla = MODELS_YOLO26_DET / f"{tag}_dla.trt"
    if dla.is_file():
        return dla
    if gpu.is_file():
        return gpu
    return dla


def resolve_yolo26_det_engine(prefer_dla: bool = False, size: str = "l") -> Path:
    """离线 det engine；默认 yolo26l GPU。"""
    p = yolo26_det_engine_path(size, prefer_dla=prefer_dla)
    if p.is_file():
        return p
    if size == "l":
        return resolve_yolo26_det_engine(prefer_dla=prefer_dla, size="s")
    if size == "s":
        return resolve_yolo26_det_engine(prefer_dla=prefer_dla, size="n")
    if size != "n":
        return resolve_yolo26_det_engine(prefer_dla=prefer_dla, size="n")
    gpu = DEFAULT_YOLO26_DET_ENGINE
    dla = DEFAULT_YOLO26_DET_DLA_ENGINE
    if prefer_dla and dla.is_file():
        return dla
    if gpu.is_file():
        return gpu
    return dla if dla.is_file() else gpu

MODELS_VITPOSE_PP = ROOT / "models" / "vitpose_plusplus"
DEFAULT_VITPOSE_PP_L_COCO_ONNX = MODELS_VITPOSE_PP / "vitposepp-l_coco_256x192.onnx"
DEFAULT_VITPOSE_PP_L_COCO_ENGINE_FP32 = (
    MODELS_VITPOSE_PP / "vitposepp-l_coco_256x192_fp32.trt"
)
DEFAULT_VITPOSE_PP_L_COCO_ENGINE_FP16 = (
    MODELS_VITPOSE_PP / "vitposepp-l_coco_256x192_fp16.trt"
)


def resolve_vitpose_pp_l_coco_engine(*, prefer_fp32: bool = True) -> Path:
    """ViTPose++-L COCO-17 engine：默认 FP32，缺失时回退 FP16。"""
    fp32 = DEFAULT_VITPOSE_PP_L_COCO_ENGINE_FP32
    fp16 = DEFAULT_VITPOSE_PP_L_COCO_ENGINE_FP16
    if prefer_fp32 and fp32.is_file():
        return fp32
    if fp16.is_file():
        return fp16
    return fp32 if prefer_fp32 else fp16


# 兼容旧名（指向 FP32 目标路径）
DEFAULT_VITPOSE_PP_L_COCO_ENGINE = DEFAULT_VITPOSE_PP_L_COCO_ENGINE_FP32
DEFAULT_VITPOSE_PP_B_COCO_ENGINE = (
    MODELS_VITPOSE_PP / "vitposepp-b_coco_256x192_fp16.trt"
)
DEFAULT_VITPOSE_PP_B_COCO_ONNX = MODELS_VITPOSE_PP / "vitposepp-b_coco_256x192.onnx"
# 离线主链默认 COCO-17（ViTPose++-L FP32）
DEFAULT_VITPOSE_PP_ENGINE = DEFAULT_VITPOSE_PP_L_COCO_ENGINE_FP32
DEFAULT_VITPOSE_PP_ONNX = DEFAULT_VITPOSE_PP_L_COCO_ONNX
DEFAULT_VITPOSE_PP_B_ENGINE = DEFAULT_VITPOSE_PP_B_COCO_ENGINE
DEFAULT_VITPOSE_PP_B_ONNX = DEFAULT_VITPOSE_PP_B_COCO_ONNX
DEFAULT_VITPOSE_PP_B_WB_ENGINE = (
    MODELS_VITPOSE_PP / "vitposepp-b_wholebody_256x192_fp16.trt"
)
DEFAULT_VITPOSE_PP_B_WB_ONNX = (
    MODELS_VITPOSE_PP / "vitposepp-b_wholebody_256x192.onnx"
)

# TAR-ViTPose（CVPR 2026，离线 stage0 试验路径）
TAR_VITPOSE_ROOT = ROOT.parent / "TARViTPose"
DEFAULT_TAR_VITPOSE_ROOT = TAR_VITPOSE_ROOT
MODELS_TAR_VITPOSE = ROOT / "models" / "tar_vitpose"
DEFAULT_TAR_VITPOSE_CONFIG = (
    TAR_VITPOSE_ROOT / "configs/posetrack17/configTARViTPoseVitB.yaml"
)
DEFAULT_TAR_VITPOSE_WEIGHTS = MODELS_TAR_VITPOSE / "tarvitpose_b_17.pt"
DEFAULT_TAR_VITPOSE_PRETRAIN = MODELS_TAR_VITPOSE / "pretrain_model_b.pth"

MODELS_DWPOSE = ROOT / "models" / "dwpose"
DEFAULT_DWPOSE_L_PTH = MODELS_DWPOSE / "dw-ll_ucoco.pth"
DEFAULT_DWPOSE_L_ONNX = MODELS_DWPOSE / "dw-ll_ucoco_256x192.onnx"
DEFAULT_DWPOSE_L_TRT = MODELS_DWPOSE / "dw-ll_ucoco_256x192_fp16.trt"


def resolve_yolo26_stereo_engine(engine_path: Path | None = None) -> Path | None:
    """若存在 batch=2 engine 则返回路径，否则 None。"""
    if engine_path is not None:
        p = Path(engine_path)
        if p.is_file() and "_b2" in p.stem:
            return p
        b2 = p.parent / f"{p.stem}_b2{p.suffix}"
        return b2 if b2.is_file() else None
    b2 = DEFAULT_YOLO26_POSE_ENGINE_B2
    return b2 if b2.is_file() else None

CALIB_DIR = ROOT / "calib"
DEFAULT_STEREO_CALIB = CALIB_DIR / "stereo_sbs.json"

BUILD_ROUTE2 = ROOT / "build" / "route2"
DEFAULT_NV12_RGB_SO = BUILD_ROUTE2 / "route2_nv12_rgb.so"

OUTPUTS = ROOT / "outputs"
CLIP_OUTPUT_ROOT = OUTPUTS / "route2_clips"

MODELS_SMOOTHNET = ROOT / "models" / "smoothnet"
DEFAULT_SMOOTHNET_2D_NPZ = MODELS_SMOOTHNET / "h36m_hrnet_2D_w16.npz"

MODELS_MUJOCO = ROOT / "models" / "mujoco"
DEFAULT_MUJOCO_COCO17_IK_MODEL = (
    MODELS_MUJOCO
    / "pose2sim_simple"
    / "Model_Pose2Sim_simple_coco17_cvt1_compat.xml"
)
DEFAULT_MUJOCO_COCO17_IK_MODEL_FALLBACK = (
    MODELS_MUJOCO / "pose2sim_simple" / "Model_Pose2Sim_simple_coco17_cvt1.xml"
)
DEFAULT_MUJOCO_WB133_IK_MODEL = MODELS_MUJOCO / "wb133_ik_humanoid.xml"
# 离线 Anipose / path6 主链默认 COCO-17 MSK IK
DEFAULT_MUJOCO_IK_MODEL = DEFAULT_MUJOCO_COCO17_IK_MODEL


def resolve_default_mujoco_ik_model() -> Path:
    """优先 compat MJCF（Jetson / 旧 MuJoCo）。"""
    if DEFAULT_MUJOCO_COCO17_IK_MODEL.is_file():
        return DEFAULT_MUJOCO_COCO17_IK_MODEL
    if DEFAULT_MUJOCO_COCO17_IK_MODEL_FALLBACK.is_file():
        return DEFAULT_MUJOCO_COCO17_IK_MODEL_FALLBACK
    if DEFAULT_MUJOCO_WB133_IK_MODEL.is_file():
        return DEFAULT_MUJOCO_WB133_IK_MODEL
    return DEFAULT_MUJOCO_COCO17_IK_MODEL
