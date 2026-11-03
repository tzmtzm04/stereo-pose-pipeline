# stereo-pose-pipeline

Real-time binocular 3D human pose from one side-by-side (SBS) camera stream.

This is **v0.99**: a runnable Jetson snapshot. Each eye keeps the highest-confidence person (no identity tracking). A paper is in preparation; this tree is the software, not a results dump.

## What it does

- Capture 3840×1080 NV12 SBS at 60 Hz (left 1920 + right 1920)
- VIC crop per eye (optional 90° mount rotate), letterbox, dual-eye TensorRT YOLO26-pose
- Emit COCO-17 keypoints in original-image coordinates and triangulated 3D
- Write a session bundle: `session.mjpeg`, `session_index.jsonl`, `session_pose.jsonl`, `session_meta.json`

Display is optional and does not have to wait for pose.

## Requirements

NVIDIA Jetson with JetPack (GStreamer NVMM, VIC, TensorRT). Python 3.10. A stereo calibration file for your cameras.

## Setup

```bash
cd stereo-pose-pipeline
export PYTHONPATH="$(pwd)"

# 1. Weights (Ultralytics YOLO26n-pose) → ONNX → TensorRT
bash scripts/route2/download_yolo26_pose_onnx.sh
bash scripts/route2/build_yolo26_pose_trt.sh

# 2. CUDA preprocess library (if not using the bundled aarch64 .so)
bash scripts/route2/build_cuda_nv12_rgb.sh

# 3. Your stereo calibration
#    copy or write calib/stereo_sbs.json  (see calib/README.md)
```

## Run

```bash
export DISPLAY=:0
bash scripts/route2/run_camera_pose3d.sh
```

Quit with `q`. Sessions land in `outputs/route2_clips/<timestamp>/`.

Useful flags (passed through to `python3 -m app.route2.main`):

| Flag | Meaning |
|------|---------|
| `--no-display` | Headless capture + infer |
| `--eye-rotate cw90` / `ccw90` | Portrait mount |
| `--calib PATH` | Calibration JSON |
| `--engine PATH` | TensorRT engine |

SDK: `from app.sdk import RealtimeSession`. Session files: `docs/realtime/session-output.md`.

## License

Apache-2.0. Third-party notices: `NOTICE`.
