# Run and build

## Default live session

```bash
export DISPLAY=:0
export PYTHONPATH="$(pwd)"
bash scripts/route2/run_camera_pose3d.sh
```

Equivalent:

```bash
python3 -m app.route2.main \
  --calib calib/stereo_sbs.json \
  --engine models/yolo26_pose/yolo26n-pose_fp16.trt \
  --device /dev/video0 \
  --width 3840 --height 1080 --fps 60 \
  --log-every 30
```

Output directory: `outputs/route2_clips/{timestamp}/`

## First-time build

```bash
bash scripts/route2/download_yolo26_pose_onnx.sh
bash scripts/route2/build_yolo26_pose_trt.sh
bash scripts/route2/build_cuda_nv12_rgb.sh
# optional GL preview library
bash scripts/route2/build_gl_display.sh
```

Put `calib/stereo_sbs.json` in place (see `calib/README.md`).

## Flags

| Flag | Role |
|------|------|
| `--clip-dir DIR` | Session output directory |
| `--no-display` | No preview window |
| `--max-frames N` | Stop after N frames |
| `--log-every N` | Log interval |
| `--no-metrics` | Disable metrics jsonl |
| `--infer-pipeline-overlap` | Overlap prep and TRT |
| `--eye-rotate cw90` / `ccw90` | Portrait mount |

Full list: `python3 -m app.route2.main --help`
