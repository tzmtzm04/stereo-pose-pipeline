# YOLO26-pose

Bundled live engines (YOLO26n-pose, TensorRT FP16):

- `yolo26n-pose_fp16.trt` — default
- `yolo26n-pose_fp16_b2.trt` — batch 2

Rebuild on another device:

```bash
bash scripts/route2/download_yolo26_pose_onnx.sh
bash scripts/route2/build_yolo26_pose_trt.sh
```
