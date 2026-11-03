# stereo-pose-pipeline

Real-time **binocular 3D human pose** on a single side-by-side (SBS) video stream.

This repository is a public placeholder for a system paper in preparation. Code for the extractable core (OpenCV-aligned GPU letterbox, dual-eye inference overlap, triangulation) will land here. Sports scoring, venue calibration, and private recordings stay out.

## Scope

Input is one NV12 SBS frame (left | right). Output is COCO-17 keypoints in **original-image coordinates** per eye, then a triangulated 3D skeleton. Capture, preprocess, infer, and display run on separate threads. Display does not sit on the pose critical path.

**Intended claims (system, not a new detector):**

- Pipeline overlap: VIC→BGR then GPU letterbox on a prep thread, dual-eye TensorRT on an infer thread, one staging buffer because CUDA graphs pin the input pointer.
- Preprocess matches the CPU reference (`cv2.resize` INTER_LINEAR / Ultralytics LetterBox) on the same BGR image. Direct NV12 letterbox is a contrast path, not the accuracy path.
- Left and right engines run in parallel. Portrait mount is the same geometry with crop+rotate in one VIC pass.

GPU letterbox that matches OpenCV is **not** claimed as original. See [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO) and [CV-CUDA #45](https://github.com/CVCUDA/CV-CUDA/issues/45). This project applies that constraint on BGR after hardware demosaic/crop.

## Status

Work in progress. Cite this repository until the preprint is up (`CITATION.cff`).

## License

Apache-2.0. Do not vendor GPL TensorRT-YOLO sources into this tree.
