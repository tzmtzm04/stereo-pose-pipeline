# Calibration

Place your stereo file at `calib/stereo_sbs.json`. The live runner looks for this path by default.

Required keys (OpenCV-style, one eye = 1920×1080 unless you change capture size):

- `image_width`, `image_height`
- `K1`, `D1`, `K2`, `D2`
- `R`, `T` (stereo extrinsics)

Do not commit production or venue calibration. This directory is empty on purpose.
