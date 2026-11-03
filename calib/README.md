# Calibration

Default live file: `calib/stereo_sbs.json` (1920×1080 per eye). The runner uses this path if you do not pass `--calib`.

Replace it with your own stereo JSON when the cameras, baseline, or resolution differ.

Required keys: `image_width`, `image_height`, `K1`, `D1`, `K2`, `D2`, `R`, `T`.
