# 推理链

描述 **capture_q 之后** 到 `session_pose.jsonl` 的路径。不含预览、PoseStore、UI 同步。

---

## 1. 数据流

```text
capture_worker
    pull NV12 SBS（3840×1080）
        ↓
    capture_q（depth = pipeline_queue_depth，默认 2）
        ↓
infer_worker.process_sbs()
    ├─ GPU letterbox（双眼 → TRT 输入尺寸）
    ├─ TRT YOLO26n-pose（parallel_eyes，双眼并行）
    ├─ 2D 后处理 + One Euro 滤波
    └─ 立体三角化 → kpts3d
        ↓
    L1PoseSessionLog.append → session_pose.jsonl
```

实现：

- `orchestration/workers/infer_worker.py`
- `core/infer/pose3d_pipeline.py` → `StereoPose3DPipeline`
- `platform/jetson/trt/` → TRT runner

---

## 2. 输入帧

| 属性 | 值 |
|------|-----|
| 布局 | NV12，SBS 3840×1080（左 1920 + 右 1920） |
| 序号 | `Nv12SbsFrame.capture_seq` |
| 来源 | GStreamer `sink_capture_sbs` appsink |

`capture_worker` 将帧送入队列；队列满时行为由 worker 策略决定（优先不堵 GStreamer）。

---

## 3. 推理配置

| 参数 | 默认 | CLI |
|------|------|-----|
| 置信度阈值 | 0.25 | `--conf` |
| One Euro | min_cutoff=1.0, beta=0.007 | `--no-2d-filter` 关闭 |
| GPU letterbox | 开（形态 B 必需） | `--no-gpu-letterbox` 会报错（VIC 路径已移除） |
| 帧内 overlap | **关** | `--infer-pipeline-overlap` |
| 融合 letterbox | 开 | `--no-infer-stereo-letterbox-fused` |
| infer 预处理 | **vic_bgr**（VIC→BGR→CPU letterbox） | `--infer-prep gpu` |

默认 **vic_bgr**：prep 线程 VIC 裁切单眼 BGR + CPU letterbox。`--infer-prep gpu` 仍为 NV12 上直接 letterbox。

标定：`--calib calib/stereo_sbs.json`，单眼分辨率须与 `eye_width × height` 一致。

---

## 4. 输出

每完成一帧 infer，向 `session_pose.jsonl` append 一行（见 [session-output.md](session-output.md)）。

内存中 infer 结果亦供进程内其它模块使用；**对外契约以 jsonl 文件为准**。

---

## 5. 计时与日志

`--log-every N`（默认 30）的打印位置取决于是否 preview：

| 模式 | 终端前缀 | 吞吐字段 | metrics |
|------|----------|----------|---------|
| 默认 preview | `ui=` | `ui_fps` / `e2e_fps` | `stage=ui_gl` |
| `--no-display` | `frame=` / `infer=` | `thr_fps` / `win_fps` | infer 分项直写 |

两种模式均可看 infer 分项（`cap`、`par`、`egl`、`lb`、`trt`、`post`）。

默认 metrics：`outputs/route2_pose3d_metrics.jsonl`（`--no-metrics` 关闭）。
