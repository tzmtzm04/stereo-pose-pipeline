# 流水线架构

**形态 B** 为唯一实现（`session_mjpeg_rec` 默认开）。
黑盒含 **落盘 + DisplayVideoRing + PoseStore**；不含 UI 叠图。

---

## 1. 总览

```text
                    ┌──► MjpegIngressWorker ──► session.mjpeg + index
                    │
v4l2src ─ mjpeg_tee ┤
                    │
                    └──► NVDEC ─ sbs_tee ─┬─► capture_q ─ infer ─┬─► session_pose.jsonl
                                          │              └──────► PoseStore
                                          │
                                          └─► display_producer ──► DisplayVideoRing
                                                                        │
                                                            黑盒外 UI ◄──┘
                                                            （叠图/窗口）
```

---

## 2. GStreamer 层

```text
v4l2src  (/dev/video0, MJPEG 3840×1080@60)
    ↓
tee name=mjpeg_tee
    │
    ├─ [probe] → MjpegIngressWorker → session 落盘
    │
    └─ nvv4l2decoder（单次 NVDEC）
            ↓
        tee name=sbs_tee
            ├─ sink_capture_sbs   → infer 支路
            └─ sink_display_*     → display_producer → DisplayVideoRing
```

实现：`platform/jetson/gst_nvmm_capture.py`

---

## 3. 应用层线程

```text
route2_mjpeg_ingress          → session.mjpeg + session_index.jsonl

route2_capture                → capture_q（内部，depth=2）
    └─ route2_infer_consumer
           letterbox → TRT → 三角化
           ├─ session_pose.jsonl
           └─ PoseStore.put_from_pose()

route2_display_producer       → DisplayVideoRing.push(DisplayTick)
                                （capture_seq + 原画；默认 NV12 fd，不含 pose）
```

编排：`orchestration/pipeline_loop.py`  
显示环：`core/display/video_ring.py`  
Pose 环：`core/display/pose_store.py`

黑盒外：`ui/*` 从 VideoRing + PoseStore 按 `capture_seq` 取数据并叠图。

---

## 4. 关键配置

| 配置 | 默认 | 作用 |
|------|------|------|
| `session_mjpeg_rec` | **开** | MJPEG ingress 录像 |
| `split_display` | session 时 **开** | 分路 + VideoRing + PoseStore |
| `pipeline_queue_depth` | 2 | `capture_q` 深度 |
| `infer_pipeline_overlap` | **关** | 帧内 prep∥TRT |

---

## 5. 启动与退出

`orchestration/session.py` → `Route2LiveSession.run()`：

1. 加载标定、构建 GStreamer pipeline
2. 注册 session 录像 + 启动 ingress
3. 创建 `PoseStore`；启动 capture / infer / display_producer
4. UI 线程消费 VideoRing + PoseStore（黑盒外）
5. 退出 → 写 `session_meta.json`；关闭 ring / store

---

## 6. 接口说明

见 [blackbox-interface.md](blackbox-interface.md)。
