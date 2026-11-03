"""prep 路：capture_q → VIC→BGR→letterbox → prep_q + prep_ready 信号。"""

from __future__ import annotations

import queue
import threading
import traceback
from collections import deque

from app.route2.session_config import Route2SessionConfig
from core.infer.stereo_pipeline import StereoInferPipeline
from orchestration.constants import ROLLING_N
from orchestration.queue_util import put_latest_to_queue, take_latest_from_queue
from core.infer.vic_bgr_stereo import VicBgrInferPrepItem
from orchestration.vic_bgr_pipeline import vic_bgr_prep_pipeline_enabled


def _release_capture_item(item: object) -> None:
    if isinstance(item, tuple) and item:
        sbs = item[0]
        if hasattr(sbs, "release"):
            try:
                sbs.release()
            except Exception:
                pass


def run_vic_bgr_prep_worker(
    cfg: Route2SessionConfig,
    infer_pipeline: StereoInferPipeline,
    capture_q: queue.Queue,
    prep_q: queue.Queue,
    prep_ready: threading.Event,
    stop: threading.Event,
) -> None:
    vic_hist: deque[float] = deque(maxlen=ROLLING_N)
    lb_hist: deque[float] = deque(maxlen=ROLLING_N)
    cap_skip_hist: deque[int] = deque(maxlen=ROLLING_N)
    prep_drop_hist: deque[int] = deque(maxlen=ROLLING_N)
    prep_count = 0

    while not stop.is_set():
        got = take_latest_from_queue(
            capture_q,
            timeout=0.2,
            release=_release_capture_item,
        )
        if got is None:
            continue
        item, cap_skipped = got
        if not isinstance(item, tuple) or len(item) != 2:
            _release_capture_item(item)
            continue
        sbs, capture_ms = item
        full = getattr(sbs, "full", None)
        if full is None or getattr(full, "dmabuf_fd", None) is None:
            _release_capture_item(item)
            continue

        from contracts.seq_util import int_attr

        full_h = int_attr(sbs, "height", default=int(full.height))
        full_w = int_attr(sbs, "width", default=int(full.width))
        eye_w = full_w // 2 if full_w > 0 else 1920
        capture_seq = int_attr(sbs, "capture_seq", default=-1)

        try:
            vic = infer_pipeline._ensure_vic_bgr(eye_w=eye_w, eye_h=full_h)
            bgr_prep = vic.prep_sbs_nv12_letterbox(full, capture_seq=capture_seq)
        except Exception as exc:
            _release_capture_item(item)
            print(f"prep 异常（跳过本帧 seq={capture_seq}）: {exc}", flush=True)
            if prep_count % cfg.log_every == 0:
                traceback.print_exc()
            continue
        finally:
            try:
                sbs.release()
            except Exception:
                pass

        prep_item = VicBgrInferPrepItem(
            prep=bgr_prep,
            capture_ms=float(capture_ms),
            capture_skipped=int(cap_skipped),
        )
        dropped = put_latest_to_queue(prep_q, prep_item)
        prep_ready.set()
        prep_count += 1
        vic_hist.append(float(bgr_prep.vic_ms))
        lb_hist.append(float(bgr_prep.lb_ms))
        cap_skip_hist.append(int(cap_skipped))
        prep_drop_hist.append(int(dropped))

        if prep_count % cfg.log_every == 0:
            vic_avg = sum(vic_hist) / len(vic_hist) if vic_hist else 0.0
            lb_avg = sum(lb_hist) / len(lb_hist) if lb_hist else 0.0
            cap_skip_avg = sum(cap_skip_hist) / len(cap_skip_hist) if cap_skip_hist else 0.0
            prep_drop_avg = sum(prep_drop_hist) / len(prep_drop_hist) if prep_drop_hist else 0.0
            print(
                f"prep={prep_count} vic={vic_avg:.1f}ms lb={lb_avg:.1f}ms "
                f"cap_skip={cap_skip_avg:.1f} prep_q_drop={prep_drop_avg:.1f} "
                f"seq={capture_seq}",
                flush=True,
            )
