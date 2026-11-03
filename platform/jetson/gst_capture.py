"""GStreamer 基础工具。"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

GST_PLAY_TIMEOUT_SEC = 30.0
GST_STOP_TIMEOUT_SEC = float(os.environ.get("ROUTE2_GST_STOP_TIMEOUT_SEC", "18"))


def gst_available() -> bool:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        return True
    except Exception as exc:
        logger.debug("GStreamer unavailable: %s", exc)
        return False


def normalize_device(camera: str | int) -> str:
    if isinstance(camera, int):
        return f"/dev/video{camera}"
    path = str(camera)
    if path.isdigit():
        return f"/dev/video{path}"
    return path


def device_held_by_current_process(device: str | int) -> bool:
    """本进程是否仍持有 V4L2 设备 fd（/proc/self/fd）。"""
    return bool(list_device_fds(device))


def list_device_fds(device: str | int) -> list[int]:
    """本进程指向该 V4L2 设备的 fd 列表。"""
    try:
        dev = os.path.realpath(normalize_device(device))
    except OSError:
        dev = normalize_device(device)
    found: list[int] = []
    proc_fd = "/proc/self/fd"
    try:
        for name in os.listdir(proc_fd):
            path = os.path.join(proc_fd, name)
            try:
                if os.path.realpath(path) == dev:
                    found.append(int(name))
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    return found


def force_close_device_fds(device: str | int) -> int:
    """最后手段：关闭本进程残留的 video fd（引擎 exit 仍占设备时）。"""
    closed = 0
    for fd in list_device_fds(device):
        try:
            os.close(fd)
            closed += 1
        except OSError:
            pass
    if closed:
        print(
            f"WARN: force_close_device_fds {normalize_device(device)} closed={closed}",
            flush=True,
        )
        time.sleep(0.3)
    return closed


def _iterate_pipeline_elements(pipeline: Any, Gst: Any) -> list[Any]:
    elements: list[Any] = []
    try:
        it = pipeline.iterate_elements()
        while True:
            result, element = it.next()
            if result != Gst.IteratorResult.OK:
                break
            elements.append(element)
    except Exception:
        pass
    return elements


def _null_elements_by_factory(
    pipeline: Any,
    Gst: Any,
    factory_name: str,
) -> None:
    """优先 NULL 指定 factory（如 v4l2src），便于尽快释放 V4L2 fd。"""
    for element in _iterate_pipeline_elements(pipeline, Gst):
        try:
            factory = element.get_factory()
            if factory is None:
                continue
            if factory.get_name() != factory_name:
                continue
            element.set_state(Gst.State.NULL)
        except Exception:
            pass


def _wait_bus_message(
    pipeline: Any,
    Gst: Any,
    msg_types: Any,
    *,
    timeout_sec: float = 3.0,
) -> bool:
    """有界等待 bus 消息（EOS/ERROR）。"""
    bus = pipeline.get_bus()
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            msg = bus.timed_pop_filtered(
                int(min(0.25, remaining) * Gst.SECOND),
                msg_types,
            )
        except Exception:
            break
        if msg is None:
            continue
        if msg.type == Gst.MessageType.EOS:
            return True
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"WARN: GStreamer bus ERROR during stop: {err} ({dbg})", flush=True)
            return False
    return False


def gst_null_v4l2src(pipeline: Any, Gst: Any) -> None:
    """release 重试：单独 NULL v4l2src 以尽快释放设备 fd。"""
    _null_elements_by_factory(pipeline, Gst, "v4l2src")


def gst_force_pipeline_null(
    pipeline: Any,
    Gst: Any,
    *,
    wait_sec: float = 4.0,
) -> bool:
    return _force_pipeline_null(pipeline, Gst, wait_sec=wait_sec)


def _null_pipeline_elements(pipeline: Any, Gst: Any) -> None:
    try:
        it = pipeline.iterate_elements()
        while True:
            result, element = it.next()
            if result != Gst.IteratorResult.OK:
                break
            try:
                element.set_state(Gst.State.NULL)
            except Exception:
                pass
    except Exception as exc:
        print(f"WARN: null_pipeline_elements: {exc}", flush=True)


def _force_pipeline_null(pipeline: Any, Gst: Any, *, wait_sec: float = 4.0) -> bool:
    try:
        pipeline.set_state(Gst.State.PAUSED)
        pipeline.get_state(int(1 * Gst.SECOND))
    except Exception:
        pass
    try:
        _null_pipeline_elements(pipeline, Gst)
        ret = pipeline.set_state(Gst.State.NULL)
        if ret == Gst.StateChangeReturn.FAILURE:
            detail = gst_drain_bus_errors(pipeline, Gst)
            print(f"WARN: force NULL FAILURE\n  {detail}", flush=True)
            return False
        ret, state, _pending = pipeline.get_state(int(max(1.0, wait_sec) * Gst.SECOND))
        nick = getattr(state, "value_nick", str(state))
        if state != Gst.State.NULL:
            print(f"WARN: force NULL 未到 NULL，state={nick}", flush=True)
            return False
        return True
    except Exception as exc:
        print(f"WARN: force_pipeline_null: {exc}", flush=True)
        return False


def gst_drain_bus_errors(pipeline: Any, Gst: Any) -> str:
    bus = pipeline.get_bus()
    lines: list[str] = []
    while True:
        msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING)
        if msg is None:
            break
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            lines.append(f"ERROR: {err} ({dbg})")
        else:
            warn, dbg = msg.parse_warning()
            lines.append(f"WARN: {warn} ({dbg})")
    return "\n  ".join(lines) if lines else ""


def gst_start_playing(
    pipeline: Any,
    Gst: Any,
    *,
    timeout_sec: float = GST_PLAY_TIMEOUT_SEC,
    pull_driven: bool = False,
) -> None:
    t0 = time.perf_counter()
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        detail = gst_drain_bus_errors(pipeline, Gst)
        raise RuntimeError(f"GStreamer set_state(PLAYING) failed.\n  {detail}")
    if pull_driven:
        time.sleep(0.8)
        detail = gst_drain_bus_errors(pipeline, Gst)
        if "ERROR:" in detail:
            raise RuntimeError(f"GStreamer bus error after PLAYING.\n  {detail}")
        return
    timeout_ns = int(timeout_sec * Gst.SECOND)
    ret, state, pending = pipeline.get_state(timeout_ns)
    elapsed = time.perf_counter() - t0
    if ret == Gst.StateChangeReturn.FAILURE or state != Gst.State.PLAYING:
        detail = gst_drain_bus_errors(pipeline, Gst)
        nick = getattr(state, "value_nick", str(state))
        raise RuntimeError(
            f"GStreamer PLAYING timeout ({timeout_sec}s), state={nick}.\n  {detail}"
        )
    logger.info("GStreamer PLAYING (%.1fs)", elapsed)


def gst_stop_pipeline(
    pipeline: Any,
    Gst: Any,
    *,
    timeout_sec: float = GST_STOP_TIMEOUT_SEC,
) -> bool:
    """EOS → PAUSED → v4l2src NULL → pipeline NULL；超时后主线程强制 NULL。"""
    if pipeline is None:
        return True

    try:
        pipeline.send_event(Gst.Event.new_flush_start())
        pipeline.send_event(Gst.Event.new_flush_stop(False))
    except Exception:
        pass
    try:
        pipeline.send_event(Gst.Event.new_eos())
    except Exception:
        pass
    _wait_bus_message(
        pipeline,
        Gst,
        Gst.MessageType.EOS | Gst.MessageType.ERROR,
        timeout_sec=min(3.0, max(1.0, float(timeout_sec) * 0.2)),
    )

    result = {"ok": False, "state": "unknown"}

    def _worker() -> None:
        try:
            ret = pipeline.set_state(Gst.State.PAUSED)
            if ret != Gst.StateChangeReturn.FAILURE:
                pipeline.get_state(int(2 * Gst.SECOND))
            _null_elements_by_factory(pipeline, Gst, "v4l2src")
            ret = pipeline.set_state(Gst.State.NULL)
            if ret == Gst.StateChangeReturn.FAILURE:
                detail = gst_drain_bus_errors(pipeline, Gst)
                print(
                    f"WARN: GStreamer set_state(NULL) FAILURE\n  {detail}",
                    flush=True,
                )
                return
            ret, state, _pending = pipeline.get_state(int(4 * Gst.SECOND))
            nick = getattr(state, "value_nick", str(state))
            result["state"] = nick
            result["ok"] = state == Gst.State.NULL
        except Exception as exc:
            print(f"WARN: gst_stop_pipeline worker: {exc}", flush=True)

    th = threading.Thread(target=_worker, name="gst_stop", daemon=True)
    th.start()
    th.join(timeout=max(1.0, float(timeout_sec)))
    if th.is_alive():
        print(
            f"WARN: GStreamer NULL 后台超时 ({timeout_sec:.0f}s)，主线程强制 NULL…",
            flush=True,
        )
    if not result["ok"]:
        _null_elements_by_factory(pipeline, Gst, "v4l2src")
        forced = _force_pipeline_null(
            pipeline,
            Gst,
            wait_sec=min(6.0, max(2.0, float(timeout_sec) * 0.45)),
        )
        if forced:
            result["ok"] = True
            result["state"] = "null"
    if not result["ok"]:
        print(
            f"WARN: GStreamer 未到 NULL，state={result['state']}",
            flush=True,
        )
    return bool(result["ok"])


def pull_sample_timeout(sink: Any, Gst: Any, timeout_sec: float = 0.2) -> Any | None:
    if sink is None:
        return None
    timeout_ns = int(timeout_sec * Gst.SECOND)
    try:
        return sink.emit("try-pull-sample", timeout_ns)
    except Exception:
        pass
    try:
        if hasattr(sink, "try_pull_sample"):
            return sink.try_pull_sample(timeout_ns)
    except Exception:
        pass
    return None
