"""NVMM dmabuf fd → Gst.Buffer（session enc 池 → appsrc）。"""

from __future__ import annotations

from typing import Any


def gst_buffer_from_dmabuf_fd(fd: int, size: int) -> Any:
    """DmaBufAllocator + DONT_CLOSE：fd 生命周期由 NvBuf 池持有。"""
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstAllocators", "1.0")
    from gi.repository import Gst, GstAllocators

    Gst.init(None)
    allocator = GstAllocators.DmaBufAllocator.new()
    flags = GstAllocators.FdMemoryFlags.DONT_CLOSE
    mem = allocator.alloc_with_flags(allocator, int(fd), int(size), flags)
    buf = Gst.Buffer.new()
    buf.append_memory(mem)
    return buf
