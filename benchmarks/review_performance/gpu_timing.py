"""Optional asynchronous OpenGL timer queries for separate diagnostic runs."""

import ctypes
import ctypes.util
from collections import deque


class GPUTimer:
    """Never wait for a query: skip sampling when all eight queries are pending."""

    def __init__(self) -> None:
        library = ctypes.util.find_library("GL")
        if library is None:
            raise RuntimeError("libGL timer queries are unavailable.")
        self.gl = ctypes.CDLL(library)
        self.gl.glGenQueries.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
        self.gl.glDeleteQueries.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
        self.gl.glBeginQuery.argtypes = (ctypes.c_uint, ctypes.c_uint)
        self.gl.glEndQuery.argtypes = (ctypes.c_uint,)
        self.gl.glGetQueryObjectiv.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
        self.gl.glGetQueryObjectui64v.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint64))
        self.ids = (ctypes.c_uint * 8)()
        self.gl.glGenQueries(8, self.ids)
        self.free = deque(self.ids)
        self.pending: deque[int] = deque()
        self.current: int | None = None
        self.milliseconds: list[float] = []

    def begin(self) -> None:
        while self.pending:
            ready = ctypes.c_int()
            self.gl.glGetQueryObjectiv(self.pending[0], 0x8867, ctypes.byref(ready))  # QUERY_RESULT_AVAILABLE
            if not ready.value:
                break
            query = self.pending.popleft()
            elapsed = ctypes.c_uint64()
            self.gl.glGetQueryObjectui64v(query, 0x8866, ctypes.byref(elapsed))
            self.milliseconds.append(elapsed.value / 1e6)
            self.free.append(query)
        if self.free:
            self.current = self.free.popleft()
            self.gl.glBeginQuery(0x88BF, self.current)  # TIME_ELAPSED

    def end(self) -> None:
        if self.current is not None:
            self.gl.glEndQuery(0x88BF)
            self.pending.append(self.current)
            self.current = None

    def close(self) -> None:
        self.gl.glDeleteQueries(8, self.ids)
