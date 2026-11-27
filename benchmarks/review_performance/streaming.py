"""Bounded PCM ingestion independent of Qt, plus incremental GPU ring writes."""

import threading
from time import monotonic

import numpy as np
from numpy.typing import NDArray
from vispy import gloo, scene
from vispy.visuals import Visual


class RollingPCM:
    """Producer owns sample ingestion; GUI copies at most the bounded dirty region.

    Falling behind overwrites old PCM. Discrete kick times are derived from the
    immutable recording event list, so no event queue can grow or lose a kick.
    """

    def __init__(self, samples: NDArray[np.float32], rate: int, seconds: float = 10) -> None:
        self.samples = samples
        self.rate = rate
        self.capacity = round(rate * seconds)
        self.history = np.zeros(self.capacity, dtype=np.float32)
        self.written = 0
        self.consumed = 0
        self.overwritten = 0
        self.maximum_ingestion_lateness_seconds = 0.0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._produce, name="experiment-pcm", daemon=True)
        self.started = 0.0
        self.scratch = np.empty(self.capacity, dtype=np.float32)

    def start(self) -> None:
        self.started = monotonic()
        self.thread.start()

    def _produce(self) -> None:
        block = max(1, self.rate // 100)  # 10 ms capture blocks, independent of draw target.
        while not self.stop_event.is_set():
            deadline = self.started + (self.written + block) / self.rate
            if self.stop_event.wait(max(0, deadline - monotonic())):
                return
            self.maximum_ingestion_lateness_seconds = max(
                self.maximum_ingestion_lateness_seconds, monotonic() - deadline
            )
            with self.lock:
                start = self.written % self.capacity
                source_start = self.written % len(self.samples)
                indices = (np.arange(block) + source_start) % len(self.samples)
                count = min(block, self.capacity - start)
                self.history[start : start + count] = self.samples[indices[:count]]
                self.history[: block - count] = self.samples[indices[count:]]
                self.written += block

    def read(self) -> tuple[int, NDArray[np.float32]]:
        with self.lock:
            first = max(self.consumed, self.written - self.capacity)
            self.overwritten += max(0, first - self.consumed)
            count = self.written - first
            start = first % self.capacity
            split = min(count, self.capacity - start)
            self.scratch[:split] = self.history[start : start + split]
            self.scratch[split:count] = self.history[: count - split]
            self.consumed = self.written
        return first, self.scratch[:count]

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("PCM ingestion did not stop.")


class StreamingLineVisual(Visual):
    """A fixed VBO: LineVisual.set_data otherwise reuploads the whole history.

    Independent sample pairs avoid a false diagonal at the circular boundary.
    Only this streaming visual needs custom shaders; static plots use Line.
    """

    def __init__(self, capacity: int, rate: int, full_upload: bool = False) -> None:
        super().__init__(
            vcode="""
            attribute vec2 a_position;
            void main() { gl_Position = $transform(vec4(a_position, 0.0, 1.0)); }
            """,
            fcode="""
            void main() { gl_FragColor = vec4(0.67, 0.68, 0.71, 1.0); }
            """,
        )
        self.capacity = capacity
        self.rate = rate
        self.full_upload = full_upload
        self.previous_end = 0
        self.points = np.zeros((capacity * 2, 2), dtype=np.float32)
        self.points[:, 0] = -1000
        self.buffer = gloo.VertexBuffer(self.points)
        self.shared_program["a_position"] = self.buffer
        self._draw_mode = "lines"
        self.previous = 0.0
        self.set_gl_state("translucent", line_width=1, line_smooth=True)

    def append(self, first: int, samples: NDArray[np.float32]) -> None:
        if not len(samples):
            return
        if first != self.previous_end:
            self.previous = float(samples[0])
        self.previous_end = first + len(samples)
        start = first % self.capacity
        split = min(len(samples), self.capacity - start)
        for source, offset, count in ((0, start, split), (split, 0, len(samples) - split)):
            if not count:
                continue
            points = self.points[offset * 2 : (offset + count) * 2]
            indices = np.arange(first + source, first + source + count)
            points[0::2, 0] = (indices - 1) / self.rate
            points[1::2, 0] = indices / self.rate
            points[1::2, 1] = samples[source : source + count]
            points[0, 1] = self.previous
            points[2::2, 1] = samples[source : source + count - 1]
            self.previous = float(samples[source + count - 1])
            if not self.full_upload:
                self.buffer.set_subdata(points, offset=offset * 2, copy=True)
        if self.full_upload:
            self.buffer.set_data(self.points, copy=True)
        self.update()

    def _prepare_transforms(self, view) -> None:
        view.view_program.vert["transform"] = view.transforms.get_transform()


StreamingLine = vars(scene.visuals)["create_visual_node"](StreamingLineVisual)
