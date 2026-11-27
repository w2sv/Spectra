"""Incremental peak envelopes in a bounded GPU ring, in capture sample time."""

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from vispy import gloo, scene
from vispy.scene import Node
from vispy.visuals import Visual
from vispy.visuals.transforms import STTransform

from spectra.analysis import KickEvent
from spectra.audio.history import PCMHistory
from spectra.ui.plots import markers
from spectra.ui.waveform import extrema_indices


class EnvelopeVisual(Visual):
    """Pairs of extrema per bucket; only dirty buckets are uploaded.

    GPU coordinates are integer sample offsets modulo a bounded period. The head
    uniform scrolls existing data without history transfers or loss of precision
    after hours of capture. Each pair uses its endpoint to unwrap continuously,
    preventing a false diagonal across the circular buffer boundary.
    """

    def __init__(self, capacity: int, block: int, sample_rate: int) -> None:
        super().__init__(
            vcode="""
            attribute vec3 a_position;
            uniform float u_head;
            uniform float u_period;
            uniform float u_rate;
            void main() {
                // Subtract bounded integer offsets directly. mod(period, period)
                // can round incorrectly with a driver's reciprocal optimization.
                float age = u_head - a_position.z;
                if (age < 0.0) age += u_period;
                float span = a_position.z - a_position.x;
                if (span < 0.0) span += u_period;
                float delta = -age - span;
                if (a_position.x < 0.0) delta = -u_period * 2.0;
                gl_Position = $transform(vec4(delta / u_rate, a_position.y, 0.0, 1.0));
            }
            """,
            fcode="""
            void main() { gl_FragColor = vec4(0.67, 0.68, 0.71, 1.0); }
            """,
        )
        self.block = block
        self.buckets = math.ceil(capacity / block) + 2
        self.period = self.buckets * block
        self.points = np.zeros((self.buckets * 2, 3), dtype=np.float32)
        self.points[:, 0] = -1
        self.buffer = gloo.VertexBuffer(self.points)
        self.shared_program["a_position"] = self.buffer
        self.shared_program["u_head"] = 0.0
        self.shared_program["u_period"] = float(self.period)
        self.shared_program["u_rate"] = float(sample_rate)
        self._draw_mode = "lines"
        self.previous_sample = 0.0
        self.uploaded_bytes = 0
        self.set_gl_state("translucent", line_smooth=True, depth_test=False)

    def clear(self) -> None:
        self.points[:, 0] = -1
        self.buffer.set_data(self.points, copy=True)
        self.previous_sample = 0.0

    def append(self, first: int, samples: NDArray[np.float32], head: int) -> None:
        if not len(samples):
            return
        if self.block == 1:
            indices = np.repeat(np.arange(len(samples), dtype=np.int64), 2)
            positions = indices.copy()
            positions[0::2] -= 1
            values = samples[indices].copy()
            values[0] = self.previous_sample
            values[2::2] = samples[:-1]
            self.previous_sample = float(samples[-1])
        else:
            leading = min(len(samples), (-first) % self.block)
            if leading:
                pair = np.sort([samples[:leading].argmin(), samples[:leading].argmax()])
                indices = np.concatenate((pair, extrema_indices(samples[leading:], self.block) + leading))
            else:
                indices = extrema_indices(samples, self.block)
            positions = indices
            values = samples[indices]
        count = len(indices) // 2
        slot = (first // self.block) % self.buckets
        split = min(count, self.buckets - slot)
        sample_positions = (positions + first % self.period) % self.period
        endpoints = np.repeat(sample_positions[1::2], 2)
        for source, offset, size in ((0, slot, split), (split, 0, count - split)):
            if not size:
                continue
            points = self.points[offset * 2 : (offset + size) * 2]
            selected = slice(source * 2, (source + size) * 2)
            points[:, 0] = sample_positions[selected]
            points[:, 1] = values[selected]
            points[:, 2] = endpoints[selected]
            self.buffer.set_subdata(points, offset=offset * 2, copy=True)
            self.uploaded_bytes += points.nbytes
        self.shared_program["u_head"] = float(head % self.period)
        self.update()

    def _prepare_draw(self, view=None) -> Literal[True]:
        if self.transforms is not None:
            self.update_gl_state(line_width=max(1, self.transforms.pixel_scale))
        return True

    def _prepare_transforms(self, view) -> None:
        view.view_program.vert["transform"] = view.transforms.get_transform()


EnvelopeNode = scene.visuals.create_visual_node(EnvelopeVisual)


class LiveWaveform:
    def __init__(self, parent: Node) -> None:
        self.parent = parent
        self.visual = None
        self.kick_markers = markers(parent)
        self.kick_markers.transform = STTransform()
        self.history: PCMHistory | None = None
        self.scratch = np.empty(0, dtype=np.float32)
        self.next_sample = 0
        self.last_stop = -1
        self.block = 0
        self.head_seconds = 0.0
        self.amplitude = 1.0
        self.marker_origin = 0.0
        self._events: tuple[KickEvent, ...] = ()
        self._marker_transform: tuple[float, float] | None = None
        self.skipped_samples = 0

    def reset(self) -> None:
        if self.visual is not None:
            self.visual.parent = None
        self.visual = None
        self.history = None
        self.next_sample = 0
        self.last_stop = -1
        self.head_seconds = 0.0
        self.marker_origin = 0.0
        self._marker_transform = None
        self.block = 0
        self._events = ()
        self.kick_markers.visible = False
        self.skipped_samples = 0
        self.amplitude = 1.0

    def update(self, history: PCMHistory, kicks: tuple[KickEvent, ...], seconds: float, physical_pixels: int) -> bool:
        samples_per_pixel = history.sample_rate * seconds / max(1, physical_pixels)
        block = 2 ** max(0, math.floor(math.log2(max(1, samples_per_pixel))))
        if history is not self.history or block != self.block:
            self.reset()
            self.history = history
            self.block = block
            self.scratch = np.empty(history.capacity, dtype=np.float32)
            self.visual = EnvelopeNode(
                history.capacity,
                block,
                history.sample_rate,
                parent=self.parent,  # pyright: ignore[reportCallIssue]
            )
        assert self.visual is not None
        if history.written == self.last_stop:
            return self._update_markers(kicks)
        result = history.copy_since(self.next_sample, self.scratch)
        if result is None or result.stop == result.first:
            return False
        if result.skipped:
            self.skipped_samples += result.skipped
            self.visual.clear()
        self.amplitude = max(self.amplitude, float(np.max(np.abs(self.scratch[: result.stop - result.first]))))
        self.visual.append(result.first, self.scratch[: result.stop - result.first], result.stop)
        # Revisit only the provisional final bucket, at most one physical pixel.
        self.next_sample = max(result.first, result.stop - result.stop % block)
        self.last_stop = result.stop
        self.head_seconds = result.stop / history.sample_rate
        self._update_markers(kicks)
        return True

    def _update_markers(self, kicks: tuple[KickEvent, ...]) -> bool:
        changed = kicks != self._events
        if changed:
            self._events = kicks
            self.marker_origin = self.head_seconds
            positions = np.array(
                [(kick.timestamp - self.marker_origin, 1.12) for kick in kicks], dtype=np.float32
            ).reshape(-1, 2)
            self.kick_markers.visible = bool(len(positions))
            if len(positions):
                self.kick_markers.set_data(pos=positions, face_color="#2bad62", size=8, edge_width=0)
        transform = (self.marker_origin - self.head_seconds, self.amplitude)
        if kicks and transform != self._marker_transform:
            self._marker_transform = transform
            self.kick_markers.transform.translate = (transform[0], 0)
            self.kick_markers.transform.scale = (1, transform[1])
            changed = True
        return changed
