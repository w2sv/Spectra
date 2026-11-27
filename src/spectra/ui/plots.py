"""Small VisPy primitives shared by the editor and live plots; GUI-thread only."""

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from vispy import app, scene
from vispy.scene import Node
from vispy.visuals.transforms import STTransform

from spectra.ui.waveform import WaveformPyramid


class PlotCanvas(scene.SceneCanvas):
    def __init__(self, background: str = "black") -> None:
        app.use_app("pyside6")
        super().__init__(keys=None, show=False, bgcolor=background, vsync=False, config={"samples": 4})
        self.unfreeze()
        root = self.scene
        assert root is not None
        self.root = root
        self.before_draw: Callable[[], None] | None = None
        self.freeze()

    def on_draw(self, event) -> None:
        # Submit only the latest CPU state when a paint actually occurs. Native
        # updates can continue while a hidden/occluded window is not drawing.
        if self.before_draw is not None:
            self.before_draw()
        super().on_draw(event)


def line(parent: Node, color: str, points: NDArray[np.float32] | None = None, *, segments: bool = False):
    # Scene visual classes are generated at runtime. Upstream constructor typing
    # omits Node keywords; suppress only those incorrect argument diagnostics.
    return scene.visuals.Line(
        parent=parent,  # pyright: ignore[reportCallIssue]
        pos=points,
        color=color,
        method="gl",
        antialias=True,
        width=1,
        connect="segments" if segments else "strip",
    )


def markers(parent: Node):
    return scene.visuals.Markers(parent=parent, scaling="fixed", antialias=1)


class ScreenLine:
    def __init__(self, parent: Node) -> None:
        self.visual = scene.visuals.Mesh(
            parent=parent,
            vertices=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32"),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype="uint32"),
            color="white",
        )
        self.visual.transform = STTransform()
        self.visual.order = 100
        self.x = 0.0
        self.width = 1
        self.visible = False
        self.color = "white"
        self._snapshot: tuple | None = None
        self.visual.visible = False

    def set(self, x: float, top: float, bottom: float, width: int, color: str, visible: bool, ratio: float) -> None:
        x = round(x * ratio) / ratio
        if not visible and not self.visible:
            return
        snapshot = (x, top, bottom, width, color, visible)
        if snapshot == self._snapshot:
            return
        self._snapshot = snapshot
        color_changed = color != self.color
        self.x, self.width, self.color, self.visible = x, width, color, visible
        self.visual.transform.scale = (width, max(1, bottom - top))
        self.visual.transform.translate = (x - width / 2, top)
        if color_changed:
            self.visual.color = color
        self.visual.visible = visible


class TimeAxis:
    def __init__(self, canvas: PlotCanvas, title: str = "Recording time (s)") -> None:
        self.camera = scene.PanZoomCamera(rect=(0, 0, 10, 1))
        self.camera.interactive = False
        self.view = scene.widgets.ViewBox(parent=canvas.root, camera=self.camera)
        self.labels = scene.visuals.Text(
            parent=self.view.scene,  # pyright: ignore[reportCallIssue]
            color="white",
            font_size=9,
            anchor_y="top",
        )
        axis_scene = self.view.scene
        assert axis_scene is not None
        self.ticks = line(axis_scene, "#aaaaaa", segments=True)
        self.title = scene.visuals.Text(
            parent=canvas.root,  # pyright: ignore[reportCallIssue]
            text=title,
            color="white",
            font_size=9,
        )
        self._key: object = None
        self._layout_key: tuple[float, float, float] | None = None

    def update(self, left: float, right: float, x: float, y: float, width: float) -> None:
        layout_key = (x, y, width)
        if layout_key != self._layout_key:
            self._layout_key = layout_key
            self.view.pos, self.view.size = (x, y), (max(1, width), 42)
            self.title.pos = (x + width / 2, y + 34)
        raw = (right - left) / max(2, width / 110)
        power = 10 ** math.floor(math.log10(raw))
        spacing = next(value * power for value in (1, 2, 5, 10) if value * power >= raw)
        if right - left >= 2:
            spacing = max(1, spacing)
        first, last = math.floor(left / spacing), math.ceil(right / spacing)
        origin = first * spacing
        self.camera.rect = (left - origin, 0, right - left, 1)
        # Do not clip a label into a misleading partial number at the view edge.
        ticks = np.arange(first, last + 1) * spacing
        margin = 28 * (right - left) / max(1, width)
        texts = [f"{value:g}" if left + margin <= value <= right - margin else "" for value in ticks]
        key = (first, last, spacing, tuple(texts))
        if key == self._key:
            return
        self._key = key
        self.labels.text = texts
        self.labels.pos = np.column_stack((ticks - origin, np.full(len(ticks), 0.72)))
        self.ticks.set_data(
            pos=np.column_stack((np.repeat(ticks - origin, 2), np.tile([0.85, 1], len(ticks)))).astype("float32")
        )

    def set_visible(self, visible: bool) -> None:
        self.view.visible = self.title.visible = visible


class RecordingWaveform:
    """A padded GPU window, updated only on cache misses or resolution changes."""

    def __init__(self, parent: Node, samples: NDArray[np.float32], sample_rate: int) -> None:
        self.samples = samples
        self.rate = sample_rate
        self.pyramid = WaveformPyramid(samples)
        self.visual = line(parent, "#aaaeb5")
        self.visual.transform = STTransform()
        self.window: tuple[int, int, int] | None = None
        self.uploads = 0
        self.uploaded_bytes = 0
        self.points = np.empty((0, 2), dtype="float32")

    def update(self, left: float, right: float, physical_pixels: int) -> bool:
        level = self.pyramid.level((right - left) * self.rate, physical_pixels)
        block = level.block if level is not None else 1
        cached = self.window
        if (
            cached is not None
            and cached[0] == block
            and cached[1] <= left * self.rate
            and cached[2] >= right * self.rate
        ):
            return False
        padding = (right - left) / 2
        first = max(0, math.floor((left - padding) * self.rate))
        last = min(len(self.samples), math.ceil((right + padding) * self.rate))
        if level is None:
            indices = np.arange(max(0, first - 2), min(len(self.samples), last + 2))
            values = self.samples[indices]
        else:
            start, end = np.searchsorted(level.indices, [first, last])
            selected = slice(max(0, int(start) - 2), min(len(level.indices), int(end) + 2))
            indices, values = level.indices[selected], level.values[selected]
        origin = int(indices[0]) if len(indices) else first
        self.points = np.empty((len(indices), 2), dtype="float32")
        self.points[:, 0] = (indices - origin) / self.rate
        self.points[:, 1] = values
        self.visual.transform.translate = (origin / self.rate, 0)
        self.visual.set_data(pos=self.points)
        self.window = (block, first, last)
        self.uploads += 1
        self.uploaded_bytes += self.points.nbytes
        return True
