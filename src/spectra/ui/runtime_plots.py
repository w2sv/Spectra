"""Raw spectral power, processed intensities and capture-time PCM in one canvas."""

import numpy as np
from numpy.typing import NDArray
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vispy import scene
from vispy.scene import Node

from spectra.analysis import KickEvent
from spectra.audio.history import PCMHistory
from spectra.ui.live_waveform import LiveWaveform
from spectra.ui.plots import PlotCanvas, TimeAxis, line
from spectra.visualization.frequency import band_colors


class BandBars:
    def __init__(self, parent: Node) -> None:
        self.mesh = scene.visuals.Mesh(parent=parent)
        self.vertices = np.empty((0, 2), dtype=np.float32)
        self.faces = np.empty((0, 3), dtype=np.uint32)
        self.colors = np.empty((0, 4), dtype=np.float32)
        self.values = np.empty(0, dtype=np.float64)
        self.bottom = 0.0

    def configure(self, edges: NDArray[np.float64], bottom: float) -> None:
        count = len(edges) - 1
        gap = np.diff(edges) * 0.08
        self.vertices = np.empty((count * 4, 2), dtype=np.float32)
        self.vertices[:, 0] = np.column_stack(
            (edges[:-1] + gap, edges[1:] - gap, edges[1:] - gap, edges[:-1] + gap)
        ).ravel()
        self.faces = (
            np.arange(count, dtype=np.uint32)[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        ).reshape(-1, 3)
        self.colors = np.repeat(np.column_stack((band_colors(count), np.ones(count))), 4, axis=0).astype("float32")
        self.bottom = bottom
        self.values = np.empty(0, dtype=np.float64)
        self.set_values(np.full(count, bottom))

    def set_values(self, values: NDArray[np.float64]) -> None:
        if np.array_equal(values, self.values):
            return
        self.values = values.copy()
        self.vertices[:, 1] = np.column_stack(
            (np.full(len(values), self.bottom), np.full(len(values), self.bottom), values, values)
        ).ravel()
        self.mesh.set_data(vertices=self.vertices, faces=self.faces, vertex_colors=self.colors)


class RuntimePlots(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(220)
        self.canvas = PlotCanvas()
        self.cameras = [scene.PanZoomCamera() for _ in range(3)]
        self.views = [scene.widgets.ViewBox(parent=self.canvas.root, camera=camera) for camera in self.cameras]
        for camera in self.cameras:
            camera.interactive = False
        spectrum_scene, intensity_scene = self.views[0].scene, self.views[1].scene
        assert spectrum_scene is not None and intensity_scene is not None
        self.spectrum = BandBars(spectrum_scene)
        self.intensity = BandBars(intensity_scene)
        wave_scene = self.views[2].scene
        assert wave_scene is not None
        self.waveform = LiveWaveform(wave_scene)
        self.zero = line(wave_scene, "#444444", np.array([[-10, 0], [0, 0]], dtype="float32"))
        self.labels = scene.visuals.Text(
            parent=self.canvas.root,  # pyright: ignore[reportCallIssue]
            color="white",
            font_size=9,
        )
        self.axis = TimeAxis(self.canvas, "Seconds before latest captured sample")
        self.waveform_enabled = True
        self.history_seconds = 5.0
        self.edges = np.geomspace(40, 16000, 25)
        self._wave_bounds: tuple | None = None
        self._history: PCMHistory | None = None
        self._kicks: tuple[KickEvent, ...] = ()
        self._features: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self.canvas.before_draw = self._prepare_frame
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)
        self.canvas.events["resize"].connect(self._resize, position="last")
        self.configure(self.edges)

    def configure(self, edges: NDArray[np.float64]) -> None:
        self.edges = edges
        logarithmic = np.log10(edges)
        self.spectrum.configure(logarithmic, -80)
        self.intensity.configure(logarithmic, 0)
        self._layout()

    def set_waveform_enabled(self, enabled: bool) -> None:
        self.waveform_enabled = enabled
        self.views[2].visible = enabled
        self.axis.set_visible(enabled)
        self._layout()

    def set_history_seconds(self, seconds: float) -> None:
        self.history_seconds = seconds
        self._layout()
        if self._history is not None and self.waveform_enabled:
            self.update_waveform(self._history, self._kicks)

    def set_features(self, spectrum: NDArray[np.float64], intensities: NDArray[np.float64]) -> None:
        self._features = (spectrum, intensities)
        self.canvas.update()

    def update_waveform(self, history: PCMHistory, kicks: tuple[KickEvent, ...]) -> None:
        self._history, self._kicks = history, kicks
        if self.waveform_enabled:
            self.canvas.update()

    def _prepare_frame(self) -> None:
        if self._features is not None:
            self.spectrum.set_values(self._features[0])
            self.intensity.set_values(self._features[1])
            self._features = None
        if self._history is not None and self.waveform_enabled:
            if self.waveform.update(
                self._history,
                self._kicks,
                self.history_seconds,
                max(1, round(self.views[2].size[0] * self.canvas.pixel_scale)),
            ):
                self._wave_range()

    def _wave_range(self) -> None:
        amplitude = self.waveform.amplitude
        bounds = (-self.history_seconds, -amplitude * 1.05, self.history_seconds, amplitude * 2.3)
        if bounds != self._wave_bounds:
            self._wave_bounds = bounds
            self.cameras[2].rect = bounds

    def clear(self) -> None:
        self._features = None
        self.spectrum.set_values(np.full(len(self.edges) - 1, -80, dtype=np.float64))
        self.intensity.set_values(np.zeros(len(self.edges) - 1, dtype=np.float64))
        self.waveform.reset()
        self._history = None
        self._kicks = ()
        self._wave_range()

    def _layout(self) -> None:
        width, height = self.canvas.size
        top_height = max(65, (height - 85) * (0.52 if self.waveform_enabled else 1))
        gap = 70
        each = max(60, (width - 100 - gap) / 2)
        for i in (0, 1):
            self.views[i].pos = (45 + i * (each + gap), 22)
            self.views[i].size = (each, top_height)
        left, right = np.log10(self.edges[[0, -1]])
        self.cameras[0].rect = (float(left), -80, float(right - left), 80)
        self.cameras[1].rect = (float(left), 0, float(right - left), 1.05)
        self.views[2].pos = (45, top_height + 68)
        self.views[2].size = (max(1, width - 60), max(35, height - top_height - 115))
        self._wave_range()
        self.axis.update(-self.history_seconds, 0, 45, height - 43, max(1, width - 60))
        texts, positions = [], []
        for index, title, ys in (
            (0, "Raw band power (dB re 1 PCM²)", [-80, -40, 0]),
            (1, "Processed LED intensity", [0, 0.5, 1]),
        ):
            view = self.views[index]
            x, y = view.pos
            texts.append(title)
            positions.append((x + each / 2, 10))
            for value in ys:
                fraction = (value + 80) / 80 if index == 0 else value / 1.05
                texts.append(f"{value:g}")
                positions.append((x - 20, y + top_height * (1 - fraction)))
            for hz, label in ((40, "40"), (250, "250"), (1000, "1k"), (4000, "4k"), (16000, "16k")):
                texts.append(label)
                positions.append((x + each * (np.log10(hz) - left) / (right - left), y + top_height + 13))
            texts.append("Frequency (Hz)")
            positions.append((x + each / 2, y + top_height + 32))
        if self.waveform_enabled:
            texts.append("Live PCM · production kick detections")
            positions.append((width / 2, top_height + 55))
        self.labels.text, self.labels.pos = texts, positions

    def _resize(self, event) -> None:
        self._layout()
