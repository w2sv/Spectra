"""Linked time plots, a single marker lane, and native horizontal navigation."""

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtWidgets import QGraphicsItem, QGraphicsLineItem, QSplitter, QVBoxLayout, QWidget

from benchmarks.annotation.analysis import PARAMETERS, AcousticAnalysis, AutomaticCandidate, Proposal
from benchmarks.annotation.corrections import (
    CorrectionEditor,
    effective_kick,
    effective_timestamp,
    uncertain_classification,
)

PLAYBACK_ANCHOR_FRACTION = 0.4


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        if spacing >= 1:
            return [str(round(value)) for value in values]
        decimals = max(1, math.ceil(-math.log10(spacing)))
        return [f"{value:.{decimals}f}" for value in values]

    def tickValues(self, minVal: float, maxVal: float, size: float) -> list[tuple[float, list[float]]]:
        levels = super().tickValues(minVal, maxVal, size)
        if maxVal - minVal >= 2:
            return [(spacing, values) for spacing, values in levels if spacing >= 1]
        return levels


class HorizontalViewBox(pg.ViewBox):
    clicked = Signal(float, QPointF, object, bool)
    manuallyNavigated = Signal()
    zoomRequested = Signal(float, float)

    def __init__(self) -> None:
        super().__init__(enableMenu=False)
        self.setMouseEnabled(x=True, y=False)

    def mouseClickEvent(self, ev) -> None:
        if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            point = ev.scenePos()
            self.clicked.emit(
                self.mapSceneToView(point).x(), point, ev.modifiers(), ev.button() == Qt.MouseButton.RightButton
            )
            ev.accept()
        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None) -> None:
        if ev.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            ev.ignore()
            return
        if ev.pos().x() != ev.lastPos().x():
            self.manuallyNavigated.emit()
        super().mouseDragEvent(ev, axis=0)

    def wheelEvent(self, ev, axis=None) -> None:
        # Match ViewBox's native wheel scaling, with navigation-specific anchoring.
        factor = 1.02 ** (ev.delta() * self.state["wheelScaleFactor"])
        self.zoomRequested.emit(factor, self.mapToView(ev.pos()).x())
        ev.accept()


def nearest_marker(points: list[tuple[str, float, float]], x: float, y: float, radius: float = 8) -> str | None:
    """Hit-test in scene pixels, so temporal tolerance scales naturally with zoom."""
    if not points:
        return None
    key, px, py = min(points, key=lambda point: (point[1] - x) ** 2 + (point[2] - y) ** 2)
    return key if (px - x) ** 2 + (py - y) ** 2 <= radius**2 else None


class Timeline(QWidget):
    clicked = Signal(float, object, bool, object)
    manuallyNavigated = Signal()
    rangeChanged = Signal(float, float)
    zoomRequested = Signal(float, float)

    def __init__(self, analysis: AcousticAnalysis, proposal: Proposal[AutomaticCandidate]) -> None:
        super().__init__()
        self.analysis = analysis
        self.proposal = proposal
        self.wave_box = HorizontalViewBox()
        self.evidence_box = HorizontalViewBox()
        self.wave = pg.PlotItem(viewBox=self.wave_box, axisItems={"bottom": TimeAxis("bottom")})
        self.evidence = pg.PlotItem(viewBox=self.evidence_box, axisItems={"bottom": TimeAxis("bottom")})
        self.wave_widget = pg.PlotWidget(plotItem=self.wave)
        self.evidence_widget = pg.PlotWidget(plotItem=self.evidence)
        self.evidence_box.setXLink(self.wave_box)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.wave_widget)
        self.splitter.addWidget(self.evidence_widget)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([600, 200])
        self.splitter.setChildrenCollapsible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)
        self.amplitude = max(0.01, float(np.max(np.abs(analysis.mono))))
        self.lane_y = self.amplitude * 1.28
        self.wave_box.setYRange(-self.amplitude * 1.05, self.amplitude * 1.48, padding=0)
        self.wave.setLabel("left", "PCM amplitude")
        self.evidence.setLabel("left", "Onset evidence")
        self.evidence.setLabel("bottom", "Recording time", units="s")
        self.wave.hideAxis("bottom")
        self.waveform = self.wave.plot(
            np.arange(len(analysis.mono), dtype=np.float64) / PARAMETERS.sample_rate,
            analysis.mono,
            pen=pg.mkPen("#aaaeb5", width=1),
            autoDownsample=True,
            autoDownsampleFactor=1.0,
            downsampleMethod="peak",
        )
        self.bass = self.evidence.plot(analysis.timestamps, analysis.bass, pen="#6595dc", autoDownsample=True)
        self.broadband = self.evidence.plot(analysis.timestamps, analysis.broadband, pen="#c48aa8", autoDownsample=True)
        self.evidence_box.setYRange(
            0, max(1.1, float(np.max(analysis.bass)), float(np.max(analysis.broadband))) * 1.1, padding=0
        )
        for curve in (self.waveform, self.bass, self.broadband):
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            # Native Qt caching avoids repainting curves just to move a cursor.
            curve.curve.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.markers = pg.ScatterPlotItem(pxMode=True, size=10)
        # PyQtGraph dispatches clicks independently of Qt's acceptedMouseButtons.
        # The ViewBox owns all hit testing, including clicks directly on markers.
        self.markers.setEnabled(False)
        self.wave.addItem(self.markers)
        self.separator = pg.InfiniteLine(self.amplitude * 1.1, angle=0, pen=pg.mkPen("#555555"))
        self.wave.addItem(self.separator)
        self.scores = pg.ScatterPlotItem(size=5, pen=None, brush="#66b283")
        self.scores.setEnabled(False)
        self.evidence.addItem(self.scores)
        self.grid = pg.PlotDataItem(pen=pg.mkPen("#60656b", width=1, style=Qt.PenStyle.DashLine), connect="pairs")
        self.grid.setZValue(-10)
        self.wave.addItem(self.grid)
        self.cursor_time = 0.0
        self.cursor_offset = 0.0
        # Parent to each ViewBox itself, outside its transformed data group.
        # Scrolling then moves only data; the playhead stays at a stable pixel.
        self.cursor_lines = [QGraphicsLineItem(box) for box in (self.wave_box, self.evidence_box)]
        for line in self.cursor_lines:
            line.setPen(pg.mkPen("#00bbdb", width=2))
            line.setZValue(1000)
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.selection_lines = [pg.InfiniteLine(0, pen=pg.mkPen("#ef72c1", width=1)) for _ in range(2)]
        self.selection_outline = pg.ScatterPlotItem(size=18, symbol="o", brush=None, pen=pg.mkPen("#ff9adb", width=2))
        self.selection_outline.setEnabled(False)
        self.wave.addItem(self.selection_outline)
        for plot, selection in zip((self.wave, self.evidence), self.selection_lines, strict=True):
            plot.hideButtons()
            plot.setMenuEnabled(False)
            box = self.wave_box if plot is self.wave else self.evidence_box
            box.setLimits(
                xMin=0,
                xMax=analysis.duration_seconds,
                minXRange=min(0.02, analysis.duration_seconds),
                maxXRange=analysis.duration_seconds,
            )
            box.disableAutoRange()
            plot.addItem(selection, ignoreBounds=True)
            selection.hide()
            selection.setZValue(19)
        self.marker_keys: list[str] = []
        colors = ("#3184e0", "#a96872", "#2bad62", "#93979d", "#e7ae40", "#dca4df")
        self.marker_pens = {color: pg.mkPen(color, width=2 if color in colors[-2:] else 1) for color in colors}
        self.marker_brushes = {color: pg.mkBrush(color) for color in colors[:4]}
        self.marker_times = np.empty(0, dtype=np.float64)
        self.inspection: dict[str, str] = {}
        self.selected: str | None = None
        for box in (self.wave_box, self.evidence_box):
            box.clicked.connect(
                lambda timestamp, point, modifiers, inspect_only, source=box: self._click(
                    timestamp, point, modifiers, inspect_only, source is self.wave_box
                )
            )
            box.manuallyNavigated.connect(self.manuallyNavigated)
            box.zoomRequested.connect(self.zoomRequested)
            box.sigResized.connect(self._position_cursor)
            box.sigXRangeChanged.connect(self._position_cursor)
        self.wave_box.sigXRangeChanged.connect(self._range_changed)
        scene = self.wave_widget.scene()
        if isinstance(scene, pg.GraphicsScene):
            scene.sigMouseMoved.connect(self._hover)
        for plot in (self.wave, self.evidence):
            plot.getAxis("left").setWidth(70)

    @property
    def interval(self) -> tuple[float, float]:
        left, right = self.wave_box.viewRange()[0]
        return float(left), float(right)

    def set_interval(self, start: float, width: float) -> None:
        width = self._clamp_width(width)
        start = min(max(0.0, start), self.analysis.duration_seconds - width)
        if not all(
            math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
            for a, b in zip(self.interval, (start, start + width), strict=True)
        ):
            self.wave_box.setXRange(start, start + width, padding=0)

    def _clamp_width(self, width: float) -> float:
        return min(self.analysis.duration_seconds, max(0.02, width))

    def _range_changed(self, *_: object) -> None:
        left, right = self.interval
        self.rangeChanged.emit(left, right)

    def hit_test(self, point: QPointF) -> str | None:
        left, right = self.interval
        first, last = np.searchsorted(self.marker_times, [left, right])
        points = []
        for index in range(max(0, int(first) - 1), min(len(self.marker_keys), int(last) + 1)):
            scene = self.wave_box.mapViewToScene(QPointF(float(self.marker_times[index]), self.lane_y))
            points.append((self.marker_keys[index], scene.x(), scene.y()))
        return nearest_marker(points, point.x(), point.y())

    def _click(self, timestamp: float, point: QPointF, modifiers: object, inspect_only: bool, waveform: bool) -> None:
        if modifiers == Qt.KeyboardModifier.ShiftModifier and not 0 <= timestamp < self.analysis.duration_seconds:
            return
        self.clicked.emit(
            min(self.analysis.duration_seconds, max(0.0, timestamp)),
            self.hit_test(point) if waveform else None,
            inspect_only,
            modifiers,
        )

    def _hover(self, point: QPointF) -> None:
        key = self.hit_test(point)
        self.wave_widget.setToolTip(self.inspection.get(key, "") if key is not None else "")

    def set_cursor(self, timestamp: float, *, offset: float = 0) -> None:
        self.cursor_time = timestamp
        self.cursor_offset = offset
        self._position_cursor()

    def _position_cursor(self, *_: object) -> None:
        for box, line in zip((self.wave_box, self.evidence_box), self.cursor_lines, strict=True):
            left, right = box.viewRange()[0]
            x = box.width() * ((self.cursor_time + self.cursor_offset - left) / (right - left))
            line.setVisible(0 <= x <= box.width())
            # Snap the fixed playhead to a device pixel, including fractional DPI.
            ratio = self.devicePixelRatioF()
            x = round(x * ratio) / ratio
            line.setLine(x, 0, x, box.height())

    def select(self, key: str | None, editor: CorrectionEditor) -> None:
        self.selected = key
        candidate = editor.candidates.get(key) if key is not None else None
        for line in self.selection_lines:
            line.setVisible(candidate is not None)
            if candidate is not None:
                line.setPos(effective_timestamp(candidate, editor.draft))
        if candidate is None:
            self.selection_outline.setData([], [])
        else:
            self.selection_outline.setData([effective_timestamp(candidate, editor.draft)], [self.lane_y])

    def update_candidates(
        self,
        editor: CorrectionEditor,
        *,
        show_nonkicks: bool,
        uncertainty: bool,
        timing: bool,
        scores: bool,
        grid: bool,
        alternative: bool,
    ) -> None:
        selected = self.selected
        ordered = sorted(
            editor.candidates.values(), key=lambda c: (effective_timestamp(c, editor.draft), c["candidate_id"])
        )
        spots = []
        self.marker_keys = []
        times = []
        self.inspection = {}
        score_times, score_values = [], []
        for candidate in ordered:
            key = candidate["candidate_id"]
            kick = effective_kick(candidate, editor.draft)
            corrected = key in editor.draft.overrides
            doubtful = uncertain_classification(candidate, editor.draft)
            if not (kick or corrected or key == selected or show_nonkicks or uncertainty and doubtful):
                continue
            timestamp = effective_timestamp(candidate, editor.draft)
            color, symbol = (
                ("#3184e0", "s")
                if corrected and kick
                else ("#a96872", "x")
                if corrected
                else ("#2bad62", "o")
                if kick
                else ("#93979d", "t")
            )
            outline = "#e7ae40" if uncertainty and doubtful else color
            # Diagnostics change one marker outline, rather than piling symbols up.
            if timing and (not candidate["localization_reliable"] or not candidate["grid_aligned"]):
                outline = "#dca4df"
            spots.append(
                {
                    "pos": (timestamp, self.lane_y),
                    "symbol": symbol,
                    "brush": self.marker_brushes[color],
                    "pen": self.marker_pens[outline],
                }
            )
            self.marker_keys.append(key)
            times.append(timestamp)
            self.inspection[key] = (
                f"{round(timestamp)} s · {'Kick' if kick else 'Non-kick'}\n{candidate['timing_reason']}\n"
                + ", ".join(candidate["uncertainty_reasons"])
            )
            score = candidate.get("positive_similarity")
            if scores and score is not None:
                score_times.append(timestamp)
                score_values.append(score)
        self.marker_times = np.asarray(times)
        self.markers.setData(spots)
        self.scores.setData(score_times, score_values)
        beats = self.proposal.get("alternative_beat_positions", []) if alternative else self.proposal["beat_positions"]
        if grid:
            x = np.repeat(beats, 2)
            self.grid.setData(x, np.tile([-self.amplitude, self.amplitude], len(beats)))
        else:
            self.grid.setData([], [])
        self.select(selected, editor)

    def follow(self, timestamp: float, *, width: float | None = None, offset: float = 0) -> None:
        timestamp = min(self.analysis.duration_seconds, max(0, timestamp))
        left, right = self.interval
        width = self._clamp_width(right - left if width is None else width)
        # Range-change signals render this same snapshot, never a newer clock read.
        self.cursor_time = timestamp
        self.cursor_offset = offset
        self.set_interval(timestamp - width * PLAYBACK_ANCHOR_FRACTION, width)
        self._position_cursor()

    def zoom(
        self, factor: float, mouse_time: float, *, playback_position: float | None = None, offset: float = 0
    ) -> None:
        left, right = self.interval
        width = self._clamp_width((right - left) * factor)
        if playback_position is not None:
            self.follow(playback_position, width=width, offset=offset)
        else:
            # Native scaling preserves the mouse anchor and applies ViewBox limits.
            self.wave_box.scaleBy(x=width / (right - left), center=QPointF(mouse_time, 0))
