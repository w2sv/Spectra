"""VisPy recording timeline; editing and DAC transport remain in the review window."""

import math
from dataclasses import dataclass
from time import monotonic
from typing import Literal

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vispy import scene
from vispy.color import Color
from vispy.visuals.transforms import STTransform

from benchmarks.annotation.analysis import PARAMETERS, AcousticAnalysis, AutomaticCandidate, Proposal
from benchmarks.annotation.corrections import (
    CorrectionEditor,
    effective_kick,
    effective_timestamp,
    uncertain_classification,
)
from benchmarks.annotation.review.styles import (
    AUTOMATIC_KICK,
    BASS,
    BEAT,
    BROADBAND,
    CANDIDATE,
    HUMAN_KICK,
    KICK_FLASH,
    MARKER_STYLES,
    PLAYBACK,
    REJECTED,
    SCORE,
    SELECTION,
    TIMING,
    UNCERTAINTY,
)
from spectra.ui.plots import PlotCanvas, RecordingWaveform, ScreenLine, TimeAxis, line, markers

PLAYBACK_ANCHOR_FRACTION = 0.4
ZOOM_TRANSITION_SECONDS = 0.15


@dataclass(frozen=True)
class ZoomTransition:
    started: float
    initial_left: float
    initial_width: float
    target_left: float
    target_width: float


def nearest_marker(points: list[tuple[str, float, float]], x: float, y: float, radius: float = 8) -> str | None:
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
        self.analysis, self.proposal = analysis, proposal
        self.zoom_transition: ZoomTransition | None = None
        self._interval = (0.0, min(10.0, analysis.duration_seconds))
        self.canvas = PlotCanvas()
        self.native = self.canvas.native
        self.native.installEventFilter(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.native)
        self.wave_camera, self.evidence_camera = scene.PanZoomCamera(), scene.PanZoomCamera()
        self.wave_camera.interactive = self.evidence_camera.interactive = False
        self.wave_view = scene.widgets.ViewBox(parent=self.canvas.root, camera=self.wave_camera)
        self.evidence_view = scene.widgets.ViewBox(parent=self.canvas.root, camera=self.evidence_camera)
        wave_scene, evidence_scene = self.wave_view.scene, self.evidence_view.scene
        assert wave_scene is not None and evidence_scene is not None
        self.wave_scene, self.evidence_scene = wave_scene, evidence_scene
        self.waveform = RecordingWaveform(wave_scene, analysis.mono, PARAMETERS.sample_rate)
        self.bass = line(evidence_scene, BASS.color)
        self.broadband = line(evidence_scene, BROADBAND.color)
        self.markers, self.scores = markers(wave_scene), markers(evidence_scene)
        self.grid = line(wave_scene, BEAT.color, segments=True)
        self.separators = line(wave_scene, "#555555", segments=True)
        self.axis = TimeAxis(self.canvas)
        self.y_labels = scene.visuals.Text(
            parent=self.canvas.root,  # pyright: ignore[reportCallIssue]
            color="white",
            font_size=9,
            anchor_x="right",
        )
        self.titles = scene.visuals.Text(
            parent=self.canvas.root,  # pyright: ignore[reportCallIssue]
            text=["PCM", "Evidence"],
            color="white",
            font_size=9,
        )
        self.playhead, self.selection = ScreenLine(self.canvas.root), ScreenLine(self.canvas.root)
        self.selection.visual.order = 90
        self.selection_outline = markers(wave_scene)
        self.amplitude = max(0.01, float(np.max(np.abs(analysis.mono))))
        self.evidence_ymax = max(1.1, float(np.max(analysis.bass)), float(np.max(analysis.broadband))) * 1.1
        self.kick_y, self.candidate_y = self.amplitude * 1.28, self.amplitude * 1.1
        self.wave_fraction = 0.75
        self.marker_keys: list[str] = []
        self.marker_times = np.empty(0, dtype=np.float64)
        self.marker_ys = np.empty(0, dtype=np.float64)
        self.marker_kicks = np.empty(0, dtype=np.bool_)
        self.marker_arrays: dict = {}
        self.score_times = np.empty(0, dtype=np.float64)
        self.score_values = np.empty(0, dtype=np.float64)
        self.marker_colors = {style.color: Color(style.color).rgba for style in MARKER_STYLES}
        self.inspection: dict[str, str] = {}
        self.selected: str | None = None
        self.selected_time: float | None = None
        self.selected_kick = False
        self.cursor_time = self.cursor_offset = 0.0
        self.flashing = False
        self.editor: CorrectionEditor | None = None
        self.marker_options: dict = {}
        self.geometry_window: tuple[float, float] | None = None
        self.beat_positions = np.empty(0, dtype=np.float64)
        self.canvas.before_draw = self._prepare_geometry
        self._layout_key: object = None
        self._press: QPointF | None = None
        self._last_mouse = QPointF()
        self._button = Qt.MouseButton.NoButton
        self._dragged = False
        self._splitting = False
        self.canvas.events["resize"].connect(self._resize, position="last")
        self._layout_views()

    @property
    def interval(self) -> tuple[float, float]:
        return self._interval

    def _clamp_width(self, width: float) -> float:
        return min(self.analysis.duration_seconds, max(0.02, width))

    def set_interval(self, start: float, width: float) -> None:
        self.cancel_zoom()
        self._set_interval(start, width)

    def _set_interval(self, start: float, width: float) -> None:
        width = self._clamp_width(width)
        start = min(max(0, start), self.analysis.duration_seconds - width)
        changed = not all(
            math.isclose(a, b, abs_tol=1e-10, rel_tol=0)
            for a, b in zip(self.interval, (start, start + width), strict=True)
        )
        self._interval = (start, start + width)
        if changed:
            self._update_view()
            self.rangeChanged.emit(start, start + width)
        self._position_cursor()
        self._position_selection()

    def _layout_views(self) -> None:
        width, height = self.canvas.size
        plot_width, available = max(1, width - 80), max(140, height - 50)
        wave_height = max(100, available * self.wave_fraction)
        self.wave_view.pos, self.wave_view.size = (75, 0), (plot_width, wave_height)
        self.evidence_view.pos, self.evidence_view.size = (
            (75, wave_height + 6),
            (plot_width, max(30, available - wave_height - 6)),
        )
        units = self.amplitude * 2.05 / max(1, wave_height - 56)
        self.candidate_y, self.kick_y = self.amplitude + units * 14, self.amplitude + units * 42
        self._update_view()
        key = (width, height, self.wave_fraction)
        if key != self._layout_key:
            self._layout_key = key
            self.separators.set_data(
                pos=np.array(
                    [
                        [0, self.amplitude],
                        [self.analysis.duration_seconds, self.amplitude],
                        [0, self.amplitude + units * 28],
                        [self.analysis.duration_seconds, self.amplitude + units * 28],
                    ],
                    dtype="float32",
                )
            )
            self.geometry_window = None
            if self.editor is not None:
                self.update_candidates(self.editor, **self.marker_options)
            labels, positions = [], []
            for evidence, values in (
                (False, [-self.amplitude, 0, self.amplitude]),
                (True, np.linspace(0, self.evidence_ymax, 4)),
            ):
                for value in values:
                    p = self.time_to_point(self.interval[0], float(value), evidence=evidence)
                    positions.append((p.x() - 8, p.y()))
                    labels.append(f"{value:.2g}")
            self.y_labels.text, self.y_labels.pos = labels, positions
            self.titles.pos = [(35, 12), (35, self.evidence_view.pos[1] + 12)]
        self._position_cursor()
        self._position_selection()

    def _update_view(self) -> None:
        left, right = self.interval
        h = self.wave_view.size[1]
        units = self.amplitude * 2.05 / max(1, h - 56)
        self.wave_camera.rect = (left, -self.amplitude * 1.05, right - left, self.amplitude * 2.05 + units * 56)
        self.evidence_camera.rect = (left, 0, right - left, self.evidence_ymax)
        self.axis.update(
            left, right, 75, self.evidence_view.pos[1] + self.evidence_view.size[1] + 3, self.wave_view.size[0]
        )
        self.canvas.update()

    def _prepare_geometry(self) -> None:
        left, right = self.interval
        self.waveform.update(left, right, max(1, round(self.wave_view.size[0] * self.canvas.pixel_scale)))
        self._update_geometry_window()

    def time_to_point(self, timestamp: float, y: float = 0, *, evidence: bool = False) -> QPointF:
        left, right = self.interval
        view = self.evidence_view if evidence else self.wave_view
        camera = self.evidence_camera if evidence else self.wave_camera
        rect = camera.rect
        assert rect is not None
        return QPointF(
            float(view.pos[0] + view.size[0] * (timestamp - left) / (right - left)),
            float(view.pos[1] + view.size[1] * (rect.top - y) / rect.height),
        )

    def point_to_time(self, point: QPointF) -> float:
        left, right = self.interval
        return left + (point.x() - self.wave_view.pos[0]) / max(1, self.wave_view.size[0]) * (right - left)

    def interaction_region(self, point: QPointF) -> Literal["waveform", "evidence", "divider"] | None:
        x, y = self.wave_view.pos
        width, height = self.wave_view.size
        if QRectF(x, y + height, width, self.evidence_view.pos[1] - y - height).contains(point):
            return "divider"
        for view in (self.wave_view, self.evidence_view):
            x, y = view.pos
            width, height = view.size
            if QRectF(x, y, width, height).contains(point):
                return "waveform" if view is self.wave_view else "evidence"
        return None

    def hit_test(self, point: QPointF) -> str | None:
        if self.interaction_region(point) != "waveform":
            return None
        timestamp = self.point_to_time(point)
        left, right = self.interval
        radius = 8 * (right - left) / max(1, self.wave_view.size[0])
        first, last = np.searchsorted(self.marker_times, [timestamp - radius, timestamp + radius])
        points = []
        for i in range(int(first), int(last)):
            p = self.time_to_point(float(self.marker_times[i]), float(self.marker_ys[i]))
            points.append((self.marker_keys[i], p.x(), p.y()))
        return nearest_marker(points, point.x(), point.y())

    def set_cursor(self, timestamp: float, *, offset: float = 0) -> None:
        self.cursor_time, self.cursor_offset = timestamp, offset
        self._position_cursor()

    def _position_cursor(self) -> None:
        left, right = self.interval
        value = self.cursor_time + self.cursor_offset
        p = self.time_to_point(value)
        style = KICK_FLASH if self.flashing else PLAYBACK
        self.playhead.set(
            p.x(),
            0,
            self.evidence_view.pos[1] + self.evidence_view.size[1],
            style.width,
            style.color,
            left <= value <= right,
            self.devicePixelRatioF(),
        )

    def set_flash(self, flashing: bool) -> None:
        if flashing != self.flashing:
            self.flashing = flashing
            self._position_cursor()

    def select(self, key: str | None, editor: CorrectionEditor) -> None:
        self.selected = key
        candidate = editor.candidates.get(key) if key else None
        self.selected_time = effective_timestamp(candidate, editor.draft) if candidate is not None else None
        self.selected_kick = effective_kick(candidate, editor.draft) if candidate is not None else False
        self._position_selection()
        self.selection_outline.visible = candidate is not None
        if candidate is not None:
            self.selection_outline.transform = STTransform(translate=(self.selected_time, 0))
            self.selection_outline.set_data(
                pos=np.array([[0, self.kick_y if self.selected_kick else self.candidate_y]], dtype="float32"),
                size=18,
                face_color=(0, 0, 0, 0),  # pyright: ignore[reportArgumentType]
                edge_color=SELECTION.color,
                edge_width=2,
            )

    def _position_selection(self) -> None:
        left, right = self.interval
        value = self.selected_time
        self.selection.set(
            self.time_to_point(value or 0).x(),
            0,
            self.evidence_view.pos[1] + self.evidence_view.size[1],
            1,
            SELECTION.color,
            value is not None and left <= value <= right,
            self.devicePixelRatioF(),
        )

    def update_candidates(
        self,
        editor: CorrectionEditor,
        *,
        show_candidates: bool,
        uncertainty: bool,
        timing: bool,
        scores: bool,
        grid: bool,
        alternative: bool,
    ) -> None:
        self.editor = editor
        self.marker_options = dict(
            show_candidates=show_candidates,
            uncertainty=uncertainty,
            timing=timing,
            scores=scores,
            grid=grid,
            alternative=alternative,
        )
        times, ys, colors, edges, sizes, symbols, widths, kicks, score_times, score_values = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.marker_keys = []
        self.inspection = {}
        for candidate in sorted(
            editor.candidates.values(), key=lambda c: (effective_timestamp(c, editor.draft), c["candidate_id"])
        ):
            key = candidate["candidate_id"]
            kick = effective_kick(candidate, editor.draft)
            corrected = key in editor.draft.overrides
            doubtful = uncertain_classification(candidate, editor.draft)
            if not (kick or corrected or key == self.selected or show_candidates or uncertainty and doubtful):
                continue
            timestamp = effective_timestamp(candidate, editor.draft)
            style = (
                HUMAN_KICK if corrected and kick else REJECTED if corrected else AUTOMATIC_KICK if kick else CANDIDATE
            )
            diagnostic = (
                TIMING
                if timing and (not candidate["localization_reliable"] or not candidate["grid_aligned"])
                else UNCERTAINTY
                if uncertainty and doubtful
                else None
            )
            self.marker_keys.append(key)
            times.append(timestamp)
            ys.append(self.kick_y if kick else self.candidate_y)
            kicks.append(kick)
            colors.append(self.marker_colors[style.color])
            edges.append(self.marker_colors[diagnostic.color if diagnostic else style.color])
            sizes.append(diagnostic.size if diagnostic else style.size)
            widths.append(2 if diagnostic else 1)
            symbols.append({"o": "disc", "s": "square", "x": "x"}[style.symbol])
            self.inspection[key] = f"{timestamp:.4f} s · {style.label}\n{candidate['timing_reason']}\n" + ", ".join(
                candidate["uncertainty_reasons"]
            )
            score = candidate.get("positive_similarity")
            if scores and score is not None:
                score_times.append(timestamp)
                score_values.append(score)
        self.marker_times = np.asarray(times, dtype=np.float64)
        self.marker_ys = np.asarray(ys, dtype=np.float64)
        self.marker_kicks = np.asarray(kicks, dtype=np.bool_)
        self.marker_arrays = dict(
            face_color=np.asarray(colors).reshape(-1, 4),
            edge_color=np.asarray(edges).reshape(-1, 4),
            size=np.asarray(sizes),
            symbol=np.asarray(symbols),
            edge_width=np.asarray(widths),
        )
        self.score_times = np.asarray(score_times)
        self.score_values = np.asarray(score_values)
        self.geometry_window = None
        self.beat_positions = np.asarray(
            self.proposal.get("alternative_beat_positions", []) if alternative else self.proposal["beat_positions"],
            dtype=np.float64,
        )
        self.grid.visible = grid and bool(len(self.beat_positions))
        self.canvas.update()
        self.select(self.selected, editor)

    def _update_geometry_window(self) -> None:
        if self.editor is None:
            return
        left, right = self.interval
        cached = self.geometry_window
        if cached is not None and cached[0] <= left and cached[1] >= right:
            return
        pad = (right - left) / 2
        start, end = max(0, left - pad), min(self.analysis.duration_seconds, right + pad)
        self.geometry_window = (start, end)
        origin = start
        first, last = np.searchsorted(self.marker_times, [start, end])
        selected = slice(int(first), int(last))
        self.markers.visible = last > first
        if last > first:
            self.markers.transform = STTransform(translate=(origin, 0))
            self.markers.set_data(
                pos=np.column_stack((self.marker_times[selected] - origin, self.marker_ys[selected])).astype("float32"),
                **{key: value[selected] for key, value in self.marker_arrays.items()},
            )
        first, last = np.searchsorted(self.score_times, [start, end])
        self.scores.visible = last > first
        if last > first:
            self.scores.transform = STTransform(translate=(origin, 0))
            self.scores.set_data(
                pos=np.column_stack((self.score_times[first:last] - origin, self.score_values[first:last])).astype(
                    "float32"
                ),
                face_color=SCORE.color,
                size=5,
            )
        first, last = np.searchsorted(self.analysis.timestamps, [start, end])
        selected = slice(max(0, int(first) - 1), min(len(self.analysis.timestamps), int(last) + 1))
        for visual, values in ((self.bass, self.analysis.bass), (self.broadband, self.analysis.broadband)):
            visual.transform = STTransform(translate=(origin, 0))
            visual.set_data(
                pos=np.column_stack((self.analysis.timestamps[selected] - origin, values[selected])).astype("float32")
            )
        beats = self.beat_positions
        beats = beats[(beats >= start) & (beats <= end)]
        self.grid.visible = self.marker_options["grid"] and bool(len(beats))
        if self.grid.visible:
            height = max(1, self.wave_view.size[1] - 56)
            ys = np.arange(0, height, 6)
            low = -self.amplitude + ys / height * self.amplitude * 2
            high = -self.amplitude + np.minimum(ys + 4, height) / height * self.amplitude * 2
            dashed = np.column_stack((low, high)).ravel()
            self.grid.transform = STTransform(translate=(origin, 0))
            self.grid.set_data(
                pos=np.column_stack((np.repeat(beats - origin, len(dashed)), np.tile(dashed, len(beats)))).astype(
                    "float32"
                )
            )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is not self.native:
            return super().eventFilter(watched, event)
        if isinstance(event, QWheelEvent):
            if event.angleDelta().y() and self.interaction_region(event.position()) in ("waveform", "evidence"):
                self.zoomRequested.emit(1.02 ** (-event.angleDelta().y() / 8), self.point_to_time(event.position()))
            return True
        if isinstance(event, QMouseEvent):
            point = event.position()
            if event.type() == QEvent.Type.MouseButtonPress:
                region = self.interaction_region(point)
                if region is None:
                    self._press = None
                    return True
                self.native.setFocus(Qt.FocusReason.MouseFocusReason)
                self._press = point
                self._last_mouse = point
                self._button = event.button()
                self._dragged = False
                self._splitting = region == "divider"
                return True
            if event.type() == QEvent.Type.MouseMove:
                if self._press is not None and self._button in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
                    if (point - self._press).manhattanLength() > 3:
                        self._dragged = True
                    if self._dragged:
                        if self._splitting:
                            self.wave_fraction = min(0.85, max(0.2, point.y() / max(1, self.canvas.size[1] - 50)))
                            self._layout_views()
                        elif point.x() != self._last_mouse.x():
                            self.manuallyNavigated.emit()
                            left, right = self.interval
                            self.set_interval(
                                left
                                - (point.x() - self._last_mouse.x()) / max(1, self.wave_view.size[0]) * (right - left),
                                right - left,
                            )
                    self._last_mouse = point
                else:
                    key = self.hit_test(point)
                    self.native.setToolTip(self.inspection.get(key, "") if key else "")
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                if (
                    self._press is not None
                    and not self._dragged
                    and not self._splitting
                    and self.interaction_region(point) in ("waveform", "evidence")
                    and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
                ):
                    timestamp = self.point_to_time(point)
                    if (
                        event.modifiers() != Qt.KeyboardModifier.ShiftModifier
                        or 0 <= timestamp < self.analysis.duration_seconds
                    ):
                        self.clicked.emit(
                            min(self.analysis.duration_seconds, max(0, timestamp)),
                            self.hit_test(point),
                            event.button() == Qt.MouseButton.RightButton,
                            event.modifiers(),
                        )
                self._press = None
                self._splitting = False
                return True
        return super().eventFilter(watched, event)

    def _resize(self, event) -> None:
        self._layout_views()

    def follow(self, timestamp: float, *, width: float | None = None, offset: float = 0) -> None:
        timestamp = min(self.analysis.duration_seconds, max(0, timestamp))
        left, right = self.interval
        width = self._clamp_width(right - left if width is None else width)
        # Range-change signals render this same snapshot, never a newer clock read.
        self.cursor_time = timestamp
        self.cursor_offset = offset
        self._set_interval(timestamp - width * PLAYBACK_ANCHOR_FRACTION, width)
        self._position_cursor()

    def zoom(
        self, factor: float, mouse_time: float, *, playback_position: float | None = None, offset: float = 0
    ) -> None:
        # The wheel event's mouse time belongs to the range still on screen.
        left, right = self.interval
        fraction = min(1.0, max(0.0, (mouse_time - left) / (right - left)))
        now = monotonic()
        self.advance_zoom(now, playback_position=playback_position, offset=offset)
        left, right = self.interval
        current_width = right - left
        transition = self.zoom_transition
        target_width = transition.target_width if transition is not None else current_width
        target_left = transition.target_left if transition is not None else left
        width = self._clamp_width(target_width * factor)
        anchor = target_left + fraction * target_width
        target_left = min(max(0.0, anchor - fraction * width), self.analysis.duration_seconds - width)
        self.zoom_transition = ZoomTransition(now, left, current_width, target_left, width)

    @property
    def target_width(self) -> float:
        left, right = self.interval
        return self.zoom_transition.target_width if self.zoom_transition is not None else right - left

    def cancel_zoom(self) -> None:
        self.zoom_transition = None

    def advance_zoom(self, now: float, *, playback_position: float | None = None, offset: float = 0) -> bool:
        transition = self.zoom_transition
        if transition is None:
            return False
        progress = min(1.0, max(0.0, (now - transition.started) / ZOOM_TRANSITION_SECONDS))
        blend = progress * progress * (3 - 2 * progress)
        width = math.exp(math.log(transition.initial_width) * (1 - blend) + math.log(transition.target_width) * blend)
        if playback_position is not None:
            self.follow(playback_position, width=width, offset=offset)
        else:
            delta = transition.target_width - transition.initial_width
            fraction = (width - transition.initial_width) / delta if delta else blend
            left = transition.initial_left + fraction * (transition.target_left - transition.initial_left)
            self._set_interval(left, width)
        if progress == 1:
            self.zoom_transition = None
        return True
