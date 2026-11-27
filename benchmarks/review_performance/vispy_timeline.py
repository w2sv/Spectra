"""SceneCanvas prototype using existing visuals and resident static geometry.

Only navigation methods are inherited from the current editor; its constructor
and graphics objects are not used. There is no production renderer interface.
"""

import math
from time import perf_counter

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vispy import app, scene
from vispy.color import Color

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
from benchmarks.annotation.review.timeline import Timeline, nearest_marker
from benchmarks.review_performance.gpu_timing import GPUTimer
from benchmarks.review_performance.metrics import Measurements
from benchmarks.review_performance.waveform import WaveformPyramid, envelope, pcm_points

# Scene visual classes are generated dynamically; use their runtime constructors
# because the upstream annotations omit Node keywords and array marker styles.


class MeasuredCanvas(scene.SceneCanvas):
    def __init__(self, measurements: Measurements) -> None:
        self.measurements = measurements
        self.gpu_timer: GPUTimer | None = None
        super().__init__(keys=None, show=False, bgcolor="black", vsync=False, config={"samples": 4})

    def on_draw(self, event) -> None:
        started = perf_counter()
        if self.gpu_timer is not None and self.measurements.active:
            self.gpu_timer.begin()
        super().on_draw(event)
        if self.gpu_timer is not None and self.measurements.active:
            self.gpu_timer.end()
        self.measurements.painted("canvas", started)


class VispyTimeline(Timeline):
    def __init__(
        self,
        analysis: AcousticAnalysis,
        proposal: Proposal[AutomaticCandidate],
        measurements: Measurements,
        strategy: str = "lod",
        axes: str = "cached",
    ) -> None:
        QWidget.__init__(self)
        app.use_app("pyside6")
        self.analysis = analysis
        self.proposal = proposal
        self.measurements = measurements
        self.strategy = strategy
        self.zoom_transition = None
        self.cursor_time = self.cursor_offset = 0.0
        self.flashing = False
        self.selected = None
        self.marker_keys = []
        self.marker_times = np.empty(0)
        self.marker_ys = np.empty(0)
        self.inspection = {}
        self.marker_colors = {style.color: Color(style.color).rgba for style in MARKER_STYLES}
        self.evidence_ymax = max(1.1, float(np.max(analysis.bass)), float(np.max(analysis.broadband))) * 1.1
        self.grid_key: object = None
        self.amplitude = max(0.01, float(np.max(np.abs(analysis.mono))))
        self.kick_y = self.amplitude * 1.28
        self.candidate_y = self.amplitude * 1.1
        self._interval = (0.0, 10.0)
        self.canvas = MeasuredCanvas(measurements)
        self.canvas.unfreeze()
        grid = self.canvas.central_widget.add_grid(margin=0, spacing=5)
        self.wave_camera = scene.PanZoomCamera()
        self.evidence_camera = scene.PanZoomCamera()
        self.wave_view = scene.widgets.ViewBox(camera=self.wave_camera)
        self.evidence_view = scene.widgets.ViewBox(camera=self.evidence_camera)
        grid.add_widget(self.wave_view, row=0, col=1)
        grid.add_widget(self.evidence_view, row=1, col=1)
        wave_scene = self.wave_view.scene
        evidence_scene = self.evidence_view.scene
        assert wave_scene is not None and evidence_scene is not None
        self.wave_scene = wave_scene
        self.evidence_scene = evidence_scene
        self.wave_view.stretch = (1, 3)
        self.evidence_view.stretch = (1, 1)
        # Navigation belongs to the shared timeline math, not VisPy's camera handlers.
        self.wave_camera.interactive = False
        self.evidence_camera.interactive = False
        self.axes_enabled = axes
        self.tick_key: object = None
        self.axis_view = None
        self.axis_camera = None
        self.tick_text = None
        self.tick_line = None
        self.y_text = None
        self.y_tick_line = None
        self.y_axis_key: object = None
        self.axis_titles = None
        self.axes = []
        if axes == "stock":
            for row, view in enumerate((self.wave_view, self.evidence_view)):
                axis = scene.AxisWidget(orientation="left", axis_label="PCM" if row == 0 else "Evidence", font_size=9)
                axis.width_max = axis.width_min = 70
                grid.add_widget(axis, row=row, col=0)
                axis.link_view(view)
                self.axes.append(axis)
            axis = scene.AxisWidget(orientation="bottom", axis_label="Recording time (s)", font_size=9)
            axis.height_max = axis.height_min = 45
            grid.add_widget(axis, row=2, col=1)
            axis.link_view(self.evidence_view)
            self.axes.append(axis)
        elif axes == "cached":
            # Profiling stock AxisWidget showed repeated text preparation even
            # when y ticks and tick strings had not changed. Keep those glyph
            # buffers resident, and scroll x ticks using the camera alone.
            gutter = scene.widgets.Widget()
            gutter.width_min = gutter.width_max = 70
            grid.add_widget(gutter, row=0, col=0, row_span=3)
            self.axis_camera = scene.PanZoomCamera(rect=(0, 0, 10, 1))
            self.axis_camera.interactive = False
            self.axis_view = scene.widgets.ViewBox(camera=self.axis_camera)
            self.axis_view.height_min = self.axis_view.height_max = 45
            grid.add_widget(self.axis_view, row=2, col=1)
            self.tick_text = vars(scene.visuals)["Text"](
                parent=self.axis_view.scene, color="white", font_size=9, anchor_y="top"
            )
            self.tick_line = self._line(self.axis_view, "#aaaaaa")
            self.tick_line.set_data(connect="segments")
            self.y_text = vars(scene.visuals)["Text"](
                parent=self.canvas.scene, color="white", font_size=9, anchor_x="right"
            )
            self.y_tick_line = vars(scene.visuals)["Line"](
                parent=self.canvas.scene, color="#aaaaaa", connect="segments"
            )
        if axes == "cached":
            self.axis_titles = vars(scene.visuals)["Text"](parent=self.canvas.scene, color="white", font_size=9)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)
        self.wave_lines: list = []
        self.pyramid = WaveformPyramid(analysis.mono, PARAMETERS.sample_rate) if strategy == "lod" else None
        self.preprocessing_seconds = self.pyramid.seconds if self.pyramid is not None else 0.0
        self.geometry_cpu_bytes = self.pyramid.bytes if self.pyramid is not None else 0
        self.waveform = self._line(self.wave_view, "#aaaeb5")
        self.wave_key: object = None
        if self.pyramid is not None:
            for level in self.pyramid.levels:
                line = self._line(self.wave_view, "#aaaeb5", level.points)
                line.visible = False
                self.wave_lines.append(line)
        elif strategy == "resident":
            points = pcm_points(analysis.mono, PARAMETERS.sample_rate, 0, len(analysis.mono))
            points[:, 0] = np.arange(len(analysis.mono), dtype="float32")
            from vispy.visuals.transforms import STTransform

            self.waveform.transform = STTransform(scale=(1 / PARAMETERS.sample_rate, 1))
            self.waveform.set_data(pos=points)
            self.geometry_cpu_bytes = points.nbytes
        self.bass = self._line(
            self.evidence_view, BASS.color, np.column_stack((analysis.timestamps, analysis.bass)).astype("float32")
        )
        self.broadband = self._line(
            self.evidence_view,
            BROADBAND.color,
            np.column_stack((analysis.timestamps, analysis.broadband)).astype("float32"),
        )
        self.markers = vars(scene.visuals)["Markers"](parent=self.wave_view.scene, scaling="fixed", antialias=1)
        self.scores = vars(scene.visuals)["Markers"](parent=self.evidence_view.scene, scaling="fixed")
        self.grid_line = self._line(self.wave_view, BEAT.color)
        self.grid_line.set_data(connect="segments")
        self.separators = self._line(self.wave_view, "#555555")
        self.separators.set_data(connect="segments")
        # Cursor geometry stays resident; only a screen-space transform moves.
        self.playhead = vars(scene.visuals)["Mesh"](
            vertices=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32"),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype="uint32"),
            color=PLAYBACK.color,
            parent=self.canvas.scene,
        )
        from vispy.visuals.transforms import STTransform

        self.playhead.transform = STTransform()
        self.cursor_snapshot: object = None
        self.selection_lines = [self._line(view, SELECTION.color) for view in (self.wave_view, self.evidence_view)]
        self.selection_outline = vars(scene.visuals)["Markers"](parent=self.wave_view.scene, scaling="fixed")
        self.marker_data: list[tuple[str, float, bool]] = []
        self.marker_options: dict = {}
        self.editor: CorrectionEditor | None = None
        self.canvas.events["mouse_wheel"].connect(self._wheel)
        self.canvas.events["mouse_move"].connect(self._move)
        self.canvas.events["mouse_release"].connect(self._release)
        self.canvas.events["resize"].connect(self._resize, position="last")
        self._set_interval(0, 10)

    @staticmethod
    def _line(view, color: str, points=None):
        return vars(scene.visuals)["Line"](
            pos=points, color=color, width=1, method="gl", antialias=True, parent=view.scene
        )

    @property
    def interval(self) -> tuple[float, float]:
        return self._interval

    def _set_interval(self, start: float, width: float) -> None:
        width = self._clamp_width(width)
        start = min(max(0, start), self.analysis.duration_seconds - width)
        changed = self._interval != (start, start + width)
        self._interval = (start, start + width)
        if changed:
            self._layout_views()
        if changed:
            self.rangeChanged.emit(start, start + width)
        self._position_cursor()

    def _layout_views(self) -> None:
        left, right = self.interval
        height = max(112, self.wave_view.size[1])
        units = self.amplitude * 2.05 / (height - 56)
        old_lanes = (self.kick_y, self.candidate_y)
        self.kick_y, self.candidate_y = self.amplitude + units * 42, self.amplitude + units * 14
        self.wave_camera.rect = (left, -self.amplitude * 1.05, right - left, self.amplitude * 2.05 + units * 56)
        self.evidence_camera.rect = (left, 0, right - left, self.evidence_ymax)
        if old_lanes != (self.kick_y, self.candidate_y):
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
            if self.editor is not None:
                self.update_candidates(self.editor, **self.marker_options)
        self._wave_geometry()
        self._update_axes()

    def _update_axes(self) -> None:
        if (
            self.axis_camera is None
            or self.axis_view is None
            or self.tick_text is None
            or self.tick_line is None
            or self.y_text is None
            or self.y_tick_line is None
        ):
            return
        left, right = self.interval
        self.axis_camera.rect = (left, 0, right - left, 1)
        raw = (right - left) / 7
        power = 10 ** math.floor(math.log10(raw))
        spacing = next(value * power for value in (1, 2, 5, 10) if value * power >= raw)
        if right - left >= 2:
            spacing = max(1, spacing)
        first = math.floor(left / spacing)
        last = math.ceil(right / spacing)
        key = (first, last, spacing)
        if key != self.tick_key:
            self.tick_key = key
            ticks = np.arange(first, last + 1) * spacing
            self.tick_text.text = [f"{value:g}" for value in ticks]
            self.tick_text.pos = np.column_stack((ticks, np.full(len(ticks), 0.72)))
            self.tick_line.set_data(
                pos=np.column_stack((np.repeat(ticks, 2), np.tile([0.85, 1], len(ticks)))).astype("float32")
            )
        key_y = (tuple(self.wave_view.size), tuple(self.evidence_view.size), tuple(self.evidence_view.pos))
        if key_y != self.y_axis_key:
            self.y_axis_key = key_y
            points, labels, segments = [], [], []
            for view, values in (
                (self.wave_view, [-self.amplitude, 0, self.amplitude]),
                (self.evidence_view, [0, 1, 2]),
            ):
                camera = self.wave_camera if view is self.wave_view else self.evidence_camera
                rect = camera.rect
                assert rect is not None
                for value in values:
                    x = view.pos[0]
                    y = view.pos[1] + view.size[1] * (rect.top - value) / rect.height
                    points.append((x - 8, y))
                    labels.append(f"{value:.2g}")
                    segments.extend([(x - 4, y), (x, y)])
            self.y_text.text = labels
            self.y_text.pos = np.asarray(points, dtype="float32")
            self.y_tick_line.set_data(pos=np.asarray(segments, dtype="float32"))
            if self.axis_titles is not None:
                self.axis_titles.text = ["PCM", "Evidence", "Recording time (s)"]
                self.axis_titles.pos = [
                    (35, 12),
                    (35, self.evidence_view.pos[1] + 12),
                    (self.wave_view.pos[0] + self.wave_view.size[0] / 2, self.canvas.size[1] - 8),
                ]

    def _wave_geometry(self) -> None:
        if self.strategy in ("resident", "stream"):
            return
        left, right = self.interval
        rate = PARAMETERS.sample_rate
        pixels = max(1, round(self.wave_view.size[0] * self.canvas.pixel_scale))
        level = self.pyramid.level((right - left) * rate, pixels) if self.pyramid is not None else None
        if level is not None and self.pyramid is not None:
            key = ("level", level.block)
            if key == self.wave_key:
                return
            self.waveform.visible = False
            for item, candidate in zip(self.wave_lines, self.pyramid.levels, strict=True):
                item.visible = candidate is level
            self.wave_key = key
            return
        first = max(0, int(left * rate) - 1)
        last = min(len(self.analysis.mono), int(right * rate) + 2)
        key = (first, last, pixels)
        if key == self.wave_key:
            return
        self.wave_key = key
        for line in self.wave_lines:
            line.visible = False
        self.waveform.visible = True
        samples_per_pixel = (last - first) / pixels
        if self.strategy == "visible" and samples_per_pixel > 2:
            block = max(1, int(samples_per_pixel))
            points = envelope(self.analysis.mono[first:last], rate, block)
            points[:, 0] += first / rate
            from vispy.visuals.transforms import STTransform

            self.waveform.transform = STTransform()
        else:
            points = pcm_points(self.analysis.mono[first:last], rate, 0, last - first)
            from vispy.visuals.transforms import STTransform

            self.waveform.transform = STTransform(translate=(first / rate, 0))
        self.waveform.set_data(pos=points)
        if self.measurements.active:
            self.measurements.rebuilds += 1
            self.measurements.cpu_geometry_bytes += points.nbytes

    def _position_cursor(self, *_: object) -> None:
        left, right = self.interval
        ratio = self.canvas.pixel_scale
        x = (
            round(
                (
                    self.wave_view.pos[0]
                    + self.wave_view.size[0] * (self.cursor_time + self.cursor_offset - left) / (right - left)
                )
                * ratio
            )
            / ratio
        )
        top = self.wave_view.pos[1]
        height = max(1, self.evidence_view.pos[1] + self.evidence_view.size[1] - top)
        width = 6 if self.flashing else 2
        snapshot = (x, top, height, width, left <= self.cursor_time + self.cursor_offset <= right)
        if snapshot != self.cursor_snapshot:
            self.cursor_snapshot = snapshot
            self.playhead.transform.scale = (width, height)
            self.playhead.transform.translate = (x - width / 2, top)
            self.playhead.visible = snapshot[-1]

    def set_flash(self, flashing: bool) -> None:
        if self.flashing == flashing:
            return
        self.flashing = flashing
        self.playhead.color = KICK_FLASH.color if flashing else PLAYBACK.color
        self._position_cursor()

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
        points, colors, edges, sizes, symbols, score_points, edge_widths = [], [], [], [], [], [], []
        self.marker_data = []
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
            points.append((timestamp, self.kick_y if kick else self.candidate_y))
            colors.append(self.marker_colors[style.color])
            edges.append(self.marker_colors[diagnostic.color if diagnostic else style.color])
            sizes.append(diagnostic.size if diagnostic else style.size)
            edge_widths.append(2 if diagnostic else 1)
            symbols.append({"o": "disc", "s": "square", "x": "x"}[style.symbol])
            self.marker_data.append((key, timestamp, kick))
            self.inspection[key] = f"{timestamp:.4f} s · {style.label}\n{candidate['timing_reason']}"
            score = candidate.get("positive_similarity")
            if scores and score is not None:
                score_points.append((timestamp, score))
        self.marker_keys = [item[0] for item in self.marker_data]
        self.marker_times = np.asarray([item[1] for item in self.marker_data])
        self.marker_ys = np.asarray([self.kick_y if item[2] else self.candidate_y for item in self.marker_data])
        self.markers.set_data(
            **dict(
                pos=np.asarray(points, dtype="float32").reshape(-1, 2),
                face_color=np.asarray(colors).reshape(-1, 4),
                edge_color=np.asarray(edges).reshape(-1, 4),
                size=np.asarray(sizes),
                symbol=np.asarray(symbols),
                edge_width=np.asarray(edge_widths),
            )
        )
        self.scores.set_data(
            pos=np.asarray(score_points, dtype="float32").reshape(-1, 2), face_color=SCORE.color, size=5
        )
        beats = self.proposal.get("alternative_beat_positions", []) if alternative else self.proposal["beat_positions"]
        grid_key = (grid, alternative, self.wave_view.size[1], self.kick_y)
        if grid_key != self.grid_key:
            self.grid_key = grid_key
            height = max(1, self.wave_view.size[1] - 56)
            ys = np.arange(0, height, 6)
            low = -self.amplitude + ys / height * self.amplitude * 2
            high = -self.amplitude + np.minimum(ys + 4, height) / height * self.amplitude * 2
            dashed = np.column_stack((low, high)).ravel()
            geometry = (
                np.column_stack((np.repeat(beats, len(dashed)), np.tile(dashed, len(beats)))).astype("float32")
                if grid
                else np.empty((0, 2), dtype="float32")
            )
            self.grid_line.set_data(pos=geometry)
        self.select(self.selected, editor)

    def select(self, key: str | None, editor: CorrectionEditor) -> None:
        self.selected = key
        candidate = editor.candidates.get(key) if key else None
        for line, camera in zip(self.selection_lines, (self.wave_camera, self.evidence_camera), strict=True):
            line.visible = candidate is not None
            if candidate is not None:
                t = effective_timestamp(candidate, editor.draft)
                rect = camera.rect
                assert rect is not None
                line.set_data(pos=np.array([[t, rect.bottom], [t, rect.top]], dtype="float32"))
        points = (
            []
            if candidate is None
            else [
                (
                    effective_timestamp(candidate, editor.draft),
                    self.kick_y if effective_kick(candidate, editor.draft) else self.candidate_y,
                )
            ]
        )
        self.selection_outline.set_data(
            **dict(
                pos=np.asarray(points, dtype="float32").reshape(-1, 2),
                size=18,
                face_color=(0, 0, 0, 0),
                edge_color=SELECTION.color,
                edge_width=2,
            )
        )

    def hit_test(self, point: QPointF) -> str | None:
        transform = self.wave_scene.node_transform(self.canvas.scene)
        left, right = self.interval
        time = left + (point.x() - self.wave_view.pos[0]) / max(1, self.wave_view.size[0]) * (right - left)
        radius = 8 * (right - left) / max(1, self.wave_view.size[0])
        first, last = np.searchsorted(self.marker_times, [time - radius, time + radius])
        points = []
        for i in range(int(first), int(last)):
            p = transform.map((self.marker_times[i], self.marker_ys[i]))
            points.append((self.marker_keys[i], float(p[0]), float(p[1])))
        return nearest_marker(points, point.x(), point.y())

    def _time_at(self, x: float) -> float:
        left, right = self.interval
        return left + (x - self.wave_view.pos[0]) / max(1, self.wave_view.size[0]) * (right - left)

    def _wheel(self, event) -> None:
        self.zoomRequested.emit(1.02 ** (-event.delta[1] * 15), self._time_at(event.pos[0]))
        event.handled = True

    def _move(self, event) -> None:
        if event.is_dragging and event.last_event is not None and event.press_event.button in (1, 3):
            dx = event.pos[0] - event.last_event.pos[0]
            if dx:
                self.manuallyNavigated.emit()
                left, right = self.interval
                self.set_interval(left - dx / max(1, self.wave_view.size[0]) * (right - left), right - left)
        else:
            key = self.hit_test(QPointF(*event.pos))
            self.canvas.native.setToolTip(self.inspection.get(key, "") if key is not None else "")

    def _release(self, event) -> None:
        if event.press_event is None or np.linalg.norm(np.asarray(event.pos) - event.press_event.pos) > 3:
            return
        modifiers = Qt.KeyboardModifier.ShiftModifier if "Shift" in event.modifiers else Qt.KeyboardModifier.NoModifier
        self.clicked.emit(
            min(self.analysis.duration_seconds, max(0, self._time_at(event.pos[0]))),
            self.hit_test(QPointF(*event.pos)),
            event.button == 2,
            modifiers,
        )

    def _resize(self, event) -> None:
        self._layout_views()
        self._position_cursor()
