"""Deterministic workloads and the interactive experiment window."""

import copy
import math
import traceback
from time import monotonic, perf_counter

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from benchmarks.annotation.analysis import PARAMETERS
from benchmarks.annotation.corrections import CorrectionEditor, effective_onsets
from benchmarks.annotation.review.playback import KickFlash, PCMPlayback, zoom_speed
from benchmarks.review_performance.baseline import create_baseline
from benchmarks.review_performance.data import Recording
from benchmarks.review_performance.metrics import Measurements, resident_bytes

SCENARIOS = (
    "follow_2.5",
    "follow_10",
    "follow_40",
    "follow_120",
    "follow_full",
    "zoom",
    "pan_zoom",
    "seek",
    "markers",
    "idle",
    "stream",
)


class MeasuredPlayback(PCMPlayback):
    def __init__(self, path, measurements: Measurements) -> None:
        self.measurements = measurements
        from benchmarks.fixtures import load_wav

        rate, samples = load_wav(path, preserve_channels=True)
        super().__init__(samples, rate)

    def _callback(self, output, frames, timing, status) -> None:
        started = perf_counter()
        try:
            super()._callback(output, frames, timing, status)
        finally:
            if self.measurements.active:
                self.measurements.callback_costs.append((perf_counter() - started) * 1000)


class ExperimentWindow(QWidget):
    def __init__(
        self, recording: Recording, backend: str, target: int, strategy: str, integrated: bool, axes: str = "cached"
    ) -> None:
        super().__init__()
        self.recording = recording
        self.backend = backend
        self.target = target
        self.measurements = Measurements()
        self.buffer_original = None
        self.buffer_set_original = None
        self._install_buffer_counter()
        self.editor = CorrectionEditor(recording.proposal, copy.deepcopy(recording.draft))
        self.initialization_started = perf_counter()
        if backend == "pyqtgraph":
            self.timeline = create_baseline(recording.analysis, recording.proposal, self.measurements)
        else:
            from benchmarks.review_performance.vispy_timeline import VispyTimeline

            self.timeline = VispyTimeline(recording.analysis, recording.proposal, self.measurements, strategy, axes)
        self.player = MeasuredPlayback(recording.audio_path, self.measurements) if integrated else None
        self.initialization_seconds = perf_counter() - self.initialization_started
        if self.player is not None:
            self.measurements.clock_position = lambda: (
                self.player.position if self.player is not None else self.position
            )
        self.following = True
        self.playing = False
        self.position = 60.0
        self.generation = 0
        self.seek_offset = 0.0
        self.flash = KickFlash()
        self.alternative = False
        self.diagnostics = True
        self.status = QLabel(
            "Space: play/pause · F: follow · wheel: zoom · drag: pan · arrows: seek · Shift-click: correct · right-click: inspect · G: grid · D: diagnostics · memory only"
        )
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.timeline, 1)
        layout.addWidget(self.status)
        self.timeline.zoomRequested.connect(self.zoom)
        self.timeline.manuallyNavigated.connect(self.unfollow)
        self.timeline.clicked.connect(self.click)
        self.timeline.rangeChanged.connect(self.speed_changed)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._tick)
        self.origin = monotonic()
        self.last_tick = self.origin
        self.last_slot = -1
        self.scenario = "interactive"
        self.last_action = -1
        self.action_interval = 0.25
        self.stop_after = math.inf
        self.done = None
        self.stream = None
        self.stream_visual = None
        self.stream_full = False
        self.stream_overwritten = 0
        self.last_memory_sample = 0.0
        self.clock_positions = np.empty(0)
        self.idle_snapshot: tuple | None = None
        self.refresh_markers()
        self.timeline.follow(self.position, width=10)
        self.setWindowTitle(f"Spectra rendering experiment — {backend} — {recording.proposal['track_identifier']}")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)

    def _install_buffer_counter(self) -> None:
        if self.backend != "vispy":
            return
        from vispy.gloo.buffer import Buffer

        original = Buffer.set_subdata
        original_set = Buffer.set_data
        self.buffer_original = original
        self.buffer_set_original = original_set
        metrics = self.measurements

        def record(data) -> None:
            if metrics.active:
                metrics.geometry_calls += 1
                metrics.geometry_bytes += data.nbytes
            else:
                metrics.initial_upload_calls += 1
                metrics.initial_upload_bytes += data.nbytes

        def counted(self, data, offset=0, copy=False):
            record(data)
            return original(self, data, offset=offset, copy=copy)

        def counted_set(self, data, copy=False):
            record(data)
            return original_set(self, data, copy=copy)

        Buffer.set_subdata = counted
        Buffer.set_data = counted_set

    def refresh_markers(self) -> None:
        self.timeline.update_candidates(
            self.editor, show_candidates=True, uncertainty=True, timing=True, scores=True, grid=True, alternative=False
        )
        self.flash.timestamps = effective_onsets(self.recording.proposal, self.editor.draft)

    def speed_changed(self, *_args) -> None:
        if self.player is not None:
            left, right = self.timeline.interval
            self.player.set_speed(zoom_speed(right - left))

    def unfollow(self) -> None:
        self.following = False

    def zoom(self, factor: float, mouse_time: float) -> None:
        if self.measurements.active:
            self.measurements.input("zoom")
        self.timeline.zoom(factor, mouse_time, playback_position=self.position if self.following else None)

    def click(self, timestamp: float, key: str | None, inspect_only: bool, modifiers: object) -> None:
        if key is not None and inspect_only:
            self.timeline.select(key, self.editor)
        elif key is not None and modifiers == Qt.KeyboardModifier.ShiftModifier:
            self.measurements.input("marker")
            self.editor.toggle(key)
            self.refresh_markers()
        else:
            self.seek(timestamp)

    def seek(self, position: float) -> None:
        self.position = min(self.recording.analysis.duration_seconds - 1, max(0, position))
        self.generation += 1
        if self.player is not None:
            self.player.seek(self.position)
        self.flash.reset()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # Native OpenGL/GraphicsView children can consume keys before the parent
        # sees them. Handle transport keys for this experiment window only.
        if (
            isinstance(watched, QWidget)
            and (watched is self or self.isAncestorOf(watched))
            and isinstance(event, QKeyEvent)
            and event.type() == QEvent.Type.KeyPress
        ):
            if event.key() in {
                Qt.Key.Key_Space,
                Qt.Key.Key_F,
                Qt.Key.Key_Left,
                Qt.Key.Key_Right,
                Qt.Key.Key_Backspace,
                Qt.Key.Key_G,
                Qt.Key.Key_D,
                Qt.Key.Key_Q,
                Qt.Key.Key_Escape,
            }:
                self.keyPressEvent(event)
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.playing = not self.playing
            if self.player is not None:
                self.player.toggle()
        elif key == Qt.Key.Key_F:
            self.following = True
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Backspace):
            left, right = self.timeline.interval
            distance = (right - left) * (0.1 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.5)
            self.seek(
                self.position
                + (-10 if key == Qt.Key.Key_Backspace else distance * (-1 if key == Qt.Key.Key_Left else 1))
            )
        elif key in (Qt.Key.Key_G, Qt.Key.Key_D):
            if key == Qt.Key.Key_G:
                self.alternative = not self.alternative
            else:
                self.diagnostics = not self.diagnostics
            self.refresh_markers()
        elif key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    def begin(self, scenario: str, duration: float, done=None, stream_full: bool = False) -> None:
        self.scenario = scenario
        self.stop_after = duration
        self.done = done
        self.origin = self.last_tick = monotonic()
        self.last_slot = self.last_action = -1
        self.seek_offset = 0.0
        self.following = True
        self.playing = scenario != "idle"
        self.seek(60)
        width = self.width_at(0)
        self.timeline.follow(self.position, width=width)
        if self.player is not None:
            self.player.set_speed(zoom_speed(width))
            if self.playing:
                self.player.play()
            else:
                self.player.pause()
        # Same source-time trajectory for both rendering-only implementations.
        # Integrate on a fixed 1 ms lattice, independent of GUI callback cadence.
        if scenario != "interactive":
            times = np.arange(math.ceil(duration * 1000) + 2) / 1000
            speeds = np.array([zoom_speed(self.width_at(float(t))) for t in times])
            self.clock_positions = 60 + np.cumsum(speeds) / 1000
        if scenario == "stream":
            if self.backend != "vispy":
                raise ValueError("Streaming is a secondary VisPy experiment.")
            from benchmarks.review_performance.streaming import RollingPCM, StreamingLine
            from benchmarks.review_performance.vispy_timeline import VispyTimeline

            assert isinstance(self.timeline, VispyTimeline)
            self.stream = RollingPCM(self.recording.analysis.mono, PARAMETERS.sample_rate)
            self.stream_full = stream_full
            self.timeline.waveform.visible = False
            for line in self.timeline.wave_lines:
                line.visible = False
            self.stream_visual = StreamingLine(
                self.stream.capacity, self.stream.rate, full_upload=stream_full, parent=self.timeline.wave_scene
            )
            self.stream.start()
        self.timer.start(0)

    def width_at(self, elapsed: float) -> float:
        if self.scenario.startswith("follow_"):
            text = self.scenario.removeprefix("follow_")
            return self.recording.analysis.duration_seconds if text == "full" else float(text)
        if self.scenario == "zoom":
            return math.exp(
                math.log(0.02)
                + (math.sin(elapsed * math.tau / 10 - math.pi / 2) + 1)
                / 2
                * math.log(self.recording.analysis.duration_seconds / 0.02)
            )
        if self.scenario == "pan_zoom":
            return 10 * math.exp(2 * math.sin(elapsed * math.tau / 3))
        return 10

    def run_actions(self, elapsed: float) -> None:
        action = int(elapsed / self.action_interval)
        while self.last_action < action:
            self.last_action += 1
            t = self.last_action * self.action_interval
            if self.scenario == "zoom" or self.scenario == "pan_zoom" and self.last_action % 2 == 0:
                self.zoom(
                    self.width_at(t) / self.timeline.target_width,
                    self.timeline.interval[0] + (self.timeline.interval[1] - self.timeline.interval[0]) * 0.65,
                )
            if self.scenario == "pan_zoom" and self.last_action % 2 == 1:
                self.following = False
                left, right = self.timeline.interval
                self.measurements.input("pan", self.origin + t)
                self.timeline.set_interval(60 + 25 * math.sin(t * 2), right - left)
            elif self.scenario == "seek" and self.last_action % 2 == 0:
                width = self.timeline.interval[1] - self.timeline.interval[0]
                self.measurements.input("seek", self.origin + t)
                # Alternate arrows at 50%/10% of the actual displayed interval.
                distance = width * (0.5 if self.last_action % 4 == 0 else -0.1)
                self.seek_offset += distance
                self.seek(self.position + distance)
            elif self.scenario == "markers":
                # Only automatic identities: toggling a human-inserted kick can
                # remove it, which would make subsequent scripts asymmetric.
                candidates = self.recording.proposal["kick_candidates"]
                visible = [c for c in candidates if abs(c["final_timestamp"] - 65) < 5]
                candidate = (visible or candidates)[self.last_action % len(visible or candidates)]
                self.measurements.input("marker", self.origin + t)
                self.editor.toggle(candidate["candidate_id"])
                self.refresh_markers()
                self.timeline.select(candidate["candidate_id"], self.editor)

    def _tick(self) -> None:
        try:
            self.tick()
        except Exception:
            traceback.print_exc()
            try:
                self.close()
            finally:
                QApplication.exit(1)

    def tick(self) -> None:
        started = perf_counter()
        now = monotonic()
        elapsed = now - self.origin
        if elapsed >= self.stop_after:
            if self.stream is not None:
                self.stream.close()
                self.stream_overwritten = self.stream.overwritten
            if self.done is not None:
                self.done()
            return
        slot = int(elapsed * self.target)
        self.last_slot = slot
        scripted_time = slot / self.target
        self.measurements.serial += 1
        self.measurements.snapshot_time = started
        if self.measurements.active:
            self.measurements.updates.append(started)
        if self.scenario != "interactive":
            self.run_actions(scripted_time)
        if self.player is not None:
            self.position = self.player.poll()
            self.playing = not self.player.paused
        elif self.playing:
            if self.scenario == "interactive":
                self.position += (now - self.last_tick) * zoom_speed(
                    self.timeline.interval[1] - self.timeline.interval[0]
                )
            else:
                self.position = (
                    float(self.clock_positions[min(len(self.clock_positions) - 1, round(scripted_time * 1000))])
                    + self.seek_offset
                )
        self.last_tick = now
        self.measurements.cursor_position = self.position
        self.render_position(now)
        if elapsed - self.last_memory_sample >= 0.5:
            self.last_memory_sample = elapsed
            if self.measurements.active:
                self.measurements.memory.append(resident_bytes())
        if self.measurements.active:
            self.measurements.update_costs.append((perf_counter() - started) * 1000)
        # Absolute-deadline scheduling with a precise single-shot timer. No 16 ms
        # production timer; missed slots are skipped rather than queued forever.
        next_slot = max(slot + 1, math.floor((monotonic() - self.origin) * self.target) + 1)
        delay = max(0, self.origin + next_slot / self.target - monotonic())
        self.timer.start(math.ceil(delay * 1000))

    def render_position(self, now: float) -> None:
        if self.stream is not None:
            first, samples = self.stream.read()
            if self.stream_visual is not None:
                self.stream_visual.append(first, samples)
            self.timeline._set_interval(max(0, self.stream.written / self.stream.rate - 10), 10)
            # _set_interval's LOD selection is disabled in the streaming scenario.
            position = self.stream.written / self.stream.rate
            self.timeline.set_cursor(position)
            self.timeline.set_flash(self.flash.update(position, playing=True, generation=0))
            return
        snapshot = (self.position, self.following, self.playing)
        animated = self.timeline.advance_zoom(now, playback_position=self.position if self.following else None)
        if snapshot != self.idle_snapshot or animated:
            if self.following:
                self.timeline.follow(self.position)
            else:
                self.timeline.set_cursor(self.position)
            self.idle_snapshot = snapshot
        self.timeline.set_flash(
            self.flash.update(
                self.position,
                playing=self.playing,
                generation=self.player.generation if self.player is not None else self.generation,
                playback_time=self.player.dac_time if self.player is not None else None,
                event_time=self.player.event_time if self.player is not None else None,
            )
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        application = QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)
        if self.stream is not None:
            self.stream.close()
        if self.player is not None:
            self.player.close()
        if self.buffer_original is not None:
            from vispy.gloo.buffer import Buffer

            Buffer.set_subdata = self.buffer_original
            if self.buffer_set_original is not None:
                Buffer.set_data = self.buffer_set_original
        if self.backend == "vispy":
            from benchmarks.review_performance.vispy_timeline import VispyTimeline

            assert isinstance(self.timeline, VispyTimeline)
            if self.timeline.canvas.gpu_timer is not None:
                self.timeline.canvas.set_current()
                self.timeline.canvas.gpu_timer.close()
            self.timeline.canvas.close()
        event.accept()
