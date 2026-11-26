"""Native review window coordinating the correction model, timeline and PCM player."""

import bisect
import copy
import hashlib
from collections.abc import Callable
from pathlib import Path
from time import monotonic

from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from benchmarks.annotation.analysis import AcousticAnalysis, AutomaticCandidate, Proposal
from benchmarks.annotation.automatic import OnsetLocalizer, classify_candidates
from benchmarks.annotation.corrections import (
    CorrectionEditor,
    ReviewDraft,
    centered_interval,
    effective_kick,
    effective_timestamp,
    load_review,
    nearby_hidden_candidates,
    proposal_checksum,
    save_review,
    uncertain_classification,
)
from benchmarks.annotation.review import DEFAULT_VIEW_DURATION
from benchmarks.annotation.review.playback import KickFlash, PCMPlayback, zoom_speed
from benchmarks.annotation.review.styles import (
    AUTOMATIC_KICK,
    BASS,
    BEAT,
    BROADBAND,
    CANDIDATE,
    HUMAN_KICK,
    PLAYBACK,
    REJECTED,
    SCORE,
    SELECTION,
    TIMING,
    UNCERTAINTY,
    ReviewLegend,
)
from benchmarks.annotation.review.timeline import Timeline

KEY_ACTIONS: dict[int, str] = {
    Qt.Key.Key_Escape: "close",
    Qt.Key.Key_Q: "close",
    Qt.Key.Key_Space: "play",
    Qt.Key.Key_Backspace: "back_10",
    Qt.Key.Key_Left: "back_half",
    Qt.Key.Key_Right: "forward_half",
    Qt.Key.Key_F: "follow",
    Qt.Key.Key_1: "reset_zoom",
    Qt.Key.Key_Comma: "previous_uncertain",
    Qt.Key.Key_Period: "next_uncertain",
    Qt.Key.Key_BracketLeft: "previous_unsupported",
    Qt.Key.Key_BracketRight: "next_unsupported",
    Qt.Key.Key_Less: "previous_corrected",
    Qt.Key.Key_Greater: "next_corrected",
}
SHIFT_KEY_ACTIONS: dict[int, str] = {
    Qt.Key.Key_Left: "back_tenth",
    Qt.Key.Key_Right: "forward_tenth",
    Qt.Key.Key_Comma: "previous_corrected",
    Qt.Key.Key_Period: "next_corrected",
}
VIEWPORT_SEEKS = {"back_half": -0.5, "forward_half": 0.5, "back_tenth": -0.1, "forward_tenth": 0.1}
LABELS = {
    "play": "Play (Space)",
    "back_10": "Back 10 s (Backspace)",
    "back_half": "Back 50% of view (Left)",
    "forward_half": "Forward 50% of view (Right)",
    "previous_uncertain": "Previous uncertain classification (,)",
    "next_uncertain": "Next uncertain classification (.)",
    "previous_unsupported": "Previous unsupported beat ([)",
    "next_unsupported": "Next unsupported beat (])",
    "undo": "Undo (Ctrl+Z)",
    "redo": "Redo (Ctrl+Y)",
    "save": "Save review (Ctrl+S)",
    "group_kick": "Similar events: kicks",
    "group_nonkick": "Similar events: non-kicks",
    "reclassify": "Reclassify using corrections",
    "previous_corrected": "Previous human correction (<)",
    "next_corrected": "Next human correction (>)",
    "clear_override": "Remove explicit classification override",
}


class Reclassification(QThread):
    completed = Signal(object, int)
    failed = Signal(str)

    def __init__(self, analysis: AcousticAnalysis, editor: CorrectionEditor, revision: int) -> None:
        super().__init__()
        self.analysis = analysis
        self.predictions = copy.deepcopy(editor.predictions)
        self.revision = revision
        references = set(editor.proposal.get("automatic_reference_ids", []))
        self.positives = [
            candidate["localized_timestamp"]
            for candidate in editor.proposal["kick_candidates"]
            if candidate["candidate_id"] in references
            and (
                candidate["candidate_id"] not in editor.draft.overrides
                or editor.draft.overrides[candidate["candidate_id"]].kick
            )
        ] + [correction.timestamp for correction in editor.draft.overrides.values() if correction.kick]
        self.negatives = [correction.timestamp for correction in editor.draft.overrides.values() if not correction.kick]

    def run(self) -> None:
        try:
            self.completed.emit(
                classify_candidates(self.analysis, self.predictions, self.positives, self.negatives), self.revision
            )
        except Exception as error:
            self.failed.emit(str(error))


class AnnotationReview(QMainWindow):
    def __init__(
        self,
        analysis: AcousticAnalysis,
        proposal: Proposal[AutomaticCandidate],
        draft: ReviewDraft,
        draft_path: Path,
        start: float = 0,
        duration: float = DEFAULT_VIEW_DURATION,
        *,
        audio_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.analysis = analysis
        self.proposal = proposal
        self.editor = CorrectionEditor(proposal, draft)
        self.draft_path = draft_path
        self.proposal_hash = proposal_checksum(proposal)
        self.stale = draft.originating_proposal_sha256 != self.proposal_hash
        self.dirty = self.stale
        self.revision = 0
        self.selected: str | None = None
        self.player: PCMPlayback | None = None
        self.worker: Reclassification | None = None
        self.close_pending = False
        self.pending_group: tuple[str, bool, tuple[str, ...]] | None = None
        self.localizer = OnsetLocalizer(analysis)
        self.playback_position = start
        self.reported_underruns = 0
        self.kick_flash = KickFlash()
        self.last_slider_update = 0.0
        self.visual_snapshot: tuple[float, float, bool] | None = None
        self.queues: dict[str, list[str]] = {}
        self.queue_positions: dict[str, list[tuple[float, str]]] = {}
        self.buttons: dict[str, QPushButton] = {}
        tempo = f"{proposal['bpm']:g} BPM" if proposal["bpm"] is not None else "BPM unknown"
        self.setWindowTitle(f"{proposal['track_identifier'].replace('_', ' ').title()} · {tempo}")
        self.resize(1440, 900)
        central = QWidget()
        self.setCentralWidget(central)
        horizontal = QHBoxLayout(central)
        left = QVBoxLayout()
        horizontal.addLayout(left, 1)
        self.range_label = QLabel()
        left.addWidget(self.range_label)
        self.timeline = Timeline(analysis, proposal)
        left.addWidget(self.timeline, 1)
        self.legend = ReviewLegend()
        left.addWidget(self.legend)
        self.unsaved_label = QLabel("Unsaved changes")
        self.unsaved_label.setStyleSheet("color: #ffcc70; font-weight: bold; padding: 3px 10px;")
        self.statusBar().addPermanentWidget(self.unsaved_label)
        self.unsaved_label.hide()
        seeking = QHBoxLayout()
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1_000_000)
        self.seek_slider.setToolTip("Seek anywhere in the recording; drag to preview, release to seek.")
        self.seek_slider.sliderReleased.connect(self._commit_seek)
        self.seek_slider.actionTriggered.connect(self._slider_action)
        self.seek_slider.valueChanged.connect(self._preview_seek)
        self.playback_label = QLabel()
        seeking.addWidget(self.seek_slider, 1)
        seeking.addWidget(self.playback_label)
        self.speed_label = QLabel()
        self.speed_label.setFixedWidth(
            self.speed_label.fontMetrics().horizontalAdvance("Zoom: 500.00x  ·  Speed: 2.00x")
        )
        seeking.addWidget(self.speed_label)
        self.adaptive_speed = QCheckBox("Adaptive speed")
        self.adaptive_speed.setChecked(True)
        self.adaptive_speed.setToolTip(
            "Adjust playback speed with zoom; 1 resets the view and speed. Pitch changes with speed."
        )
        self.adaptive_speed.toggled.connect(self._adaptive_toggled)
        seeking.addWidget(self.adaptive_speed)
        left.addLayout(seeking)
        self.inspection = QLabel("Right-click a marker to select and inspect it without editing.")
        self.inspection.setWordWrap(True)
        left.addWidget(self.inspection)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(295)
        scroll.setMaximumWidth(350)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = QWidget()
        controls = QVBoxLayout(panel)
        controls.setSpacing(6)
        scroll.setWidget(panel)
        horizontal.addWidget(scroll)
        hint = QLabel("<b>Click: seek playback<br>Shift + Click: correct or add kick</b>")
        hint.setWordWrap(True)
        controls.addWidget(hint)
        self._section(controls, "Playback", ["play", "back_10", "back_half", "forward_half"])
        self.follow_checkbox = QCheckBox("Follow playback (F)")
        self.follow_checkbox.setChecked(True)
        self.follow_checkbox.setToolTip(
            "F returns to playback at 40% of the plot width where possible. Drag to explore independently; "
            "wheel zoom follows playback or the mouse pointer in free navigation."
        )
        self.follow_checkbox.toggled.connect(self._follow_toggled)
        controls.addWidget(self.follow_checkbox)
        review_controls = self._section(
            controls,
            "Review",
            [
                "previous_uncertain",
                "next_uncertain",
                "previous_unsupported",
                "next_unsupported",
            ],
        )
        history = QHBoxLayout()
        self._button(history, "undo")
        self._button(history, "redo")
        review_controls.addLayout(history)
        self._button(review_controls, "save")
        self._section(controls, "Classification", ["group_kick", "group_nonkick", "reclassify"])
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress = QLabel()
        self.progress.setWordWrap(True)
        progress_layout.addWidget(self.progress)
        controls.addWidget(progress_group)
        self.advanced_toggle = QPushButton("Advanced ▸")
        self.advanced_toggle.setCheckable(True)
        controls.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        self.overlays: dict[str, QCheckBox] = {}
        for key, label in [
            ("candidates", "Show kick candidates"),
            ("uncertainty", "Show classification uncertainty"),
            ("grid", "Show beat grid"),
            ("scores", "Show similarity scores"),
            ("timing", "Show timing diagnostics"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(key in ("grid", "candidates"))
            checkbox.toggled.connect(self.refresh_markers)
            self.overlays[key] = checkbox
            advanced_layout.addWidget(checkbox)
        self.flash_checkbox = QCheckBox("Flash on kicks")
        self.flash_checkbox.setChecked(True)
        self.flash_checkbox.toggled.connect(self._flash_toggled)
        advanced_layout.addWidget(self.flash_checkbox)
        self.grid_choice = QComboBox()
        self.grid_choice.addItems(["Primary beat grid", "Alternative beat grid"])
        self.grid_choice.setEnabled(bool(proposal.get("alternative_beat_positions")))
        self.grid_choice.setCurrentIndex(
            1 if proposal.get("selected_grid") == "alternative" and proposal.get("alternative_beat_positions") else 0
        )
        self.grid_choice.currentIndexChanged.connect(self.refresh)
        advanced_layout.addWidget(self.grid_choice)
        for action in ("previous_corrected", "next_corrected", "clear_override"):
            self._button(advanced_layout, action)
        advanced_layout.addWidget(QLabel("Visual playback cursor offset (ms)"))
        self.cursor_offset = QDoubleSpinBox()
        self.cursor_offset.setRange(-1000, 1000)
        self.cursor_offset.setDecimals(1)
        self.cursor_offset.setSuffix(" ms")
        self.cursor_offset.setToolTip(
            "Positive moves only the displayed cursor forward. Audio, seeks and annotations are unchanged."
        )
        self.cursor_offset.valueChanged.connect(lambda _: self._render_position())
        advanced_layout.addWidget(self.cursor_offset)
        self.advanced.setVisible(False)
        self.advanced_toggle.toggled.connect(self._advanced_toggled)
        controls.addWidget(self.advanced)
        controls.addStretch()
        self.statusBar().setSizeGripEnabled(True)
        self.statusBar().showMessage("Predictions are unreviewed. Save corrections with Ctrl+S.")
        self.timeline.clicked.connect(self.timeline_click)
        self.timeline.manuallyNavigated.connect(self._manually_navigated)
        self.timeline.zoomRequested.connect(self._zoom)
        self.timeline.rangeChanged.connect(self._range_changed)
        self.timeline.set_interval(start, duration)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.update_playback)
        if audio_path is not None:
            try:
                self.player = PCMPlayback.from_wav(audio_path)
                self.player.seek(start)
                self._set_speed_from_zoom()
            except Exception as error:
                self.statusBar().showMessage(f"Playback unavailable: {error}. Review edits remain available.")
        self.refresh()
        self._follow_toggled(self.follow_checkbox.isChecked())
        self.timer.start()

    @property
    def draft(self) -> ReviewDraft:
        return self.editor.draft

    @property
    def alternative_grid(self) -> bool:
        return self.grid_choice.currentIndex() == 1

    def _button(self, layout: QBoxLayout, action: str) -> None:
        button = QPushButton(LABELS[action])
        tooltip = LABELS[action]
        if action in ("back_half", "forward_half"):
            tooltip += "; Shift + " + ("Left: back 10%" if action == "back_half" else "Right: forward 10%")
            tooltip += ". Uses the displayed viewport duration, independently of playback speed."
        if action == "redo":
            tooltip += "; Ctrl+Shift+Z also redoes"
        button.setToolTip(tooltip)
        button.clicked.connect(lambda _checked=False, name=action: self.perform_action(name))
        self.buttons[action] = button
        layout.addWidget(button)

    def _section(self, layout: QVBoxLayout, title: str, actions: list[str]) -> QVBoxLayout:
        group = QGroupBox(title)
        inner = QVBoxLayout(group)
        inner.setSpacing(3)
        for action in actions:
            self._button(inner, action)
        layout.addWidget(group)
        return inner

    def _advanced_toggled(self, expanded: bool) -> None:
        self.advanced.setVisible(expanded)
        self.advanced_toggle.setText("Advanced ▾" if expanded else "Advanced ▸")
        self._inspect()

    def _range_changed(self, left: float, right: float) -> None:
        text = f"Visible range: {round(left)}–{round(right)} s"
        if self.range_label.text() != text:
            self.range_label.setText(text)
        self._set_speed_from_zoom()

    def _set_speed_from_zoom(self) -> None:
        if self.player is not None:
            left, right = self.timeline.interval
            self.player.set_speed(zoom_speed(right - left) if self.adaptive_speed.isChecked() else 1.0)

    def _adaptive_toggled(self, enabled: bool) -> None:
        self._set_speed_from_zoom()

    def _manually_navigated(self) -> None:
        self.timeline.cancel_zoom()
        self.follow_checkbox.setChecked(False)

    def _follow_toggled(self, enabled: bool) -> None:
        self.timeline.cancel_zoom()
        if enabled:
            self.visual_snapshot = None
            self.update_playback(force=True)
            if self.player is None:
                self._render_position()

    def _restore_following(self) -> None:
        if self.follow_checkbox.isChecked():
            self._follow_toggled(True)
        else:
            self.follow_checkbox.setChecked(True)

    def _zoom(self, factor: float, mouse_time: float) -> None:
        position = self.player.position if self.player is not None else self.playback_position
        self.playback_position = position
        self.timeline.zoom(
            factor,
            mouse_time,
            playback_position=position if self.follow_checkbox.isChecked() else None,
            offset=self.cursor_offset.value() / 1000,
        )

    def _seek(self, destination: float) -> None:
        self.timeline.cancel_zoom()
        player = self.player
        if player is not None:
            destination = min(self.analysis.duration_seconds, max(0.0, destination))
            self._playback_action(lambda: player.seek(destination))

    def _preview_seek(self, value: int) -> None:
        if self.seek_slider.isSliderDown():
            self.playback_label.setText(
                f"Seek: {round(value / 1_000_000 * self.analysis.duration_seconds)} / {round(self.analysis.duration_seconds)} s"
            )

    def _commit_seek(self) -> None:
        destination = self.seek_slider.sliderPosition() / 1_000_000 * self.analysis.duration_seconds
        self._seek(destination)

    def _slider_action(self, _: int) -> None:
        if not self.seek_slider.isSliderDown():
            # Qt adjusts sliderPosition before this signal, value afterward.
            self._commit_seek()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
            and isinstance(event, QKeyEvent)
            and self.isActiveWindow()
            and QApplication.activePopupWidget() is None
            and QApplication.activeModalWidget() is None
        ):
            focus = QApplication.focusWidget()
            if (
                isinstance(focus, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox))
                and event.key() != Qt.Key.Key_Escape
            ):
                return super().eventFilter(watched, event)
            modifiers = event.modifiers()
            action = None
            if modifiers == Qt.KeyboardModifier.ControlModifier:
                control_actions: dict[int, str] = {Qt.Key.Key_Z: "undo", Qt.Key.Key_Y: "redo", Qt.Key.Key_S: "save"}
                action = control_actions.get(event.key())
            elif modifiers == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
                action = "redo" if event.key() == Qt.Key.Key_Z else None
            elif modifiers in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier):
                action = KEY_ACTIONS.get(event.key())
                if modifiers == Qt.KeyboardModifier.ShiftModifier:
                    action = SHIFT_KEY_ACTIONS.get(event.key(), action)
            if action == "close" and modifiers != Qt.KeyboardModifier.NoModifier:
                action = None
            if action is not None:
                event.accept()
                if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                    self.perform_action(action)
                return True
        return super().eventFilter(watched, event)

    def refresh(self, *_: object) -> None:
        candidates = list(self.editor.candidates.values())
        ordered = sorted(candidates, key=lambda c: (effective_timestamp(c, self.draft), c["candidate_id"]))
        kicks = sorted(effective_timestamp(c, self.draft) for c in ordered if effective_kick(c, self.draft))
        self.kick_flash.timestamps = kicks
        self.kick_flash.reset()
        self.timeline.set_flash(False)
        self.unsaved_label.setVisible(self.dirty or self.stale)
        # Build queues only after edits/grid changes, never in the playback timer.

        self.queues = {"uncertain": [], "unsupported": [], "corrected": []}
        for candidate in ordered:
            key = candidate["candidate_id"]
            if key in self.draft.overrides:
                self.queues["corrected"].append(key)
                continue
            if uncertain_classification(candidate, self.draft):
                self.queues["uncertain"].append(key)
            timestamp = effective_timestamp(candidate, self.draft)
            index = bisect.bisect_left(kicks, timestamp)
            covered = any(abs(kicks[i] - timestamp) <= 0.050 for i in (index - 1, index) if 0 <= i < len(kicks))
            if ("alternative" if self.alternative_grid else "primary") in candidate[
                "unsupported_grids"
            ] and not covered:
                self.queues["unsupported"].append(key)
        self.progress.setText(
            f"{len(kicks)} effective kicks\n{len(self.draft.overrides)} human corrections\n"
            f"{len(self.queues['uncertain'])} uncertain classifications\n{len(self.queues['unsupported'])} unsupported beats\n"
            "Corrections do not measure reviewed coverage."
        )
        self.queue_positions = {
            kind: [(effective_timestamp(self.editor.candidates[key], self.draft), key) for key in keys]
            for kind, keys in self.queues.items()
        }
        self.refresh_markers()
        self._range_changed(*self.timeline.interval)
        self.refresh_actions()
        self._inspect()
        if self.stale:
            self.statusBar().showMessage(
                "STALE REVIEW: matched decisions retained. Inspect new predictions and save before finalizing."
            )

    def refresh_markers(self, *_: object) -> None:
        self.timeline.select(self.selected, self.editor)
        self.timeline.update_candidates(
            self.editor,
            show_candidates=self.overlays["candidates"].isChecked(),
            uncertainty=self.overlays["uncertainty"].isChecked(),
            timing=self.overlays["timing"].isChecked(),
            scores=self.overlays["scores"].isChecked(),
            grid=self.overlays["grid"].isChecked(),
            alternative=self.alternative_grid,
        )
        entries = [AUTOMATIC_KICK, HUMAN_KICK, CANDIDATE, REJECTED, PLAYBACK, SELECTION, BASS, BROADBAND]
        if not self.overlays["candidates"].isChecked():
            entries.remove(CANDIDATE)
        if self.overlays["uncertainty"].isChecked():
            entries.append(UNCERTAINTY)
        if self.overlays["timing"].isChecked():
            entries.append(TIMING)
        if self.overlays["grid"].isChecked():
            entries.append(BEAT)
        if self.overlays["scores"].isChecked():
            entries.append(SCORE)
        self.legend.set_entries(tuple(entries))

    def navigation_target(self, kind: str, direction: int) -> str | None:
        positions = self.queue_positions[kind]
        selected = self.editor.candidates.get(self.selected) if self.selected is not None else None
        if selected is not None:
            anchor = (effective_timestamp(selected, self.draft), selected["candidate_id"])
            index = (
                bisect.bisect_right(positions, anchor) if direction > 0 else bisect.bisect_left(positions, anchor) - 1
            )
        else:
            timestamp = self.timeline.interval[0]
            index = (
                bisect.bisect_left(positions, (timestamp, ""))
                if direction > 0
                else bisect.bisect_right(positions, (timestamp, chr(0x10FFFF))) - 1
            )
        return positions[index][1] if 0 <= index < len(positions) else None

    def action_enabled(self, action: str) -> bool:
        if action in ("play", "back_10") or action in VIEWPORT_SEEKS:
            return self.player is not None
        if action == "undo":
            return bool(self.editor.history)
        if action == "redo":
            return bool(self.editor.redo_history)
        if action == "reclassify":
            return bool(self.draft.overrides) and self.worker is None
        if action == "clear_override":
            return self.selected in self.draft.overrides
        if action.startswith("group_"):
            return self.selected is not None and bool(self.editor.group_targets(self.selected))
        if action.startswith(("previous_", "next_")):
            direction, kind = action.split("_", 1)
            return self.navigation_target(kind, 1 if direction == "next" else -1) is not None
        return True

    def refresh_actions(self) -> None:
        for action, button in self.buttons.items():
            enabled = self.action_enabled(action)
            if button.isEnabled() != enabled:
                button.setEnabled(enabled)
        self.buttons["play"].setText(
            "Pause (Space)" if self.player is not None and not self.player.paused else LABELS["play"]
        )
        for action in ("group_kick", "group_nonkick"):
            pending = self.pending_group
            self.buttons[action].setText(
                f"Confirm {len(pending[2])} {'kicks' if pending[1] else 'non-kicks'}"
                if pending is not None and pending[1] == (action == "group_kick")
                else LABELS[action]
            )
        self.seek_slider.setEnabled(self.player is not None)

    def _inspect(self) -> None:
        candidate = self.editor.candidates.get(self.selected) if self.selected is not None else None
        if candidate is None:
            self.inspection.setText("Right-click a marker to select and inspect it without editing.")
            return
        text = f"Selected: {'Kick' if effective_kick(candidate, self.draft) else 'Rejected candidate' if self.selected in self.draft.overrides else 'Kick candidate'} at {round(effective_timestamp(candidate, self.draft))} s"
        if self.advanced_toggle.isChecked():
            text += (
                f" · {effective_timestamp(candidate, self.draft):.4f} s · cluster {candidate['cluster_id']}\n{candidate['timing_reason']} · "
                + ", ".join(candidate["uncertainty_reasons"])
            )
        self.inspection.setText(text)

    def edited(self, message: str) -> None:
        self.dirty = self.stale or self.editor.dirty
        self.revision += 1
        self.pending_group = None
        self.statusBar().showMessage(message)
        self.refresh()

    def timeline_click(self, timestamp: float, key: str | None, inspect_only: bool, modifiers: object) -> None:
        if inspect_only:
            if key is not None:
                self.selected = key
                self.refresh_markers()
                self.refresh_actions()
                self._inspect()
            return
        if modifiers == Qt.KeyboardModifier.ShiftModifier:
            self.correct_or_insert(timestamp, key)
        else:
            if self.selected is not None:
                self.selected = None
                self.refresh_markers()
                self.refresh_actions()
                self._inspect()
            self._seek(timestamp)

    def correct_or_insert(self, timestamp: float, key: str | None) -> None:
        self.selected = None
        if key is not None:
            self.editor.toggle(key)
            self.edited(
                "Kick removed." if key not in self.editor.candidates else "Existing event corrected; onset preserved."
            )
            return
        nearby = nearby_hidden_candidates(self.editor, timestamp, set(self.timeline.marker_keys))
        if nearby:
            # A close timestamp alone is insufficient: acoustic localization must
            # agree with the existing event. Otherwise request explicit selection.
            localized, reliable, _, _ = self.localizer.localize(timestamp)
            appropriate = [
                k
                for k in nearby
                if reliable and abs(self.editor.candidates[k]["localized_timestamp"] - localized) <= 0.015
            ]
            if len(appropriate) == 1:
                self.editor.toggle(appropriate[0])
                self.edited("Hidden candidate recovered as a kick; existing onset preserved.")
                return
            self.selected = nearby[0]
            self.refresh_markers()
            self.refresh_actions()
            self._inspect()
            self.statusBar().showMessage(
                "Possible nearby event highlighted. Shift-click its marker to correct it, or right-click another marker to inspect. No duplicate inserted."
            )
            return
        self.insert_manual(timestamp)

    def insert_manual(self, timestamp: float) -> None:
        if not 0 <= timestamp < self.analysis.duration_seconds:
            return
        localized, reliable, reason, increase = self.localizer.localize(timestamp)
        key = "manual-" + hashlib.sha256(f"{self.proposal['audio_sha256']}:{timestamp:.9f}".encode()).hexdigest()[:20]
        candidate: AutomaticCandidate = {
            "candidate_id": key,
            "timestamp": timestamp,
            "novelty_timestamp": timestamp,
            "localized_timestamp": localized,
            "final_timestamp": localized,
            "localization_reliable": reliable,
            "timing_reason": reason,
            "low_frequency_increase": increase,
            "provenance": "manual",
            "bass_evidence": 0.0,
            "prominence": 0.0,
            "broadband_evidence": 0.0,
            "nearest_beat_distance_seconds": None,
            "grid_aligned": False,
            "unsupported_grids": [],
            "automatic_kick": False,
            "low_confidence": True,
            "cluster_id": None,
            "uncertainty_reasons": [],
            "classification": "unclassified",
        }
        if self.editor.insert(candidate):
            self.selected = None
            self.edited("New kick inserted; onset localized automatically.")
        else:
            nearest = min(
                self.editor.candidates.values(), key=lambda c: abs(effective_timestamp(c, self.draft) - localized)
            )
            self.selected = nearest["candidate_id"]
            self.refresh_markers()
            self.refresh_actions()
            self._inspect()
            self.statusBar().showMessage(
                "Duplicate prevented: existing event within 30 ms of localized onset highlighted."
            )

    def _playback_action(self, action: Callable[[], None]) -> None:
        was_paused = self.player is None or self.player.paused
        try:
            action()
        except Exception as error:
            self.playback_failed(error)
        self.kick_flash.reset()
        self.update_playback(force=True)
        if self.player is not None and self.player.paused != was_paused:
            self.refresh_actions()

    def playback_failed(self, error: Exception) -> None:
        self.kick_flash.reset()
        self.timeline.set_flash(False)
        player, self.player = self.player, None
        if player is not None:
            try:
                player.close()
            except Exception:
                pass
        self.statusBar().showMessage(f"Playback stopped: {error}. Annotation edits are retained.")
        self.refresh_actions()

    def perform_action(self, action: str) -> None:
        if not self.action_enabled(action):
            return
        try:
            if action == "close":
                self.close()
            elif action == "play" and self.player is not None:
                self._playback_action(self.player.toggle)
            elif (action == "back_10" or action in VIEWPORT_SEEKS) and self.player is not None:
                left, right = self.timeline.interval
                distance = -10.0 if action == "back_10" else (right - left) * VIEWPORT_SEEKS[action]
                self._seek(self.player.position + distance)
            elif action == "follow":
                self._restore_following()
            elif action == "reset_zoom":
                left, right = self.timeline.interval
                self._zoom(DEFAULT_VIEW_DURATION / self.timeline.target_width, (left + right) / 2)
            elif action.startswith(("previous_", "next_")):
                direction, kind = action.split("_", 1)
                key = self.navigation_target(kind, 1 if direction == "next" else -1)
                if key is not None:
                    self.selected = key
                    start, width = centered_interval(
                        effective_timestamp(self.editor.candidates[key], self.draft), self.analysis.duration_seconds
                    )
                    self.follow_checkbox.setChecked(False)
                    self.timeline.set_interval(start, width)
                    if self.player is not None:
                        self._playback_action(lambda: self._pause_seek(start))
                    self.refresh_markers()
                    self.refresh_actions()
                    self._inspect()
            elif action in ("undo", "redo"):
                if action == "undo":
                    self.editor.undo()
                else:
                    self.editor.redo()
                if self.selected not in self.editor.candidates:
                    self.selected = None
                self.edited("Correction undone." if action == "undo" else "Correction redone.")
            elif action == "clear_override" and self.selected is not None:
                self.editor.clear_override(self.selected)
                removed = self.selected not in self.editor.candidates
                if removed:
                    self.selected = None
                self.edited("Manual kick removed." if removed else "Explicit classification override removed.")
            elif action == "save":
                save_review(self.draft_path, self.draft, self.proposal)
                self.editor.mark_saved()
                self.dirty = self.stale = False
                self.unsaved_label.hide()
                self.statusBar().showMessage("Review saved · undo history cleared")
                self.refresh_actions()
            elif action.startswith("group_") and self.selected is not None:
                kick = action == "group_kick"
                request = (self.selected, kick, tuple(self.editor.group_targets(self.selected)))
                if self.pending_group == request:
                    count = self.editor.group_correct(self.selected, kick)
                    self.edited(f"Corrected {count} similar events; earlier human decisions preserved.")
                else:
                    self.pending_group = request
                    self.statusBar().showMessage(
                        f"Will change {len(request[2])} uncorrected events. Click the same action again to confirm."
                    )
                    self.refresh_actions()
            elif action == "reclassify":
                self.worker = Reclassification(self.analysis, self.editor, self.revision)
                self.worker.completed.connect(self._classified)
                self.worker.failed.connect(
                    lambda error: self.statusBar().showMessage(f"Reclassification failed: {error}")
                )
                self.worker.finished.connect(self._worker_finished)
                self.worker.start()
                self.statusBar().showMessage("Reclassifying in the background… Edits remain available.")
                self.refresh_actions()
        except (OSError, ValueError, RuntimeError) as error:
            self.statusBar().showMessage(str(error))

    def _pause_seek(self, timestamp: float) -> None:
        if self.player is not None:
            self.player.pause()
            self.player.seek(timestamp)

    def _classified(self, candidates: list[AutomaticCandidate], revision: int) -> None:
        if self.close_pending:
            return
        if revision != self.revision:
            self.statusBar().showMessage(
                "Corrections changed during reclassification. Result discarded; run again to include them."
            )
            return
        changed = self.editor.replace_predictions(candidates)
        self.edited(f"Changed {changed} automatic predictions. Explicit human decisions preserved.")

    def _worker_finished(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None
        self.refresh_actions()
        if self.close_pending:
            self.close()

    def _flash_toggled(self, enabled: bool) -> None:
        self.kick_flash.reset()
        self.timeline.set_flash(False)

    def _render_position(self) -> None:
        offset = self.cursor_offset.value() / 1000
        following = self.follow_checkbox.isChecked()
        snapshot = (self.playback_position, offset, following)
        if self.timeline.advance_zoom(
            monotonic(), playback_position=self.playback_position if following else None, offset=offset
        ):
            self.visual_snapshot = snapshot
            if not following:
                self.timeline.set_cursor(self.playback_position, offset=offset)
            return
        if snapshot == self.visual_snapshot:
            return
        self.visual_snapshot = snapshot
        if following:
            self.timeline.follow(self.playback_position, offset=offset)
        else:
            self.timeline.set_cursor(self.playback_position, offset=offset)

    def _update_seek_controls(self, position: float, *, force: bool) -> None:
        if self.seek_slider.isSliderDown():
            return
        now = monotonic()
        if force or now - self.last_slider_update >= 0.08:
            self.last_slider_update = now
            value = round(position / self.analysis.duration_seconds * 1_000_000)
            if self.seek_slider.value() != value:
                with QSignalBlocker(self.seek_slider):
                    self.seek_slider.setValue(value)
        text = f"Playback: {round(position)} / {round(self.analysis.duration_seconds)} s"
        if self.playback_label.text() != text:
            self.playback_label.setText(text)

    def update_playback(self, *, force: bool = False) -> None:
        player = self.player
        if player is None:
            text = f"Playback unavailable / {round(self.analysis.duration_seconds)} s"
            if self.playback_label.text() != text:
                self.playback_label.setText(text)
            self._render_position()
            self._update_speed_label()
            return
        try:
            was_paused = player.paused
            position = player.poll()
            self.playback_position = position
            self._render_position()
            flashing = self.kick_flash.update(
                position,
                playing=not player.paused,
                generation=player.generation,
                playback_time=player.dac_time,
                event_time=player.event_time,
            )
            self.timeline.set_flash(flashing and self.flash_checkbox.isChecked())
            self._update_speed_label()
            self._update_seek_controls(position, force=force or was_paused != player.paused)
            if was_paused != player.paused:
                self.refresh_actions()
            if player.underruns > self.reported_underruns:
                self.reported_underruns = player.underruns
                self.statusBar().showMessage(
                    f"Audio output underrun ({self.reported_underruns}); "
                    + ("switched to reliable buffering." if player.reliable_buffering else "playback continues.")
                )
        except Exception as error:
            self.playback_failed(error)

    def _update_speed_label(self) -> None:
        left, right = self.timeline.interval
        zoom = DEFAULT_VIEW_DURATION / (right - left)
        speed = self.player.speed if self.player is not None else 1.0
        text = f"Zoom: {zoom:.2f}x  ·  Speed: {speed:.2f}x"
        if self.speed_label.text() != text:
            self.speed_label.setText(text)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.dirty and not self.close_pending:
            choice = QMessageBox.question(
                self,
                "Unsaved review",
                "Save annotation changes before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if choice == QMessageBox.StandardButton.Save:
                self.perform_action("save")
                if self.dirty:
                    event.ignore()
                    return
        if self.worker is not None:
            self.close_pending = True
            central = self.centralWidget()
            if central is not None:
                central.setEnabled(False)
            if self.player is not None:
                self._playback_action(self.player.pause)
            self.statusBar().showMessage("Waiting for background reclassification to finish before closing…")
            event.ignore()
            return
        self.timer.stop()

        if self.player is not None:
            try:
                self.player.close()
            except Exception:
                pass
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        event.accept()


def show_review(
    analysis: AcousticAnalysis,
    proposal: Proposal[AutomaticCandidate],
    draft_path: Path,
    start: float,
    duration: float,
    *,
    root: Path,
) -> None:
    draft = load_review(draft_path, proposal)
    app = QApplication.instance() or QApplication([])
    view = AnnotationReview(
        analysis,
        proposal,
        draft,
        draft_path,
        start,
        duration,
        audio_path=root / "audio" / f"{proposal['track_identifier']}.wav",
    )
    view.show()
    app.exec()
