"""Native review window coordinating the correction model, timeline and PCM player."""

import bisect
import copy
import hashlib
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
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
from benchmarks.annotation.review.playback import PCMPlayback
from benchmarks.annotation.review.timeline import Timeline

KEY_ACTIONS: dict[int, str] = {
    Qt.Key.Key_Space: "play",
    Qt.Key.Key_Backspace: "back_10",
    Qt.Key.Key_Left: "back_5",
    Qt.Key.Key_Right: "forward_5",
    Qt.Key.Key_F: "follow",
    Qt.Key.Key_Comma: "previous_uncertain",
    Qt.Key.Key_Period: "next_uncertain",
    Qt.Key.Key_BracketLeft: "previous_unsupported",
    Qt.Key.Key_BracketRight: "next_unsupported",
    Qt.Key.Key_Less: "previous_corrected",
    Qt.Key.Key_Greater: "next_corrected",
}
SHIFT_KEY_ACTIONS: dict[int, str] = {
    Qt.Key.Key_Left: "back_1",
    Qt.Key.Key_Right: "forward_1",
    Qt.Key.Key_Comma: "previous_corrected",
    Qt.Key.Key_Period: "next_corrected",
}
RELATIVE_SEEKS = {"back_10": -10, "back_5": -5, "forward_5": 5, "back_1": -1, "forward_1": 1}
LABELS = {
    "play": "Play (Space)",
    "back_10": "Back 10 s (Backspace)",
    "back_5": "Back 5 s (Left)",
    "forward_5": "Forward 5 s (Right)",
    "previous_uncertain": "Previous uncertain classification (,)",
    "next_uncertain": "Next uncertain classification (.)",
    "previous_unsupported": "Previous unsupported beat ([)",
    "next_unsupported": "Next unsupported beat (])",
    "undo": "Undo correction (Ctrl+Z)",
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
        duration: float = 10,
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
        self.dirty = False
        self.revision = 0
        self.selected: str | None = None
        self.player: PCMPlayback | None = None
        self.worker: Reclassification | None = None
        self.close_pending = False
        self.pending_group: tuple[str, bool, tuple[str, ...]] | None = None
        self.localizer = OnsetLocalizer(analysis)
        self.playback_position = start
        self.reported_underruns = 0
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
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        left.addWidget(self.legend)
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
        self._section(controls, "Playback", ["play", "back_10", "back_5", "forward_5"])
        self.follow_checkbox = QCheckBox("Follow playback (F)")
        self.follow_checkbox.setChecked(True)
        self.follow_checkbox.setToolTip(
            "F returns to playback at 40% of the plot width where possible. Drag to explore independently; "
            "wheel zoom follows playback or the mouse pointer in free navigation."
        )
        self.follow_checkbox.toggled.connect(self._follow_toggled)
        controls.addWidget(self.follow_checkbox)
        self._section(
            controls,
            "Review",
            ["previous_uncertain", "next_uncertain", "previous_unsupported", "next_unsupported", "undo", "save"],
        )
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
            ("nonkicks", "Show non-kick candidates"),
            ("uncertainty", "Show classification uncertainty"),
            ("grid", "Show beat grid"),
            ("scores", "Show similarity scores"),
            ("timing", "Show timing diagnostics"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(key == "grid")
            checkbox.toggled.connect(self.refresh_markers)
            self.overlays[key] = checkbox
            advanced_layout.addWidget(checkbox)
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
        advanced_layout.addWidget(self.cursor_offset)
        self.advanced.setVisible(False)
        self.advanced_toggle.toggled.connect(self._advanced_toggled)
        controls.addWidget(self.advanced)
        controls.addStretch()
        self.statusBar().setSizeGripEnabled(True)
        self.statusBar().showMessage("Predictions are unreviewed. Save corrections with Ctrl+S.")
        self.timeline.clicked.connect(self.timeline_click)
        self.timeline.manuallyNavigated.connect(lambda: self.follow_checkbox.setChecked(False))
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
        self.follow_timer = QTimer(self)
        self.follow_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.follow_timer.setInterval(33)
        self.follow_timer.timeout.connect(self.follow_playback)
        if audio_path is not None:
            try:
                self.player = PCMPlayback.from_wav(audio_path)
                self.player.seek(start)
            except Exception as error:
                self.statusBar().showMessage(f"Playback unavailable: {error}. Review edits remain available.")
        self.refresh()
        self._follow_toggled(self.follow_checkbox.isChecked())
        self.timer.start()
        self.follow_timer.start()

    @property
    def draft(self) -> ReviewDraft:
        return self.editor.draft

    @property
    def alternative_grid(self) -> bool:
        return self.grid_choice.currentIndex() == 1

    def _button(self, layout: QVBoxLayout, action: str) -> None:
        button = QPushButton(LABELS[action])
        tooltip = LABELS[action]
        if action in ("back_5", "forward_5"):
            tooltip += "; Shift + " + ("Left: back 1 s" if action == "back_5" else "Right: forward 1 s")
        button.setToolTip(tooltip)
        button.clicked.connect(lambda _checked=False, name=action: self.perform_action(name))
        self.buttons[action] = button
        layout.addWidget(button)

    def _section(self, layout: QVBoxLayout, title: str, actions: list[str]) -> None:
        group = QGroupBox(title)
        inner = QVBoxLayout(group)
        inner.setSpacing(3)
        for action in actions:
            self._button(inner, action)
        layout.addWidget(group)

    def _advanced_toggled(self, expanded: bool) -> None:
        self.advanced.setVisible(expanded)
        self.advanced_toggle.setText("Advanced ▾" if expanded else "Advanced ▸")
        self._inspect()

    def _range_changed(self, left: float, right: float) -> None:
        self.range_label.setText(
            f"Visible range: {round(left)}–{round(right)} s" + (" · unsaved review" if self.dirty else "")
        )

    def _follow_toggled(self, enabled: bool) -> None:
        if enabled:
            self.update_playback()
            self.follow_playback()

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
            self.isActiveWindow()
            and event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
            and isinstance(event, QKeyEvent)
        ):
            focus = QApplication.focusWidget()
            if isinstance(focus, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
                return super().eventFilter(watched, event)
            modifiers = event.modifiers()
            action = None
            if modifiers == Qt.KeyboardModifier.ControlModifier:
                action = "undo" if event.key() == Qt.Key.Key_Z else "save" if event.key() == Qt.Key.Key_S else None
            elif modifiers in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier):
                action = KEY_ACTIONS.get(event.key())
                if modifiers == Qt.KeyboardModifier.ShiftModifier:
                    action = SHIFT_KEY_ACTIONS.get(event.key(), action)
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
            show_nonkicks=self.overlays["nonkicks"].isChecked(),
            uncertainty=self.overlays["uncertainty"].isChecked(),
            timing=self.overlays["timing"].isChecked(),
            scores=self.overlays["scores"].isChecked(),
            grid=self.overlays["grid"].isChecked(),
            alternative=self.alternative_grid,
        )
        entries = [
            "● Green: automatic kick",
            "■ Blue: human kick",
            "× Red: human rejection",
            "Pink: selection",
            "Cyan: playback",
            "Blue / rose: bass / broadband",
        ]
        if self.overlays["nonkicks"].isChecked():
            entries.append("Gray triangle: non-kick")
        if self.overlays["uncertainty"].isChecked():
            entries.append("Orange outline: classification uncertainty")
        if self.overlays["timing"].isChecked():
            entries.append("Lilac outline: timing / grid discrepancy")
        if self.overlays["grid"].isChecked():
            entries.append("Dashed: estimated beat")
        if self.overlays["scores"].isChecked():
            entries.append("Green dots: similarity")
        self.legend.setText(" · ".join(entries))

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
        if action == "play" or action in RELATIVE_SEEKS:
            return self.player is not None
        if action == "undo":
            return bool(self.editor.history)
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
            button.setEnabled(self.action_enabled(action))
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
        text = f"Selected: {'Kick' if effective_kick(candidate, self.draft) else 'Non-kick'} at {round(effective_timestamp(candidate, self.draft))} s"
        if self.advanced_toggle.isChecked():
            text += (
                f" · {effective_timestamp(candidate, self.draft):.4f} s · cluster {candidate['cluster_id']}\n{candidate['timing_reason']} · "
                + ", ".join(candidate["uncertainty_reasons"])
            )
        self.inspection.setText(text)

    def edited(self, message: str) -> None:
        self.dirty = True
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
            if key is not None:
                self.selected = key
                self.refresh_markers()
                self.refresh_actions()
                self._inspect()
            self._seek(timestamp)

    def correct_or_insert(self, timestamp: float, key: str | None) -> None:
        if key is not None:
            self.selected = key
            self.editor.toggle(key)
            self.edited("Existing event corrected; onset and identity preserved.")
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
                self.selected = appropriate[0]
                self.editor.toggle(self.selected)
                self.edited("Hidden non-kick recovered as a kick; existing onset preserved.")
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
            self.selected = key
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
        try:
            action()
        except Exception as error:
            self.playback_failed(error)
        self.update_playback()
        self.follow_playback()
        self.refresh_actions()

    def playback_failed(self, error: Exception) -> None:
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
            if action == "play" and self.player is not None:
                self._playback_action(self.player.toggle)
            elif action in RELATIVE_SEEKS and self.player is not None:
                self._seek(self.player.position + RELATIVE_SEEKS[action])
            elif action == "follow":
                self._restore_following()
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
            elif action == "undo":
                self.editor.undo()
                if self.selected not in self.editor.candidates:
                    self.selected = None
                self.edited("Correction undone.")
            elif action == "clear_override" and self.selected is not None:
                self.editor.clear_override(self.selected)
                self.edited("Explicit classification override removed.")
            elif action == "save":
                save_review(self.draft_path, self.draft, self.proposal)
                self.dirty = self.stale = False
                self.statusBar().showMessage(
                    "Review saved. Finalization still requires explicit full-recording confirmation."
                )
                self._range_changed(*self.timeline.interval)
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

    def update_playback(self) -> None:
        if self.player is None:
            self.playback_label.setText(f"Playback unavailable / {round(self.analysis.duration_seconds)} s")
            return
        try:
            was_paused = self.player.paused
            position = self.player.poll()
            self.playback_position = position
            if not self.follow_checkbox.isChecked() or self.player.paused:
                self.timeline.set_cursor(position, offset=self.cursor_offset.value() / 1000)
            if not self.seek_slider.isSliderDown():
                with QSignalBlocker(self.seek_slider):
                    self.seek_slider.setValue(round(position / self.analysis.duration_seconds * 1_000_000))
                self.playback_label.setText(f"Playback: {round(position)} / {round(self.analysis.duration_seconds)} s")
            if was_paused != self.player.paused:
                self.refresh_actions()
            if self.player.underruns > self.reported_underruns:
                self.reported_underruns = self.player.underruns
                self.statusBar().showMessage(
                    f"Audio output underrun ({self.reported_underruns}); "
                    + ("switched to reliable buffering." if self.player.reliable_buffering else "playback continues.")
                )
        except Exception as error:
            self.playback_failed(error)

    def follow_playback(self) -> None:
        if self.player is not None and self.follow_checkbox.isChecked():
            # The cursor timer polls the clock. Scroll and render that snapshot
            # together, including while paused after explicit navigation.
            self.timeline.follow(self.playback_position, offset=self.cursor_offset.value() / 1000)

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
        self.follow_timer.stop()
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
