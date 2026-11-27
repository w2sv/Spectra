"""Focused Qt interaction and review-category regressions; no audio hardware."""

import copy
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("vispy")
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QKeyEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QLineEdit, QMenu, QMessageBox

from benchmarks.annotation.analysis import AutomaticCandidate, analyze_audio, generate_proposal
from benchmarks.annotation.automatic import automatic_proposal
from benchmarks.annotation.corrections import (
    Correction,
    effective_kick,
    effective_timestamp,
    load_review,
    new_review,
    uncertain_classification,
)
from benchmarks.annotation.review.styles import AUTOMATIC_KICK, KICK_FLASH
from benchmarks.annotation.review.timeline import PLAYBACK_ANCHOR_FRACTION, nearest_marker
from benchmarks.annotation.review.window import AnnotationReview
from benchmarks.fixtures import synthetic_fixtures


class DeterministicPlayback:
    """Audio-free player whose clock advances only when the test requests it."""

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.position = 0.0
        self.paused = True
        self.underruns = 0
        self.generation = 0
        self.reliable_buffering = False
        self.seeks: list[float] = []
        self.speed = 1.0
        self.target_speed = 1.0
        self.dac_time = 0.0

    def set_speed(self, speed: float) -> None:
        self.target_speed = speed

    def event_time(self, position: float) -> float | None:
        return position

    def seek(self, position: float) -> None:
        self.generation += 1
        self.position = min(self.duration, max(0.0, position))
        self.seeks.append(self.position)
        if self.position == self.duration:
            self.paused = True

    def toggle(self) -> None:
        self.generation += 1
        if self.paused and self.position == self.duration:
            self.position = 0
        self.paused = not self.paused

    def pause(self) -> None:
        self.paused = True

    def poll(self) -> float:
        self.dac_time = self.position
        return self.position

    def close(self) -> None:
        self.pause()


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def review_data():
    fixture = next(f for f in synthetic_fixtures() if f.identifier == "four_on_floor")
    analysis = analyze_audio(np.tile(fixture.load_audio(), (4, 1)), fixture.sample_rate)
    proposal = automatic_proposal(analysis, generate_proposal(analysis, "test", "hash", fixture.sample_rate, 120))
    return analysis, proposal


@pytest.fixture
def review(application, tmp_path: Path, review_data):
    analysis, original = review_data
    proposal = copy.deepcopy(original)
    view = AnnotationReview(analysis, proposal, new_review(proposal), tmp_path / "review.json", 0, 2)
    view.show()
    view.activateWindow()
    application.processEvents()
    yield view
    view.dirty = False
    view.close()
    application.processEvents()


@pytest.fixture
def playback(review: AnnotationReview, monkeypatch: pytest.MonkeyPatch):
    player = DeterministicPlayback(review.analysis.duration_seconds)
    monkeypatch.setattr(review, "player", player)
    review.timer.stop()
    review.refresh_actions()
    return player


def wheel(view: AnnotationReview, mouse_time: float, delta: int = 120, *, evidence: bool = False) -> None:
    widget = view.timeline.native
    point = view.timeline.time_to_point(mouse_time, evidence=evidence).toPoint()
    event = QWheelEvent(
        QPointF(point),
        QPointF(widget.mapToGlobal(point)),
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


def marker_point(view: AnnotationReview, key: str):
    timestamp = effective_timestamp(view.editor.candidates[key], view.draft)
    y = view.timeline.kick_y if effective_kick(view.editor.candidates[key], view.draft) else view.timeline.candidate_y
    return view.timeline.time_to_point(timestamp, y).toPoint()


def test_screen_space_hit_testing_and_plain_click_vs_shift_click(review: AnnotationReview) -> None:
    assert nearest_marker([("a", 10, 20), ("b", 30, 20)], 17, 20) == "a"
    assert nearest_marker([("a", 10, 20)], 19, 20) is None
    assert nearest_marker([("a", 10, 20)], 10, 29) is None
    key = next(
        k
        for k in review.timeline.marker_keys
        if 0.2 < effective_timestamp(review.editor.candidates[k], review.draft) < 1.8
    )
    original = copy.deepcopy(review.editor.candidates[key])
    point = marker_point(review, key)
    viewport = review.timeline.native
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=point)
    assert not review.draft.overrides and review.selected is None
    QTest.mouseClick(viewport, Qt.MouseButton.RightButton, pos=point)
    assert review.selected == key and not review.draft.overrides
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, point)
    assert key in review.draft.overrides and effective_kick(original, review.draft) != original["automatic_kick"]
    assert review.selected is None
    assert not review.timeline.selection.visible
    assert review.editor.candidates[key] == original
    assert key in review.timeline.marker_keys  # rejected markers remain visible/selectable
    review.perform_action("undo")
    assert not review.draft.overrides
    count = len(review.editor.candidates)
    # Panning must never become a click, even with Shift held.
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, point)
    QTest.mouseMove(viewport, point + QPointF(70, 0).toPoint(), 30)
    QTest.mouseRelease(
        viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, point + QPointF(70, 0).toPoint()
    )
    assert not review.draft.overrides and len(review.editor.candidates) == count
    assert not review.follow_checkbox.isChecked()


def test_canvas_margins_do_not_seek_edit_inspect_pan_or_zoom(
    review: AnnotationReview, playback: DeterministicPlayback
) -> None:
    timeline = review.timeline
    viewport = timeline.native
    playback.position = 10
    review.perform_action("follow")
    before = timeline.interval
    draft = copy.deepcopy(review.draft)
    wave_y = round(timeline.wave_view.size[1] / 2)
    evidence_bottom = round(timeline.evidence_view.pos[1] + timeline.evidence_view.size[1])
    for point in (
        QPoint(25, wave_y),  # Y labels and titles.
        QPoint(viewport.width() - 2, wave_y),
        QPoint(300, evidence_bottom + 15),  # Time axis and labels.
        QPoint(300, viewport.height() - 2),
    ):
        assert timeline.interaction_region(QPointF(point)) is None
        assert timeline.hit_test(QPointF(point)) is None
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseClick(viewport, Qt.MouseButton.RightButton, pos=point)
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, point)
        event = QWheelEvent(
            QPointF(point),
            QPointF(viewport.mapToGlobal(point)),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(viewport, event)
        # A drag that starts outside cannot become a pan after entering the plot.
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=point)
        inside = timeline.time_to_point(10).toPoint()
        QTest.mouseMove(viewport, inside, 10)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=inside)
        assert timeline.interval == before and timeline.zoom_transition is None
        assert not playback.seeks and review.selected is None
        assert review.draft == draft and review.follow_checkbox.isChecked()


def test_plot_and_divider_drags_continue_outside_their_initial_bounds(
    review: AnnotationReview, playback: DeterministicPlayback
) -> None:
    timeline = review.timeline
    viewport = timeline.native
    playback.position = 10
    review.perform_action("follow")
    before = timeline.interval
    point = timeline.time_to_point(10).toPoint()
    outside = QPoint(20, point.y())
    QTest.mousePress(viewport, Qt.MouseButton.MiddleButton, pos=point)
    QTest.mouseMove(viewport, outside, 10)
    QTest.mouseRelease(viewport, Qt.MouseButton.MiddleButton, pos=outside)
    assert timeline.interval[0] > before[0] and not review.follow_checkbox.isChecked()
    assert not playback.seeks and not review.draft.manual_events

    review.perform_action("follow")
    before = timeline.interval
    fraction = timeline.wave_fraction
    point = QPoint(300, round(timeline.wave_view.size[1] + 3))
    assert timeline.interaction_region(QPointF(point)) == "divider"
    outside = point + QPoint(-280, -60)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseMove(viewport, outside, 10)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=outside)
    assert timeline.wave_fraction < fraction
    assert timeline.interval == before and review.follow_checkbox.isChecked()
    assert not playback.seeks and not review.draft.manual_events


def test_hidden_recovery_insertion_duplicate_undo_and_persistence(review: AnnotationReview) -> None:
    candidate = next(c for c in review.editor.candidates.values() if c["automatic_kick"])
    key = candidate["candidate_id"]
    candidate["automatic_kick"] = False
    review.overlays["candidates"].setChecked(False)
    review.refresh()
    assert key not in review.timeline.marker_keys
    review.correct_or_insert(candidate["localized_timestamp"], None)
    assert review.selected is None and review.draft.overrides[key].kick
    assert len(review.draft.manual_events) == 0
    review.perform_action("save")
    assert load_review(review.draft_path, review.proposal).overrides[key].kick
    review.perform_action("undo")
    assert review.draft.overrides[key].kick  # Saving protects prior corrections.
    assert not review.editor.history and not review.editor.redo_history and not review.dirty
    # An isolated manually identified sound uses the production onset localizer.
    review.insert_manual(0.1)
    assert len(review.draft.manual_events) == 1
    review.insert_manual(0.11)
    assert len(review.draft.manual_events) == 1
    review.perform_action("undo")
    assert not review.draft.manual_events


def test_uncertainty_is_classification_only_and_navigation_reveals_one_event(review: AnnotationReview) -> None:
    base = copy.deepcopy(next(iter(review.editor.candidates.values())))
    for reason in ("uncertain onset", "off-grid prediction", "unsupported beat"):
        candidate: AutomaticCandidate = {
            **base,
            "classification": "likely",
            "positive_similarity": 0.99,
            "uncertainty_reasons": [reason],
        }
        assert not uncertain_classification(candidate, review.draft)
    cases: list[AutomaticCandidate] = [
        {**base, "classification": "ambiguous", "uncertainty_reasons": []},
        {**base, "classification": "unclassified", "uncertainty_reasons": []},
        {**base, "positive_similarity": 0.84, "uncertainty_reasons": []},
        {**base, "positive_similarity": 0.9, "negative_similarity": 0.85, "uncertainty_reasons": []},
    ]
    for candidate in cases:
        assert uncertain_classification(candidate, review.draft)
        review.draft.overrides[base["candidate_id"]] = Correction(True, base["final_timestamp"], base["timestamp"])
        assert not uncertain_classification(candidate, review.draft)
        review.draft.overrides.clear()
    candidates = list(review.editor.candidates.values())[:3]
    for index, candidate in enumerate(candidates):
        candidate.update(automatic_kick=False, classification="ambiguous", final_timestamp=0.1 + index * 0.1)
    review.overlays["candidates"].setChecked(False)
    review.refresh()
    before = set(review.timeline.marker_keys)
    review.selected = None
    review.timeline.set_interval(0, 2)
    review.perform_action("next_uncertain")
    assert review.selected == candidates[0]["candidate_id"]
    assert set(review.timeline.marker_keys) - before == {review.selected}
    review.perform_action("next_uncertain")
    assert review.selected == candidates[1]["candidate_id"]
    assert review.timeline.interval == pytest.approx((0, 2))


def test_group_confirmation_and_background_result_revision_guard(review: AnnotationReview) -> None:
    review.selected = next(k for k, c in review.editor.candidates.items() if c["cluster_id"] is not None)
    review.perform_action("group_nonkick")
    assert not review.draft.overrides and review.pending_group is not None
    review.perform_action("group_nonkick")
    assert review.draft.overrides
    snapshot = copy.deepcopy(review.draft)
    review._classified(copy.deepcopy(review.editor.predictions), review.revision - 1)
    assert review.draft == snapshot
    review.perform_action("undo")
    assert not review.draft.overrides


def test_close_cancel_and_following_are_independent_of_annotation_data(
    review: AnnotationReview, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposal = copy.deepcopy(review.proposal)
    review.timeline.follow(1.6)
    assert review.timeline.interval == pytest.approx((0.8, 2.8))
    fixed_x = review.timeline.playhead.x - review.timeline.wave_view.pos[0]
    review.timeline.follow(1.8)
    assert (review.timeline.playhead.x - review.timeline.wave_view.pos[0]) == fixed_x
    review.timeline.follow(0)
    assert review.timeline.interval == pytest.approx((0, 2))
    assert (review.timeline.playhead.x - review.timeline.wave_view.pos[0]) == 0
    review.timeline.follow(100)
    left, right = review.timeline.interval
    assert right == review.analysis.duration_seconds and right - left == pytest.approx(2)
    assert (review.timeline.playhead.x - review.timeline.wave_view.pos[0]) == pytest.approx(
        round(review.timeline.wave_view.size[0] * review.devicePixelRatioF()) / review.devicePixelRatioF()
    )
    assert review.proposal == proposal
    review.dirty = True
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Cancel)
    assert not review.close()
    assert review.isVisible()


def test_reclassification_runs_in_qt_worker_and_preserves_explicit_decisions(
    review: AnnotationReview, application
) -> None:
    from PySide6.QtCore import QElapsedTimer

    key = next(iter(review.editor.candidates))
    review.editor.toggle(key)
    review.edited("Test correction")
    correction = copy.deepcopy(review.draft.overrides[key])
    review.perform_action("reclassify")
    assert review.worker is not None and not review.buttons["reclassify"].isEnabled()
    elapsed = QElapsedTimer()
    elapsed.start()
    while review.worker is not None and elapsed.elapsed() < 5000:
        QTest.qWait(10)
        application.processEvents()
    assert review.worker is None
    assert review.draft.prediction_updates
    assert review.draft.overrides[key] == correction


def test_follow_ranges_cursor_alignment_and_visual_offset(application, review_data) -> None:
    from benchmarks.annotation.review.timeline import Timeline

    analysis, proposal = review_data
    timeline = Timeline(replace(analysis, duration_seconds=317), proposal)
    timeline.resize(1100, 700)
    timeline.show()
    application.processEvents()
    try:
        timeline.set_interval(0, 10)
        for position, expected in [(100, (96, 106)), (2, (0, 10)), (315, (307, 317))]:
            timeline.follow(position)
            assert timeline.interval == pytest.approx(expected)
            expected_x = timeline.time_to_point(position).x()
            assert timeline.playhead.x == pytest.approx(expected_x, abs=1 / timeline.devicePixelRatioF())
            assert timeline.playhead.visible
        timeline.follow(100)
        fixed_x = timeline.playhead.x - timeline.wave_view.pos[0]
        assert fixed_x == pytest.approx(timeline.wave_view.size[0] * PLAYBACK_ANCHOR_FRACTION, abs=1)
        for position in (100.01, 100.03, 100.09):
            timeline.follow(position)
            assert (timeline.playhead.x - timeline.wave_view.pos[0]) == fixed_x
        timeline.follow(100, offset=0.25)
        assert timeline.interval == pytest.approx((96, 106))
        assert timeline.cursor_time == 100
        assert (timeline.playhead.x - timeline.wave_view.pos[0]) > fixed_x
        timeline.set_interval(0, 1000)
        timeline.follow(100)
        assert timeline.interval == pytest.approx((0, 317))
        assert (timeline.playhead.x - timeline.wave_view.pos[0]) == pytest.approx(
            timeline.wave_view.size[0] * 100 / 317, abs=1
        )
        timeline.set_interval(1, 0)
        assert timeline.interval[1] - timeline.interval[0] == pytest.approx(0.02)
    finally:
        timeline.close()


@pytest.mark.parametrize("following", [True, False])
@pytest.mark.parametrize("playing", [True, False])
def test_relative_seeks_use_player_clock_and_preserve_mode_and_zoom(
    review: AnnotationReview, playback: DeterministicPlayback, following: bool, playing: bool
) -> None:
    review.follow_checkbox.setChecked(following)
    review.timeline.set_interval(4, 2)
    playback.position = 16
    playback.paused = not playing
    review.playback_position = 1  # Deliberately stale GUI snapshot.
    for action, expected in [
        ("forward_half", 17),
        ("back_half", 16),
        ("back_tenth", 15.8),
        ("forward_tenth", 16),
        ("back_10", 6),
    ]:
        before = review.timeline.interval
        review.perform_action(action)
        assert playback.position == pytest.approx(expected)
        assert playback.paused is (not playing)
        assert review.follow_checkbox.isChecked() is following
        assert review.timeline.interval == pytest.approx((expected - 0.8, expected + 1.2) if following else before)
    review.perform_action("back_10")
    assert playback.position == 0
    assert playback.paused is (not playing)
    playback.position = playback.duration - 1
    review.perform_action("forward_half")
    assert playback.position == playback.duration and playback.paused
    assert review.timeline.interval[1] - review.timeline.interval[0] == pytest.approx(2)


@pytest.mark.parametrize("following", [True, False])
def test_click_and_slider_seeks_follow_mode_without_changing_annotations(
    review: AnnotationReview, playback: DeterministicPlayback, following: bool
) -> None:
    review.follow_checkbox.setChecked(following)
    review.timeline.set_interval(0, 2)
    playback.paused = False
    widget = review.timeline.native
    point = review.timeline.time_to_point(1.5).toPoint()
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=point)
    assert playback.position == pytest.approx(1.5, abs=0.01)
    assert not playback.paused and not review.draft.overrides
    assert review.timeline.interval[0] == pytest.approx(playback.position - 0.8 if following else 0)
    before = review.timeline.interval
    review.seek_slider.setSliderDown(True)
    review.seek_slider.setSliderPosition(500_000)
    assert len(playback.seeks) == 1
    review.seek_slider.setSliderDown(False)  # Emits sliderReleased, committing once.
    assert len(playback.seeks) == 2
    assert playback.position == pytest.approx(playback.duration / 2)
    assert review.timeline.interval == pytest.approx(
        (playback.position - 0.8, playback.position + 1.2) if following else before
    )
    assert review.follow_checkbox.isChecked() is following


@pytest.mark.parametrize("evidence", [False, True])
def test_drag_is_free_navigation_without_playback_or_annotation_changes(
    review: AnnotationReview, playback: DeterministicPlayback, evidence: bool
) -> None:
    playback.position = 10
    playback.paused = False
    review.perform_action("follow")
    before = review.timeline.interval
    widget = review.timeline.native if evidence else review.timeline.native
    point = review.timeline.time_to_point(10, evidence=evidence).toPoint()
    viewport = widget
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, point)
    assert review.follow_checkbox.isChecked()
    assert review.timeline.interval == before
    QTest.mouseMove(viewport, point + QPoint(70, 0), 30)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, point + QPoint(70, 0))
    assert not review.follow_checkbox.isChecked()
    assert playback.position == 10 and not playback.paused and not playback.seeks
    assert not review.draft.overrides and not review.draft.manual_events
    left, right = review.timeline.interval
    assert left < before[0] and right - left == pytest.approx(2)
    rect = review.timeline.evidence_camera.rect
    assert rect is not None and (rect.left, rect.right) == pytest.approx((left, right))
    review.follow_checkbox.setChecked(True)
    assert review.timeline.interval == pytest.approx(before)
    review.follow_checkbox.setChecked(False)
    review.timeline.set_interval(1, 2)
    review.update_playback()
    assert not review.timeline.playhead.visible
    review.perform_action("follow")
    assert review.timeline.interval == pytest.approx(before)
    assert playback.position == 10 and not playback.paused


@pytest.mark.parametrize("following", [True, False])
@pytest.mark.parametrize("playing", [True, False])
def test_wheel_zoom_anchors_and_boundaries(
    review: AnnotationReview,
    playback: DeterministicPlayback,
    following: bool,
    playing: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [10.0]
    monkeypatch.setattr("benchmarks.annotation.review.timeline.monotonic", lambda: now[0])
    monkeypatch.setattr("benchmarks.annotation.review.window.monotonic", lambda: now[0])

    def finish_zoom() -> None:
        now[0] += 0.151
        review.update_playback()

    playback.position = 12
    playback.paused = not playing
    review.update_playback()
    review.follow_checkbox.setChecked(following)
    review.timeline.set_interval(8, 10)
    for evidence in (False, True):
        left, right = review.timeline.interval
        mouse_time = left + (right - left) * 0.7
        wheel(review, mouse_time, evidence=evidence)
        finish_zoom()
        new_left, new_right = review.timeline.interval
        width = new_right - new_left
        assert width < right - left
        if following:
            assert new_left == pytest.approx(playback.position - width * PLAYBACK_ANCHOR_FRACTION)
        else:
            assert (mouse_time - new_left) / width == pytest.approx(0.7, abs=0.01)
        assert review.follow_checkbox.isChecked() is following
        assert playback.position == 12 and playback.paused is (not playing) and not playback.seeks
    for position in (0.1, playback.duration - 0.1):
        playback.position = position
        review.timeline.set_interval(0 if position < 1 else playback.duration - 2, 2)
        wheel(review, position, -120)
        finish_zoom()
        left, right = review.timeline.interval
        assert 0 <= left < right <= playback.duration
    # A huge zoom out is constrained to the full recording; zoom in has a floor.
    review._zoom(1e6, playback.position)
    finish_zoom()
    assert review.timeline.interval == pytest.approx((0, playback.duration))
    review._zoom(1e-6, playback.position)
    finish_zoom()
    left, right = review.timeline.interval
    assert right - left == pytest.approx(0.02)
    assert 0 <= left < right <= playback.duration


def test_shortcuts_focus_and_override_do_not_duplicate_actions(
    review: AnnotationReview, playback: DeterministicPlayback, application
) -> None:
    playback.position = 15
    for focus in (review.timeline.native, review.buttons["back_half"], review.seek_slider):
        focus.setFocus()
        application.processEvents()
        playback.position = 15
        count = len(playback.seeks)
        QTest.keyClick(focus, Qt.Key.Key_Left)
        assert playback.position == 14 and len(playback.seeks) == count + 1
        QTest.keyClick(focus, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        assert playback.position == pytest.approx(14.2)
        QTest.keyClick(focus, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
        assert playback.position == 14
        QTest.keyClick(focus, Qt.Key.Key_Right)
        assert playback.position == 15
        QTest.keyClick(focus, Qt.Key.Key_Backspace)
        assert playback.position == 5
        QTest.keyClick(focus, Qt.Key.Key_Space)
        assert not playback.paused
        QTest.keyClick(focus, Qt.Key.Key_Space)
        assert playback.paused
        review.follow_checkbox.setChecked(False)
        before = review.timeline.interval
        review.timeline.set_interval(0, 2)
        QTest.keyClick(focus, Qt.Key.Key_F)
        assert review.follow_checkbox.isChecked()
        assert review.timeline.interval == pytest.approx(before)
        assert playback.position == 5 and playback.paused
        QTest.keyClick(focus, Qt.Key.Key_F)
        assert review.timeline.interval == pytest.approx(before)
    count = len(playback.seeks)
    event = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier)
    assert review.eventFilter(review.seek_slider, event)
    assert len(playback.seeks) == count
    entry = QLineEdit(review)
    entry.show()
    entry.setFocus()
    application.processEvents()
    review.follow_checkbox.setChecked(False)
    QTest.keyClick(entry, Qt.Key.Key_F)
    assert entry.text() == "f" and not review.follow_checkbox.isChecked()


@pytest.mark.parametrize(
    "selected, alternative, expected",
    [("primary", True, 0), ("alternative", True, 1), ("alternative", False, 0), (None, True, 0)],
)
def test_default_grid_and_unsupported_queue_share_hypothesis(
    application, review_data, tmp_path: Path, selected: str | None, alternative: bool, expected: int
) -> None:
    analysis, original = review_data
    proposal = copy.deepcopy(original)
    proposal["selected_grid"] = selected
    proposal["beat_positions"] = [0.3, 0.8]
    proposal["alternative_beat_positions"] = [0.4, 0.9] if alternative else []
    # Isolate unsupported candidates from effective kicks.
    for index, candidate in enumerate(proposal["kick_candidates"]):
        candidate["automatic_kick"] = False
        candidate["unsupported_grids"] = ["primary" if index % 2 == 0 else "alternative"]
    draft = new_review(proposal)
    before = copy.deepcopy(draft)
    view = AnnotationReview(analysis, proposal, draft, tmp_path / "grid.json")
    try:
        assert view.overlays["grid"].isChecked()
        assert view.grid_choice.currentIndex() == expected
        assert all(not box.isChecked() for name, box in view.overlays.items() if name not in ("grid", "candidates"))
        for index in (expected, 1 - expected) if alternative else (expected,):
            view.grid_choice.setCurrentIndex(index)
            hypothesis = "alternative" if index else "primary"
            assert set(view.queues["unsupported"]) == {
                c["candidate_id"] for c in proposal["kick_candidates"] if hypothesis in c["unsupported_grids"]
            }
            beats = proposal["alternative_beat_positions"] if index else proposal["beat_positions"]
            assert view.timeline.grid.visible
            assert view.timeline.beat_positions == pytest.approx(beats)
        assert any(entry.label == "Estimated beat" for entry in view.legend.entries)
        view.overlays["grid"].setChecked(False)
        assert all(entry.label != "Estimated beat" for entry in view.legend.entries)
        assert view.draft == before and not view.dirty
    finally:
        view.close()


def test_playback_actions_disabled_without_audio(review: AnnotationReview) -> None:
    for action in ("play", "back_10", "back_half", "forward_half", "back_tenth", "forward_tenth"):
        assert not review.action_enabled(action)
        review.perform_action(action)
    assert not review.seek_slider.isEnabled()


@pytest.mark.parametrize("following", [True, False])
def test_corrections_and_inspection_leave_playback_and_navigation_untouched(
    review: AnnotationReview, playback: DeterministicPlayback, following: bool
) -> None:
    playback.position = 1
    playback.paused = False
    review.follow_checkbox.setChecked(following)
    review.update_playback()
    key = next(k for k in review.timeline.marker_keys if 0.2 < review.editor.candidates[k]["final_timestamp"] < 1.8)
    point = marker_point(review, key)
    viewport = review.timeline.native
    before = review.timeline.interval
    QTest.mouseClick(viewport, Qt.MouseButton.RightButton, pos=point)
    assert review.selected == key and not review.draft.overrides
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, point)
    assert key in review.draft.overrides
    assert playback.position == 1 and not playback.paused and not playback.seeks
    assert review.timeline.interval == before and review.follow_checkbox.isChecked() is following
    review.perform_action("undo")
    assert not review.draft.overrides
    assert playback.position == 1 and not playback.paused and review.timeline.interval == before


def test_review_navigation_pauses_seeks_and_f_returns_without_seeking(
    review: AnnotationReview, playback: DeterministicPlayback
) -> None:
    candidates = list(review.editor.candidates.values())[:3]
    for index, candidate in enumerate(candidates):
        candidate.update(automatic_kick=False, classification="ambiguous", final_timestamp=5 + index * 0.1)
    review.refresh()
    playback.paused = False
    review.selected = None
    for candidate in candidates:
        review.perform_action("next_uncertain")
        assert review.selected == candidate["candidate_id"]
        assert review.selected in review.timeline.marker_keys
        assert playback.paused and not review.follow_checkbox.isChecked()
        left, right = review.timeline.interval
        assert right - left == pytest.approx(2) and playback.position == left
    position = playback.position
    seeks = list(playback.seeks)
    review.perform_action("follow")
    assert review.follow_checkbox.isChecked() and playback.paused
    assert playback.position == position and playback.seeks == seeks
    assert review.timeline.interval == pytest.approx((position - 0.8, position + 1.2))


def test_visual_frame_uses_one_snapshot_and_skips_unchanged_paused_ranges(
    review: AnnotationReview, playback: DeterministicPlayback, monkeypatch: pytest.MonkeyPatch
) -> None:
    playback.position = 5
    playback.paused = False
    review.perform_action("follow")
    fixed_x = review.timeline.playhead.x - review.timeline.wave_view.pos[0]
    playback.position = 6
    calls = []

    def poll() -> float:
        calls.append(playback.position)
        playback.position = 7  # Subsequent reads must not affect this visual frame.
        return 6

    monkeypatch.setattr(playback, "poll", poll)
    review.update_playback()
    assert calls == [6]
    assert review.timeline.interval == pytest.approx((5.2, 7.2))
    assert review.timeline.cursor_time == 6
    assert (review.timeline.playhead.x - review.timeline.wave_view.pos[0]) == fixed_x
    changes: list[tuple[float, float]] = []
    review.timeline.rangeChanged.connect(lambda left, right: changes.append((left, right)))
    playback.paused = True
    review.update_playback()
    review.update_playback()
    assert not changes


def test_save_boundary_failure_and_history_shortcuts(review: AnnotationReview, monkeypatch: pytest.MonkeyPatch) -> None:
    key = next(iter(review.editor.candidates))
    review.correct_or_insert(0, key)
    edited = copy.deepcopy(review.draft)
    assert review.dirty and review.unsaved_label.isVisible()
    focus = review.timeline.native
    QTest.keyClick(focus, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert not review.dirty and not review.buttons["undo"].isEnabled() and review.buttons["redo"].isEnabled()
    for modifiers, keycode in [
        (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier, Qt.Key.Key_Z),
        (Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_Y),
    ]:
        QTest.keyClick(focus, keycode, modifiers)
        assert review.draft == edited and review.dirty
        review.perform_action("undo")
    review.perform_action("redo")
    review.correct_or_insert(0, key)
    review.perform_action("undo")
    histories = copy.deepcopy((review.editor.history, review.editor.redo_history, review.draft))
    original_path = review.draft_path
    review.draft_path = original_path.parent  # Atomic replacement of a directory must fail.
    review.perform_action("save")
    assert (review.editor.history, review.editor.redo_history, review.draft) == histories
    assert review.dirty and review.buttons["undo"].isEnabled() and review.buttons["redo"].isEnabled()
    review.draft_path = original_path
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Cancel)
    assert not review.close()
    assert (review.editor.history, review.editor.redo_history, review.draft) == histories
    QTest.keyClick(focus, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
    assert review.statusBar().currentMessage() == "Review saved · undo history cleared"
    assert not review.dirty and not review.unsaved_label.isVisible()
    assert not review.buttons["undo"].isEnabled() and not review.buttons["redo"].isEnabled()
    assert not review.editor.history and not review.editor.redo_history
    review.perform_action("undo")
    assert review.draft == edited
    review.correct_or_insert(0, key)
    assert review.dirty
    review.perform_action("undo")
    assert not review.dirty
    review.perform_action("redo")
    assert review.dirty
    loaded = load_review(review.draft_path, review.proposal)
    from benchmarks.annotation.corrections import CorrectionEditor

    reopened = CorrectionEditor(review.proposal, loaded)
    assert not reopened.history and not reopened.redo_history and not reopened.dirty


def test_lanes_separate_coincident_markers_and_follow_classification(review: AnnotationReview) -> None:
    candidates = list(review.editor.candidates.values())[:2]
    kick, candidate = candidates
    for event in candidates:
        event["final_timestamp"] = 1
    kick["automatic_kick"] = True
    candidate["automatic_kick"] = False
    review.refresh()
    timeline = review.timeline
    for event in candidates:
        key = event["candidate_id"]
        point = QPointF(marker_point(review, key))
        assert timeline.hit_test(point) == key
    kick_point = timeline.time_to_point(1, timeline.kick_y)
    candidate_point = timeline.time_to_point(1, timeline.candidate_y)
    assert abs(kick_point.y() - candidate_point.y()) == pytest.approx(28, abs=0.5)
    key = kick["candidate_id"]
    review.correct_or_insert(1, key)
    assert timeline.marker_ys[timeline.marker_keys.index(key)] == timeline.candidate_y
    review.overlays["candidates"].setChecked(False)
    assert key in timeline.marker_keys and candidate["candidate_id"] not in timeline.marker_keys
    review.perform_action("undo")
    assert timeline.marker_ys[timeline.marker_keys.index(key)] == timeline.kick_y
    review.perform_action("redo")
    assert timeline.marker_ys[timeline.marker_keys.index(key)] == timeline.candidate_y
    review.timeline_click(1, key, True, Qt.KeyboardModifier.NoModifier)
    review.timeline_click(1, key, False, Qt.KeyboardModifier.NoModifier)
    assert review.selected is None and not timeline.selection.visible


def test_effective_kick_index_drives_both_cursor_flashes(
    review: AnnotationReview, playback: DeterministicPlayback
) -> None:
    # Isolate automatic, human-promoted, rejected, and manual events.
    events = list(review.editor.candidates.values())
    for event in events:
        event["automatic_kick"] = False
    events[0].update(automatic_kick=True, final_timestamp=1)
    events[1]["final_timestamp"] = 2
    events[2]["final_timestamp"] = 3
    review.editor.toggle(events[1]["candidate_id"])
    review.editor.set_classification(events[2]["candidate_id"], False)
    manual: AutomaticCandidate = {
        **events[0],
        "candidate_id": "manual-flash",
        "timestamp": 0.1,
        "final_timestamp": 0.1,
        "provenance": "manual",
    }
    assert review.editor.insert(manual)
    review.edited("Test")
    assert review.kick_flash.timestamps == [0.1, 1, 2]
    playback.paused = False
    for timestamp in review.kick_flash.timestamps:
        playback.position = timestamp - 0.01
        review.kick_flash.reset()
        review.update_playback()
        playback.position = timestamp + 0.01
        review.update_playback()
        assert review.timeline.flashing
        assert review.timeline.playhead.color == KICK_FLASH.color and review.timeline.playhead.width == 6
        assert KICK_FLASH.pen().color().saturation() == 255
        assert KICK_FLASH.pen().color().green() > QColor(AUTOMATIC_KICK.color).green()
        review.update_playback()  # A repeated snapshot neither restarts nor expires the pulse.
        assert review.timeline.flashing
        playback.position = timestamp + 0.074
        review.update_playback()
        assert review.timeline.flashing
        playback.position = timestamp + 0.076
        review.update_playback()
        assert not review.timeline.flashing
        assert review.timeline.playhead.width == 2
    playback.position = 3.01
    review.update_playback()
    assert not review.timeline.flashing
    review._seek(0.99)
    playback.position = 1.01
    review.update_playback()
    assert review.timeline.flashing
    review.flash_checkbox.setChecked(False)
    assert not review.timeline.flashing
    review.flash_checkbox.setChecked(True)
    assert not review.timeline.flashing
    review._seek(2.01)
    assert not review.timeline.flashing
    review.correct_or_insert(2, events[1]["candidate_id"])
    assert 2 not in review.kick_flash.timestamps and not review.timeline.flashing


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("following", [False, True])
def test_seek_only_refreshes_existing_selection(
    review: AnnotationReview,
    playback: DeterministicPlayback,
    monkeypatch: pytest.MonkeyPatch,
    selected: bool,
    following: bool,
) -> None:
    playback.paused = False
    review.follow_checkbox.setChecked(following)
    if selected:
        review.timeline_click(1, next(iter(review.editor.candidates)), True, Qt.KeyboardModifier.NoModifier)
    scatter = review.timeline.marker_arrays
    inspection = review.timeline.inspection
    refreshes = [Mock(wraps=getattr(review, name)) for name in ("refresh_markers", "refresh_actions", "_inspect")]
    for name, refresh in zip(("refresh_markers", "refresh_actions", "_inspect"), refreshes, strict=True):
        monkeypatch.setattr(review, name, refresh)
    before = review.timeline.interval
    review.timeline_click(1.5, None, False, Qt.KeyboardModifier.NoModifier)
    assert all(refresh.call_count == int(selected) for refresh in refreshes)
    assert review.selected is None and playback.seeks == [1.5] and not playback.paused
    assert review.timeline.interval == pytest.approx((0.7, 2.7) if following else before)
    if not selected:
        assert review.timeline.marker_arrays is scatter and review.timeline.inspection is inspection


def test_clear_manual_override_removes_event_and_selection(review: AnnotationReview) -> None:
    review.insert_manual(0.1)
    event = review.draft.manual_events[0]
    key = event["candidate_id"]
    timestamp = effective_timestamp(event, review.draft)
    review.perform_action("save")
    review.timeline_click(timestamp, key, True, Qt.KeyboardModifier.NoModifier)
    assert review.selected == key
    review.perform_action("clear_override")
    assert len(review.editor.history) == 1 and review.selected is None
    assert not review.draft.manual_events and key not in review.draft.overrides and key not in review.editor.candidates
    assert key not in review.timeline.marker_keys and timestamp not in review.kick_flash.timestamps
    assert not review.timeline.selection.visible
    review.perform_action("undo")
    assert not review.dirty and key in review.editor.candidates and review.draft.overrides[key].kick
    review.perform_action("redo")
    assert review.dirty and key not in review.editor.candidates and not review.draft.manual_events


@pytest.mark.parametrize("key", [Qt.Key.Key_Escape, Qt.Key.Key_Q])
@pytest.mark.parametrize("outcome", ["clean", "save", "discard", "cancel", "failed_save"])
def test_exit_shortcuts_use_save_discard_cancel_and_cleanup(
    review: AnnotationReview,
    playback: DeterministicPlayback,
    monkeypatch: pytest.MonkeyPatch,
    key: Qt.Key,
    outcome: str,
) -> None:
    review.perform_action("save")
    path = review.draft_path
    saved_bytes = path.read_bytes()
    if outcome != "clean":
        first, second = list(review.editor.candidates)[:2]
        review.correct_or_insert(0, first)
        review.correct_or_insert(0, second)
        review.perform_action("undo")
    before = copy.deepcopy((review.draft, review.editor.history, review.editor.redo_history))
    choice = {
        "save": QMessageBox.StandardButton.Save,
        "failed_save": QMessageBox.StandardButton.Save,
        "discard": QMessageBox.StandardButton.Discard,
    }.get(outcome, QMessageBox.StandardButton.Cancel)
    question = Mock(return_value=choice)
    monkeypatch.setattr(QMessageBox, "question", question)
    if outcome == "failed_save":
        review.draft_path = path.parent  # Replacing a directory must fail.
    playback.paused = False
    QTest.keyClick(review.timeline.native, key)
    assert question.call_count == (outcome != "clean")
    remains_open = outcome in ("cancel", "failed_save")
    assert review.isVisible() is remains_open
    assert playback.paused is (not remains_open)
    if remains_open:
        assert review.dirty and (review.draft, review.editor.history, review.editor.redo_history) == before
    if outcome == "save":
        assert load_review(path, review.proposal) == before[0]
        assert not review.dirty and not review.editor.history and not review.editor.redo_history
    else:
        assert path.read_bytes() == saved_bytes


def test_exit_shortcuts_defer_to_text_modifiers_and_popups(review: AnnotationReview, application) -> None:
    entry = QLineEdit(review)
    entry.show()
    entry.setFocus()
    application.processEvents()
    QTest.keyClick(entry, Qt.Key.Key_Q)
    assert entry.text() == "q" and review.isVisible()
    review.timeline.native.setFocus()
    for modifiers in (
        Qt.KeyboardModifier.ShiftModifier,
        Qt.KeyboardModifier.ControlModifier,
        Qt.KeyboardModifier.AltModifier,
    ):
        QTest.keyClick(review.timeline.native, Qt.Key.Key_Q, modifiers)
        assert review.isVisible()
    menu = QMenu(review)
    menu.addAction("Inspect")
    combo = QComboBox(review)
    combo.addItems(["Primary", "Alternative"])
    combo.show()
    for popup in (menu, combo):
        if isinstance(popup, QMenu):
            popup.popup(review.mapToGlobal(QPoint(20, 20)))
        else:
            popup.showPopup()
        application.processEvents()
        active = QApplication.activePopupWidget()
        assert active is not None
        QTest.keyClick(active, Qt.Key.Key_Escape)
        assert review.isVisible() and QApplication.activePopupWidget() is None
    dialog = QDialog(review)
    dialog.setModal(True)
    dialog.show()
    application.processEvents()
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    assert not dialog.isVisible() and review.isVisible()


def test_escape_cancels_the_actual_unsaved_dialog(review: AnnotationReview) -> None:
    review.correct_or_insert(0, next(iter(review.editor.candidates)))
    before = copy.deepcopy((review.draft, review.editor.history))
    dialogs = []

    def dismiss() -> None:
        dialog = QApplication.activeModalWidget()
        dialogs.append(dialog)
        if isinstance(dialog, QMessageBox):
            QTest.keyClick(dialog, Qt.Key.Key_Escape)

    QTimer.singleShot(0, dismiss)
    QTest.keyClick(review.timeline.native, Qt.Key.Key_Q)
    assert len(dialogs) == 1 and isinstance(dialogs[0], QMessageBox)
    assert review.isVisible() and review.dirty and (review.draft, review.editor.history) == before


@pytest.mark.parametrize("following", [False, True])
def test_zoom_redirection_adaptive_speed_reset_and_navigation(
    review: AnnotationReview, playback: DeterministicPlayback, monkeypatch: pytest.MonkeyPatch, following: bool
) -> None:
    now = [100.0]
    monkeypatch.setattr("benchmarks.annotation.review.timeline.monotonic", lambda: now[0])
    monkeypatch.setattr("benchmarks.annotation.review.window.monotonic", lambda: now[0])
    playback.position = 12
    playback.paused = False
    review.follow_checkbox.setChecked(following)
    review.timeline.set_interval(8, 8)
    review._zoom(0.5, 12)
    now[0] += 0.075
    review.update_playback()
    before = review.timeline.interval
    assert 4 < before[1] - before[0] < 8
    previous = review.timeline.zoom_transition
    assert previous is not None
    fraction = 0.8
    target_anchor = previous.target_left + fraction * previous.target_width
    mouse_time = before[0] + fraction * (before[1] - before[0])
    now[0] += 0.025  # A wheel event arrives before the next animation frame.
    review._zoom(0.5, mouse_time)
    assert review.timeline.target_width == pytest.approx(
        2
    )  # Accumulate against the target, not the intermediate width.
    redirected = review.timeline.zoom_transition
    assert redirected is not None and 4 < redirected.initial_width < before[1] - before[0]
    if not following:
        assert redirected.target_left + fraction * redirected.target_width == pytest.approx(target_anchor)
    assert review.timeline.interval == pytest.approx(
        (redirected.initial_left, redirected.initial_left + redirected.initial_width)
    )
    now[0] += 0.151
    review.update_playback()
    assert review.timeline.interval[1] - review.timeline.interval[0] == pytest.approx(2)
    assert playback.target_speed == pytest.approx((2 / 10) ** 0.5)
    assert "Speed: 1.00x" in review.speed_label.text()  # Submitted targets are not audible rates.
    playback.speed = 0.6
    review.update_playback()
    assert "Speed: 0.60x" in review.speed_label.text()
    before = review.timeline.interval
    review.adaptive_speed.setChecked(False)
    assert playback.target_speed == 1 and review.timeline.interval == before
    review.adaptive_speed.setChecked(True)
    assert playback.target_speed == pytest.approx((2 / 10) ** 0.5)
    QTest.keyClick(review.timeline.native, Qt.Key.Key_1)
    assert review.timeline.zoom_transition is not None
    now[0] += 0.151
    review.update_playback()
    assert review.timeline.interval[1] - review.timeline.interval[0] == pytest.approx(10)
    assert playback.target_speed == pytest.approx(1)
    assert playback.position == 12 and not playback.paused and review.follow_checkbox.isChecked() is following
    review._zoom(0.5, 12)
    now[0] += 0.075
    review.update_playback()
    width = review.timeline.interval[1] - review.timeline.interval[0]
    review._seek(14)
    assert review.timeline.zoom_transition is None
    assert review.timeline.interval[1] - review.timeline.interval[0] == pytest.approx(width)
    review._zoom(0.5, 14)
    review.timeline.manuallyNavigated.emit()
    assert review.timeline.zoom_transition is None and not review.follow_checkbox.isChecked()
    review._zoom(0.5, 14)
    review.perform_action("follow")
    assert review.timeline.zoom_transition is None and review.follow_checkbox.isChecked()


def test_keyboard_seeking_uses_displayed_zoom_with_adaptive_speed_disabled(
    review: AnnotationReview, playback: DeterministicPlayback, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [100.0]
    monkeypatch.setattr("benchmarks.annotation.review.timeline.monotonic", lambda: now[0])
    monkeypatch.setattr("benchmarks.annotation.review.window.monotonic", lambda: now[0])
    review.follow_checkbox.setChecked(False)
    review.adaptive_speed.setChecked(False)
    review.timeline.set_interval(4, 10)
    playback.position = 16
    playback.paused = False
    playback.speed = 0.25
    review._zoom(0.25, 9)
    now[0] += 0.075
    review.update_playback()
    before = review.timeline.interval
    assert before[1] - before[0] == pytest.approx(5)
    assert review.timeline.target_width == pytest.approx(2.5)
    QTest.keyClick(review.timeline.native, Qt.Key.Key_Right)
    assert playback.position == pytest.approx(18.5)
    assert review.timeline.interval == pytest.approx(before) and not playback.paused
    assert review.timeline.zoom_transition is None
    assert not review.follow_checkbox.isChecked() and not review.adaptive_speed.isChecked()
    review.timeline.set_interval(0, 0.5)
    QTest.keyClick(review.timeline.native, Qt.Key.Key_Left)
    assert playback.position == pytest.approx(18.25)
    QTest.keyClick(review.timeline.native, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert playback.position == pytest.approx(18.3)
    QTest.keyClick(review.timeline.native, Qt.Key.Key_Backspace)
    assert playback.position == pytest.approx(8.3)


def test_timeline_click_restores_transport_focus_after_text_input(
    review: AnnotationReview, playback: DeterministicPlayback, application
) -> None:
    entry = QLineEdit(review)
    entry.show()
    entry.setFocus()
    application.processEvents()
    assert QApplication.focusWidget() is entry
    QTest.mouseClick(review.timeline.native, Qt.MouseButton.LeftButton, pos=review.timeline.time_to_point(1).toPoint())
    assert QApplication.focusWidget() is review.timeline.native
    QTest.keyClick(review.timeline.native, Qt.Key.Key_Space)
    assert not playback.paused
