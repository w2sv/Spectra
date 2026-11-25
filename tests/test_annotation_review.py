"""Focused Qt interaction and review-category regressions; no audio hardware."""

import copy
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QMessageBox

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
        self.reliable_buffering = False
        self.seeks: list[float] = []

    def seek(self, position: float) -> None:
        self.position = min(self.duration, max(0.0, position))
        self.seeks.append(self.position)
        if self.position == self.duration:
            self.paused = True

    def toggle(self) -> None:
        if self.paused and self.position == self.duration:
            self.position = 0
        self.paused = not self.paused

    def pause(self) -> None:
        self.paused = True

    def poll(self) -> float:
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
    review.follow_timer.stop()
    review.refresh_actions()
    return player


def wheel(view: AnnotationReview, mouse_time: float, delta: int = 120, *, evidence: bool = False) -> None:
    box = view.timeline.evidence_box if evidence else view.timeline.wave_box
    widget = view.timeline.evidence_widget if evidence else view.timeline.wave_widget
    point = widget.mapFromScene(box.mapViewToScene(QPointF(mouse_time, 0)))
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
    QApplication.sendEvent(widget.viewport(), event)


def marker_point(view: AnnotationReview, key: str):
    timestamp = effective_timestamp(view.editor.candidates[key], view.draft)
    scene = view.timeline.wave_box.mapViewToScene(QPointF(timestamp, view.timeline.lane_y))
    return view.timeline.wave_widget.mapFromScene(scene)


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
    viewport = review.timeline.wave_widget.viewport()
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=point)
    assert not review.draft.overrides
    QTest.mouseClick(viewport, Qt.MouseButton.RightButton, pos=point)
    assert review.selected == key and not review.draft.overrides
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, point)
    assert key in review.draft.overrides and not effective_kick(original, review.draft)
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


def test_hidden_recovery_insertion_duplicate_undo_and_persistence(review: AnnotationReview) -> None:
    candidate = next(c for c in review.editor.candidates.values() if c["automatic_kick"])
    key = candidate["candidate_id"]
    candidate["automatic_kick"] = False
    review.refresh()
    assert key not in review.timeline.marker_keys
    review.correct_or_insert(candidate["localized_timestamp"], None)
    assert review.selected == key and review.draft.overrides[key].kick
    assert len(review.draft.manual_events) == 0
    review.perform_action("save")
    assert load_review(review.draft_path, review.proposal).overrides[key].kick
    review.perform_action("undo")
    assert not review.draft.overrides
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
    fixed_x = review.timeline.cursor_lines[0].line().x1()
    review.timeline.follow(1.8)
    assert review.timeline.cursor_lines[0].line().x1() == fixed_x
    review.timeline.follow(0)
    assert review.timeline.interval == pytest.approx((0, 2))
    assert review.timeline.cursor_lines[0].line().x1() == 0
    review.timeline.follow(100)
    left, right = review.timeline.interval
    assert right == review.analysis.duration_seconds and right - left == pytest.approx(2)
    assert review.timeline.cursor_lines[0].line().x1() == pytest.approx(
        round(review.timeline.wave_box.width() * review.devicePixelRatioF()) / review.devicePixelRatioF()
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
            for box, line in zip((timeline.wave_box, timeline.evidence_box), timeline.cursor_lines, strict=True):
                expected_x = box.width() * (position - expected[0]) / 10
                assert line.line().x1() == pytest.approx(expected_x, abs=1 / timeline.devicePixelRatioF())
                assert line.isVisible()
            scene_x = [line.mapToScene(line.line().p1()).x() for line in timeline.cursor_lines]
            assert scene_x[0] == pytest.approx(scene_x[1], abs=1 / timeline.devicePixelRatioF())
        timeline.follow(100)
        fixed_x = timeline.cursor_lines[0].line().x1()
        assert fixed_x == pytest.approx(timeline.wave_box.width() * PLAYBACK_ANCHOR_FRACTION, abs=1)
        for position in (100.01, 100.03, 100.09):
            timeline.follow(position)
            assert timeline.cursor_lines[0].line().x1() == fixed_x
        timeline.follow(100, offset=0.25)
        assert timeline.interval == pytest.approx((96, 106))
        assert timeline.cursor_time == 100
        assert timeline.cursor_lines[0].line().x1() > fixed_x
        timeline.set_interval(0, 1000)
        timeline.follow(100)
        assert timeline.interval == pytest.approx((0, 317))
        assert timeline.cursor_lines[0].line().x1() == pytest.approx(timeline.wave_box.width() * 100 / 317, abs=1)
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
    for action, expected in [("forward_5", 21), ("back_5", 16), ("back_1", 15), ("forward_1", 16), ("back_10", 6)]:
        before = review.timeline.interval
        review.perform_action(action)
        assert playback.position == expected
        assert playback.paused is (not playing)
        assert review.follow_checkbox.isChecked() is following
        assert review.timeline.interval == pytest.approx((expected - 0.8, expected + 1.2) if following else before)
    review.perform_action("back_10")
    assert playback.position == 0
    assert playback.paused is (not playing)
    playback.position = playback.duration - 1
    review.perform_action("forward_5")
    assert playback.position == playback.duration and playback.paused
    assert review.timeline.interval[1] - review.timeline.interval[0] == pytest.approx(2)


@pytest.mark.parametrize("following", [True, False])
def test_click_and_slider_seeks_follow_mode_without_changing_annotations(
    review: AnnotationReview, playback: DeterministicPlayback, following: bool
) -> None:
    review.follow_checkbox.setChecked(following)
    review.timeline.set_interval(0, 2)
    playback.paused = False
    widget = review.timeline.wave_widget
    point = widget.mapFromScene(review.timeline.wave_box.mapViewToScene(QPointF(1.5, 0)))
    QTest.mouseClick(widget.viewport(), Qt.MouseButton.LeftButton, pos=point)
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
    widget = review.timeline.evidence_widget if evidence else review.timeline.wave_widget
    box = review.timeline.evidence_box if evidence else review.timeline.wave_box
    point = widget.mapFromScene(box.mapViewToScene(QPointF(10, 0)))
    viewport = widget.viewport()
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
    assert review.timeline.evidence_box.viewRange()[0] == pytest.approx((left, right))
    review.follow_checkbox.setChecked(True)
    assert review.timeline.interval == pytest.approx(before)
    review.follow_checkbox.setChecked(False)
    review.timeline.set_interval(1, 2)
    review.update_playback()
    assert all(not line.isVisible() for line in review.timeline.cursor_lines)
    review.perform_action("follow")
    assert review.timeline.interval == pytest.approx(before)
    assert playback.position == 10 and not playback.paused


@pytest.mark.parametrize("following", [True, False])
@pytest.mark.parametrize("playing", [True, False])
def test_wheel_zoom_anchors_and_boundaries(
    review: AnnotationReview, playback: DeterministicPlayback, following: bool, playing: bool
) -> None:
    playback.position = 12
    playback.paused = not playing
    review.update_playback()
    review.follow_checkbox.setChecked(following)
    review.timeline.set_interval(8, 10)
    for evidence in (False, True):
        left, right = review.timeline.interval
        mouse_time = left + (right - left) * 0.7
        wheel(review, mouse_time, evidence=evidence)
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
        left, right = review.timeline.interval
        assert 0 <= left < right <= playback.duration
    # A huge zoom out is constrained to the full recording; zoom in has a floor.
    review._zoom(1e6, playback.position)
    assert review.timeline.interval == pytest.approx((0, playback.duration))
    review._zoom(1e-6, playback.position)
    left, right = review.timeline.interval
    assert right - left == pytest.approx(0.02)
    assert 0 <= left < right <= playback.duration


def test_shortcuts_focus_and_override_do_not_duplicate_actions(
    review: AnnotationReview, playback: DeterministicPlayback, application
) -> None:
    playback.position = 15
    for focus in (review.timeline.wave_widget, review.buttons["back_5"], review.seek_slider):
        focus.setFocus()
        application.processEvents()
        playback.position = 15
        count = len(playback.seeks)
        QTest.keyClick(focus, Qt.Key.Key_Left)
        assert playback.position == 10 and len(playback.seeks) == count + 1
        QTest.keyClick(focus, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        assert playback.position == 11
        QTest.keyClick(focus, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
        assert playback.position == 10
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
        assert all(not box.isChecked() for name, box in view.overlays.items() if name != "grid")
        for index in (expected, 1 - expected) if alternative else (expected,):
            view.grid_choice.setCurrentIndex(index)
            hypothesis = "alternative" if index else "primary"
            assert set(view.queues["unsupported"]) == {
                c["candidate_id"] for c in proposal["kick_candidates"] if hypothesis in c["unsupported_grids"]
            }
            beats = proposal["alternative_beat_positions"] if index else proposal["beat_positions"]
            assert view.timeline.grid.getData()[0] == pytest.approx(np.repeat(beats, 2))
        assert "estimated beat" in view.legend.text()
        view.overlays["grid"].setChecked(False)
        assert "estimated beat" not in view.legend.text()
        assert view.draft == before and not view.dirty
    finally:
        view.close()


def test_playback_actions_disabled_without_audio(review: AnnotationReview) -> None:
    for action in ("play", "back_10", "back_5", "forward_5", "back_1", "forward_1"):
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
    viewport = review.timeline.wave_widget.viewport()
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


def test_follow_timers_share_snapshot_and_skip_unchanged_boundary_ranges(
    review: AnnotationReview, playback: DeterministicPlayback
) -> None:
    playback.position = 5
    playback.paused = False
    review.perform_action("follow")
    before = review.timeline.interval
    fixed_x = review.timeline.cursor_lines[0].line().x1()
    playback.position = 6
    review.update_playback()
    # Polling must not move the cursor independently of its following viewport.
    assert review.timeline.interval == before and review.timeline.cursor_time == 5
    playback.position = 7  # Scrolling must consume the polled snapshot, not reread the clock.
    review.follow_playback()
    assert review.timeline.interval == pytest.approx((5.2, 7.2))
    assert review.timeline.cursor_time == 6
    assert review.timeline.cursor_lines[0].line().x1() == fixed_x
    playback.position = 0
    review.update_playback()
    review.follow_playback()
    changes: list[tuple[float, float]] = []
    review.timeline.rangeChanged.connect(lambda left, right: changes.append((left, right)))
    review.follow_playback()
    review.follow_playback()
    assert not changes
