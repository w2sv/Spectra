"""Review navigation anchors, grid provenance, decisions, and shared UI actions."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from benchmarks.annotation_analysis import Proposal, analyze_audio
from benchmarks.annotation_editing import (
    DraftEditor,
    UnsupportedBeatNavigator,
    beat_grid_identity,
    load_draft,
    new_draft,
    save_draft,
)
from benchmarks.annotation_templates import TemplateDefinition


def proposal(beats: list[float], duration: float = 10.0) -> Proposal:
    return {
        "track_identifier": "test",
        "audio_sha256": "0" * 64,
        "annotation_algorithm_version": 2,
        "analysis_parameters": {},
        "bpm": 120,
        "beat_grid_phase_seconds": 0.0,
        "beat_positions": beats,
        "unsupported_beat_positions": beats.copy(),
        "kick_candidates": [],
        "beat_grid_validation": [],
        "duration_seconds": duration,
        "source_sample_rate": 48_000,
        "informational": {},
        "alternative_phase_seconds": 0.25,
        "alternative_beat_positions": [0.25, 0.75, 1.25],
    }


def test_forward_backward_navigation_keeps_selected_beat_as_anchor() -> None:
    current = proposal([0.05, 0.2, 0.35, 1.0, 9.95])
    draft = new_draft(current)
    navigator = UnsupportedBeatNavigator()
    navigator.use_grid(beat_grid_identity(current, False))
    start = 0.0
    for expected in current["beat_positions"]:
        window = navigator.jump(1, current["beat_positions"], draft, start, 10)
        assert window is not None
        start, width = window
        assert navigator.selected == expected
        assert width == 2 and 0 <= start <= 8
    assert navigator.jump(1, current["beat_positions"], draft, start, 10) is None
    assert navigator.selected == 9.95
    for expected in reversed(current["beat_positions"][:-1]):
        assert navigator.jump(-1, current["beat_positions"], draft, start, 10) is not None
        assert navigator.selected == expected
    assert navigator.jump(-1, current["beat_positions"], draft, 0, 10) is None
    assert navigator.selected == 0.05


@pytest.mark.parametrize(("beat", "duration", "window"), [(0.01, 10, (0, 2)), (9.99, 10, (8, 2)), (1.4, 1.5, (0, 1.5))])
def test_navigation_shifts_windows_at_recording_boundaries(
    beat: float, duration: float, window: tuple[float, float]
) -> None:
    current = proposal([beat], duration)
    navigator = UnsupportedBeatNavigator()
    assert navigator.jump(1, [beat], new_draft(current), 0, duration) == window


def test_annotations_within_50ms_are_excluded_and_undo_recalculates() -> None:
    current = proposal([0.5, 0.55, 0.551, 1.0])
    draft = new_draft(current)
    draft.kick_onsets = [0.5]
    editor = DraftEditor(draft, TemplateDefinition("test", "0" * 64), current)
    navigator = UnsupportedBeatNavigator()
    navigator.use_grid(beat_grid_identity(current, False))
    assert navigator.remaining(current["beat_positions"], editor.draft) == [0.551, 1.0]
    editor.add(0.97)
    assert navigator.remaining(current["beat_positions"], editor.draft) == [0.551]
    assert editor.undo()
    assert navigator.remaining(current["beat_positions"], editor.draft) == [0.551, 1.0]


def test_decisions_persist_and_both_presence_and_absence_are_undoable(tmp_path: Path) -> None:
    current = proposal([0.2, 0.4, 0.6])
    identity = beat_grid_identity(current, False)
    editor = DraftEditor(new_draft(current), TemplateDefinition("test", "0" * 64), current)
    navigator = UnsupportedBeatNavigator()
    navigator.use_grid(identity)
    editor.review_unsupported_beat(identity, 0.2, False)
    assert editor.draft.kick_onsets == []
    editor.review_unsupported_beat(identity, 0.4, True)
    assert editor.draft.kick_onsets == [0.4]
    assert editor.draft.unsupported_beat_reviews[identity] == {"0.2": "gap", "0.4": "kick"}
    path = tmp_path / "draft.json"
    save_draft(path, editor.draft, 10)
    reloaded = load_draft(path, current)
    assert navigator.remaining(current["beat_positions"], reloaded) == [0.6]
    assert editor.undo()
    assert editor.draft.kick_onsets == []
    assert navigator.remaining(current["beat_positions"], editor.draft) == [0.4, 0.6]
    assert editor.undo()
    assert navigator.remaining(current["beat_positions"], editor.draft) == [0.2, 0.4, 0.6]


def test_decisions_are_specific_to_the_originating_grid() -> None:
    current = proposal([0.0, 0.5, 1.0])
    editor = DraftEditor(new_draft(current), TemplateDefinition("test", "0" * 64), current)
    navigator = UnsupportedBeatNavigator()
    primary, alternative = beat_grid_identity(current, False), beat_grid_identity(current, True)
    navigator.use_grid(primary)
    editor.review_unsupported_beat(primary, 0.0, False)
    navigator.selected = 0.5
    navigator.use_grid(alternative)
    assert navigator.selected is None
    assert navigator.remaining(current.get("alternative_beat_positions", []), editor.draft) == [0.25, 0.75, 1.25]
    editor.review_unsupported_beat(alternative, 0.25, False)
    navigator.use_grid(primary)
    assert navigator.remaining(current["beat_positions"], editor.draft) == [0.5, 1.0]
    current["beat_grid_phase_seconds"] = 0.1
    navigator.use_grid(beat_grid_identity(current, False))
    assert navigator.remaining(current["beat_positions"], editor.draft) == [0.0, 0.5, 1.0]


def test_review_buttons_and_keyboard_share_navigation_and_decision_handlers(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent

    from benchmarks.annotation_review import AnnotationReview
    from benchmarks.review_playback import MpvPlayback

    current = proposal([0.1, 0.2, 0.3, 3.9], duration=4)
    current["alternative_beat_positions"] = [0.25 + i * 0.5 for i in range(8)]
    analysis = analyze_audio(np.zeros((4 * 48_000, 2), dtype=np.float32), 48_000)
    view = AnnotationReview(analysis, current, new_draft(current), tmp_path / "draft.json", 0, 4)
    # A real IPC connection verifies pause/seek commands without audio output.
    import json
    import socket

    client, server = socket.socketpair()
    server.setblocking(False)
    player = MpvPlayback(client)
    view.player = player
    try:
        original_templates = asdict(view.editor.templates)
        view.on_key(KeyEvent("key_press_event", view.figure.canvas, "]"))
        assert view.unsupported_navigator.selected == 0.1
        commands = [json.loads(line)["command"] for line in server.recv(65_536).splitlines()]
        assert ["set_property", "pause", True] in commands
        assert commands[-1] == ["seek", 0.0, "absolute+exact"]
        button = view.controls.buttons["next_unsupported"]
        view.figure.canvas.draw()
        x, y = button.ax.transAxes.transform((0.5, 0.5))
        for event_name in ("button_press_event", "button_release_event"):
            event = MouseEvent(event_name, view.figure.canvas, x, y, button=MouseButton.LEFT)
            view.figure.canvas.callbacks.process(event_name, event)
        assert view.unsupported_navigator.selected == 0.2
        view.on_key(KeyEvent("key_press_event", view.figure.canvas, "y"))
        assert view.draft.kick_onsets == [0.2]
        assert "provisional" in view.status
        view.on_key(KeyEvent("key_press_event", view.figure.canvas, "ctrl+z"))
        assert view.draft.kick_onsets == []
        assert 0.2 in view.remaining_unsupported()
        assert asdict(view.editor.templates) == original_templates
        assert current["kick_candidates"] == []
        assert "Confirmed kicks: 0" in view.unsupported_progress()
        view.perform_action("kick_absent")
        assert 0.2 not in view.remaining_unsupported()
        view.perform_action("save")
        assert (
            load_draft(tmp_path / "draft.json", current).unsupported_beat_reviews == view.draft.unsupported_beat_reviews
        )
        view.perform_action("grid")
        assert view.unsupported_navigator.selected is None
        assert view.remaining_unsupported() == current.get("alternative_beat_positions")
        view.perform_action("grid")
        assert view.remaining_unsupported() == [0.1, 0.3, 3.9]
        view.perform_action("reclassify")
        assert view.remaining_unsupported() == [0.1, 0.3, 3.9]
        view.cursor_time = 0.31
        view.perform_action("add")
        assert view.remaining_unsupported() == [0.1, 3.9]
        view.perform_action("undo")
        assert view.remaining_unsupported() == [0.1, 0.3, 3.9]
        for expected in (0.1, 0.3, 3.9):
            view.perform_action("next_unsupported")
            assert view.unsupported_navigator.selected == expected
        assert view.current_interval() == (2.0, 4.0)
        view.perform_action("next_unsupported")
        assert "No further unsupported beats" in view.status
        assert view.unsupported_navigator.selected == 3.9
    finally:
        view.close()
        server.close()
        plt.close(view.figure)
