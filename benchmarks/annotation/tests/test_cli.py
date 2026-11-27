"""CLI workflows, headless operations and safe proposal replacement."""

import copy
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.io import wavfile

from benchmarks.annotation import cli
from benchmarks.annotation.analysis import analyze_audio, generate_proposal
from benchmarks.annotation.automatic import AUTOMATIC_ALGORITHM_VERSION, automatic_proposal
from benchmarks.annotation.corrections import CorrectionEditor, load_review, new_review, proposal_checksum, save_review
from benchmarks.annotation.validation import ValidationStrategy
from benchmarks.fixtures import file_checksum
from benchmarks.tracks import Track


@pytest.fixture
def recording(tmp_path: Path):
    (tmp_path / "audio").mkdir()
    audio = tmp_path / "audio/test.wav"
    samples = np.zeros((48_000, 2), dtype=np.float32)
    wavfile.write(audio, 48_000, samples)
    analysis = analyze_audio(samples, 48_000)
    proposal = automatic_proposal(analysis, generate_proposal(analysis, "test", file_checksum(audio), 48_000, 120))
    track = Track(
        "test", "artist", "title", 120, "test", "local", "development", "test.wav", file_checksum(audio), 48_000, 1.0
    )
    cli.write_json(tmp_path / "proposals/test.json", proposal)
    (tmp_path / "annotations.json").write_text("[]")
    return track, proposal, analysis


@pytest.mark.parametrize("selection", [None, *ValidationStrategy, "all"])
def test_validation_selection_and_export(
    selection: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    validate = Mock(return_value={"result": "validation"})
    monkeypatch.setattr(cli, "validate", validate)
    output = tmp_path / "validation.json"
    command = ["--validate", "--json", str(output)]
    if selection is not None:
        command.extend(["--strategy", selection])
    cli.main(command)
    validate.assert_called_once_with(selection or ValidationStrategy.AUTOMATIC_TEMPLATE_MATCHING)
    assert json.loads(output.read_text()) == json.loads(capsys.readouterr().out) == validate.return_value


@pytest.mark.parametrize(
    "arguments",
    [
        ["--strategy", "all"],
        ["--track", "test", "--validate"],
        ["--track", "test", "--json", "ignored.json"],
        ["--track", "test", "--prepare-only", "--start", "0"],
        ["--track", "test", "--finalize", "--regenerate"],
    ],
)
def test_incompatible_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("prepare", [False, True])
def test_track_defaults_to_review_with_explicit_preparation(
    recording, monkeypatch: pytest.MonkeyPatch, prepare: bool
) -> None:
    track, _, _ = recording
    run = Mock(return_value={})
    monkeypatch.setattr(cli, "read_catalog", lambda: [track])
    monkeypatch.setattr(cli, "annotate_track", run)
    command = (
        ["--track", "test", "--prepare-only"] if prepare else ["--track", "test", "--start", "0.2", "--duration", "0.5"]
    )
    cli.main(command)
    assert run.call_args.args[0] == track
    assert run.call_args.kwargs["mode"] == ("prepare" if prepare else "review")
    assert (run.call_args.kwargs["start"], run.call_args.kwargs["duration"]) == ((0, 10) if prepare else (0.2, 0.5))


def test_preparation_and_diagnostics_are_headless(recording, tmp_path: Path) -> None:
    track, _, _ = recording
    script = """
import json, sys
from pathlib import Path
from benchmarks.annotation.cli import annotate_track
from benchmarks.tracks import Track
root = Path(sys.argv[2])
track = Track(**json.loads(sys.argv[1]))
annotate_track(track, mode="prepare", root=root)
annotate_track(track, mode="diagnostics", root=root)
assert list((root / "proposals/diagnostics").glob("*.png"))
assert not any(name.startswith(("PySide6", "vispy", "sounddevice")) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, json.dumps(asdict(track)), str(tmp_path)], check=True, timeout=60)


@pytest.mark.parametrize("human", [False, True])
def test_outdated_cache_regenerates_predictions_but_protects_human_corrections(
    recording, tmp_path: Path, human: bool
) -> None:
    track, current, _ = recording
    cached = copy.deepcopy(current)
    cached["annotation_algorithm_version"] = AUTOMATIC_ALGORITHM_VERSION - 1
    cached["kick_candidates"].append({**cached["kick_candidates"][0], "candidate_id": "obsolete", "timestamp": 0.25})
    editor = CorrectionEditor(cached, new_review(cached))
    editor.replace_predictions(copy.deepcopy(editor.predictions))
    key = next(iter(editor.candidates))
    if human:
        editor.toggle(key)
    path, draft_path = tmp_path / "proposals/test.json", tmp_path / "drafts/test.json"
    cli.write_json(path, cached)
    save_review(draft_path, editor.draft, cached)
    before = (path.read_bytes(), draft_path.read_bytes())
    if human:
        with pytest.raises(ValueError, match="--regenerate explicitly"):
            cli.annotate_track(track, mode="prepare", root=tmp_path)
        assert (path.read_bytes(), draft_path.read_bytes()) == before
    cli.annotate_track(track, mode="prepare", regenerate=human, root=tmp_path)
    adopted = json.loads(path.read_text())
    loaded = load_review(draft_path, adopted)
    assert adopted["annotation_algorithm_version"] == AUTOMATIC_ALGORITHM_VERSION
    assert adopted["kick_candidates"] == json.loads(json.dumps(current["kick_candidates"]))
    assert loaded.overrides == editor.draft.overrides
    assert not loaded.prediction_updates and "obsolete" not in loaded.candidate_ids
    assert loaded.originating_proposal_sha256 != proposal_checksum(adopted)
    assert draft_path.read_bytes() == before[1]
    save_review(draft_path, loaded, adopted)
    assert loaded.annotation_algorithm_version == adopted["annotation_algorithm_version"]
    assert loaded.originating_proposal_sha256 == proposal_checksum(adopted)


def test_unmatched_regeneration_and_cancelled_reset_never_replace_review_or_proposal(
    recording,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    track, original, _ = recording
    editor = CorrectionEditor(original, new_review(original))
    editor.toggle(next(iter(editor.candidates)))
    path, draft_path = tmp_path / "proposals/test.json", tmp_path / "drafts/test.json"
    save_review(draft_path, editor.draft, original)
    before = (path.read_bytes(), draft_path.read_bytes())
    changed = copy.deepcopy(original)
    changed["kick_candidates"] = []
    monkeypatch.setattr(cli, "automatic_proposal", lambda *args: changed)
    with pytest.raises(ValueError, match="Unmatched human corrections"):
        cli.annotate_track(track, regenerate=True, mode="prepare", root=tmp_path)
    assert (path.read_bytes(), draft_path.read_bytes()) == before
    monkeypatch.setattr("builtins.input", lambda _: "cancel")
    with pytest.raises(ValueError, match="reset cancelled"):
        cli.annotate_track(track, regenerate=True, reset=True, mode="prepare", root=tmp_path)
    assert (path.read_bytes(), draft_path.read_bytes()) == before
    assert not draft_path.with_suffix(".before-reset.json").exists()
    monkeypatch.setattr("builtins.input", lambda _: "RESET test DRAFT")
    cli.annotate_track(track, reset=True, mode="prepare", root=tmp_path)
    assert draft_path.with_suffix(".before-reset.json").read_bytes() == before[1]
    assert not load_review(draft_path, original).overrides
    with pytest.raises(ValueError, match="Backup already exists"):
        cli.annotate_track(track, reset=True, mode="prepare", root=tmp_path)


def test_outdated_acoustic_cache_without_a_draft_regenerates(recording, tmp_path: Path) -> None:
    track, _, analysis = recording
    old = generate_proposal(analysis, track.identifier, track.sha256, 48_000, 120)
    old["kick_candidates"] = [
        {
            "timestamp": 0.5,
            "bass_evidence": 1.0,
            "prominence": 1.0,
            "broadband_evidence": 0.0,
            "nearest_beat_distance_seconds": 0.0,
            "grid_aligned": True,
        }
    ]
    path = tmp_path / "proposals/test.json"
    cli.write_json(path, old)
    cli.annotate_track(track, mode="prepare", root=tmp_path)
    assert json.loads(path.read_text())["annotation_algorithm_version"] == AUTOMATIC_ALGORITHM_VERSION
    assert not (tmp_path / "drafts/test.json").exists()
