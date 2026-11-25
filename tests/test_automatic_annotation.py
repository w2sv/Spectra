"""Automatic numerical invariants and correction persistence, without audio output."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.annotation.analysis import (
    AcousticAnalysis,
    AutomaticCandidate,
    Proposal,
    analyze_audio,
    generate_proposal,
)
from benchmarks.annotation.automatic import OnsetLocalizer, automatic_proposal, feature_matrix
from benchmarks.annotation.corrections import (
    CorrectionEditor,
    centered_interval,
    effective_kick,
    effective_onsets,
    effective_timestamp,
    load_review,
    migrate_legacy,
    navigate_candidate,
    new_review,
    proposal_checksum,
    save_review,
)
from benchmarks.annotation.templates import event_feature
from benchmarks.fixtures import AudioFixture, synthetic_fixtures

type Clean = tuple[AudioFixture, AcousticAnalysis, Proposal, Proposal[AutomaticCandidate]]


@pytest.fixture(scope="module")
def clean() -> Clean:
    fixture = next(fixture for fixture in synthetic_fixtures() if fixture.identifier == "four_on_floor")
    analysis = analyze_audio(fixture.load_audio(), fixture.sample_rate)
    acoustic = generate_proposal(analysis, fixture.identifier, "a" * 64, fixture.sample_rate, 120)
    return fixture, analysis, acoustic, automatic_proposal(analysis, acoustic)


def test_discovery_is_deterministic_and_does_not_need_ground_truth(clean: Clean) -> None:
    _, analysis, acoustic, proposal = clean
    again = automatic_proposal(analysis, acoustic)
    assert proposal_checksum(again) == proposal_checksum(proposal)
    assert len(proposal.get("automatic_reference_ids", [])) > 0
    assert any(c["automatic_kick"] for c in proposal["kick_candidates"])
    originals = [c["timestamp"] for c in proposal["kick_candidates"] if c["provenance"] == "acoustic"]
    assert originals == [c["timestamp"] for c in acoustic["kick_candidates"]]
    assert len({c["candidate_id"] for c in proposal["kick_candidates"]}) == len(proposal["kick_candidates"])
    for c in proposal["kick_candidates"]:
        if c["provenance"] != "acoustic":
            assert all(abs(c["timestamp"] - old["timestamp"]) > 0.030 for old in acoustic["kick_candidates"])


def test_batched_features_and_localization(clean: Clean) -> None:
    fixture, analysis, _, _ = clean
    times = np.asarray([0, 0.5, 0.98, analysis.duration_seconds - 0.001])
    features, valid = feature_matrix(analysis, times)
    for index, time in enumerate(times):
        feature = event_feature(analysis, float(time))
        assert valid[index] == (feature is not None)
        if feature is not None:
            np.testing.assert_allclose(features[index], feature, atol=1e-14)
    localizer = OnsetLocalizer(analysis)
    for onset in fixture.kick_onsets:
        localized, reliable, _, _ = localizer.localize(onset - 0.015)
        assert reliable and abs(localized - onset) < 0.010


def test_silent_grid_candidates_are_selectable_but_never_automatic_kicks() -> None:
    analysis = analyze_audio(np.zeros((96_000, 2), dtype=np.float32), 48_000)
    proposal = automatic_proposal(analysis, generate_proposal(analysis, "silence", "s", 48_000, 120))
    assert proposal["kick_candidates"]
    assert all(c["provenance"] == "beat_grid_only" and not c["automatic_kick"] for c in proposal["kick_candidates"])
    assert all(not c["localization_reliable"] for c in proposal["kick_candidates"])


def test_overrides_group_undo_reclassification_and_persistence(clean: Clean, tmp_path: Path) -> None:
    proposal: Proposal[AutomaticCandidate] = copy.deepcopy(clean[3])
    editor = CorrectionEditor(proposal, new_review(proposal))
    first = proposal["kick_candidates"][0]
    key = first["candidate_id"]
    original_time = effective_timestamp(first, editor.draft)
    original_kick = effective_kick(first, editor.draft)
    editor.toggle(key)
    editor.toggle(key)
    assert key in editor.draft.overrides and effective_kick(first, editor.draft) == original_kick
    assert effective_timestamp(first, editor.draft) == original_time
    count = len(editor.group_targets(key))
    assert editor.group_correct(key, False) == count
    assert editor.draft.overrides[key].provenance == "individual"
    assert all(c.provenance.startswith("cluster:") for k, c in editor.draft.overrides.items() if k != key)
    assert editor.undo() and len(editor.draft.overrides) == 1
    predictions = copy.deepcopy(proposal["kick_candidates"])
    for c in predictions:
        c["automatic_kick"] = not c["automatic_kick"]
        c["final_timestamp"] += 0.001
    changed = editor.replace_predictions(predictions)
    assert changed == len(predictions) - 1
    assert effective_timestamp(editor.candidates[key], editor.draft) == original_time
    path = tmp_path / "review.json"
    save_review(path, editor.draft, proposal)
    loaded = load_review(path, proposal)
    assert effective_onsets(proposal, loaded) == effective_onsets(proposal, editor.draft)
    assert loaded.prediction_updates
    assert editor.undo() and not editor.draft.prediction_updates
    assert len(editor.draft.overrides) == 1


def test_regeneration_keeps_exact_decisions_and_rejects_unmatched(clean: Clean, tmp_path: Path) -> None:
    proposal = copy.deepcopy(clean[3])
    editor = CorrectionEditor(proposal, new_review(proposal))
    key = proposal["kick_candidates"][0]["candidate_id"]
    editor.toggle(key)
    path = tmp_path / "draft.json"
    save_review(path, editor.draft, proposal)
    other = copy.deepcopy(proposal)
    other["informational"]["annotation_processing_seconds"] = 123.4
    assert proposal_checksum(other) == proposal_checksum(proposal)
    other["kick_candidates"][0]["final_timestamp"] += 0.010
    loaded = load_review(path, other)
    assert effective_timestamp(other["kick_candidates"][0], loaded) == editor.draft.overrides[key].timestamp
    other["kick_candidates"].pop(0)
    with pytest.raises(ValueError, match="Unmatched human corrections"):
        load_review(path, other)
    assert json.loads(path.read_text())["overrides"][key]
    other["audio_sha256"] = "different"
    with pytest.raises(ValueError, match="checksum"):
        load_review(path, other)


def test_unified_unsupported_navigation_preserves_close_events_and_grid_decisions(clean: Clean) -> None:
    base = copy.deepcopy(clean[3]["kick_candidates"][0])
    candidates: list[AutomaticCandidate] = []
    for i, time in enumerate([0.01, 0.2, 0.35, 3.99]):
        candidates.append(
            {
                **base,
                "candidate_id": str(i),
                "timestamp": time,
                "final_timestamp": time,
                "automatic_kick": False,
                "unsupported_grids": ["primary"],
                "uncertainty_reasons": ["unsupported beat"],
            }
        )
    candidates.append(
        {
            **base,
            "candidate_id": "alt",
            "timestamp": 0.25,
            "final_timestamp": 0.25,
            "automatic_kick": False,
            "unsupported_grids": ["alternative"],
        }
    )
    proposal = copy.deepcopy(clean[3])
    proposal["kick_candidates"] = candidates
    draft = new_review(proposal)
    selected = None
    for expected in ("0", "1", "2", "3"):
        selected = navigate_candidate(candidates, draft, "unsupported", 1, selected, 0)
        assert selected == expected
    assert navigate_candidate(candidates, draft, "unsupported", 1, selected, 0) is None
    assert navigate_candidate(candidates, draft, "unsupported", -1, selected, 0) == "2"
    assert centered_interval(0.01, 4) == (0, 2)
    assert centered_interval(3.99, 4) == (2, 2)
    assert centered_interval(0.5, 1) == (0, 1)
    editor = CorrectionEditor(proposal, draft)
    editor.toggle("1")
    assert navigate_candidate(candidates, draft, "unsupported", 1, "0", 0) == "2"
    assert (
        navigate_candidate(candidates, draft, "unsupported", 1, None, 0, "alternative") is None
    )  # covered within 50ms
    assert editor.undo()
    assert navigate_candidate(candidates, editor.draft, "unsupported", 1, None, 0, "alternative") == "alt"


def test_legacy_migration_is_explicit_and_keeps_original_bytes(clean: Clean, tmp_path: Path) -> None:
    proposal = copy.deepcopy(clean[3])
    candidate = proposal["kick_candidates"][0]
    path = tmp_path / "draft.json"
    original = json.dumps(
        {
            "track_identifier": proposal["track_identifier"],
            "audio_sha256": proposal["audio_sha256"],
            "kick_onsets": [candidate["final_timestamp"]],
        }
    )
    path.write_text(original)
    with pytest.raises(ValueError, match="Legacy draft retained"):
        load_review(path, proposal)
    migrated = migrate_legacy(path, proposal)
    assert effective_onsets(proposal, migrated)
    assert path.with_suffix(".legacy.json").read_text() == original
    assert migrated.overrides[candidate["candidate_id"]].provenance == "legacy migration"


def test_manual_insertion_deduplicates_and_undoes(clean: Clean) -> None:
    proposal = copy.deepcopy(clean[3])
    editor = CorrectionEditor(proposal, new_review(proposal))
    candidate = copy.deepcopy(proposal["kick_candidates"][0])
    candidate["candidate_id"] = "manual-new"
    candidate["final_timestamp"] = 0.100
    candidate["timestamp"] = 0.100
    candidate["provenance"] = "manual"
    assert editor.insert(candidate)
    assert editor.draft.overrides["manual-new"].provenance == "manual insertion"
    assert not editor.insert({**candidate, "candidate_id": "duplicate", "final_timestamp": 0.120})
    assert editor.undo() and not editor.draft.manual_events


def test_finalization_exports_effective_predictions_only_after_review(
    clean: Clean,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scipy.io import wavfile

    from benchmarks.annotation.cli import finalize_annotation
    from benchmarks.fixtures import file_checksum
    from benchmarks.tracks import Track

    fixture, _, _, original = clean
    root = tmp_path
    (root / "audio").mkdir()
    (root / "proposals").mkdir()
    audio = root / "audio" / "four_on_floor.wav"
    samples = fixture.load_audio()
    wavfile.write(audio, fixture.sample_rate, samples)
    proposal = copy.deepcopy(original)
    proposal["audio_sha256"] = file_checksum(audio)
    track = Track(
        "four_on_floor",
        "test",
        "test",
        120,
        "test",
        "test",
        "development",
        audio.name,
        proposal["audio_sha256"],
        fixture.sample_rate,
        len(samples) / fixture.sample_rate,
    )
    editor = CorrectionEditor(proposal, new_review(proposal))
    editor.toggle(proposal["kick_candidates"][0]["candidate_id"])
    save_review(root / "drafts" / "four_on_floor.json", editor.draft, proposal)
    (root / "proposals" / "four_on_floor.json").write_text(json.dumps(proposal))
    manifest = root / "annotations.json"
    manifest.write_text("[]")
    monkeypatch.setattr("builtins.input", lambda _: "no")
    with pytest.raises(ValueError, match="not explicitly confirmed"):
        finalize_annotation(track, root)
    assert manifest.read_text() == "[]"
    monkeypatch.setattr("builtins.input", lambda _: "REVIEWED four_on_floor ENTIRE RECORDING")
    finalize_annotation(track, root)
    assert json.loads(manifest.read_text())[0]["confirmed_kick_onsets"] == effective_onsets(proposal, editor.draft)
