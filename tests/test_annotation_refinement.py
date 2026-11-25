"""Numerical signatures, provenance, diagnostics, and benchmark completeness."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from scipy.io import wavfile

from benchmarks.annotation.analysis import Candidate, Proposal, analyze_audio
from benchmarks.annotation.templates import (
    TemplateDefinition,
    alternative_phase,
    classification,
    event_feature,
    load_templates,
    refine_candidates,
    save_templates,
)
from benchmarks.fixtures import file_checksum, synthetic_fixture
from benchmarks.kick_detection import main as benchmark_main
from benchmarks.tracks import Track, collection_status, reviewed_fixtures


def candidate(
    timestamp: float,
    label: Literal["unclassified", "likely", "ambiguous", "unlikely"] = "unclassified",
    aligned: bool = True,
    refined: float | None = None,
) -> Candidate:
    return {
        "timestamp": timestamp,
        "bass_evidence": 1.0,
        "prominence": 0.5,
        "broadband_evidence": 0.3,
        "nearest_beat_distance_seconds": 0.0,
        "grid_aligned": aligned,
        "classification": label,
        "refined_timestamp": refined,
    }


def proposal(candidates: list[Candidate], positive_count: int = 0) -> Proposal:
    return {
        "track_identifier": "test",
        "audio_sha256": "0" * 64,
        "annotation_algorithm_version": 2,
        "analysis_parameters": {},
        "bpm": 120.0,
        "beat_grid_phase_seconds": 0.0,
        "beat_positions": [0.0, 0.5, 1.0, 1.5],
        "unsupported_beat_positions": [],
        "kick_candidates": candidates,
        "beat_grid_validation": [],
        "duration_seconds": 2.0,
        "source_sample_rate": 48_000,
        "informational": {},
        "positive_template_count": positive_count,
    }


def test_features_preserve_spectral_shape_and_normalize_global_amplitude() -> None:
    fixture = synthetic_fixture("features", "clean", (0.5, 1.0))
    samples = fixture.load_audio()
    analysis = analyze_audio(samples, fixture.sample_rate)
    feature = event_feature(analysis, 0.5)
    quieter = event_feature(analyze_audio(samples * np.float32(0.2), fixture.sample_rate), 0.5)
    assert feature is not None and quieter is not None
    assert feature.shape == (37 * 12,)
    assert np.linalg.norm(feature) == pytest.approx(1.0)
    np.testing.assert_allclose(feature, quieter, atol=1e-7)
    powers = analysis.template_band_power.copy()
    powers[:, 3] *= 9
    changed = event_feature(replace(analysis, template_band_power=powers), 0.5)
    assert changed is not None and not np.allclose(feature, changed)
    empty = analyze_audio(np.zeros((48_000, 2), dtype=np.float32), 48_000)
    assert event_feature(empty, 0.5) is None
    assert event_feature(replace(empty, template_band_power=np.ones_like(empty.template_band_power)), 0.5) is None
    powers = np.ones_like(empty.template_band_power)
    boundary = event_feature(replace(empty, template_band_power=powers), 0.0)
    assert boundary is not None
    np.testing.assert_array_equal(boundary.reshape(37, 12)[:4], np.zeros((4, 12)))


@pytest.mark.parametrize(
    ("positive", "negative", "has_negative", "expected"),
    [
        (0.85, None, False, "likely"),
        (0.85, 0.8, True, "ambiguous"),
        (0.7, None, False, "ambiguous"),
        (0.69, None, False, "unlikely"),
        (1.0, 0.9, True, "likely"),
        (None, None, False, "unlikely"),
        (0.95, None, True, "ambiguous"),
    ],
)
def test_similarity_classification(
    positive: float | None, negative: float | None, has_negative: bool, expected: str
) -> None:
    assert classification(positive, negative, has_negative) == expected


def test_cosine_alignment_uses_template_onset_and_flags_search_boundary() -> None:
    fixture = synthetic_fixture("alignment", "clean", (0.5, 2.0, 3.5))
    analysis = analyze_audio(fixture.load_audio(), fixture.sample_rate)
    templates = TemplateDefinition("alignment", "0" * 64, [0.5])
    refined, _, _ = refine_candidates(analysis, [candidate(1.985, aligned=False), candidate(3.455)], templates)
    assert refined[0]["timestamp"] == 1.985
    assert refined[0].get("refined_timestamp") == pytest.approx(2.0, abs=0.0025)
    assert refined[0].get("positive_similarity") == pytest.approx(1.0, abs=1e-6)
    assert refined[0].get("classification") == "likely"
    assert not refined[0]["grid_aligned"]
    assert not refined[0].get("alignment_at_search_boundary")
    assert refined[1].get("alignment_at_search_boundary")
    zeros = replace(analysis, template_band_power=np.zeros_like(analysis.template_band_power))
    invalid, invalid_refs, _ = refine_candidates(zeros, [candidate(0.985)], templates)
    assert invalid_refs == [0.5]
    assert invalid[0].get("positive_similarity") is None
    assert invalid[0].get("classification") != "likely"


def test_template_persistence_checks_audio_identity_and_timestamps(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"original recording bytes")
    checksum = file_checksum(audio)
    path = tmp_path / "templates.json"
    templates = TemplateDefinition("track", checksum, [0.125, 1.0], [0.75])
    save_templates(path, templates, 2.0)
    assert load_templates(path, "track", checksum, 2.0, audio) == templates
    with pytest.raises(ValueError, match="checksum"):
        load_templates(path, "other", checksum, 2.0)
    with pytest.raises(ValueError, match="unique and sorted"):
        save_templates(path, TemplateDefinition("track", checksum, [1.0, 0.5]), 2.0)
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_templates(path, "track", checksum, 2.0, audio)


def test_alternative_phase_is_half_a_beat_without_changing_events() -> None:
    assert alternative_phase(0.4, 120) == pytest.approx(0.15)
    assert alternative_phase(0.0, 150) == pytest.approx(0.2)


def test_diagnostic_ranking_selects_distinct_intervals_and_uses_likely_support() -> None:
    from benchmarks.annotation.diagnostics import diagnostic_sections

    candidates = [candidate(i * 0.5, "likely") for i in range(4)]
    candidates += [candidate(10 + i * 0.5, "likely") for i in range(18)]
    candidates += [candidate(20.1 + i * 0.2, "ambiguous", aligned=False) for i in range(10)]
    current = proposal(candidates, 1)
    current["duration_seconds"] = 30.0
    current["beat_positions"] = [i * 0.5 for i in range(60)]
    current["beat_grid_validation"] = [
        {
            "start_seconds": 0.0,
            "end_seconds": 30.0,
            "local_phase_seconds": 0.04,
            "phase_difference_seconds": 0.04,
            "drift_flagged": True,
        }
    ]
    selected = diagnostic_sections(current)
    assert selected["regular"] is not None and selected["regular"]["start"] == 10
    assert selected["challenging"] is not None and selected["challenging"]["start"] == 20
    assert selected["phase_discrepancy"] is not None and selected["phase_discrepancy"]["start"] == 0


def test_complete_and_partial_catalog_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path
    (root / "audio").mkdir()
    audio = root / "audio" / "available.wav"
    wavfile.write(audio, 48_000, np.zeros((48_000, 2), dtype=np.int16))
    available = Track(
        "available",
        "Artist",
        "Track",
        120,
        "test",
        "local",
        "development",
        "available.wav",
        file_checksum(audio),
        48_000,
        1.0,
    )
    missing = replace(available, identifier="missing", filename="missing.wav")
    (root / "annotations.json").write_text(
        json.dumps(
            [
                {
                    "track_identifier": "available",
                    "audio_sha256": available.sha256,
                    "confirmed_bpm": 120,
                    "confirmed_kick_onsets": [0.5],
                    "annotation_algorithm_version": 2,
                    "review_status": "reviewed",
                }
            ]
        )
    )
    monkeypatch.setattr("benchmarks.kick_detection.read_catalog", lambda: [available, missing])
    monkeypatch.setattr("benchmarks.kick_detection.reviewed_fixtures", lambda tracks: reviewed_fixtures(tracks, root))
    monkeypatch.setattr(
        "benchmarks.kick_detection.collection_status",
        lambda tracks, fixtures: collection_status(tracks, fixtures, root),
    )
    output = tmp_path / "result.json"
    command = ["benchmark", "--dataset", "techno", "--partition", "development", "--json", str(output)]
    monkeypatch.setattr("sys.argv", command)
    with pytest.raises(SystemExit) as error:
        benchmark_main()
    assert error.value.code == 2
    assert not output.exists()
    monkeypatch.setattr("sys.argv", [*command, "--allow-partial"])
    benchmark_main()
    report = json.loads(output.read_text())
    status = report["collection_completeness"]
    assert report["result_scope"] == "partial"
    assert not status["complete"]
    assert status["requested_track_identifiers"] == ["available", "missing"]
    assert status["evaluated_track_identifiers"] == ["available"]
    assert status["missing_audio"] == status["missing_reviewed_annotations"] == ["missing"]
    monkeypatch.setattr("sys.argv", [*command, "--fixture", "available"])
    benchmark_main()
    assert json.loads(output.read_text())["result_scope"] == "complete"
