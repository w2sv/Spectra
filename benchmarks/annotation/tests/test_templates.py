"""Signature features, supplied-reference matching and annotation diagnostics."""

from dataclasses import replace
from typing import Literal

import numpy as np
import pytest

from benchmarks.annotation.analysis import Candidate, Proposal, analyze_audio
from benchmarks.annotation.templates import alternative_phase, classification, compare_signatures, event_feature
from benchmarks.fixtures import synthetic_fixture


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
    positive = [(0.5, event_feature(analysis, 0.5))]
    refined = compare_signatures(analysis, [candidate(1.985, aligned=False), candidate(3.455)], positive, [])
    assert refined[0]["timestamp"] == 1.985
    assert refined[0].get("refined_timestamp") == pytest.approx(2.0, abs=0.0025)
    assert refined[0].get("positive_similarity") == pytest.approx(1.0, abs=1e-6)
    assert refined[0].get("classification") == "likely"
    assert not refined[0]["grid_aligned"]
    assert not refined[0].get("alignment_at_search_boundary")
    assert refined[1].get("alignment_at_search_boundary")
    zeros = replace(analysis, template_band_power=np.zeros_like(analysis.template_band_power))
    invalid = compare_signatures(zeros, [candidate(0.985)], [(0.5, event_feature(zeros, 0.5))], [])
    assert invalid[0].get("positive_similarity") is None
    assert invalid[0].get("classification") != "likely"


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
