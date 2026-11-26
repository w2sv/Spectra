"""Acoustic evidence, recording-time coordinates and beat-grid associations."""

from dataclasses import replace

import numpy as np
import pytest

from benchmarks.annotation.analysis import (
    Candidate,
    analyze_audio,
    associate_candidates,
    circular_phase_difference,
    estimate_phase,
    validate_grid,
)


def test_centered_spectral_timestamps_and_pre_onset_evidence() -> None:
    samples = np.zeros((48_000, 2), dtype=np.float32)
    samples[24_000] = 1
    analysis = analyze_audio(samples, 48_000)
    assert analysis.timestamps[0] == 0
    assert np.diff(analysis.timestamps) == pytest.approx(np.full(199, 0.005))
    # Centered Hann-window power peaks at the impulse's actual timestamp.
    assert analysis.timestamps[np.argmax(analysis.band_power.sum(axis=1))] == 0.5
    # Positive log differences anticipate a physical onset because centered
    # windows already include future samples; this is evidence, not latency.
    peak_time = analysis.timestamps[np.argmax(analysis.bass)]
    assert 0.5 - 2048 / (2 * 48_000) <= peak_time < 0.5
    silence = analyze_audio(np.zeros_like(samples), 48_000)
    assert not np.any(silence.band_novelty)
    assert not np.any(silence.bass)
    converted = analyze_audio(samples[::2], 24_000)
    assert converted.timestamps[np.argmax(converted.band_power.sum(axis=1))] == 0.5


def test_phase_estimation_circular_difference_and_local_grid_drift() -> None:
    times = np.arange(12_000, dtype=np.float64) * 0.005
    evidence = np.zeros(len(times), dtype=np.float64)
    # Alternating +/-20ms evidence resolves the scoring plateau uniquely.
    for i, beat in enumerate(np.arange(0.125, 60, 0.5)):
        phase_shift = 0 if beat < 30 else 0.08
        event = beat + phase_shift + (-0.02 if i % 2 else 0.02)
        evidence[round(event / 0.005)] = 1
    phase = estimate_phase(times, evidence, 120, 0, 30)
    assert phase == pytest.approx(0.125)
    assert circular_phase_difference(0.01, 0.49, 0.5) == pytest.approx(0.02)
    assert circular_phase_difference(0.49, 0.01, 0.5) == pytest.approx(-0.02)
    empty = analyze_audio(np.zeros((4800, 2), dtype=np.float32), 48_000)
    sections = validate_grid(replace(empty, timestamps=times, bass=evidence, duration_seconds=60), 120, phase)
    assert [section["drift_flagged"] for section in sections] == [False, True]
    assert sections[1]["phase_difference_seconds"] == pytest.approx(0.08)


def test_peak_association_keeps_acoustic_time_and_all_off_grid_peaks() -> None:
    def candidate(timestamp: float, evidence: float) -> Candidate:
        return {
            "timestamp": timestamp,
            "bass_evidence": evidence,
            "prominence": 0.5,
            "broadband_evidence": 0.1,
            "nearest_beat_distance_seconds": None,
            "grid_aligned": False,
        }

    peaks = [candidate(0.98, 0.5), candidate(1.03, 0.9), candidate(1.25, 0.8)]
    result, unsupported = associate_candidates(peaks, np.array([1, 1.06, 1.5], dtype=np.float64))
    assert len(result) == 3
    assert [peak["timestamp"] for peak in result] == [0.98, 1.03, 1.25]
    assert [peak["grid_aligned"] for peak in result] == [False, True, False]
    assert unsupported == [1.06, 1.5]
    assert result[1]["nearest_beat_distance_seconds"] == pytest.approx(0.03)
    assert not any(peak["grid_aligned"] for peak in peaks)
