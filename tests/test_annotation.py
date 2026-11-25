"""Numerical timeline, grid association, review integrity, and partition boundaries."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from benchmarks.annotation.analysis import (
    Candidate,
    analyze_audio,
    associate_candidates,
    circular_phase_difference,
    estimate_phase,
    validate_grid,
)
from benchmarks.annotation.cli import finalize_annotation
from benchmarks.fixtures import AudioFixture, file_checksum, verify_checksum
from benchmarks.kick_detection import grouped_aggregates
from benchmarks.metrics import evaluate
from benchmarks.prepare_tracks import matches_track, prepare
from benchmarks.tracks import Track, reviewed_fixtures, validate_timestamps


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


@pytest.mark.parametrize("timestamps", [[float("nan")], [float("inf")], [-0.01], [2.0], [1.0, 1.0], [1.0, 0.5]])
def test_annotation_timestamp_integrity(timestamps: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_timestamps(timestamps, 2.0)


def catalog_track(
    *, sha256: str | None = None, sample_rate: int | None = None, duration_seconds: float | None = None
) -> Track:
    return Track(
        identifier="test",
        artist="HardTraX",
        title="No Trust",
        bpm=150,
        category="distorted",
        source="https://example.invalid/source",
        partition="development",
        filename="test.wav",
        sha256=sha256,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
    )


def test_recording_identification_and_checksum_integrity(tmp_path: Path) -> None:
    track = catalog_track()
    assert matches_track(Path("hardtrax - A Grim Premonition - 02 No Trust.WAV"), track)
    assert not matches_track(Path("HardTraX - A Grim Premonition - 01 A Grim Premonition.wav"), track)
    assert not matches_track(Path("Other artist - No Trust.wav"), track)
    path = tmp_path / "track.wav"
    path.write_bytes(b"downloaded original")
    checksum = file_checksum(path)
    assert verify_checksum(path, checksum) == checksum
    path.write_bytes(b"modified recording")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checksum(path, checksum)


def test_preparation_preserves_complete_files_and_rejects_overwrites(tmp_path: Path) -> None:
    source, root = tmp_path / "source", tmp_path / "catalog"
    source.mkdir()
    root.mkdir()
    track = catalog_track()
    (root / "tracks.json").write_text(json.dumps([asdict(track)]))
    original = source / "HardTraX - Album - 02 No Trust.wav"
    wavfile.write(original, 24_000, np.zeros((24_000, 2), dtype=np.int16))
    prepare(source, root=root)
    copied = root / "audio" / "test.wav"
    assert original.read_bytes() == copied.read_bytes()
    prepare(source, root=root)
    prepare(None, verify=True, root=root)
    copied.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="different checksum"):
        prepare(source, root=root)


def test_group_aggregates_pool_counts_and_keep_category_dataset_boundaries() -> None:
    audio = np.zeros((48_000, 2), dtype=np.float32)
    fixtures = [
        AudioFixture("one", "shared", audio, 48_000, (0.5,)),
        AudioFixture("two", "shared", audio, 48_000, (0.5,), dataset="techno", partition="development"),
        AudioFixture("three", "shared", audio, 48_000, (), dataset="techno", partition="development"),
    ]
    evaluations = [
        {"50": evaluate([0.5], [0.51], 1, 0.05)},
        {"50": evaluate([0.5], [], 1, 0.05)},
        {"50": evaluate([], [0.5, 0.6], 1, 0.05)},
    ]
    datasets, categories, partitions = grouped_aggregates(fixtures, evaluations)
    assert datasets["techno"]["fixture_count"] == 2
    assert datasets["techno"]["total_duration_seconds"] == 2
    assert datasets["techno"]["ground_truth_kick_count"] == 1
    assert datasets["techno"]["metrics_by_tolerance_ms"]["50"]["false_positives"] == 2
    assert datasets["techno"]["metrics_by_tolerance_ms"]["50"]["false_negatives"] == 1
    assert categories["synthetic"]["shared"]["fixture_count"] == 1
    assert categories["techno"]["shared"]["fixture_count"] == 2
    assert partitions["development"] == datasets["techno"]


def test_finalization_requires_explicit_review_and_verified_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    (root / "audio").mkdir()
    (root / "drafts").mkdir()
    (root / "annotations.json").write_text("[]")
    path = root / "audio" / "test.wav"
    wavfile.write(path, 24_000, np.zeros((48_000, 2), dtype=np.int16))
    track = catalog_track(sha256=file_checksum(path), sample_rate=24_000, duration_seconds=2.0)
    draft = dict(
        track_identifier="test",
        audio_sha256=track.sha256,
        annotation_algorithm_version=1,
        confirmed_bpm=150,
        kick_onsets=[0.5, 1.0],
        manually_confirmed_kick_onsets=[],
        review_status="draft",
    )
    (root / "drafts" / "test.json").write_text(json.dumps(draft))
    monkeypatch.setattr("builtins.input", lambda _: "no")
    with pytest.raises(ValueError, match="not explicitly confirmed"):
        finalize_annotation(track, root)
    assert json.loads((root / "annotations.json").read_text()) == []
    monkeypatch.setattr("builtins.input", lambda _: "REVIEWED test ENTIRE RECORDING")
    finalize_annotation(track, root)
    fixtures, missing = reviewed_fixtures([track], root)
    assert not missing
    assert fixtures[0].kick_onsets == (0.5, 1.0)
    assert fixtures[0].dataset == "techno"
    assert fixtures[0].partition == "development"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fixtures[0].load_audio()
