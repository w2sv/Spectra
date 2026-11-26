"""Catalog identity, timestamp integrity, track preparation and dataset reporting."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from benchmarks.fixtures import file_checksum, verify_checksum
from benchmarks.kick_detection import main as benchmark_main
from benchmarks.prepare_tracks import matches_track, prepare
from benchmarks.tracks import Track, collection_status, reviewed_fixtures, validate_timestamps


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
