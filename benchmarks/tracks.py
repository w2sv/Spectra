"""Fixed recording catalog, file integrity, and explicitly reviewed ground truth."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from benchmarks.fixtures import AudioFixture, Partition, load_wav, verify_checksum
from spectra.audio import AudioSamples

TECHNO = Path(__file__).resolve().parent / "datasets" / "techno"


@dataclass(frozen=True)
class Track:
    identifier: str
    artist: str
    title: str
    bpm: float | None
    category: str
    source: str
    partition: Partition
    filename: str
    sha256: str | None
    sample_rate: int | None
    duration_seconds: float | None


def read_catalog(root: Path = TECHNO) -> list[Track]:
    tracks = [Track(**record) for record in json.loads((root / "tracks.json").read_text(encoding="utf-8"))]
    if len({track.identifier for track in tracks}) != len(tracks):
        raise ValueError("Catalog identifiers must be unique.")
    for track in tracks:
        if track.filename != f"{track.identifier}.wav" or Path(track.filename).name != track.filename:
            raise ValueError("Catalog audio filenames must be <identifier>.wav without directories.")
        if track.partition not in ("development", "evaluation"):
            raise ValueError("Techno tracks require development or evaluation partitions.")
    return tracks


def validate_timestamps(timestamps: list[float], duration: float) -> None:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Recording duration must be finite and positive.")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < duration for value in timestamps
    ):
        raise ValueError("Annotation timestamps must be finite and within the original recording duration.")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("Annotation timestamps must be unique and sorted.")


def verified_audio(track: Track, root: Path = TECHNO) -> tuple[int, AudioSamples]:
    """Verify file identity before loading or using acquisition metadata."""
    path = root / "audio" / track.filename
    verify_checksum(path, track.sha256)
    rate, samples = load_wav(path)
    if rate != track.sample_rate or len(samples) / rate != track.duration_seconds:
        raise ValueError(f"{track.identifier}: audio metadata disagrees with catalog.")
    return rate, samples


def reviewed_fixtures(tracks: list[Track], root: Path = TECHNO) -> tuple[list[AudioFixture], list[str]]:
    records = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    annotations = {record["track_identifier"]: record for record in records}
    if len(annotations) != len(records):
        raise ValueError("Reviewed annotation track identifiers must be unique.")
    fixtures, missing = [], []
    for track in tracks:
        path = root / "audio" / track.filename
        record = annotations.get(track.identifier)
        problems = []
        if not path.is_file():
            problems.append(f"missing audio; source: {track.source}")
        if record is None or record.get("review_status") != "reviewed":
            problems.append("missing fully reviewed annotations")
        if problems or record is None:
            missing.append(f"{track.identifier}: {'; '.join(problems)}")
            continue
        verify_checksum(path, track.sha256)
        if record["audio_sha256"] != track.sha256:
            raise ValueError(f"{track.identifier}: reviewed annotation checksum disagrees with catalog.")
        if track.sample_rate is None or track.duration_seconds is None:
            raise ValueError(f"{track.identifier}: incomplete acquisition metadata.")
        validate_timestamps(record["confirmed_kick_onsets"], track.duration_seconds)
        bpm = record["confirmed_bpm"]
        if type(bpm) not in (int, float) or not math.isfinite(bpm) or bpm <= 0:
            raise ValueError(f"{track.identifier}: confirmed BPM must be finite and positive.")
        fixtures.append(
            AudioFixture(
                track.identifier,
                track.category,
                path,
                track.sample_rate,
                tuple(record["confirmed_kick_onsets"]),
                {"annotation_algorithm_version": record["annotation_algorithm_version"], "confirmed_bpm": bpm},
                dataset="techno",
                partition=track.partition,
                audio_sha256=track.sha256,
            )
        )
    return fixtures, missing
