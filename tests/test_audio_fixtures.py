"""Integrity of the small, version-controlled PCM reference collection."""

import hashlib
import json
import wave
from pathlib import Path


def test_curated_audio_matches_manifest() -> None:
    directory = Path(__file__).parent / "fixtures" / "audio"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert {entry["filename"] for entry in manifest["fixtures"]} == {path.name for path in directory.glob("*.wav")}
    for entry in manifest["fixtures"]:
        path = directory / entry["filename"]
        with path.open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == entry["sha256"]
        assert path.stat().st_size == entry["size_bytes"]
        with wave.open(str(path), "rb") as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (48_000, 2, 2)
            assert audio.getnframes() == entry["frame_count"]
            assert audio.getnframes() / audio.getframerate() == entry["duration_seconds"]
