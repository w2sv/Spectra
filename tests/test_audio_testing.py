"""Offline export contracts using the development helper's synthetic audio."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


def test_analyze_audio_excerpt(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "audio_test.py"
    audio = tmp_path / "smoke.wav"
    subprocess.run([sys.executable, str(script), "synth", str(audio)], check=True, capture_output=True)
    command = [sys.executable, str(script), "analyze", str(audio), "--start", "0.25", "--duration", "3.5"]
    reports = []
    for name in ("first", "repeated"):
        output = tmp_path / f"{name}.json"
        subprocess.run([*command, "--output", str(output)], check=True, capture_output=True)
        reports.append(json.loads(output.read_text(encoding="utf-8")))
    first, repeated = reports

    assert first == repeated
    assert set(first) == {
        "input",
        "sha256",
        "sample_rate",
        "start_frame",
        "frame_count",
        "timestamp_origin",
        "detector_overrides",
        "fft_size",
        "hop_size",
        "analysis_source_sha256",
        "detections",
    }
    assert first["input"] == str(audio.resolve())
    assert first["sha256"] == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert first["sample_rate"] == 48_000
    assert first["start_frame"] == 12_000
    assert first["frame_count"] == 168_000
    assert first["detector_overrides"] == {}
    assert set(first["analysis_source_sha256"]) == {"kick.py", "spectrum.py"}

    # Check timing conventions without fixing the detector's chosen kick locations.
    detections = first["detections"]
    assert detections
    previous = -1.0
    for detection in detections:
        assert set(detection) == {"timestamp", "source_timestamp", "strength"}
        timestamp = detection["timestamp"]
        center_frame = timestamp * first["sample_rate"]
        assert first["fft_size"] / 2 <= center_frame <= first["frame_count"] - first["fft_size"] / 2
        hop = (center_frame - first["fft_size"] / 2) / first["hop_size"]
        assert hop == pytest.approx(round(hop))
        assert timestamp > previous
        assert detection["source_timestamp"] == pytest.approx(timestamp + first["start_frame"] / first["sample_rate"])
        assert math.isfinite(detection["strength"]) and detection["strength"] > 0
        previous = timestamp
