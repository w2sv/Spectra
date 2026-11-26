"""Causal offline benchmarking, reproducible fixtures and annotated WAV input."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from benchmarks.fixtures import load_manifest, synthetic_fixture
from benchmarks.kick_detection import detect_offline, resample_causally
from spectra.analysis import KickDetectionConfig, KickOnsetDetector, SpectrumAnalyzer
from spectra.audio import DEFAULT_READ_FRAMES


def test_audio_timestamps_use_live_read_availability_and_preserve_detector_history() -> None:
    fixture = synthetic_fixture("test", "test", (0.513, 1.23, 1.41, 2.67))
    samples = fixture.load_audio()
    run = detect_offline(samples, fixture.sample_rate)
    analyzer = SpectrumAnalyzer()
    detector = KickOnsetDetector(analyzer.band_edges)
    # Feeding one whole chunk is independent of live read scheduling, and
    # checks that availability reporting never changes detector audio time.
    expected = [event for spectrum in analyzer.process(samples) if (event := detector.process(spectrum)) is not None]
    assert len(run.detections) == len(expected) > 0
    for actual, event in zip(run.detections, expected, strict=True):
        assert actual.spectrum_center_timestamp == event.timestamp
        assert actual.strength == event.strength
        window_end = round(event.timestamp * analyzer.sample_rate + analyzer.fft_size / 2)
        read_end = min(
            ((window_end + DEFAULT_READ_FRAMES - 1) // DEFAULT_READ_FRAMES) * DEFAULT_READ_FRAMES, len(samples)
        )
        assert actual.timestamp == read_end / analyzer.sample_rate
        assert actual.timestamp >= window_end / analyzer.sample_rate
    # No artificial tail padding may create an early or incomplete window.
    assert not detect_offline(samples[: analyzer.fft_size - 1], fixture.sample_rate).detections


@pytest.mark.parametrize("source_rate", [24_000, 44_100, 96_000])
def test_resampling_is_causal_and_retains_filter_delay(source_rate: int) -> None:
    samples = np.random.default_rng(7).normal(0, 0.1, (source_rate, 2)).astype(np.float32)
    converted, delay = resample_causally(samples, source_rate, 48_000)
    prefix, prefix_delay = resample_causally(samples[: source_rate // 2], source_rate, 48_000)
    np.testing.assert_array_equal(prefix, converted[: len(prefix)])
    assert prefix_delay == delay > 0
    # A delayed impulse exposes lookahead or accidental delay compensation.
    impulse = np.zeros_like(samples)
    impulse[source_rate // 2] = 1.0
    response, _ = resample_causally(impulse, source_rate, 48_000)
    assert not np.any(response[:24_000])
    peak = int(np.argmax(response[:, 0])) / 48_000
    assert peak == pytest.approx(0.5 + delay, abs=1 / 48_000)


def test_fixture_generation_and_offline_detection_are_reproducible() -> None:
    first = synthetic_fixture("repeat", "rumble", (0.5, 1.0, 2.0), background="rumble")
    second = synthetic_fixture("repeat", "rumble", (0.5, 1.0, 2.0), background="rumble")
    np.testing.assert_array_equal(first.load_audio(), second.load_audio())
    assert first.kick_onsets == tuple(position / first.sample_rate for position in (24_000, 48_000, 96_000))
    config = KickDetectionConfig()
    run = detect_offline(first.load_audio(), first.sample_rate, config)
    repeated = detect_offline(second.load_audio(), second.sample_rate, config)
    assert run.detections == repeated.detections
    assert run.detections
    assert run.resampling_delay_seconds == repeated.resampling_delay_seconds == 0.0


def test_external_annotated_wav_uses_the_same_pipeline(tmp_path: Path) -> None:
    rate = 24_000
    samples = np.zeros(rate, dtype=np.int16)
    time = np.arange(rate // 4) / rate
    samples[rate // 2 : 3 * rate // 4] = (12_000 * np.sin(2 * np.pi * 65 * time) * np.exp(-time / 0.05)).astype(
        np.int16
    )
    wavfile.write(tmp_path / "kick.wav", rate, samples)
    manifest = tmp_path / "annotations.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "identifier": "external",
                    "category": "wav",
                    "audio": "kick.wav",
                    "sample_rate": rate,
                    "kick_onsets": [0.5],
                }
            ]
        ),
        encoding="utf-8",
    )
    fixture = load_manifest(manifest)[0]
    audio = fixture.load_audio()
    np.testing.assert_array_equal(audio[:, 0], samples.astype(np.float32) / 32768)
    np.testing.assert_array_equal(audio[:, 0], audio[:, 1])
    run = detect_offline(audio, fixture.sample_rate)
    assert run.detections
    assert run.detections[0].timestamp >= fixture.kick_onsets[0]
