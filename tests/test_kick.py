"""Deterministic raw-spectrum bass transients, without capture or Qt."""

import numpy as np

from spectra.analysis import KickDetectionConfig, KickOnsetDetector, SpectrumAnalyzer, SpectrumResult


def spectrum(timestamp: float, bass: float = 0.0, high: float = 0.0) -> SpectrumResult:
    power = np.zeros(24, dtype=np.float64)
    power[0] = bass
    power[-1] = high
    return SpectrumResult(timestamp, power)


def test_silence_noise_and_high_frequency_transients_are_ignored() -> None:
    detector = KickOnsetDetector(np.geomspace(40, 16_000, 25))
    for index in range(100):
        result = spectrum(index * 0.01, bass=(index % 2) * 1e-8, high=float(index % 2))
        assert detector.process(result) is None


def test_sustained_bass_and_small_fluctuations_do_not_retrigger() -> None:
    detector = KickOnsetDetector(np.geomspace(40, 16_000, 25))
    assert detector.process(spectrum(0.0)) is None
    assert detector.process(spectrum(0.01, bass=1e-3)) is not None
    for index in range(2, 100):
        bass = 1e-3 * (1 + 0.01 * np.sin(index))
        assert detector.process(spectrum(index * 0.01, bass=bass)) is None


def test_sudden_bass_increase_produces_an_onset() -> None:
    detector = KickOnsetDetector(np.geomspace(40, 16_000, 25))
    assert detector.process(spectrum(0.0)) is None

    event = detector.process(spectrum(0.1, bass=1e-3))

    assert event is not None
    assert event.timestamp == 0.1
    assert event.strength > 1


def test_refractory_period_suppresses_duplicate_transients() -> None:
    detector = KickOnsetDetector(np.geomspace(40, 16_000, 25), KickDetectionConfig(refractory_seconds=0.2))
    detector.process(spectrum(0.0))
    assert detector.process(spectrum(0.01, bass=1e-3)) is not None
    assert detector.process(spectrum(0.03)) is None
    assert detector.process(spectrum(0.1, bass=1e-2)) is None
    assert detector.process(spectrum(0.12)) is None
    # A suppressed transient must not extend the refractory period.
    assert detector.process(spectrum(0.25, bass=1e-3)) is not None


def test_distinct_quieter_transients_and_new_stream_reset() -> None:
    detector = KickOnsetDetector(np.geomspace(40, 16_000, 25))
    detector.process(spectrum(0.0))
    first = detector.process(spectrum(0.01, bass=1e-3))
    detector.process(spectrum(0.03))
    second = detector.process(spectrum(0.5, bass=1e-4))

    assert first is not None and second is not None
    assert second.timestamp == 0.5
    detector.reset()
    assert detector.process(spectrum(0.0)) is None
    assert detector.process(spectrum(0.01, bass=1e-3)) == first


def test_falling_bass_with_energy_moving_between_bands_is_not_an_onset() -> None:
    detector = KickOnsetDetector(np.geomspace(40, 16_000, 25))
    detector.process(spectrum(0.0, bass=1e-3))
    power = np.zeros(24, dtype=np.float64)
    power[0] = 4e-4
    power[1] = 1e-4

    assert detector.process(SpectrumResult(0.01, power)) is None

    # Rejecting a redistribution must still allow a subsequent bass attack.
    assert detector.process(spectrum(0.8, bass=1e-3)) is not None


def test_pcm_tone_has_an_attack_but_no_release_onset() -> None:
    sample_rate = 48_000
    mono = np.zeros(2 * sample_rate, dtype=np.float32)
    timeline = np.arange(sample_rate) / sample_rate
    mono[sample_rate // 2 : 3 * sample_rate // 2] = 0.15 * np.sin(2 * np.pi * 65 * timeline)
    analyzer = SpectrumAnalyzer(sample_rate=sample_rate)
    detector = KickOnsetDetector(analyzer.band_edges)
    events = []
    for result in analyzer.process(np.column_stack((mono, mono))):
        event = detector.process(result)
        if event is not None:
            events.append(event)

    assert len(events) == 1
    # A window-center timestamp may precede the tone's first nonzero sample.
    assert abs(events[0].timestamp - 0.5) <= analyzer.fft_size / (2 * sample_rate)
