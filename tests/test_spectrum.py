"""Deterministic spectral measurements and stream continuity."""

from itertools import cycle

import numpy as np
import pytest

from spectra.analysis import SpectrumAnalyzer, SpectrumResult


@pytest.mark.parametrize("frequency", [100.0, 1000.0, 5000.0])
def test_sine_frequency_and_mean_square_power(frequency: float) -> None:
    analyzer = SpectrumAnalyzer()
    time = np.arange(analyzer.fft_size) / analyzer.sample_rate
    sine = np.sin(2 * np.pi * frequency * time)
    # Distinct channel amplitudes also verify the arithmetic stereo downmix.
    samples = np.column_stack((0.8 * sine, 0.4 * sine)).astype(np.float32)

    result = analyzer.process(samples)[0]

    band = int(np.searchsorted(analyzer.band_edges, frequency) - 1)
    assert int(np.argmax(result.band_energies)) == band
    nearby_power = result.band_energies[max(0, band - 1) : band + 2].sum()
    assert nearby_power > 0.99 * result.band_energies.sum()
    assert result.band_energies.sum() == pytest.approx(0.6**2 / 2, rel=0.02)


def test_chunk_boundaries_hops_reset_and_result_ownership() -> None:
    analyzer = SpectrumAnalyzer()
    frame_count = analyzer.fft_size + 23 * analyzer.hop_size + 137
    time = np.arange(frame_count) / analyzer.sample_rate
    signal = (0.2 + time) * np.sin(2 * np.pi * 1000 * time) + 0.1 * np.sin(2 * np.pi * 100 * time)
    samples = np.column_stack((signal, 0.5 * signal)).astype(np.float32)
    reference = analyzer.process(samples)
    expected_count = 1 + (frame_count - analyzer.fft_size) // analyzer.hop_size
    assert len(reference) == expected_count
    expected_times = (analyzer.fft_size / 2 + np.arange(expected_count) * analyzer.hop_size) / analyzer.sample_rate
    np.testing.assert_array_equal([result.timestamp for result in reference], expected_times)
    saved_energies = reference[0].band_energies.copy()

    analyzer.reset()
    streamed: list[SpectrumResult] = []
    offset = 0
    for size in cycle((4095, 1, 1024, 17, 513, 5000)):
        chunk = samples[offset : offset + size]
        offset += len(chunk)
        streamed.extend(analyzer.process(chunk))
        count = max(0, 1 + (offset - analyzer.fft_size) // analyzer.hop_size)
        assert len(streamed) == count
        if offset == frame_count:
            break

    for expected, actual in zip(reference, streamed, strict=True):
        assert actual.timestamp == expected.timestamp
        np.testing.assert_array_equal(actual.band_energies, expected.band_energies)
    np.testing.assert_array_equal(reference[0].band_energies, saved_energies)
    assert not reference[0].band_energies.flags.writeable
    assert not analyzer.band_edges.flags.writeable


def test_silence_is_finite_zero_power() -> None:
    analyzer = SpectrumAnalyzer()
    assert analyzer.process(np.empty((0, 2), dtype=np.float32)) == []
    results = analyzer.process(np.zeros((analyzer.fft_size + analyzer.hop_size, 2), dtype=np.float32))

    assert len(results) == 2
    assert analyzer.band_edges.shape == (25,)
    for result in results:
        assert result.band_energies.dtype == np.float64
        assert result.band_energies.shape == (24,)
        assert np.all(np.isfinite(result.band_energies))
        np.testing.assert_array_equal(result.band_energies, np.zeros(24))


def test_dc_and_nyquist_power_are_not_doubled() -> None:
    # Include virtually the full frequency range, retaining logarithmic bands.
    analyzer = SpectrumAnalyzer(min_frequency=1e-9, max_frequency=24_000)
    signal = 0.25 + 0.5 * (-1.0) ** np.arange(analyzer.fft_size)
    samples = np.column_stack((signal, signal)).astype(np.float32)

    result = analyzer.process(samples)[0]

    assert result.band_energies.sum() == pytest.approx(0.25**2 + 0.5**2, rel=1e-8)
