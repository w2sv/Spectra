"""Fixed-reference intensity bounds and deterministic audio-time envelopes."""

from dataclasses import replace

import numpy as np

from spectra.visualization.intensity import BandIntensityProcessor, BandIntensitySettings


def test_bounds_silence_reset_and_snapshot_ownership() -> None:
    processor = BandIntensityProcessor(4)
    first = processor.process(np.array([0, 1e-20, 1e-3, 1e6], dtype=np.float64), 0.0)
    saved = first.copy()
    assert np.all(np.isfinite(first))
    assert np.all((first >= 0) & (first <= 1))
    assert first[0] == first[1] == 0
    assert first[-1] == 1
    silence = processor.process(np.zeros(4), 4.0)
    np.testing.assert_array_equal(silence, np.zeros(4))
    np.testing.assert_array_equal(first, saved)
    assert not first.flags.writeable
    processor.reset()
    np.testing.assert_array_equal(processor.process(np.zeros(4), 0.0), np.zeros(4))


def test_separate_attack_release_and_settings_continuity() -> None:
    settings = BandIntensitySettings(frequency_balance_db=0, attack_seconds=0.1, release_seconds=0.4)
    processor = BandIntensityProcessor(2, settings)
    np.testing.assert_array_equal(processor.process(np.array([0.0, 1.0]), 0.0), [0, 1])

    changed = processor.process(np.array([1.0, 0.0]), 0.1)
    np.testing.assert_allclose(
        changed, [1 - np.exp(-0.1 / settings.attack_seconds), np.exp(-0.1 / settings.release_seconds)]
    )

    processor.update_settings(replace(settings, sensitivity_db=-12, release_seconds=0.2))
    # Changing settings does not replace accumulated state or advance audio time.
    np.testing.assert_allclose(processor.process(np.zeros(2), 0.1), changed)
    released = processor.process(np.zeros(2), 0.2)
    assert np.all((released > 0) & (released < changed))


def test_smoothing_is_consistent_across_update_intervals() -> None:
    settings = BandIntensitySettings(frequency_balance_db=0, attack_seconds=0.1, release_seconds=0.4)
    final = []
    for interval in (0.005, 0.02):
        processor = BandIntensityProcessor(2, settings)
        result = processor.process(np.zeros(2), 0.0)
        elapsed = 0.0
        for energies, duration in ((np.array([1.0, 1e-3]), 0.2), (np.zeros(2), 0.4)):
            for index in range(1, round(duration / interval) + 1):
                result = processor.process(energies, elapsed + index * interval)
            elapsed += duration
        final.append(result)
    np.testing.assert_allclose(final[0], final[1], rtol=1e-12, atol=1e-12)
