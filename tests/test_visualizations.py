"""Audio timing and composition contracts, without fixed visual appearances."""

import numpy as np

from spectra.analysis import KickEvent
from spectra.visualization.combined import CombinedVisualization
from spectra.visualization.frequency import FrequencyVisualization, frequency_frame
from spectra.visualization.pulses import TravelingPulses


def test_pulse_travels_from_its_onset_and_expires() -> None:
    intensities = np.zeros(24, dtype=np.float64)
    kick = KickEvent(2.0, 1.0)
    pulses = TravelingPulses(duration_seconds=1.0)

    initial = pulses.render(intensities, (kick,), 2.0)
    advanced = pulses.render(intensities, (), 2.5)

    assert np.any(initial)
    assert np.argmax(advanced.max(axis=1)) > np.argmax(initial.max(axis=1))
    # Receiving the event later must not restart it at the beginning.
    delayed = TravelingPulses(duration_seconds=1.0)
    np.testing.assert_array_equal(delayed.render(intensities, (kick,), 2.5), advanced)
    np.testing.assert_array_equal(pulses.render(intensities, (), 3.0), np.zeros((300, 3), dtype=np.uint8))


def test_overlapping_pulses_produce_a_valid_frame_and_reset_to_dark() -> None:
    intensities = np.zeros(24, dtype=np.float64)
    first, second = KickEvent(0.0, 1.0), KickEvent(0.05, 1.0)
    pulses = TravelingPulses()
    frame = pulses.render(intensities, (first, second), 0.3)
    expected = np.maximum(
        TravelingPulses().render(intensities, (first,), 0.3),
        TravelingPulses().render(intensities, (second,), 0.3),
    )

    assert frame.shape == (300, 3)
    assert frame.dtype == np.uint8
    np.testing.assert_array_equal(frame, expected)
    pulses.reset()
    assert not np.any(pulses.render(intensities, (), 0.4))


def test_combined_composes_both_visualizations_without_changing_inputs() -> None:
    intensities = np.linspace(0.0, 1.0, 24)
    original = intensities.copy()
    kicks = (KickEvent(0.0, 1.0),)
    combined = CombinedVisualization()
    frequency = FrequencyVisualization().render(intensities, kicks, 0.3)
    pulses = TravelingPulses().render(intensities, kicks, 0.3)

    frame = combined.render(intensities, kicks, 0.3)

    assert frame.shape == (300, 3)
    assert frame.dtype == np.uint8
    np.testing.assert_array_equal(frame, np.maximum(frequency, pulses))
    np.testing.assert_array_equal(intensities, original)
    combined.reset()
    np.testing.assert_array_equal(combined.render(intensities, (), 0.4), frequency)


def test_frequency_mode_and_inactive_combined_preserve_frequency_frame() -> None:
    intensities = np.linspace(0.0, 1.0, 24)
    expected = frequency_frame(intensities)
    frequency = FrequencyVisualization()

    np.testing.assert_array_equal(frequency.render(intensities, (KickEvent(0.0, 1.0),), 0.0), expected)
    frequency.reset()
    np.testing.assert_array_equal(frequency.render(intensities, (), 1.0), expected)
    np.testing.assert_array_equal(CombinedVisualization().render(intensities, (), 1.0), expected)
