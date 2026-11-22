"""Only the framebuffer's deterministic data and silence contracts."""

import numpy as np

from spectra.visualization.frequency import frequency_frame


def test_frame_covers_every_led_in_frequency_order() -> None:
    led_count = 300
    next_led = 0
    # Illuminate each band independently to check its region without choosing a palette.
    for band_energies in np.eye(24, dtype=np.float64):
        frame = frequency_frame(band_energies, led_count=led_count)

        assert frame.shape == (led_count, 3)
        assert frame.dtype == np.uint8
        assigned = np.flatnonzero(np.any(frame, axis=1))
        assert assigned.size > 0
        np.testing.assert_array_equal(assigned, np.arange(next_led, next_led + assigned.size))
        next_led += assigned.size
    assert next_led == led_count


def test_silence_is_unlit() -> None:
    frame = frequency_frame(np.zeros(24, dtype=np.float64))

    np.testing.assert_array_equal(frame, np.zeros((300, 3), dtype=np.uint8))
