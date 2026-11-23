"""Audio-timed traveling kick pulses, independent of capture and presentation."""

import numpy as np
from numpy.typing import NDArray

from spectra.analysis import KickEvent


class TravelingPulses:
    """Render independent pulses traveling from LED zero to the strip's end.

    duration_seconds includes the tail leaving the strip. width_leds is the
    distance behind the bright head over which brightness falls linearly to
    zero. Color uses RGB components in [0, 255]; brightness scales them by
    [0, 1]. Render with increasing audio-relative time, resetting for a new
    stream. No wall clock, audio backend, or presentation framework is used.
    """

    def __init__(
        self,
        led_count: int = 300,
        *,
        duration_seconds: float = 0.75,
        width_leds: float = 60.0,
        color: tuple[int, int, int] = (255, 255, 255),
        brightness: float = 1.0,
    ) -> None:
        if led_count <= 0:
            raise ValueError("LED count must be positive.")
        if not np.isfinite(duration_seconds) or duration_seconds <= 0 or not np.isfinite(width_leds) or width_leds <= 0:
            raise ValueError("Pulse duration and width must be finite and positive.")
        if len(color) != 3 or any(not 0 <= component <= 255 for component in color):
            raise ValueError("Pulse color must contain three RGB components between 0 and 255.")
        if not 0 <= brightness <= 1:
            raise ValueError("Pulse brightness must be between 0 and 1.")

        self._led_count = led_count
        self._duration_seconds = duration_seconds
        self._width_leds = width_leds
        self._color = np.asarray(color, dtype=np.float64) * brightness
        self._positions = np.arange(led_count, dtype=np.float64)
        self._onsets: list[float] = []

    def reset(self) -> None:
        """Discard active and pending pulses when beginning or clearing a stream."""
        self._onsets.clear()

    def render(
        self,
        intensities: NDArray[np.float64],
        kicks: tuple[KickEvent, ...],
        timestamp: float,
    ) -> NDArray[np.uint8]:
        """Return a (led_count, 3) uint8 frame; continuous intensities are unused.

        New kicks retain their original timestamps, so late arrivals begin
        partway through their journey. Per-channel maximum avoids overflow
        when pulses overlap. Unlit regions stay black. Expired pulses are
        discarded; future onsets remain pending until their audio time.
        """
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Render timestamps must be finite and nonnegative.")
        for kick in kicks:
            if not np.isfinite(kick.timestamp) or kick.timestamp < 0:
                raise ValueError("Onset timestamps must be finite and nonnegative.")
            self._onsets.append(kick.timestamp)

        self._onsets = [onset for onset in self._onsets if timestamp - onset < self._duration_seconds]
        composed = np.zeros((self._led_count, 3), dtype=np.uint8)
        for onset in self._onsets:
            age = timestamp - onset
            if age < 0:
                continue
            head = age / self._duration_seconds * (self._led_count - 1 + self._width_leds)
            distance = head - self._positions
            intensity = np.where(distance >= 0, np.clip(1 - distance / self._width_leds, 0, 1), 0)
            pulse = np.rint(intensity[:, None] * self._color).astype(np.uint8)
            np.maximum(composed, pulse, out=composed)
        return composed
