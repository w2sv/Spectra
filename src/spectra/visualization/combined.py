"""Frequency colors with traveling pulses, composed without additive clipping."""

import numpy as np
from numpy.typing import NDArray

from spectra.analysis import KickEvent
from spectra.visualization.frequency import FrequencyVisualization
from spectra.visualization.pulses import TravelingPulses


class CombinedVisualization:
    """Reuse the two concrete visualizations on the same audio timeline."""

    def __init__(self, led_count: int = 300) -> None:
        self._frequency = FrequencyVisualization(led_count=led_count)
        self._pulses = TravelingPulses(led_count=led_count)

    def render(
        self,
        intensities: NDArray[np.float64],
        kicks: tuple[KickEvent, ...],
        timestamp: float,
    ) -> NDArray[np.uint8]:
        return np.maximum(
            self._frequency.render(intensities, kicks, timestamp),
            self._pulses.render(intensities, kicks, timestamp),
        )

    def reset(self) -> None:
        self._frequency.reset()
        self._pulses.reset()
