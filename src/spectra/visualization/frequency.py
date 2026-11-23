"""Frequency colors, diagnostic decibels, and intensity-to-RGB mapping."""

from colorsys import hsv_to_rgb

import numpy as np
from numpy.typing import NDArray

from spectra.analysis import KickEvent

POWER_FLOOR = 1e-12  # -120 dB relative to unit mean-square PCM power.


def power_to_decibels(power: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert mean-square PCM power to dB relative to power 1, with a finite floor."""
    return 10.0 * np.log10(np.maximum(power, POWER_FLOOR))


def band_colors(
    band_count: int,
    *,
    low_hue: float = 0.0,
    high_hue: float = 2.0 / 3.0,
) -> NDArray[np.float64]:
    """Return full-brightness RGB colors in frequency order, shape (band_count, 3).

    Components are float64 values in [0, 1], retaining precision for later
    brightness scaling. The default hues progress from red to blue.
    """
    if band_count <= 0:
        raise ValueError("Band count must be positive.")
    if not 0 <= low_hue <= 1 or not 0 <= high_hue <= 1:
        raise ValueError("Hue endpoints must lie between 0 and 1.")

    hues = np.linspace(low_hue, high_hue, band_count)
    return np.array([hsv_to_rgb(float(hue), 1.0, 1.0) for hue in hues], dtype=np.float64)


def frequency_frame(
    intensities: NDArray[np.float64],
    *,
    led_count: int = 300,
    low_hue: float = 0.0,
    high_hue: float = 2.0 / 3.0,
) -> NDArray[np.uint8]:
    """Map ordered bands to a complete (led_count, 3) uint8 RGB framebuffer.

    Hue runs from red at the lowest band to blue at the highest by default.
    Normalized intensities in [0, 1] directly control LED brightness.
    Every LED receives one band, in order; region sizes differ by at most one.
    Intensity processing belongs to BandIntensityProcessor, not this renderer.
    """
    if intensities.ndim != 1 or len(intensities) == 0 or led_count <= 0:
        raise ValueError("Provide a nonempty intensity vector and a positive LED count.")
    if not np.all(np.isfinite(intensities)) or np.any((intensities < 0) | (intensities > 1)):
        raise ValueError("Band intensities must be finite values between 0 and 1.")
    colors = band_colors(len(intensities), low_hue=low_hue, high_hue=high_hue)
    colors = np.rint(colors * intensities[:, None] * 255).astype(np.uint8)
    band_indices = np.arange(led_count) * len(intensities) // led_count
    return colors[band_indices]


class FrequencyVisualization:
    """Adapt frequency_frame to the visualization interface without animation."""

    def __init__(self, led_count: int = 300) -> None:
        if led_count <= 0:
            raise ValueError("LED count must be positive.")
        self._led_count = led_count

    def render(
        self,
        intensities: NDArray[np.float64],
        kicks: tuple[KickEvent, ...],
        timestamp: float,
    ) -> NDArray[np.uint8]:
        return frequency_frame(intensities, led_count=self._led_count)

    def reset(self) -> None:
        """Frequency colors have no animation state to clear."""
