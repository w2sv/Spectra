"""A fixed frequency-to-hue and power-to-brightness mapping."""

from colorsys import hsv_to_rgb

import numpy as np
from numpy.typing import NDArray

POWER_FLOOR = 1e-12  # -120 dB relative to unit mean-square PCM power.


def power_to_decibels(power: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert mean-square PCM power to dB relative to power 1, with a finite floor."""
    return 10.0 * np.log10(np.maximum(power, POWER_FLOOR))


def frequency_frame(
    band_energies: NDArray[np.float64],
    *,
    led_count: int = 300,
    min_db: float = -60.0,
    max_db: float = -6.0,
    low_hue: float = 0.0,
    high_hue: float = 2.0 / 3.0,
) -> NDArray[np.uint8]:
    """Map ordered bands to a complete (led_count, 3) uint8 RGB framebuffer.

    Hue runs from red at the lowest band to blue at the highest by default.
    Brightness is linear in dB between min_db (off) and max_db (full).
    Every LED receives one band, in order; region sizes differ by at most one.
    No frame-dependent normalization, smoothing, or clipping of source power occurs.
    """
    if band_energies.ndim != 1 or len(band_energies) == 0 or led_count <= 0:
        raise ValueError("Provide a nonempty band-energy vector and a positive LED count.")
    if not np.isfinite(min_db) or not np.isfinite(max_db) or min_db >= max_db:
        raise ValueError("Brightness limits must be finite, with min_db < max_db.")
    if not 0 <= low_hue <= 1 or not 0 <= high_hue <= 1:
        raise ValueError("Hue endpoints must lie between 0 and 1.")

    brightness = np.clip((power_to_decibels(band_energies) - min_db) / (max_db - min_db), 0.0, 1.0)
    hues = np.linspace(low_hue, high_hue, len(band_energies))
    colors = np.array([hsv_to_rgb(float(hue), 1.0, float(level)) for hue, level in zip(hues, brightness, strict=True)])
    colors = np.rint(colors * 255).astype(np.uint8)
    band_indices = np.arange(led_count) * len(band_energies) // led_count
    return colors[band_indices]
