"""Fixed-reference band balancing and audio-time attack/release envelopes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spectra.visualization.frequency import power_to_decibels


@dataclass(frozen=True)
class BandIntensitySettings:
    """Immutable LED response settings, independent of capture and presentation.

    frequency_balance_db is the high-minus-low gain difference across ordered
    logarithmic bands, centered at zero gain. Positive values attenuate low
    frequencies and boost high ones. Time constants are in audio seconds.
    """

    min_db: float = -60.0
    max_db: float = -6.0
    sensitivity_db: float = 0.0
    frequency_balance_db: float = 6.0
    attack_seconds: float = 0.02
    release_seconds: float = 0.18

    def __post_init__(self) -> None:
        values = (
            self.min_db,
            self.max_db,
            self.sensitivity_db,
            self.frequency_balance_db,
            self.attack_seconds,
            self.release_seconds,
        )
        if not np.all(np.isfinite(values)) or self.min_db >= self.max_db:
            raise ValueError("Intensity settings must be finite, with min_db < max_db.")
        if self.attack_seconds < 0 or self.release_seconds < 0:
            raise ValueError("Attack and release times must be nonnegative; zero means instantaneous.")


class BandIntensityProcessor:
    """Convert raw mean-square power into smoothed float64 intensities in [0, 1].

    Band positions are uniformly spaced in logarithmic frequency, as supplied
    by SpectrumAnalyzer. Gains are fixed offsets in dB, never adaptive to
    incoming power. Each update uses exp(-elapsed_audio_time / time_constant).
    Settings changes preserve the envelope and take effect on the next result.
    """

    def __init__(self, band_count: int, settings: BandIntensitySettings | None = None) -> None:
        if band_count <= 0:
            raise ValueError("Band count must be positive.")
        self._band_count = band_count
        self._balance_positions = np.linspace(-0.5, 0.5, band_count) if band_count > 1 else np.zeros(1)
        self._settings = BandIntensitySettings() if settings is None else settings
        self._last_timestamp: float | None = None
        self._intensities = np.zeros(band_count, dtype=np.float64)

    def update_settings(self, settings: BandIntensitySettings) -> None:
        """Replace the settings without resetting the accumulated envelope."""
        self._settings = settings

    def reset(self) -> None:
        """Begin a new stream; its first result is applied without startup delay."""
        self._last_timestamp = None
        self._intensities = np.zeros(self._band_count, dtype=np.float64)

    def process(self, band_energies: NDArray[np.float64], timestamp: float) -> NDArray[np.float64]:
        """Consume one spectrum in audio-time order and return a read-only snapshot.

        Input and output shapes are (band_count,). Timestamps are seconds from
        the stream start and must not decrease. Returned snapshots stay valid
        after later updates or reset(). No wall-clock timing is used.
        """
        if band_energies.shape != (self._band_count,) or not np.all(np.isfinite(band_energies)):
            raise ValueError("Expected a finite band-energy vector matching the configured band count.")
        if np.any(band_energies < 0):
            raise ValueError("Spectral power must be nonnegative.")
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Audio timestamps must be finite and nonnegative.")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Audio timestamps must not decrease; reset for a new stream.")

        settings = self._settings
        levels = power_to_decibels(band_energies) + settings.sensitivity_db
        levels += settings.frequency_balance_db * self._balance_positions
        target = np.clip((levels - settings.min_db) / (settings.max_db - settings.min_db), 0.0, 1.0)
        target[band_energies == 0] = 0.0

        if self._last_timestamp is None:
            intensities = target
        else:
            elapsed = timestamp - self._last_timestamp
            time_constants = np.where(target > self._intensities, settings.attack_seconds, settings.release_seconds)
            decay = np.zeros(self._band_count, dtype=np.float64)
            smooth = time_constants > 0
            decay[smooth] = np.exp(-elapsed / time_constants[smooth])
            intensities = np.clip(target + decay * (self._intensities - target), 0.0, 1.0)

        # Exponential release is asymptotic. Snap only off-target tails below
        # 0.01% intensity to zero, well below the RGB framebuffer's resolution.
        intensities[(target == 0) & (intensities < 1e-4)] = 0.0
        intensities.setflags(write=False)
        self._intensities = intensities
        self._last_timestamp = timestamp
        return intensities
