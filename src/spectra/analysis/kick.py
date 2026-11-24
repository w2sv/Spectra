"""Causal bass-onset detection from raw spectral power."""

from dataclasses import dataclass
from math import expm1

import numpy as np
from numpy.typing import NDArray

from spectra.analysis.spectrum import SpectrumResult


@dataclass(frozen=True)
class KickDetectionConfig:
    """Bass selection, novelty threshold, and audio-time adaptation parameters."""

    min_frequency: float = 40.0
    max_frequency: float = 160.0
    min_energy: float = 1e-6
    min_flux: float = 0.35
    threshold_multiplier: float = 2.0
    adaptation_seconds: float = 1.0
    refractory_seconds: float = 0.2

    def __post_init__(self) -> None:
        parameters = (
            self.min_frequency,
            self.max_frequency,
            self.min_energy,
            self.min_flux,
            self.threshold_multiplier,
            self.adaptation_seconds,
            self.refractory_seconds,
        )
        if (
            not np.all(np.isfinite(parameters))
            or not 0 <= self.min_frequency < self.max_frequency
            or self.min_energy <= 0
            or self.min_flux <= 0
            or self.adaptation_seconds <= 0
            or self.threshold_multiplier < 0
            or self.refractory_seconds < 0
        ):
            raise ValueError(
                "Use a valid bass range, positive energy/flux/adaptation, and nonnegative multiplier/refractory."
            )


@dataclass(frozen=True)
class KickEvent:
    """A bass transient at an audio-relative spectrum-window center timestamp.

    strength is positive log-power flux divided by its adaptive threshold:
    values exceed one and indicate threshold exceedance, not confidence or velocity.
    """

    timestamp: float
    strength: float


class KickOnsetDetector:
    """Detect rising bass novelty, independently of LED response settings.

    Power is fractionally selected at the bass-range boundaries, compressed
    with log1p(power / min_energy), and positive band differences are summed.
    The threshold is max(min_flux, recent_mean + multiplier * recent_deviation).
    Both recent statistics use causal, audio-time exponential updates; the
    current flux is tested before updating them. A detection also requires
    total selected bass power to increase: redistribution between bands during
    a decay or release can otherwise create positive flux without an attack.
    Only a rising threshold crossing outside the refractory period can emit
    an event. Flux statistics still include releases and other fluctuations.

    min_energy gates summed bass power in squared PCM amplitude units (the
    default 1e-6 is -60 dB relative to unit power). min_flux is a floor on
    summed positive log-power differences, before adaptive thresholding.

    These are kick-like bass transients: sustained notes stay quiet, but other
    abrupt low-frequency sounds can also trigger. No beat prediction is used.
    """

    def __init__(
        self,
        band_edges: NDArray[np.float64],
        config: KickDetectionConfig = KickDetectionConfig(),
    ) -> None:
        if (
            band_edges.ndim != 1
            or len(band_edges) < 2
            or not np.all(np.isfinite(band_edges))
            or np.any(np.diff(band_edges) <= 0)
        ):
            raise ValueError("Expected finite, increasing frequency-band boundaries.")
        overlap = np.minimum(band_edges[1:], config.max_frequency) - np.maximum(band_edges[:-1], config.min_frequency)
        self._bass_weights = np.maximum(overlap, 0.0) / np.diff(band_edges)
        if not np.any(self._bass_weights):
            raise ValueError("The bass range must overlap the supplied frequency bands.")
        self._min_energy = config.min_energy
        self._min_flux = config.min_flux
        self._threshold_multiplier = config.threshold_multiplier
        self._adaptation_seconds = config.adaptation_seconds
        self._refractory_seconds = config.refractory_seconds
        self._previous_log_power: NDArray[np.float64] | None = None
        self._previous_bass_energy = 0.0
        self._previous_timestamp: float | None = None
        self._flux_mean = 0.0
        self._flux_deviation = 0.0
        self._last_onset: float | None = None
        self._above_threshold = False

    def reset(self) -> None:
        """Discard the baseline, adaptation history, and refractory state."""
        self._previous_log_power = None
        self._previous_bass_energy = 0.0
        self._previous_timestamp = None
        self._flux_mean = 0.0
        self._flux_deviation = 0.0
        self._last_onset = None
        self._above_threshold = False

    def process(self, spectrum: SpectrumResult) -> KickEvent | None:
        """Consume every spectrum in strictly increasing audio-time order."""
        power = spectrum.band_energies
        timestamp = spectrum.timestamp
        if power.shape != self._bass_weights.shape or not np.all(np.isfinite(power)) or np.any(power < 0):
            raise ValueError("Expected finite, nonnegative raw power matching the configured bands.")
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Audio timestamps must be finite and nonnegative.")
        if self._previous_timestamp is not None and timestamp <= self._previous_timestamp:
            raise ValueError("Audio timestamps must increase; reset for a new stream.")

        bass_power = power * self._bass_weights
        bass_energy = float(bass_power.sum())
        log_power = np.log1p(bass_power / self._min_energy)
        event = None
        if self._previous_log_power is not None and self._previous_timestamp is not None:
            flux = float(np.maximum(log_power - self._previous_log_power, 0.0).sum())
            threshold = max(self._min_flux, self._flux_mean + self._threshold_multiplier * self._flux_deviation)
            above = bass_energy >= self._min_energy and bass_energy > self._previous_bass_energy and flux > threshold
            ready = self._last_onset is None or timestamp - self._last_onset >= self._refractory_seconds
            if above and not self._above_threshold and ready:
                event = KickEvent(timestamp, flux / threshold)
                self._last_onset = timestamp
            self._above_threshold = above

            alpha = -expm1(-(timestamp - self._previous_timestamp) / self._adaptation_seconds)
            difference = flux - self._flux_mean
            self._flux_mean += alpha * difference
            self._flux_deviation += alpha * (abs(difference) - self._flux_deviation)

        self._previous_log_power = log_power
        self._previous_bass_energy = bass_energy
        self._previous_timestamp = timestamp
        return event
