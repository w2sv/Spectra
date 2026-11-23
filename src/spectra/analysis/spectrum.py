"""Streaming, overlapping FFT analysis with logarithmic power bands."""

from dataclasses import dataclass

import numpy as np
from numpy.fft import rfft, rfftfreq
from numpy.typing import NDArray
from scipy.signal.windows import hann


@dataclass(frozen=True)
class SpectrumResult:
    """A completed window's spectrum.

    timestamp is the window center in seconds from the stream's first frame.
    band_energies is a read-only float64 array of shape (band_count,), indexed
    by SpectrumAnalyzer.band_edges. Values are integrated mean-square power
    in squared PCM amplitude units, not power density per Hz or decibels.
    An in-range sine of peak amplitude A contributes approximately A**2 / 2.
    Published arrays are independent of subsequent processing and resets.
    """

    timestamp: float
    band_energies: NDArray[np.float64]


class SpectrumAnalyzer:
    """Analyze successive (frames, 2) float32 chunks with an arithmetic mono mix.

    A periodic Hann window and real FFT produce one-sided bin powers:
    abs(FFT(x * window))**2 / (fft_size * sum(window**2)), doubled except
    at DC and, for even FFT sizes, Nyquist. The full spectrum sums to
    sum((x * window)**2) / sum(window**2), a window-weighted mean-square.

    Each bin's power is distributed uniformly over its frequency cell,
    bounded by adjacent bin midpoints (clipped to DC and Nyquist). Bands
    sum fractional overlaps with these cells. This conserves in-range
    power and avoids empty bands caused by bin-center-only assignment;
    it does not improve the FFT's underlying frequency resolution.

    No partial window is emitted or zero-padded. The caller owns execution;
    this class neither captures audio nor starts workers.
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        fft_size: int = 4096,
        hop_size: int = 512,
        min_frequency: float = 40.0,
        max_frequency: float = 16_000.0,
        band_count: int = 24,
    ) -> None:
        if sample_rate <= 0 or fft_size < 2 or not 1 <= hop_size <= fft_size:
            raise ValueError("Use a positive sample rate, FFT size >= 2, and hop size between 1 and the FFT size.")
        if band_count <= 0 or not 0 < min_frequency < max_frequency <= sample_rate / 2:
            raise ValueError("Use positive band counts and 0 < minimum frequency < maximum frequency <= Nyquist.")

        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._hop_size = hop_size
        self._window: NDArray[np.float64] = hann(fft_size, sym=False)
        self._power_scale = np.full(fft_size // 2 + 1, 2.0 / (fft_size * np.sum(self._window**2)))
        self._power_scale[0] *= 0.5
        if fft_size % 2 == 0:
            self._power_scale[-1] *= 0.5

        self._band_edges: NDArray[np.float64] = np.geomspace(min_frequency, max_frequency, band_count + 1)
        self._band_edges.setflags(write=False)
        frequencies = rfftfreq(fft_size, d=1.0 / sample_rate)
        bin_edges = np.concatenate(([0.0], (frequencies[:-1] + frequencies[1:]) / 2, [sample_rate / 2]))
        overlaps = np.minimum(self._band_edges[1:, None], bin_edges[None, 1:]) - np.maximum(
            self._band_edges[:-1, None], bin_edges[None, :-1]
        )
        self._band_weights = np.maximum(overlaps, 0.0) / np.diff(bin_edges)

        self._buffer = np.empty(fft_size, dtype=np.float64)
        self._buffered = 0
        self._frames_received = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def fft_size(self) -> int:
        return self._fft_size

    @property
    def hop_size(self) -> int:
        return self._hop_size

    @property
    def band_edges(self) -> NDArray[np.float64]:
        """Read-only float64 boundaries in Hz, shape (band_count + 1,)."""
        return self._band_edges

    def reset(self) -> None:
        """Discard buffered samples and restart timestamps for a new stream."""
        self._buffered = 0
        self._frames_received = 0

    def process(self, samples: NDArray[np.float32]) -> list[SpectrumResult]:
        """Consume stereo PCM and return every completed window in time order.

        Chunk boundaries do not affect results. Only one FFT window is
        buffered, even for large input chunks. Empty chunks return no results.
        """
        if samples.ndim != 2 or samples.shape[1] != 2 or samples.dtype != np.float32:
            raise ValueError("Expected stereo float32 PCM with shape (frames, 2).")

        results: list[SpectrumResult] = []
        offset = 0
        while offset < len(samples):
            count = min(self._fft_size - self._buffered, len(samples) - offset)
            self._buffer[self._buffered : self._buffered + count] = samples[offset : offset + count].mean(
                axis=1, dtype=np.float64
            )
            self._buffered += count
            self._frames_received += count
            offset += count

            if self._buffered == self._fft_size:
                spectrum = rfft(self._buffer * self._window)
                power = (spectrum.real**2 + spectrum.imag**2) * self._power_scale
                energies = self._band_weights @ power
                energies.setflags(write=False)
                timestamp = (self._frames_received - self._fft_size / 2) / self._sample_rate
                results.append(SpectrumResult(timestamp, energies))

                self._buffered = self._fft_size - self._hop_size
                self._buffer[: self._buffered] = self._buffer[self._hop_size :]

        return results
