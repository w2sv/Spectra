"""Bounded, variable-rate PCM interpolation in original recording coordinates."""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from spectra.audio import AudioSamples

MIN_SPEED = 0.25
MAX_SPEED = 2.0
RATE_TRANSITION_SECONDS = 0.2
_INTERPOLATION_TAPS = 48
_INTERPOLATION_PHASES = 512
_FILTER_LEVELS = 33
_RENDER_CHUNK = 1024


@dataclass(frozen=True)
class SampleSpan:
    """A linear rate ramp followed by a constant rate, measured in output frames."""

    first_sample: float
    frames: int
    speed: float = 1.0
    speed_step: float = 0.0
    ramp_frames: int = 0

    def source_at(self, frame: float) -> float:
        ramp = min(max(frame, 0.0), self.ramp_frames)
        return self.first_sample + self.speed * frame + self.speed_step * ramp * (frame - 0.5 * ramp)

    def speed_at(self, frame: float) -> float:
        return self.speed + self.speed_step * min(max(frame, 0.0), self.ramp_frames)

    def frame_at(self, sample: float) -> float:
        distance = sample - self.first_sample
        ramp_distance = self.source_at(self.ramp_frames) - self.first_sample
        if distance >= ramp_distance:
            return self.ramp_frames + (distance - ramp_distance) / self.speed_at(self.ramp_frames)
        # This form avoids cancellation when a ramp is almost constant.
        return 2 * distance / (self.speed + math.sqrt(max(0.0, self.speed**2 + 2 * self.speed_step * distance)))


@lru_cache(maxsize=1)
def _interpolation_filters() -> NDArray[np.float32]:
    """Phase/rate tables avoid designing filters in the audio callback (~3 MB)."""
    offsets = np.arange(_INTERPOLATION_TAPS) - (_INTERPOLATION_TAPS // 2 - 1)
    phases = np.arange(_INTERPOLATION_PHASES + 1) / _INTERPOLATION_PHASES
    cutoff = 1 / np.linspace(1, MAX_SPEED, _FILTER_LEVELS)
    kernels = cutoff[:, None, None] * np.sinc(cutoff[:, None, None] * (offsets - phases[:, None]))
    # Center the window on the fractional position too. The endpoint phases must
    # be shifted copies, otherwise crossing an integer sample creates a click.
    distance = (offsets - phases[:, None]) / (_INTERPOLATION_TAPS / 2)
    window = np.where(np.abs(distance) < 1, 0.5 * (1 + np.cos(np.pi * distance)), 0)
    kernels *= window
    kernels /= kernels.sum(axis=-1, keepdims=True)
    result = kernels.astype(np.float32).reshape(-1, _INTERPOLATION_TAPS)
    result.flags.writeable = False
    return result


class VariableRatePCM:
    """Callback-owned source position/ramp; the GUI only publishes target_speed.

    Windowed-sinc interpolation includes a rate-dependent low-pass filter when
    accelerating. Bilinear table interpolation keeps phase/filter changes smooth.
    Normal-speed integer-position playback copies the original PCM exactly.
    """

    def __init__(self, samples: AudioSamples, sample_rate: int) -> None:
        self.samples = np.ascontiguousarray(samples)
        self.sample_rate = sample_rate
        self.position = 0.0
        self.speed = 1.0
        self.target_speed = 1.0
        self.ramp_target = 1.0
        self.remaining = 0
        self.step = 0.0
        self.filters = _interpolation_filters()
        self.steps = np.arange(_RENDER_CHUNK + 1, dtype=np.float64)
        self.offsets = np.arange(_INTERPOLATION_TAPS) - (_INTERPOLATION_TAPS // 2 - 1)
        self.indices = np.empty((_RENDER_CHUNK, _INTERPOLATION_TAPS), dtype=np.int64)
        self.weights = np.empty((_RENDER_CHUNK, _INTERPOLATION_TAPS), dtype=np.float32)
        self.upper = np.empty_like(self.weights)
        self.scratch = np.empty_like(self.weights)
        self.values = np.empty_like(self.weights)

    def reset(self, position: float, speed: float) -> None:
        self.position = position
        self.speed = speed
        self.ramp_target = speed
        self.remaining = 0
        self.step = 0.0

    def _span(self, frames: int) -> SampleSpan:
        target = self.target_speed
        if target != self.ramp_target:
            self.ramp_target = target
            self.remaining = max(1, round(RATE_TRANSITION_SECONDS * self.sample_rate))
            self.step = (target - self.speed) / self.remaining
        return SampleSpan(self.position, frames, self.speed, self.step, self.remaining)

    def _advance_rate(self, span: SampleSpan, frames: int) -> None:
        self.speed = span.speed_at(frames)
        self.remaining = max(0, self.remaining - frames)
        if not self.remaining:
            self.speed = self.ramp_target
            self.step = 0.0

    def advance_paused_rate(self, seconds: float) -> None:
        if not self.remaining and self.speed == self.target_speed:
            return
        frames = max(0, round(seconds * self.sample_rate))
        self._advance_rate(self._span(frames), frames)

    def render(self, output: AudioSamples) -> SampleSpan:
        span = self._span(len(output))
        if span.speed == 1 and not self.remaining and self.position.is_integer():
            first = int(self.position)
            count = min(len(output), len(self.samples) - first)
            output[:count] = self.samples[first : first + count]
            output[count:] = 0
        else:
            count = 0
            for start in range(0, len(output), _RENDER_CHUNK):
                size = min(_RENDER_CHUNK, len(output) - start)
                frames = self.steps[:size] + start
                ramp = np.minimum(frames, span.ramp_frames)
                positions = span.first_sample + span.speed * frames + span.speed_step * ramp * (frames - 0.5 * ramp)
                valid = int(np.searchsorted(positions, len(self.samples)))
                if valid:
                    speeds = span.speed + span.speed_step * ramp[:valid]
                    self._interpolate(positions[:valid], speeds, output[start : start + valid])
                    count += valid
                if valid < size:
                    output[start + valid :] = 0
                    break
        self.position = min(float(len(self.samples)), span.source_at(count))
        self._advance_rate(span, count)
        return SampleSpan(span.first_sample, count, span.speed, span.speed_step, span.ramp_frames)

    def _interpolate(self, positions: NDArray[np.float64], speeds: NDArray[np.float64], output: AudioSamples) -> None:
        count = len(positions)
        base = np.floor(positions).astype(np.int64)
        phase = (positions - base) * _INTERPOLATION_PHASES
        phase_index = phase.astype(np.int64)
        phase_mix = (phase - phase_index).astype(np.float32)[:, None]
        level = (np.maximum(1.0, speeds) - 1) / (MAX_SPEED - 1) * (_FILTER_LEVELS - 1)
        level_index = np.minimum(level.astype(np.int64), _FILTER_LEVELS - 2)
        level_mix = (level - level_index).astype(np.float32)[:, None]
        row = level_index * (_INTERPOLATION_PHASES + 1) + phase_index
        weights, upper, scratch = self.weights[:count], self.upper[:count], self.scratch[:count]
        np.take(self.filters, row, axis=0, out=weights, mode="clip")
        np.take(self.filters, row + 1, axis=0, out=scratch, mode="clip")
        scratch -= weights
        scratch *= phase_mix
        weights += scratch
        np.take(self.filters, row + _INTERPOLATION_PHASES + 1, axis=0, out=upper, mode="clip")
        np.take(self.filters, row + _INTERPOLATION_PHASES + 2, axis=0, out=scratch, mode="clip")
        scratch -= upper
        scratch *= phase_mix
        upper += scratch
        upper -= weights
        upper *= level_mix
        weights += upper
        indices = self.indices[:count]
        np.add(base[:, None], self.offsets, out=indices)
        np.clip(indices, 0, len(self.samples) - 1, out=indices)
        channels = self.samples.shape[1]
        indices *= channels
        values = self.values[:count]
        for channel in range(channels):
            np.take(self.samples, indices, out=values, mode="clip")
            values *= weights
            np.sum(values, axis=1, out=output[:, channel])
            indices += 1
