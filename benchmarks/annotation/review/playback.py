"""In-memory PCM playback, with position measured in the PortAudio clock domain.

The GUI owns transport transitions and publishes rate targets. The callback owns
resampling and bounded, immutable timing records. Abort (not draining stop)
precedes transport discontinuities; rate changes leave the stream running.
"""

import math
from bisect import bisect_right
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.signal import resample_poly

from benchmarks.annotation.review import DEFAULT_VIEW_DURATION
from benchmarks.annotation.review.resampling import MAX_SPEED, MIN_SPEED, SampleSpan, VariableRatePCM
from benchmarks.fixtures import load_wav
from spectra.audio import AudioSamples

if TYPE_CHECKING:
    import sounddevice as sd

_KICK_FLASH_SECONDS = 0.075


def zoom_speed(visible_duration: float) -> float:
    return min(MAX_SPEED, max(MIN_SPEED, math.sqrt(visible_duration / DEFAULT_VIEW_DURATION)))


class KickFlash:
    """Cross effective kicks in recording time; expire pulses in audible DAC time."""

    def __init__(self) -> None:
        self.timestamps: list[float] = []
        self.previous: float | None = None
        self.generation: int | None = None
        self.until = 0.0

    def reset(self) -> None:
        self.previous = None
        self.until = 0.0

    def update(
        self,
        position: float,
        *,
        playing: bool,
        generation: int,
        playback_time: float | None = None,
        event_time: Callable[[float], float | None] | None = None,
    ) -> bool:
        previous = self.previous
        if not playing or generation != self.generation or previous is None or position < previous:
            self.until = 0.0
        else:
            index = bisect_right(self.timestamps, position) - 1
            if index >= 0 and previous < self.timestamps[index] <= position:
                crossing = event_time(self.timestamps[index]) if event_time is not None else self.timestamps[index]
                self.until = crossing + _KICK_FLASH_SECONDS if crossing is not None else 0.0
        self.generation = generation
        self.previous = position if playing else None
        return playing and (position if playback_time is None else playback_time) < self.until


class CallbackTime(Protocol):
    """PortAudio's callback timing structure, in stream-clock seconds."""

    outputBufferDacTime: float


@dataclass(frozen=True)
class BufferTiming:
    generation: int
    span: SampleSpan
    dac_time: float


class PlaybackClock:
    def __init__(self, rate: int, sample_count: int, duration: float) -> None:
        self.rate = rate
        self.sample_count = sample_count
        self.duration = duration
        self.generation = 0
        self.frozen_sample = 0.0
        self.frozen_speed = 1.0
        self.playing = False
        self.timings: deque[BufferTiming] = deque(maxlen=256)

    def position(self, stream_time: float) -> float:
        return self.snapshot(stream_time)[0]

    def snapshot(self, stream_time: float) -> tuple[float, float]:
        sample = float(self.frozen_sample)
        speed = self.frozen_speed
        if self.playing:
            # Future buffers must not replace the buffer currently being heard.
            # A snapshot avoids deque iteration racing callback append operations.
            for timing in reversed(tuple(self.timings)):
                if timing.generation == self.generation and timing.dac_time <= stream_time:
                    frame = min(timing.span.frames, max(0.0, (stream_time - timing.dac_time) * self.rate))
                    sample = timing.span.source_at(frame)
                    speed = timing.span.speed_at(frame)
                    break
        return min(self.duration, max(0.0, sample / self.rate)), speed

    def event_time(self, position: float) -> float | None:
        sample = position * self.rate
        for timing in reversed(tuple(self.timings)):
            if timing.generation == self.generation and timing.span.first_sample <= sample <= timing.span.source_at(
                timing.span.frames
            ):
                return timing.dac_time + timing.span.frame_at(sample) / self.rate
        return None

    def reset(self, position: float, *, playing: bool, speed: float | None = None) -> None:
        self.generation += 1
        self.frozen_sample = min(float(self.sample_count), max(0.0, position * self.rate))
        if speed is not None:
            self.frozen_speed = speed
        self.timings.clear()
        self.playing = playing


class PCMPlayback:
    def __init__(self, samples: AudioSamples, sample_rate: int) -> None:
        import sounddevice as sd

        self.audio = sd
        if sample_rate <= 0 or samples.ndim != 2 or not len(samples) or samples.shape[1] < 1:
            raise ValueError("Playback requires nonempty PCM shaped (frames, channels) and a positive rate.")
        if samples.dtype != np.float32 or not np.all(np.isfinite(samples)):
            raise ValueError("Playback requires finite float32 PCM.")
        self.duration = len(samples) / sample_rate
        self.source_rate = sample_rate
        self.rate = sample_rate
        self.samples = samples
        self.stream: sd.OutputStream | None = None
        self.speed = 1.0
        self.dac_time = 0.0
        self.paused_rate_time = monotonic()
        self.underruns = 0
        self.reliable_buffering = False
        self.callback_error: Exception | None = None
        self.closed = False
        self.eof = False
        channels = samples.shape[1]
        try:
            sd.check_output_settings(channels=channels, dtype="float32", samplerate=self.rate)
        except sd.PortAudioError:
            device = sd.query_devices(kind="output")
            supported_rate = round(device["default_samplerate"])
            sd.check_output_settings(channels=channels, dtype="float32", samplerate=supported_rate)
            divisor = math.gcd(sample_rate, supported_rate)
            self.samples = np.ascontiguousarray(
                resample_poly(samples, supported_rate // divisor, sample_rate // divisor, axis=0), dtype=np.float32
            )
            self.rate = supported_rate
        self.clock = PlaybackClock(self.rate, len(self.samples), self.duration)
        self.resampler = VariableRatePCM(self.samples, self.rate)
        # Desktop profiling repeatedly exhausted the backend's tiny low-latency
        # buffers during Qt painting. The DAC clock accounts for queued output.
        self.stream = self._open_stream("high")

    def _open_stream(self, latency: str | float) -> sd.OutputStream:
        return self.audio.OutputStream(
            samplerate=self.rate,
            channels=self.samples.shape[1],
            dtype="float32",
            blocksize=0,
            latency=latency,
            clip_off=True,
            dither_off=True,
            callback=self._callback,
        )

    @classmethod
    def from_wav(cls, path: Path) -> PCMPlayback:
        rate, samples = load_wav(path, preserve_channels=True)
        return cls(samples, rate)

    @property
    def paused(self) -> bool:
        return not self.clock.playing

    @property
    def generation(self) -> int:
        return self.clock.generation

    @property
    def position(self) -> float:
        if self.stream is None or self.paused:
            return self.clock.position(0)
        return self.clock.position(self.stream.time)

    @property
    def submitted_sample(self) -> float:
        return self.resampler.position

    def set_speed(self, speed: float) -> None:
        if not math.isfinite(speed):
            raise ValueError("Playback speed must be finite.")
        speed = min(MAX_SPEED, max(MIN_SPEED, speed))
        if math.isclose(speed, 1.0, abs_tol=1e-9):
            speed = 1.0
        if not math.isclose(speed, self.resampler.target_speed, rel_tol=1e-9):
            if self.paused:
                self._update_paused_speed()
            self.resampler.target_speed = speed

    def event_time(self, position: float) -> float | None:
        return self.clock.event_time(position)

    def _callback(self, output: AudioSamples, frames: int, timing: CallbackTime, status: sd.CallbackFlags) -> None:
        generation = self.clock.generation
        try:
            span = self.resampler.render(output)
            self.clock.timings.append(BufferTiming(generation, span, timing.outputBufferDacTime))
            if status.output_underflow:
                self.underruns += 1
            if self.submitted_sample >= len(self.samples):
                self.eof = True
                # Only natural completion drains the final buffer. Pause/seek abort.
                raise self.audio.CallbackStop
        except self.audio.CallbackStop:
            raise
        except Exception as error:
            output.fill(0)
            self.callback_error = error
            raise self.audio.CallbackAbort from error

    def play(self) -> None:
        if self.closed or self.stream is None:
            raise RuntimeError("Playback stream is closed.")
        if not self.paused:
            return
        self._update_paused_speed()
        position = self.position
        if position >= self.duration:
            position = 0.0
        self.stream.abort()
        self.clock.reset(position, playing=True)
        self.resampler.reset(self.clock.frozen_sample, self.speed)
        self.eof = False
        self.callback_error = None
        try:
            self.stream.start()
        except Exception:
            self.clock.reset(position, playing=False)
            raise

    def pause(self) -> None:
        if self.paused or self.stream is None:
            return
        # Abort discards queued audio without waiting for it to drain. Reading
        # the same clock after abort estimates the final audible sample.
        position = self.position
        try:
            self.stream.abort()
            position, self.speed = self.clock.snapshot(self.stream.time)
        finally:
            self.clock.reset(position, playing=False, speed=self.speed)
            self.resampler.reset(self.clock.frozen_sample, self.speed)
            self.paused_rate_time = monotonic()

    def seek(self, position: float) -> None:
        if not math.isfinite(position):
            raise ValueError("Seek requires a finite recording timestamp.")
        playing = not self.paused
        self.pause()
        self.clock.reset(min(self.duration, max(0.0, position)), playing=False)
        # Seeking to EOF stays at EOF. Space subsequently restarts at zero.
        if playing and position < self.duration:
            self.play()

    def toggle(self) -> None:
        if self.paused:
            self.play()
        else:
            self.pause()

    def poll(self) -> float:
        if self.paused or self.stream is None:
            self._update_paused_speed()
            position = self.clock.position(0)
        else:
            self.dac_time = self.stream.time
            position, self.speed = self.clock.snapshot(self.dac_time)
        if self.callback_error is not None:
            error = self.callback_error
            self.pause()
            raise RuntimeError(f"Audio callback failed: {error}") from error
        if not self.paused and self.stream is not None:
            if self.eof and (position >= self.duration or not self.stream.active):
                self.pause()
                self.clock.reset(self.duration, playing=False)
                return self.duration
            if not self.stream.active:
                self.pause()
                if not self.eof:
                    raise RuntimeError("Audio device stopped unexpectedly. Review edits are retained.")
            elif self.underruns >= 3 and not self.reliable_buffering:
                # A Python callback also needs the GIL during native Qt painting.
                # Retry once with reliable buffering; the DAC clock accounts
                # for the larger queue, so this needs no cursor compensation.
                self.pause()
                self.stream.close()
                self.stream = None
                self.reliable_buffering = True
                self.stream = self._open_stream(0.1)
                self.play()
                position = self.position
        return position

    def _update_paused_speed(self) -> None:
        now = monotonic()
        self.resampler.advance_paused_rate(now - self.paused_rate_time)
        self.paused_rate_time = now
        self.speed = self.resampler.speed
        self.clock.frozen_speed = self.speed

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.pause()
        finally:
            self.closed = True
            if self.stream is not None:
                self.stream.close()
