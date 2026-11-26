"""DAC clock and transition invariants, without requiring an audio device."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import sounddevice as sd
from scipy.io import wavfile

from benchmarks.annotation.review.playback import BufferTiming, PCMPlayback, PlaybackClock
from benchmarks.annotation.review.resampling import SampleSpan, VariableRatePCM
from benchmarks.fixtures import load_wav


@dataclass
class CallbackTime:
    outputBufferDacTime: float


def test_dac_clock_waits_for_audibility_and_bounds_each_buffer() -> None:
    clock = PlaybackClock(1000, 3000, 3)
    clock.reset(1, playing=True)
    clock.timings.append(BufferTiming(clock.generation, SampleSpan(1000, 100), 50.1))
    clock.timings.append(BufferTiming(clock.generation, SampleSpan(1100, 100), 50.2))
    assert clock.position(50.05) == 1  # samples submitted, not yet audible
    assert clock.position(50.15) == pytest.approx(1.05)
    assert clock.position(50.25) == pytest.approx(1.15)
    assert clock.position(51) == 1.2  # do not extrapolate beyond submitted audio
    old_generation = clock.generation
    clock.reset(2, playing=False)
    clock.timings.append(BufferTiming(old_generation, SampleSpan(0, 100), 51))
    assert clock.position(60) == 2
    clock.reset(2, playing=True)
    assert clock.position(60) == 2  # reset discards timing history


class OutputStream:
    """Only the PortAudio boundary is simulated; production PCM/callback paths run."""

    def __init__(self, **_: object) -> None:
        self.time = 100.0
        self.active = False
        self.closed = False
        self.aborts = 0

    def start(self) -> None:
        self.active = True

    def abort(self) -> None:
        self.active = False
        self.aborts += 1

    def close(self) -> None:
        self.closed = True


def test_pcm_transients_pause_seek_resume_and_actual_end(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = OutputStream()
    monkeypatch.setattr(sd, "check_output_settings", lambda **_: None)
    monkeypatch.setattr(sd, "OutputStream", lambda **_: stream)
    samples = np.zeros((2000, 2), np.float32)
    samples[[100, 600, 1100], 0] = 0.75
    samples[[100, 600, 1100], 1] = -0.25
    player = PCMPlayback(samples, 1000)
    output = np.empty((500, 2), np.float32)
    player.play()
    player._callback(output, 500, CallbackTime(100.1), sd.CallbackFlags())
    np.testing.assert_array_equal(output, samples[:500])
    stream.time = 100.2
    assert player.poll() == pytest.approx(0.1)  # identifiable transient at DAC
    player.pause()
    paused = player.position
    stream.time += 10
    assert player.position == paused
    for _ in range(5):
        player.play()
        first = player.submitted_sample
        player._callback(output, 500, CallbackTime(stream.time + 0.1), sd.CallbackFlags())
        stream.time += 0.25
        assert player.poll() == pytest.approx(first / 1000 + 0.15)
        player.pause()
    player.play()
    player.seek(1.9)
    assert not player.paused and player.position == 1.9
    with pytest.raises(sd.CallbackStop):
        player._callback(output, 500, CallbackTime(stream.time + 0.1), sd.CallbackFlags())
    assert not np.any(output[100:])
    stream.time += 0.15
    assert player.poll() == pytest.approx(1.95)
    stream.time += 0.1
    assert player.poll() == 2 and player.paused
    player.toggle()
    assert player.position == 0 and not player.paused
    player.seek(0.6)
    player.pause()
    assert player.position == 0.6
    player.seek(0.4)
    assert player.paused and player.position == 0.4
    player.close()
    player.close()
    assert stream.closed and not stream.active


def test_original_channel_layout_and_amplitude_and_resampled_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = np.array([0, 2000, -1000, 32767], dtype=np.int16)
    path = tmp_path / "mono.wav"
    wavfile.write(path, 4000, source)
    rate, samples = load_wav(path, preserve_channels=True)
    assert samples.shape == (4, 1)
    np.testing.assert_array_equal(samples[:, 0], source.astype(np.float32) / 32768)
    calls = []

    def settings(**values: object) -> None:
        calls.append(values)
        if values["samplerate"] == 4000:
            raise sd.PortAudioError("unsupported rate")

    monkeypatch.setattr(sd, "check_output_settings", settings)
    monkeypatch.setattr(sd, "query_devices", lambda **_: {"default_samplerate": 8000})
    monkeypatch.setattr(sd, "OutputStream", lambda **_: OutputStream())
    player = PCMPlayback(samples, rate)
    assert player.rate == 8000 and player.duration == 0.001
    assert player.samples.shape == (8, 1)
    player.seek(0.0005)
    assert player.position == 0.0005
    player.close()


def test_underrun_recovery_and_device_failure_freeze_audible_position(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = OutputStream(), OutputStream()
    streams = iter([first, second])
    requested = []

    def open_stream(**settings: object):
        requested.append(settings["latency"])
        return next(streams)

    monkeypatch.setattr(sd, "check_output_settings", lambda **_: None)
    monkeypatch.setattr(sd, "OutputStream", open_stream)
    player = PCMPlayback(np.zeros((3000, 2), np.float32), 1000)
    player.play()
    player._callback(np.empty((500, 2), np.float32), 500, CallbackTime(100), sd.CallbackFlags())
    first.time = 100.2
    player.underruns = 3
    assert player.poll() == pytest.approx(0.2)
    assert player.reliable_buffering and first.closed
    assert requested == ["high", 0.1]
    second.active = False
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        player.poll()
    assert player.paused and player.position == pytest.approx(0.2)
    player.close()


def test_kick_flash_crossings_expiration_and_discontinuities() -> None:
    from benchmarks.annotation.review.playback import KickFlash

    flash = KickFlash()
    flash.timestamps = [1, 1.02, 2, 3]
    assert not flash.update(0.9, playing=True, generation=1)
    assert flash.update(1.01, playing=True, generation=1)
    assert flash.update(1.04, playing=True, generation=1)  # Latest crossed kick wins.
    assert flash.update(1.094, playing=True, generation=1)
    assert not flash.update(1.096, playing=True, generation=1)
    assert not flash.update(2.2, playing=True, generation=1)  # Entire pulse missed.
    assert not flash.update(3.01, playing=True, generation=2)  # Seek/recovery generation.
    assert not flash.update(0.9, playing=True, generation=2)  # Backward discontinuity.
    assert flash.update(1, playing=True, generation=2)
    assert not flash.update(1.01, playing=False, generation=2)
    assert not flash.update(2.01, playing=True, generation=2)  # Resuming seeds crossings.
    flash.reset()  # Editing must never create a crossing.
    assert not flash.update(3.01, playing=True, generation=2)


@pytest.mark.parametrize(
    "duration, expected",
    [(0.1, 0.25), (0.625, 0.25), (2.5, 0.5), (5, 2**-0.5), (10, 1), (20, 2**0.5), (40, 2), (100, 2)],
)
def test_zoom_speed_mapping(duration: float, expected: float) -> None:
    from benchmarks.annotation.review.playback import zoom_speed

    assert zoom_speed(duration) == pytest.approx(expected)


@pytest.mark.parametrize("speed", [0.25, 0.5, 1.0, 2.0])
def test_fractional_source_positions_and_end_boundaries(speed: float) -> None:
    samples = np.full((4000, 2), [0.75, -0.25], dtype=np.float32)
    pcm = VariableRatePCM(samples, 1000)
    pcm.reset(100.25, speed)
    pcm.target_speed = speed
    for frames in (1, 37, 64, 125):
        before = pcm.position
        output = np.empty((frames, 2), np.float32)
        span = pcm.render(output)
        assert span.first_sample == before
        assert pcm.position == pytest.approx(before + frames * speed)
        np.testing.assert_allclose(output, np.broadcast_to([0.75, -0.25], output.shape), atol=1e-6)
    pcm.reset(3990, speed)
    output = np.empty((100, 2), np.float32)
    span = pcm.render(output)
    assert span.frames == round(10 / speed) and pcm.position == 4000
    assert not np.any(output[span.frames :])


def test_rate_ramps_are_continuous_and_independent_of_callback_partitioning() -> None:
    samples = np.random.default_rng(4).normal(0, 0.1, (10_000, 2)).astype(np.float32)
    whole, split = VariableRatePCM(samples, 1000), VariableRatePCM(samples, 1000)
    for pcm in (whole, split):
        pcm.reset(100, 1.0)
        pcm.target_speed = 2.0
    expected = np.empty((1200, 2), np.float32)
    span = whole.render(expected)
    actual = np.empty_like(expected)
    offset = 0
    for frames in (7, 53, 128, 333, 679):
        before = split.position
        part = split.render(actual[offset : offset + frames])
        assert part.first_sample == before
        offset += frames
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert split.position == whole.position == pytest.approx(2400)
    assert split.speed == whole.speed == 2
    assert span.source_at(100) == 225 and span.speed_at(100) == 1.5
    for frame in (0, 30, 100, 200, 800, 1200):
        assert span.frame_at(span.source_at(frame)) == pytest.approx(frame)
    split.target_speed = 0.25
    next_span = split.render(actual[:50])
    assert next_span.first_sample == 2400 and next_span.speed == 2
    speed = split.speed
    split.target_speed = 1.0
    redirected = split.render(actual[:50])
    assert redirected.speed == speed
    assert redirected.first_sample == next_span.source_at(50)


def test_acceleration_filters_aliases_without_changing_channel_gains() -> None:
    time = np.arange(5000)
    samples = np.column_stack((np.sin(2 * np.pi * 0.04 * time), np.sin(2 * np.pi * 0.4 * time))).astype(np.float32)
    pcm = VariableRatePCM(samples, 1000)
    pcm.reset(1000.0, 2.0)
    pcm.target_speed = 2.0
    output = np.empty((1000, 2), np.float32)
    pcm.render(output)
    rms = np.sqrt(np.mean(output**2, axis=0))
    assert rms[0] == pytest.approx(2**-0.5, abs=0.01)
    assert rms[1] < 0.005


def test_variable_rate_clock_uses_audible_ramps_not_latest_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("benchmarks.annotation.review.playback.monotonic", lambda: 0.0)
    stream = OutputStream()
    opened = []
    monkeypatch.setattr(sd, "check_output_settings", lambda **_: None)
    monkeypatch.setattr(sd, "OutputStream", lambda **kwargs: opened.append(kwargs) or stream)
    player = PCMPlayback(np.zeros((5000, 2), np.float32), 1000)
    player.play()
    player.set_speed(2)
    output = np.empty((250, 2), np.float32)
    player._callback(output, 250, CallbackTime(100.1), sd.CallbackFlags())
    player.set_speed(0.25)
    player._callback(output, 250, CallbackTime(100.35), sd.CallbackFlags())
    stream.time = 100.2
    assert player.poll() == pytest.approx(0.125)
    assert player.speed == pytest.approx(1.5)
    assert player.event_time(0.125) == pytest.approx(100.2)
    player.pause()
    assert player.position == pytest.approx(0.125)
    player.play()
    player._callback(output, 250, CallbackTime(100.3), sd.CallbackFlags())
    assert player.clock.timings[-1].span.first_sample == pytest.approx(125)
    assert player.clock.timings[-1].span.speed == pytest.approx(1.5)
    generation = player.generation
    player.seek(2)
    assert player.position == 2 and not player.paused and player.generation > generation
    assert player.event_time(0.125) is None
    player._callback(output, 250, CallbackTime(100.4), sd.CallbackFlags())
    assert player.clock.timings[-1].span.first_sample == 2000
    assert len(opened) == 1  # Rate changes never reopen the output stream.
    player.close()


def test_kick_flash_duration_uses_dac_time_at_variable_speed() -> None:
    from benchmarks.annotation.review.playback import KickFlash

    flash = KickFlash()
    flash.timestamps = [1]
    assert not flash.update(0.9, playing=True, generation=1, playback_time=50)

    def crossing(position: float) -> float:
        return 50.2

    assert flash.update(1.001, playing=True, generation=1, playback_time=50.21, event_time=crossing)
    assert not flash.update(1.02, playing=True, generation=1, playback_time=50.28, event_time=crossing)
    flash.reset()
    assert not flash.update(0.9, playing=True, generation=1, playback_time=50)
    assert not flash.update(1.2, playing=True, generation=1, playback_time=51, event_time=crossing)


def test_paused_speed_targets_ramp_without_advancing_the_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [10.0]
    monkeypatch.setattr("benchmarks.annotation.review.playback.monotonic", lambda: now[0])
    stream = OutputStream()
    monkeypatch.setattr(sd, "check_output_settings", lambda **_: None)
    monkeypatch.setattr(sd, "OutputStream", lambda **_: stream)
    player = PCMPlayback(np.zeros((5000, 2), np.float32), 1000)
    player.seek(1.25)
    now[0] = 20  # Time before receiving a target must not count towards its ramp.
    player.set_speed(0.5)
    assert player.speed == 1
    now[0] = 20.1
    assert player.poll() == 1.25 and player.speed == pytest.approx(0.75)
    player.set_speed(2)
    now[0] = 20.2
    assert player.poll() == 1.25 and player.speed == pytest.approx(1.375)
    now[0] = 20.3
    assert player.speed == pytest.approx(1.375)  # No GUI poll before resuming.
    player.play()
    assert player.position == 1.25 and player.speed == 2
    player._callback(np.empty((100, 2), np.float32), 100, CallbackTime(100.1), sd.CallbackFlags())
    stream.time = 100.15
    assert player.poll() == pytest.approx(1.35) and player.speed == 2
    player.close()
