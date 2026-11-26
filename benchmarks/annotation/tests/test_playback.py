"""DAC clock and transition invariants, without requiring an audio device."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import sounddevice as sd
from scipy.io import wavfile

from benchmarks.annotation.review.playback import BufferTiming, PCMPlayback, PlaybackClock
from benchmarks.fixtures import load_wav


@dataclass
class CallbackTime:
    outputBufferDacTime: float


def test_dac_clock_waits_for_audibility_and_bounds_each_buffer() -> None:
    clock = PlaybackClock(1000, 3000, 3)
    clock.reset(1, playing=True)
    clock.timings.append(BufferTiming(clock.generation, 1000, 100, 50.1))
    clock.timings.append(BufferTiming(clock.generation, 1100, 100, 50.2))
    assert clock.position(50.05) == 1  # samples submitted, not yet audible
    assert clock.position(50.15) == pytest.approx(1.05)
    assert clock.position(50.25) == pytest.approx(1.15)
    assert clock.position(51) == 1.2  # do not extrapolate beyond submitted audio
    old_generation = clock.generation
    clock.reset(2, playing=False)
    clock.timings.append(BufferTiming(old_generation, 0, 100, 51))
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
    assert player.poll() == 0.2
    assert player.reliable_buffering and first.closed
    assert requested == ["high", 0.1]
    second.active = False
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        player.poll()
    assert player.paused and player.position == 0.2
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
