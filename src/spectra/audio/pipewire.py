"""Output-monitor capture through PipeWire's PulseAudio compatibility server."""

import json
import select
import subprocess
from threading import Event
from typing import IO

from spectra.audio import DEFAULT_READ_FRAMES, DEFAULT_SAMPLE_RATE, AudioSamples, AudioSourceError, AudioSourceInfo
from spectra.audio.pcm import read_float32_frames

DEFAULT_OUTPUT_MONITOR = AudioSourceInfo("@DEFAULT_MONITOR@", "Default output monitor")


class _CancellableStream:
    """Adapt a raw pipe to the PCM reader without blocking cancellation."""

    def __init__(self, stream: IO[bytes], cancelled: Event) -> None:
        self._stream = stream
        self._cancelled = cancelled

    def read(self, size: int = -1) -> bytes:
        while not self._cancelled.is_set():
            readable, _, _ = select.select([self._stream], [], [], 0.1)
            if readable:
                return self._stream.read(size)
        raise AudioSourceError("Audio capture was stopped.")


def discover_output_monitors() -> list[AudioSourceInfo]:
    """List output monitors by persistent source name, never numeric index."""
    try:
        result = subprocess.run(
            ["pactl", "--format=json", "list", "sinks"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except FileNotFoundError as error:
        raise AudioSourceError("pactl is missing. Install Ubuntu's pulseaudio-utils package.") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip()
        raise AudioSourceError(
            f"Cannot discover output monitors. Is PipeWire's PulseAudio server running? {detail}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise AudioSourceError("Timed out while discovering output monitors.") from error
    except OSError as error:
        raise AudioSourceError(f"Cannot run pactl: {error}") from error

    try:
        sinks = json.loads(result.stdout)
        if not isinstance(sinks, list):
            raise ValueError("Expected a list of output devices.")
        monitors: list[AudioSourceInfo] = []
        for sink in sinks:
            identifier = sink["monitor_source"]
            description = sink["description"]
            if not isinstance(identifier, str) or not identifier or not isinstance(description, str):
                raise ValueError("Expected an output-monitor source name and description.")
            monitors.append(AudioSourceInfo(identifier, f"Monitor of {description}"))
        return monitors
    except (ValueError, KeyError, TypeError) as error:
        raise AudioSourceError(f"Invalid output-monitor data from pactl: {error}") from error


class PipeWireSource:
    """Blocking 48 kHz stereo capture; the caller owns start/read/stop.

    No routing or playback settings are changed. The default monitor is
    resolved by the server at capture start; explicit source names can be
    selected using discovery results.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 2

    def __init__(self, source: AudioSourceInfo = DEFAULT_OUTPUT_MONITOR) -> None:
        self.info = source
        self._process: subprocess.Popen[bytes] | None = None
        self._cancelled = Event()

    def start(self) -> None:
        """Launch parec with raw little-endian float32, left/right PCM."""
        if self._process is not None:
            raise AudioSourceError("Audio capture has already started.")
        self._cancelled.clear()

        try:
            # Inherit stderr so server errors stay visible without an undrained pipe.
            self._process = subprocess.Popen(
                [
                    "parec",
                    f"--device={self.info.identifier}",
                    "--raw",
                    "--format=float32le",
                    f"--rate={self.sample_rate}",
                    f"--channels={self.channels}",
                    "--channel-map=front-left,front-right",
                    "--latency-msec=50",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as error:
            raise AudioSourceError("parec is missing. Install Ubuntu's pulseaudio-utils package.") from error
        except OSError as error:
            raise AudioSourceError(f"Cannot start parec: {error}") from error

    def read(self, frames: int = DEFAULT_READ_FRAMES) -> AudioSamples:
        """Read complete frames, or report capture failure after draining PCM."""
        if frames <= 0:
            raise ValueError("Frame count must be positive.")
        process = self._process
        if process is None or process.stdout is None:
            raise AudioSourceError("Start audio capture before reading samples.")

        try:
            stream = _CancellableStream(process.stdout, self._cancelled)
            samples = read_float32_frames(stream, frames, self.channels)
        except (OSError, ValueError) as error:
            self.stop()
            raise AudioSourceError(f"Cannot read captured PCM: {error}") from error

        if len(samples) == 0:
            # EOF can precede process exit very briefly; allow it to be reaped.
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            self.stop()
            raise AudioSourceError(
                f"parec stopped unexpectedly (exit status {process.returncode}). "
                "Check that PipeWire's PulseAudio server and the selected output are available."
            )
        return samples

    def request_stop(self) -> None:
        """Wake a blocked read within 100 ms; the owner performs cleanup."""
        self._cancelled.set()

    def stop(self) -> None:
        """Terminate and reap parec, killing it if graceful shutdown times out."""
        process = self._process
        if process is None:
            return

        try:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()
            self._process = None
