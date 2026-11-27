"""LED animation and DDP cadence independent of Qt painting and capture cadence."""

from collections import deque
from dataclasses import dataclass, replace
from threading import Event, Lock, Thread
from time import monotonic

import numpy as np
from numpy.typing import NDArray

from spectra.analysis import KickEvent
from spectra.capture import CaptureResult
from spectra.output.ddp import DDPOutput
from spectra.visualization import Visualization
from spectra.visualization.combined import CombinedVisualization
from spectra.visualization.frequency import FrequencyVisualization
from spectra.visualization.pulses import TravelingPulses

STALE_SECONDS = 0.5


@dataclass(frozen=True)
class OutputSettings:
    fps: int = 30
    mode: int = 0
    mode_revision: int = 0
    mode_started_at: float | None = None
    capturing: bool = False
    generation: int = 0
    address: str | None = None
    output_revision: int = 0


@dataclass(frozen=True)
class OutputState:
    frame: NDArray[np.uint8]
    enabled: bool = False
    status: str = "Disabled"
    error: str | None = None


class LedOutputWorker:
    """Own visualization state and UDP on one thread; publish a latest preview.

    Capture publishes an immutable result and at most one event in a short lock.
    Neither capture nor output waits for the GUI. The pending event deque is
    bounded like CaptureWorker's presentation queue; stale continuous values are
    superseded. The GUI changes immutable settings, never touches the socket.
    """

    def __init__(self, led_count: int = 300) -> None:
        self.led_count = led_count
        self._black = np.zeros((led_count, 3), dtype=np.uint8)
        self._black.setflags(write=False)
        self._state = OutputState(self._black)
        self._settings = OutputSettings()
        self._lock = Lock()
        self._wake, self._stop = Event(), Event()
        self._latest: CaptureResult | None = None
        self._kicks: deque[KickEvent] = deque(maxlen=32)
        self._thread = Thread(target=self._run, name="spectra-led-output")
        self._visualization: Visualization = FrequencyVisualization(led_count=led_count)
        self._output: DDPOutput | None = None
        self._applied_output = -1
        self._generation = -1
        self._mode_revision = 0
        self._blackout_pending = False
        self._rendered_time = 0.0

    @property
    def state(self) -> OutputState:
        return self._state

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def publish(self, result: CaptureResult, kick: KickEvent | None) -> None:
        with self._lock:
            if not self._settings.capturing:
                return
            self._latest = result
            if kick is not None:
                self._kicks.append(kick)
        # New continuous input does not wake a render early or change its FPS.

    def set_capturing(self, capturing: bool) -> None:
        with self._lock:
            self._settings = replace(
                self._settings, capturing=capturing, generation=self._settings.generation + 1, mode_started_at=None
            )
            self._latest = None
            self._kicks.clear()
        self._wake.set()

    def set_frame_rate(self, fps: int) -> None:
        if fps <= 0:
            raise ValueError("Frame rate must be positive.")
        with self._lock:
            self._settings = replace(self._settings, fps=fps)
        self._wake.set()

    def set_mode(self, mode: int) -> None:
        if mode not in (0, 1, 2):
            raise ValueError("Unknown LED visualization.")
        with self._lock:
            if mode == self._settings.mode:
                return
            anchor = (
                None
                if self._latest is None
                else self._latest.spectrum.timestamp + monotonic() - self._latest.produced_at
            )
            self._settings = replace(
                self._settings, mode=mode, mode_revision=self._settings.mode_revision + 1, mode_started_at=anchor
            )
            self._kicks.clear()
        self._wake.set()

    def set_output(self, address: str | None) -> None:
        with self._lock:
            self._settings = replace(
                self._settings, address=address, output_revision=self._settings.output_revision + 1
            )
        self._wake.set()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            raise RuntimeError("LED output did not stop within the shutdown timeout.")

    def _close_output(self) -> None:
        output, self._output = self._output, None
        self._blackout_pending = False
        if output is not None:
            try:
                sent = output.close()
            except OSError as error:
                self._state = replace(self._state, enabled=False, status=f"Disabled · blackout failed: {error}")
            else:
                self._state = replace(
                    self._state, enabled=False, status="Disabled" if sent else "Disabled · blackout dropped"
                )

    def _configure(self, settings: OutputSettings) -> None:
        if settings.generation != self._generation:
            self._generation = settings.generation
            self._visualization.reset()
            self._rendered_time = 0
        if settings.mode_revision != self._mode_revision:
            self._mode_revision = settings.mode_revision
            self._visualization = (
                FrequencyVisualization(led_count=self.led_count)
                if settings.mode == 0
                else TravelingPulses(led_count=self.led_count)
                if settings.mode == 1
                else CombinedVisualization(led_count=self.led_count)
            )
        if settings.output_revision != self._applied_output:
            self._applied_output = settings.output_revision
            self._close_output()
            if settings.address is not None:
                try:
                    self._output = DDPOutput(settings.address, led_count=self.led_count)
                except (ValueError, OSError) as error:
                    self._state = replace(self._state, enabled=False, status=f"Output error: {error}")
                else:
                    self._blackout_pending = True
                    self._state = replace(self._state, enabled=True, status="Enabled")

    def _render(
        self,
        settings: OutputSettings,
        latest: CaptureResult | None,
        kicks: tuple[KickEvent, ...],
        now: float,
    ) -> None:
        active = settings.capturing and latest is not None and now - latest.produced_at < STALE_SECONDS
        if not active:
            self._visualization.reset()
            self._rendered_time = 0
            frame = self._black
        else:
            assert latest is not None
            # Extrapolate in audio time using elapsed wall time since production,
            # so delayed publication cannot restart or shift a kick pulse.
            self._rendered_time = max(self._rendered_time, latest.spectrum.timestamp + now - latest.produced_at)
            kicks = tuple(
                k for k in kicks if settings.mode_started_at is None or k.timestamp > settings.mode_started_at
            )
            frame = self._visualization.render(latest.intensities, kicks, self._rendered_time)
            frame.setflags(write=False)
        # Active silence still runs at the selected cadence. Inactive capture
        # sends one best-effort blackout, also when enabling a new destination.
        if self._output is not None and (active or self._blackout_pending):
            self._blackout_pending = active
            try:
                sent = self._output.send(frame)
            except OSError as error:
                self._close_output()
                self._state = replace(self._state, enabled=False, status=f"Output error: {error}")
            else:
                self._state = replace(
                    self._state, status="UDP sent · delivery unverified" if sent else "Frame dropped · UDP buffer busy"
                )
        self._state = replace(self._state, frame=frame)

    def _run(self) -> None:
        deadline = monotonic()
        try:
            while not self._stop.is_set():
                self._wake.clear()
                with self._lock:
                    settings, latest = self._settings, self._latest
                    kicks = tuple(self._kicks)
                    self._kicks.clear()
                self._configure(settings)
                now = monotonic()
                self._render(settings, latest, kicks, now)
                interval = 1 / settings.fps
                deadline = max(deadline + interval, now)
                if self._wake.wait(max(0, deadline - monotonic())):
                    deadline = monotonic()
        except Exception as error:
            self._state = OutputState(self._black, error=str(error) or type(error).__name__)
        finally:
            self._close_output()
