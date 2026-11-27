"""Capture and analysis on a caller-controlled thread, without Qt."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from time import monotonic

import numpy as np
from numpy.typing import NDArray

from spectra.analysis import KickEvent, KickOnsetDetector, SpectrumAnalyzer, SpectrumResult
from spectra.audio import AudioSource, AudioSourceError
from spectra.audio.history import PCMHistory
from spectra.visualization.intensity import BandIntensityProcessor, BandIntensitySettings


@dataclass(frozen=True)
class CaptureResult:
    """Matching raw spectrum, read-only float64 intensities, and session kick count.

    Intensities have shape (bands,). kick_count includes all detections,
    including older events evicted from the bounded pending-event buffer.
    produced_at is monotonic wall time, distinct from spectrum/kick audio time.
    """

    spectrum: SpectrumResult
    intensities: NDArray[np.float64]
    kick_count: int = 0
    produced_at: float = field(kw_only=True)


class CaptureWorker:
    """Own a source and analyzer, publishing only the latest continuous result.

    The caller starts this worker once and requests cancellation when done.
    All source lifecycle operations run on the worker except request_stop(),
    whose audio contract explicitly allows cross-thread cancellation.
    """

    def __init__(
        self,
        source: AudioSource,
        settings: BandIntensitySettings | None = None,
        *,
        on_result: Callable[[CaptureResult, KickEvent | None], None] | None = None,
    ) -> None:
        settings = BandIntensitySettings() if settings is None else settings
        self._source = source
        self._analyzer = SpectrumAnalyzer(sample_rate=source.sample_rate)
        self._processor = BandIntensityProcessor(len(self._analyzer.band_edges) - 1, settings)
        self._kick_detector = KickOnsetDetector(self._analyzer.band_edges)
        self._settings = settings
        self.history = PCMHistory(source.sample_rate)
        self._history_kicks: deque[KickEvent] = deque(maxlen=self.history.capacity // self._analyzer.hop_size + 2)
        self.waveform_kicks: tuple[KickEvent, ...] = ()
        self._on_result = on_result
        self._stop_requested = Event()
        self._lock = Lock()
        self._latest: CaptureResult | None = None
        self._pending_kicks: deque[KickEvent] = deque(maxlen=32)
        self._kick_count = 0
        self._error: str | None = None
        self._thread = Thread(target=self._run, name="spectra-capture")

    @property
    def band_edges(self) -> NDArray[np.float64]:
        return self._analyzer.band_edges

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def request_stop(self) -> None:
        """Request shutdown without blocking the presentation thread."""
        self._stop_requested.set()
        self._source.request_stop()

    def join(self, timeout: float = 3.0) -> None:
        """Wait a bounded time for cleanup, reporting an unresponsive source."""
        self._thread.join(timeout)
        if self.is_alive:
            raise AudioSourceError("Audio capture did not stop within the shutdown timeout.")

    def update_settings(self, settings: BandIntensitySettings) -> None:
        """Publish an immutable settings snapshot; only the worker applies it."""
        with self._lock:
            self._settings = settings

    def take_latest(self) -> tuple[CaptureResult | None, tuple[KickEvent, ...], str | None]:
        """Consume the latest frame and drain pending events atomically.

        Keep the newest 32 events if presentation stalls; the session count
        includes every detection even if that bounded buffer overflows.
        Terminal errors remain available.
        """
        with self._lock:
            result = self._latest
            self._latest = None
            events = tuple(self._pending_kicks)
            self._pending_kicks.clear()
            return result, events, self._error

    def _run(self) -> None:
        try:
            if self._stop_requested.is_set():
                return
            self._source.start()
            while not self._stop_requested.is_set():
                samples = self._source.read()
                self.history.append(samples)
                results = self._analyzer.process(samples)
                oldest = max(0, self.history.written - self.history.capacity) / self.history.sample_rate
                pruned = False
                while self._history_kicks and self._history_kicks[0].timestamp < oldest:
                    self._history_kicks.popleft()
                    pruned = True
                if pruned:
                    self.waveform_kicks = tuple(self._history_kicks)
                for result in results:
                    kick = self._kick_detector.process(result)
                    with self._lock:
                        settings = self._settings
                    self._processor.update_settings(settings)
                    intensities = self._processor.process(result.band_energies, result.timestamp)
                    if kick is not None:
                        self._history_kicks.append(kick)
                        self.waveform_kicks = tuple(self._history_kicks)
                    with self._lock:
                        if kick is not None:
                            self._pending_kicks.append(kick)
                            self._kick_count += 1
                        latest = CaptureResult(result, intensities, self._kick_count, produced_at=monotonic())
                        self._latest = latest
                    if self._on_result is not None:
                        self._on_result(latest, kick)
        except Exception as error:
            if not self._stop_requested.is_set():
                with self._lock:
                    self._error = str(error) or type(error).__name__
        finally:
            try:
                self._source.stop()
            except Exception as error:
                with self._lock:
                    self._error = f"Cannot release audio capture: {error}"
