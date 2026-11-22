"""Capture and analysis on a caller-controlled thread, without Qt."""

from threading import Event, Lock, Thread

import numpy as np
from numpy.typing import NDArray

from spectra.analysis import SpectrumAnalyzer, SpectrumResult
from spectra.audio import AudioSource, AudioSourceError


class CaptureWorker:
    """Own a source and analyzer, publishing only the latest continuous result.

    The caller starts this worker once and requests cancellation when done.
    All source lifecycle operations run on the worker except request_stop(),
    whose audio contract explicitly allows cross-thread cancellation.
    """

    def __init__(self, source: AudioSource) -> None:
        self._source = source
        self._analyzer = SpectrumAnalyzer(sample_rate=source.sample_rate)
        self._stop_requested = Event()
        self._lock = Lock()
        self._latest: SpectrumResult | None = None
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

    def take_latest(self) -> tuple[SpectrumResult | None, str | None]:
        """Consume the current result; terminal errors remain available."""
        with self._lock:
            result = self._latest
            self._latest = None
            return result, self._error

    def _run(self) -> None:
        try:
            if self._stop_requested.is_set():
                return
            self._source.start()
            while not self._stop_requested.is_set():
                results = self._analyzer.process(self._source.read())
                if results:
                    with self._lock:
                        self._latest = results[-1]
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
