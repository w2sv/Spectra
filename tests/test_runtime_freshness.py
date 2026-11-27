"""One presentation freshness gate covers plots, kick feedback and LED preview."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from spectra.analysis import KickEvent, SpectrumResult
from spectra.audio import AudioSourceInfo
from spectra.audio.history import PCMHistory
from spectra.capture import CaptureResult
from spectra.output.worker import OutputState
from spectra.ui.window import SpectraWindow


class CaptureSnapshot:
    """Controllable capture delivery, retaining history between GUI polls."""

    def __init__(self) -> None:
        self.band_edges = np.geomspace(40, 16000, 25)
        self.history = PCMHistory(48000)
        self.waveform_kicks: tuple[KickEvent, ...] = ()
        self.latest: CaptureResult | None = None
        self.pending: tuple[KickEvent, ...] = ()
        self.is_alive = False

    def start(self) -> None:
        self.is_alive = True

    def request_stop(self) -> None:
        self.is_alive = False

    def join(self) -> None:
        pass

    def take_latest(self) -> tuple[CaptureResult | None, tuple[KickEvent, ...], None]:
        result, events = self.latest, self.pending
        self.latest, self.pending = None, ()
        return result, events, None

    def publish(self, produced_at: float, kick: KickEvent | None = None) -> None:
        self.history.append(np.ones((1024, 2), dtype=np.float32))
        if kick is not None:
            self.waveform_kicks += (kick,)
            self.pending = (kick,)
        self.latest = CaptureResult(
            SpectrumResult(1, np.ones(24)), np.ones(24), len(self.waveform_kicks), produced_at=produced_at
        )


def test_stale_capture_stays_clear_recovers_and_restarts_without_previous_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    now = [10.0]
    monkeypatch.setattr("spectra.ui.window.monotonic", lambda: now[0])
    first, second = CaptureSnapshot(), CaptureSnapshot()
    sessions = iter((first, second))
    monkeypatch.setattr("spectra.ui.window.CaptureWorker", lambda *_args, **_kwargs: next(sessions))
    source = AudioSourceInfo("test.monitor", "Test monitor")
    window = SpectraWindow([source], source)
    window._timer.stop()
    # Hold an old, nonblack output snapshot to ensure stale GUI polls cannot
    # restore it while the independently scheduled output worker catches up.
    window._led_output.close()
    monkeypatch.setattr(window._led_output, "_state", OutputState(np.full((300, 3), 255, dtype=np.uint8)))

    def assert_clear() -> None:
        assert window._plots._history is None and not window._plots._kicks
        assert window._plots._features is None
        assert np.all(window._plots.spectrum.values == -80)
        assert not np.any(window._plots.intensity.values)
        assert not np.any(window._leds._frame)
        assert not window._kick_indicator.styleSheet()

    try:
        window._start_capture()
        window._update_capture()
        assert_clear()
        first.publish(now[0], KickEvent(0.01, 1))
        window._update_capture()
        assert window._plots._history is first.history and window._plots._kicks == first.waveform_kicks
        assert window._plots._features is not None
        assert np.any(window._leds._frame) and window._kick_indicator.styleSheet()
        now[0] += 0.51
        for _ in range(3):
            window._update_capture()
            assert_clear()
        # Consuming a result produced before the timeout cannot renew freshness.
        first.publish(now[0] - 0.51, KickEvent(0.02, 1))
        window._update_capture()
        assert_clear()
        window._waveform_enabled.setChecked(False)
        window._waveform_enabled.setChecked(True)
        assert_clear()

        first.publish(now[0], KickEvent(0.03, 1))
        window._update_capture()
        assert window._plots._history is first.history and window._plots._kicks == first.waveform_kicks
        assert np.any(window._leds._frame) and window._kick_indicator.styleSheet()
        window._stop_capture()
        assert_clear()
        window._update_capture()
        assert window._worker is None
        assert_clear()

        window._start_capture()
        window._update_capture()
        assert_clear()
        second.publish(now[0])  # A new session has no preceding session's kick markers.
        window._update_capture()
        assert window._plots._history is second.history and not window._plots._kicks
        assert not window._kick_indicator.styleSheet()
    finally:
        window.shutdown()
        window.close()
        app.processEvents()
