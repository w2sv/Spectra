"""Actual UDP output continues without GUI polling and preserves frame semantics."""

import socket
from time import monotonic, sleep

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from spectra.analysis import KickEvent, SpectrumResult
from spectra.audio import AudioSourceInfo
from spectra.capture import CaptureResult
from spectra.output import ddp
from spectra.output.worker import LedOutputWorker
from spectra.ui.window import SpectraWindow
from spectra.visualization.frequency import frequency_frame
from spectra.visualization.pulses import TravelingPulses


def receive_matching(receiver: socket.socket, expected: np.ndarray | None = None) -> np.ndarray:
    deadline = monotonic() + 2
    while monotonic() < deadline:
        packet = receiver.recv(2048)
        frame = np.frombuffer(packet[10:], dtype=np.uint8).reshape(300, 3)
        if expected is None and np.any(frame) or expected is not None and np.array_equal(frame, expected):
            return frame
    raise AssertionError("Expected UDP framebuffer did not arrive")


def test_wled_and_preview_share_output_frames_without_gui_driving_transmission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = AudioSourceInfo("test.monitor", "Test monitor")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(2)
        monkeypatch.setattr(ddp, "DDP_PORT", receiver.getsockname()[1])
        window = SpectraWindow([source], source)
        window._timer.stop()
        try:
            window._wled_address.setText("127.0.0.1")
            window._wled_enabled.setChecked(True)
            black = np.zeros((300, 3), dtype=np.uint8)
            receive_matching(receiver, black)
            intensities = np.linspace(0.2, 0.8, 24)
            worker = window._led_output
            worker.set_capturing(True)
            worker.publish(CaptureResult(SpectrumResult(1, np.ones(24)), intensities, produced_at=monotonic()), None)
            expected = frequency_frame(intensities)
            # No Qt event processing: real packets still arrive at output cadence.
            receive_matching(receiver, expected)
            receive_matching(receiver, expected)
            window._update_output(fresh=True)
            np.testing.assert_array_equal(window._leds._frame, expected)
            window._wled_enabled.setChecked(False)
            receive_matching(receiver, black)
            window._update_output(fresh=True)
            np.testing.assert_array_equal(window._leds._frame, expected)
            worker.publish(
                CaptureResult(SpectrumResult(1.1, np.ones(24)), intensities[::-1].copy(), produced_at=monotonic()), None
            )
            deadline = monotonic() + 2
            while not np.array_equal(worker.state.frame, frequency_frame(intensities[::-1])) and monotonic() < deadline:
                sleep(0.002)
            window._update_output(fresh=True)
            np.testing.assert_array_equal(window._leds._frame, frequency_frame(intensities[::-1]))
        finally:
            window.shutdown()
            window.close()
            app.processEvents()


def test_frame_rate_changes_preserve_active_pulses_and_stale_capture_blacks_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(2)
        monkeypatch.setattr(ddp, "DDP_PORT", receiver.getsockname()[1])
        worker = LedOutputWorker()
        worker.start()
        try:
            worker.set_mode(1)
            worker.set_capturing(True)
            worker.set_output("127.0.0.1")
            values = np.zeros(24)
            worker.publish(
                CaptureResult(SpectrumResult(1, values), values, produced_at=monotonic()), KickEvent(0.99, 1)
            )
            receive_matching(receiver)
            for fps in (60, 90, 120, 30):
                worker.set_frame_rate(fps)
                receive_matching(receiver)
            # The worker itself clears stale capture, even without GUI updates.
            receive_matching(receiver, np.zeros((300, 3), dtype=np.uint8))
            receiver.settimeout(0.12)
            with pytest.raises(TimeoutError):
                receiver.recv(2048)
            receiver.settimeout(2)
            # A new capture resets the previous stream's mode-switch time anchor.
            worker.publish(CaptureResult(SpectrumResult(100, values), values, produced_at=monotonic()), None)
            worker.set_mode(2)
            worker.set_capturing(False)
            worker.set_capturing(True)
            worker.publish(
                CaptureResult(SpectrumResult(1, values), values, produced_at=monotonic()), KickEvent(0.99, 1)
            )
            receive_matching(receiver)
        finally:
            worker.close()
        assert not worker.is_alive


def test_rapid_mode_round_trip_discards_pulses_and_pending_kicks(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [10.0]
    monkeypatch.setattr("spectra.output.worker.monotonic", lambda: now[0])
    worker = LedOutputWorker()
    worker.set_capturing(True)
    worker.set_mode(2)
    values = np.zeros(24)

    def tick() -> None:
        # Exercise a complete worker iteration without scheduler-dependent timing.
        with worker._lock:
            settings, latest = worker._settings, worker._latest
            kicks = tuple(worker._kicks)
            worker._kicks.clear()
        worker._configure(settings)
        worker._render(settings, latest, kicks, now[0])

    worker.publish(CaptureResult(SpectrumResult(1, values), values, produced_at=now[0]), KickEvent(0.99, 1))
    tick()
    assert np.any(worker.state.frame)
    visualization = worker._visualization
    worker.set_mode(2)  # Reassigning the requested mode must preserve animation.
    worker.set_frame_rate(120)
    now[0] += 0.02
    tick()
    assert worker._visualization is visualization and np.any(worker.state.frame)

    worker.publish(CaptureResult(SpectrumResult(1.02, values), values, produced_at=now[0]), KickEvent(1.01, 1))
    worker.set_mode(0)
    worker.set_mode(2)  # Both transitions occur before the worker applies settings.
    now[0] += 0.02
    tick()
    assert not np.any(worker.state.frame)
    tick()  # No replay on the subsequent iteration either.
    assert not np.any(worker.state.frame)
    worker.publish(CaptureResult(SpectrumResult(1.06, values), values, produced_at=now[0]), KickEvent(1.05, 1))
    tick()
    assert np.any(worker.state.frame)


def test_inactive_output_suspends_but_active_silence_keeps_transmitting(monkeypatch: pytest.MonkeyPatch) -> None:
    black = np.zeros((300, 3), dtype=np.uint8)

    def assert_suspended(receiver: socket.socket) -> None:
        receiver.settimeout(0.12)
        with pytest.raises(TimeoutError):
            receiver.recv(2048)
        receiver.settimeout(2)

    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as first,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as second,
    ):
        for receiver in (first, second):
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(2)
        monkeypatch.setattr(ddp, "DDP_PORT", first.getsockname()[1])
        worker = LedOutputWorker()
        worker.start()
        try:
            worker.set_output("127.0.0.1")
            receive_matching(first, black)
            assert_suspended(first)
            worker.set_frame_rate(60)
            worker.set_mode(2)
            assert_suspended(first)

            # Reconfiguration blacks out the old destination and initializes the new one.
            monkeypatch.setattr(ddp, "DDP_PORT", second.getsockname()[1])
            worker.set_output("127.0.0.1")
            receive_matching(first, black)
            receive_matching(second, black)
            assert_suspended(first)
            assert_suspended(second)
            worker.set_capturing(True)
            values = np.zeros(24)
            worker.publish(
                CaptureResult(SpectrumResult(1, values), values, produced_at=monotonic() - 0.51), KickEvent(0.99, 1)
            )
            assert_suspended(second)  # Delayed stale delivery must not resume DDP.
            worker.publish(CaptureResult(SpectrumResult(1, values), values, produced_at=monotonic()), None)
            for _ in range(3):
                receive_matching(second, black)
            worker.set_capturing(False)
            receive_matching(second, black)
            assert_suspended(second)
            worker.set_output(None)
            receive_matching(second, black)
            assert_suspended(second)
            worker.set_output("127.0.0.1")
            receive_matching(second, black)
            assert_suspended(second)
        finally:
            worker.close()
        receive_matching(second, black)
        assert_suspended(second)


def test_delayed_publication_preserves_audio_timed_pulses_and_mode_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [10.2]
    monkeypatch.setattr("spectra.output.worker.monotonic", lambda: now[0])
    worker = LedOutputWorker()
    worker.set_capturing(True)
    worker.set_mode(1)
    values = np.zeros(24)
    result = CaptureResult(SpectrumResult(1.04, values), values, produced_at=10)
    kick = KickEvent(1, 1)
    worker.publish(result, kick)  # Published 200 ms after production.
    worker._configure(worker._settings)
    now[0] = 10.25
    worker._render(worker._settings, result, (kick,), now[0])
    expected = TravelingPulses().render(values, (kick,), 1.29)
    assert np.any(expected)
    np.testing.assert_array_equal(worker.state.frame, expected)

    now[0] = 10.3
    worker.set_mode(0)
    worker.set_mode(1)
    worker._configure(worker._settings)
    # Audio time at the switch is 1.34, so a delayed event at 1.30 is old.
    worker._render(worker._settings, result, (KickEvent(1.30, 1),), now[0])
    assert not np.any(worker.state.frame)
