"""The GUI delivers rendered frames to its preview and a local UDP receiver."""

import socket
from time import monotonic

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from spectra.audio import AudioSourceInfo
from spectra.output import ddp
from spectra.ui.window import SpectraWindow
from spectra.visualization.frequency import frequency_frame


def test_wled_mirrors_rendered_frames_and_disabling_preserves_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QApplication.instance() or QApplication([])
    source = AudioSourceInfo("test.monitor", "Test monitor")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        monkeypatch.setattr(ddp, "DDP_PORT", receiver.getsockname()[1])
        window = SpectraWindow([source], source)
        window._timer.stop()
        try:
            window._wled_address.setText("127.0.0.1")
            window._wled_enabled.setChecked(True)
            receiver.recv(2048)  # Enabling output publishes the current, initially black frame.

            intensities = np.linspace(0.2, 0.8, 24)
            window._latest_intensities = intensities
            window._latest_audio_timestamp = 1.0
            window._last_result_at = monotonic()
            window._render_led_frame(window._last_result_at)

            expected = frequency_frame(intensities)
            packet = receiver.recv(2048)
            received = np.frombuffer(packet[10:], dtype=np.uint8).reshape(300, 3)
            np.testing.assert_array_equal(received, expected)
            np.testing.assert_array_equal(window._leds._frame, expected)

            window._wled_enabled.setChecked(False)
            black = receiver.recv(2048)
            black_frame = np.frombuffer(black[10:], dtype=np.uint8).reshape(300, 3)
            np.testing.assert_array_equal(black_frame, np.zeros_like(expected))
            np.testing.assert_array_equal(window._leds._frame, expected)

            # Rendering keeps updating the actual preview widget after disabling output.
            window._latest_intensities = intensities[::-1].copy()
            window._render_led_frame(monotonic())
            np.testing.assert_array_equal(window._leds._frame, frequency_frame(intensities[::-1]))
        finally:
            window.shutdown()
            window.close()
            application.processEvents()
