"""Main-thread presentation and controls for live spectral measurements."""

from dataclasses import replace
from time import monotonic

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spectra.audio import AudioSourceInfo
from spectra.audio.pipewire import PipeWireSource
from spectra.capture import CaptureWorker
from spectra.output.worker import STALE_SECONDS, LedOutputWorker
from spectra.ui.led_preview import LedPreview
from spectra.ui.runtime_plots import RuntimePlots
from spectra.visualization.frequency import power_to_decibels
from spectra.visualization.intensity import BandIntensitySettings

SPECTRUM_MIN_DB = -80.0
SPECTRUM_MAX_DB = 0.0
KICK_FLASH_SECONDS = 0.12


class SpectraWindow(QMainWindow):
    """Poll a latest-result slot; all widgets and painting stay on this thread."""

    def __init__(
        self,
        sources: list[AudioSourceInfo],
        selected: AudioSourceInfo,
        discovery_error: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Spectra")
        self.resize(1000, 760)
        self.setMinimumSize(680, 600)
        self._worker: CaptureWorker | None = None
        self._stopping = False
        self._closing = False
        self._last_produced_at: float | None = None
        self._intensity_settings = BandIntensitySettings()
        self._kick_flash_until: float | None = None
        self._kick_count = 0
        self._kick_flash_style = (
            f"background-color: {self.palette().color(QPalette.ColorRole.Highlight).name()};"
            f"color: {self.palette().color(QPalette.ColorRole.HighlightedText).name()};"
        )

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Audio source:"))
        self._sources = QComboBox()
        for source in sources:
            self._sources.addItem(source.display_name, source)
            if source.identifier == selected.identifier:
                self._sources.setCurrentIndex(self._sources.count() - 1)
        controls.addWidget(self._sources, 1)
        self._start_button = QPushButton("Start")
        self._start_button.clicked.connect(self._start_capture)
        controls.addWidget(self._start_button)
        self._stop_button = QPushButton("Stop")
        self._stop_button.clicked.connect(self._stop_capture)
        controls.addWidget(self._stop_button)
        self._status = QLabel("Stopped")
        controls.addWidget(self._status)
        layout.addLayout(controls)
        self._error = QLabel(f"Error: {discovery_error}" if discovery_error else "")
        self._error.setWordWrap(True)
        self._error.setVisible(bool(discovery_error))
        layout.addWidget(self._error)

        self._sensitivity = QDoubleSpinBox()
        self._frequency_balance = QDoubleSpinBox()
        self._attack = QSpinBox()
        self._release = QSpinBox()
        for control, value in (
            (self._sensitivity, self._intensity_settings.sensitivity_db),
            (self._frequency_balance, self._intensity_settings.frequency_balance_db),
        ):
            control.setRange(-24, 24)
            control.setDecimals(1)
            control.setSingleStep(1)
            control.setSuffix(" dB")
            control.setValue(value)
        for control, maximum, step, seconds in (
            (self._attack, 500, 5, self._intensity_settings.attack_seconds),
            (self._release, 3000, 20, self._intensity_settings.release_seconds),
        ):
            control.setRange(0, maximum)
            control.setSingleStep(step)
            control.setSuffix(" ms")
            control.setValue(round(seconds * 1000))
        self._sensitivity.setToolTip("Additional gain for LED brightness; the spectrum graph stays raw.")
        self._frequency_balance.setToolTip(
            "High-minus-low gain across log-spaced bands. Positive values reduce bass and boost highs equally."
        )
        self._attack.setToolTip("Intensity rise time constant; zero makes increases instantaneous.")
        self._release.setToolTip("Intensity fall time constant; zero makes decreases instantaneous.")
        processing_controls = QGridLayout()
        for row, column, label, control in (
            (0, 0, "LED sensitivity:", self._sensitivity),
            (0, 2, "Attack:", self._attack),
            (1, 0, "Frequency balance:", self._frequency_balance),
            (1, 2, "Release:", self._release),
        ):
            processing_controls.addWidget(QLabel(label), row, column)
            processing_controls.addWidget(control, row, column + 1)
            control.valueChanged.connect(self._update_intensity_settings)
        processing_controls.setColumnStretch(1, 1)
        processing_controls.setColumnStretch(3, 1)
        layout.addLayout(processing_controls)

        kick_feedback = QHBoxLayout()
        self._kick_indicator = QLabel("Kick")
        self._kick_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._kick_indicator.setContentsMargins(8, 3, 8, 3)
        kick_feedback.addWidget(self._kick_indicator)
        self._kick_count_label = QLabel("Kicks: 0")
        kick_feedback.addWidget(self._kick_count_label)
        kick_feedback.addStretch()
        kick_feedback.addWidget(QLabel("Visualization:"))
        self._visualization_selector = QComboBox()
        self._visualization_selector.addItems(["Frequency Spectrum", "Kick Pulses", "Combined"])
        kick_feedback.addWidget(self._visualization_selector)
        kick_feedback.addWidget(QLabel("FPS:"))
        self._frame_rate_selector = QComboBox()
        self._frame_rate_selector.addItems(["30", "60", "90", "120"])
        self._frame_rate_selector.setToolTip(
            "Target LED frame rate; transmission continues independently of GUI painting."
        )
        kick_feedback.addWidget(self._frame_rate_selector)
        layout.addLayout(kick_feedback)

        wled_controls = QHBoxLayout()
        wled_controls.addWidget(QLabel("WLED output:"))
        self._wled_address = QLineEdit()
        self._wled_address.setPlaceholderText("IPv4 address, e.g. 192.168.1.50")
        self._wled_address.setMinimumWidth(180)
        self._wled_address.setMaximumWidth(300)
        wled_controls.addWidget(self._wled_address)
        self._wled_enabled = QCheckBox("Enable WLED")
        self._wled_enabled.toggled.connect(self._set_wled_enabled)
        wled_controls.addWidget(self._wled_enabled)
        self._wled_status = QLabel("Disabled")
        self._wled_status.setWordWrap(True)
        wled_controls.addWidget(self._wled_status, 1)
        layout.addLayout(wled_controls)

        waveform_controls = QHBoxLayout()
        self._waveform_enabled = QCheckBox("Live PCM and kick markers")
        self._waveform_enabled.setChecked(True)
        waveform_controls.addWidget(self._waveform_enabled)
        waveform_controls.addWidget(QLabel("History:"))
        self._waveform_seconds = QSpinBox()
        self._waveform_seconds.setRange(1, 10)
        self._waveform_seconds.setValue(5)
        self._waveform_seconds.setSuffix(" s")
        waveform_controls.addWidget(self._waveform_seconds)
        waveform_controls.addStretch()
        layout.addLayout(waveform_controls)
        self._plots = RuntimePlots()
        self._plots.setToolTip("Kick markers use the detector's spectrum-window center, not physical onset time.")
        self._waveform_enabled.toggled.connect(self._plots.set_waveform_enabled)
        self._waveform_seconds.valueChanged.connect(self._plots.set_history_seconds)
        layout.addWidget(self._plots, 1)

        self._leds = LedPreview()
        self._led_output = LedOutputWorker(self._leds.led_count)
        self._visualization_selector.currentIndexChanged.connect(self._change_visualization)
        layout.addWidget(QLabel(f"Virtual LED strip · {self._leds.led_count} LEDs"))
        layout.addWidget(self._leds)
        self.setCentralWidget(content)

        # Terminate application on 'Esc' keystroke
        self._close_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._close_shortcut.activated.connect(self.close)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._update_capture)
        self._frame_rate_selector.currentIndexChanged.connect(self._update_frame_rate)
        self._update_frame_rate()
        self._led_output.start()
        self._timer.start()
        self._update_controls()

    def _update_frame_rate(self) -> None:
        fps = int(self._frame_rate_selector.currentText())
        self._timer.setInterval(round(1000 / fps))
        self._led_output.set_frame_rate(fps)

    def _update_controls(self) -> None:
        idle = self._worker is None and not self._closing
        self._sources.setEnabled(idle)
        self._start_button.setEnabled(idle)
        self._stop_button.setEnabled(self._worker is not None and not self._stopping and not self._closing)

    def _update_intensity_settings(self) -> None:
        self._intensity_settings = replace(
            self._intensity_settings,
            sensitivity_db=self._sensitivity.value(),
            frequency_balance_db=self._frequency_balance.value(),
            attack_seconds=self._attack.value() / 1000,
            release_seconds=self._release.value() / 1000,
        )
        if self._worker is not None:
            self._worker.update_settings(self._intensity_settings)

    def _clear_display(self) -> None:
        self._plots.clear()
        self._leds.set_frame(np.zeros((self._leds.led_count, 3), dtype=np.uint8))
        self._last_produced_at = None
        self._clear_kick_flash()

    def _set_wled_enabled(self, enabled: bool) -> None:
        self._wled_address.setEnabled(not enabled)
        self._led_output.set_output(self._wled_address.text() if enabled else None)

    def _change_visualization(self, index: int) -> None:
        self._led_output.set_mode(index)

    def _update_output(self, *, fresh: bool) -> None:
        state = self._led_output.state
        if fresh:
            self._leds.set_frame(state.frame)
        if self._wled_status.text() != state.status:
            self._wled_status.setText(state.status)
        if self._wled_enabled.isChecked() and state.status.startswith("Output error:"):
            with QSignalBlocker(self._wled_enabled):
                self._wled_enabled.setChecked(False)
            self._wled_address.setEnabled(True)
        if state.error is not None:
            self._error.setText(f"LED visualization failed: {state.error}")
            self._error.show()

    def _clear_kick_flash(self) -> None:
        self._kick_indicator.setStyleSheet("")
        self._kick_flash_until = None

    def _start_capture(self) -> None:
        selected = self._sources.currentData()
        if self._worker is not None or not isinstance(selected, AudioSourceInfo):
            return
        worker = CaptureWorker(PipeWireSource(selected), self._intensity_settings, on_result=self._led_output.publish)
        self._clear_display()
        self._led_output.set_capturing(True)
        self._kick_count = 0
        self._kick_count_label.setText("Kicks: 0")
        self._plots.configure(worker.band_edges)
        self._worker = worker
        self._error.clear()
        self._error.hide()
        self._update_controls()
        self._status.setText("Starting…")
        worker.start()

    def _stop_capture(self) -> None:
        if self._worker is not None and not self._stopping:
            self._stopping = True
            self._led_output.set_capturing(False)
            self._status.setText("Stopping…")
            self._update_controls()
            self._clear_display()
            self._worker.request_stop()

    def _update_capture(self) -> None:
        now = monotonic()
        if self._kick_flash_until is not None and now >= self._kick_flash_until:
            self._clear_kick_flash()
        worker = self._worker
        if worker is None:
            self._update_output(fresh=False)
            return
        # Observe termination before taking the snapshot so a final error
        # cannot arrive between the snapshot and discarding the worker.
        finished = not worker.is_alive
        result, kicks, error = worker.take_latest()
        if result is not None:
            self._kick_count = result.kick_count
            self._kick_count_label.setText(f"Kicks: {self._kick_count}")
            self._last_produced_at = result.produced_at
        if error:
            self._error.setText(f"Error: {error}")
            self._error.show()
            self._status.setText("Capture failed")
        fresh = (
            not self._stopping
            and error is None
            and not finished
            and self._last_produced_at is not None
            and now - self._last_produced_at < STALE_SECONDS
        )
        if fresh:
            if result is not None:
                levels = np.clip(power_to_decibels(result.spectrum.band_energies), SPECTRUM_MIN_DB, SPECTRUM_MAX_DB)
                self._plots.set_features(levels, result.intensities)
                self._status.setText("Capturing")
            if kicks:
                self._kick_indicator.setStyleSheet(self._kick_flash_style)
                self._kick_flash_until = now + KICK_FLASH_SECONDS
            self._plots.update_waveform(worker.history, worker.waveform_kicks)
        elif self._last_produced_at is not None:
            self._clear_display()
            if error is None and not self._stopping and not finished:
                self._status.setText("Waiting for audio…")
        self._update_output(fresh=fresh)
        if finished:
            self._led_output.set_capturing(False)
            worker.join()
            self._worker = None
            self._stopping = False
            self._status.setText("Capture failed" if error else "Stopped")
            self._update_controls()
            self._clear_display()
            if self._closing:
                self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._wled_enabled.setChecked(False)
        if self._worker is not None:
            self._closing = True
            self._stop_capture()
            event.ignore()
        else:
            self._timer.stop()
            self._led_output.close()
            self._plots.canvas.close()
            event.accept()

    def shutdown(self) -> None:
        """Bounded cleanup if the application exits outside the window-close path."""
        self._timer.stop()
        try:
            if self._worker is not None:
                self._stop_capture()
                self._worker.join()
                self._worker = None
                self._stopping = False
            self._update_controls()
        finally:
            self._wled_enabled.setChecked(False)
            self._led_output.close()
            self._plots.canvas.close()
