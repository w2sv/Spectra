"""Main-thread presentation and controls for live spectral measurements."""

from dataclasses import replace
from time import monotonic

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QColor, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spectra.audio import AudioSourceInfo
from spectra.audio.pipewire import PipeWireSource
from spectra.capture import CaptureWorker
from spectra.ui.led_preview import LedPreview
from spectra.visualization.frequency import band_colors, frequency_frame, power_to_decibels
from spectra.visualization.intensity import BandIntensitySettings

FRAME_RATE = 30
SPECTRUM_MIN_DB = -80.0
SPECTRUM_MAX_DB = 0.0
STALE_RESULT_TIMEOUT = 0.5


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
        self.resize(1000, 600)
        self.setMinimumSize(680, 400)
        self._worker: CaptureWorker | None = None
        self._stopping = False
        self._closing = False
        self._band_count = 0
        self._last_result_at: float | None = None
        self._intensity_settings = BandIntensitySettings()

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
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

        self._plot = pg.PlotWidget(background=self.palette().color(QPalette.ColorRole.Base).name())
        self._plot.setLabel("left", "Band power", units="dB re 1 PCM²")
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.setTitle("Frequency spectrum")
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.disableAutoRange()
        self._plot.getViewBox().setYRange(SPECTRUM_MIN_DB, SPECTRUM_MAX_DB, padding=0)
        ticks = [(40, "40"), (100, "100"), (250, "250"), (1000, "1k"), (4000, "4k"), (16_000, "16k")]
        self._plot.getAxis("bottom").setTicks([[(float(np.log10(hz)), label) for hz, label in ticks]])
        self._plot.getViewBox().setXRange(float(np.log10(ticks[0][0])), float(np.log10(ticks[-1][0])), padding=0.03)
        for axis in ("left", "bottom"):
            self._plot.getAxis(axis).setPen(self.palette().color(QPalette.ColorRole.Text))
            self._plot.getAxis(axis).setTextPen(self.palette().color(QPalette.ColorRole.Text))
        # BarGraphItem does not transform log coordinates; map edges explicitly.
        self._bars = pg.BarGraphItem(
            x0=[],
            x1=[],
            y0=SPECTRUM_MIN_DB,
            height=[],
            pen=None,
        )
        self._plot.addItem(self._bars)
        layout.addWidget(self._plot, 1)

        self._leds = LedPreview()
        layout.addWidget(QLabel(f"Virtual LED strip · {self._leds.led_count} LEDs"))
        layout.addWidget(self._leds)
        self.setCentralWidget(content)

        # Terminate application on 'Esc' keystroke
        self._close_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._close_shortcut.activated.connect(self.close)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_capture)
        self._timer.start(round(1000 / FRAME_RATE))
        self._update_controls()

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
        self._bars.setOpts(height=np.zeros(self._band_count))
        self._leds.set_frame(np.zeros((self._leds.led_count, 3), dtype=np.uint8))
        self._last_result_at = None

    def _start_capture(self) -> None:
        selected = self._sources.currentData()
        if self._worker is not None or not isinstance(selected, AudioSourceInfo):
            return
        worker = CaptureWorker(PipeWireSource(selected), self._intensity_settings)
        self._clear_display()
        edges = np.log10(worker.band_edges)
        self._band_count = len(edges) - 1
        gap = np.diff(edges) * 0.08
        brushes = [
            pg.mkBrush(QColor.fromRgbF(float(rgb[0]), float(rgb[1]), float(rgb[2])))
            for rgb in band_colors(self._band_count)
        ]
        self._bars.setOpts(
            x0=edges[:-1] + gap,
            x1=edges[1:] - gap,
            height=np.zeros(self._band_count),
            brushes=brushes,
        )
        self._plot.getViewBox().setXRange(float(edges[0]), float(edges[-1]), padding=0.03)
        self._worker = worker
        self._error.clear()
        self._error.hide()
        self._update_controls()
        self._status.setText("Starting…")
        worker.start()

    def _stop_capture(self) -> None:
        if self._worker is not None and not self._stopping:
            self._stopping = True
            self._status.setText("Stopping…")
            self._update_controls()
            self._clear_display()
            self._worker.request_stop()

    def _update_capture(self) -> None:
        worker = self._worker
        if worker is None:
            return
        # Observe termination before taking the snapshot so a final error
        # cannot arrive between the snapshot and discarding the worker.
        finished = not worker.is_alive
        result, error = worker.take_latest()
        if error:
            self._error.setText(f"Error: {error}")
            self._error.show()
            self._status.setText("Capture failed")
        if result is not None and not self._stopping and error is None:
            levels = np.clip(power_to_decibels(result.spectrum.band_energies), SPECTRUM_MIN_DB, SPECTRUM_MAX_DB)
            self._bars.setOpts(height=levels - SPECTRUM_MIN_DB)
            self._leds.set_frame(frequency_frame(result.intensities, led_count=self._leds.led_count))
            self._last_result_at = monotonic()
            self._status.setText("Capturing")
        elif self._last_result_at is not None and monotonic() - self._last_result_at >= STALE_RESULT_TIMEOUT:
            self._clear_display()
            if error is None and not self._stopping:
                self._status.setText("Waiting for audio…")
        if finished:
            worker.join()
            self._worker = None
            self._stopping = False
            self._status.setText("Capture failed" if error else "Stopped")
            self._update_controls()
            self._clear_display()
            if self._closing:
                self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None:
            self._closing = True
            self._stop_capture()
            event.ignore()
        else:
            self._timer.stop()
            event.accept()

    def shutdown(self) -> None:
        """Bounded cleanup if the application exits outside the window-close path."""
        self._timer.stop()
        if self._worker is not None:
            self._stop_capture()
            self._worker.join()
            self._worker = None
            self._stopping = False
        self._update_controls()
