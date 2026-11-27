"""Recording fidelity and incremental GPU payload integrity, without a display."""

import numpy as np
from vispy.scene import Node

from spectra.analysis import KickEvent
from spectra.audio.history import PCMHistory
from spectra.ui.live_waveform import EnvelopeVisual, LiveWaveform
from spectra.ui.plots import RecordingWaveform
from spectra.ui.waveform import WaveformPyramid, extrema_indices


def test_extrema_preserve_peak_positions_order_and_final_short_bucket() -> None:
    samples = np.zeros(67, dtype=np.float32)
    samples[[3, 17, 64, 66]] = [1, -0.8, -1, 0.9]
    indices = extrema_indices(samples, 16)
    assert all(index in indices for index in (3, 17, 64, 66))
    assert np.all(np.diff(indices) >= 0)
    pyramid = WaveformPyramid(samples)
    assert pyramid.level(1000, 1000) is None
    level = pyramid.level(1000, 50)
    assert level is not None and level.block <= 20
    for level in pyramid.levels:
        assert level.values.min() == samples.min() and level.values.max() == samples.max()


def test_recording_window_retains_original_samples_and_precise_local_coordinates() -> None:
    samples = np.sin(np.arange(48000 * 4) * 0.1).astype(np.float32)
    waveform = RecordingWaveform(Node(), samples, 48000)
    assert waveform.update(3.9, 3.92, 2000)
    assert waveform.window is not None and waveform.window[0] == 1
    origin = round(waveform.visual.transform.translate[0] * 48000)
    indices = origin + np.arange(len(waveform.points))
    np.testing.assert_array_equal(waveform.points[:, 1], samples[indices])
    np.testing.assert_allclose(waveform.points[:, 0], np.arange(len(indices)) / 48000, atol=1e-8)
    assert not waveform.update(3.905, 3.925, 2000)
    assert waveform.update(1, 3, 1000)
    assert waveform.window is not None and waveform.window[0] > 1
    assert waveform.points[:, 1].max() > 0.99


def test_history_wrap_and_oversized_chunks_preserve_arithmetic_mono() -> None:
    history = PCMHistory(1000, 0.037)
    source = np.column_stack((np.arange(210), np.arange(210) * 0.5)).astype(np.float32)
    for chunk in (source[:13], source[13:50], source[50:]):
        history.append(chunk)
    scratch = np.empty(history.capacity, dtype=np.float32)
    result = history.copy_since(0, scratch)
    assert result is not None and result.stop == 210 and result.first == 173 and result.skipped == 173
    np.testing.assert_array_equal(scratch, source[-37:].mean(axis=1))
    assert history.samples.nbytes == 37 * 4
    history._sequence += 1  # An in-progress producer write must invalidate a read.
    assert history.copy_since(result.stop, scratch) is None
    history._sequence += 1
    empty = history.copy_since(result.stop, scratch)
    assert empty is not None and empty.stop == empty.first


def test_incremental_envelope_wrap_preserves_peaks_and_sample_time_after_long_capture() -> None:
    visual = EnvelopeVisual(capacity=64, block=4, sample_rate=1000)
    base = 10**12
    for first in range(0, 160, 16):
        values = np.arange(first, first + 16, dtype=np.float32)
        visual.append(base + first, values, base + first + 16)
    points = visual.points[visual.points[:, 0] >= 0]
    head = (base + 160) % visual.period
    ends = -np.mod(head - points[:, 2] + visual.period, visual.period)
    relative = ends - np.mod(points[:, 2] - points[:, 0] + visual.period, visual.period)
    recent = relative >= -64
    assert points[recent, 1].max() == 159
    np.testing.assert_array_equal(relative[recent], points[recent, 1] - 160)
    assert visual.uploaded_bytes == 10 * 8 * 3 * 4

    history = PCMHistory(1000, 0.064)
    waveform = LiveWaveform(Node())
    samples = np.column_stack((np.arange(160), np.arange(160))).astype(np.float32)
    history.append(samples[:64])
    assert waveform.update(history, (), 0.064, 16)
    first_visual = waveform.visual
    assert first_visual is not None
    uploaded = first_visual.uploaded_bytes
    assert waveform.update(history, (KickEvent(0.015, 1),), 0.064, 16)
    assert waveform.kick_markers.visible and first_visual.uploaded_bytes == uploaded
    history.append(samples[64:])
    assert waveform.update(history, (), 0.064, 16)
    assert waveform.visual is first_visual and waveform.skipped_samples > 0
    assert waveform.last_stop == 160 and waveform.head_seconds == 0.16
    assert waveform.visual is not None
    assert len(waveform.visual.points) <= history.capacity * 2


def test_hidden_view_coalesces_updates_until_geometry_is_drawn(monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    from spectra.ui.runtime_plots import RuntimePlots

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QApplication.instance() or QApplication([])
    plots = RuntimePlots()
    history = PCMHistory(1000, 0.064)
    try:
        for index in range(100):
            history.append(np.full((16, 2), index / 100, dtype=np.float32))
            plots.set_features(np.full(24, -index, dtype=np.float64), np.full(24, index / 100))
            plots.update_waveform(history, ())
        assert plots.waveform.visual is None  # No deferred GPU FIFO per capture update.
        plots._prepare_frame()
        assert plots.waveform.last_stop == history.written
        np.testing.assert_array_equal(plots.spectrum.values, np.full(24, -99))
        np.testing.assert_array_equal(plots.intensity.values, np.full(24, 0.99))
    finally:
        plots.canvas.close()
        plots.close()
        application.processEvents()
