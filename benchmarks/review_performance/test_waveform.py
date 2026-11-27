"""Protect transient fidelity and geometry accounting, not experimental colors."""

import numpy as np

from benchmarks.review_performance.metrics import Measurements
from benchmarks.review_performance.waveform import WaveformPyramid, envelope, pcm_points


def test_envelope_preserves_extrema_sample_times_and_incomplete_tail() -> None:
    samples = np.zeros(67, dtype="float32")
    samples[[3, 17, 64, 66]] = [1, -0.8, -1, 0.9]
    points = envelope(samples, 48000, 16)
    for index in [3, 17, 64, 66]:
        assert any(np.isclose(time, index / 48000) and value == samples[index] for time, value in points)
    assert np.all(np.diff(points[:, 0]) >= 0)


def test_lod_uses_physical_pixels_and_original_pcm_at_close_zoom() -> None:
    samples = np.linspace(-1, 1, 1031, dtype="float32")
    pyramid = WaveformPyramid(samples, 48000)
    assert pyramid.level(1000, 1000) is None
    level = pyramid.level(1000, 100)
    assert level is not None and level.block <= 10
    points = pcm_points(samples, 48000, 37, 100)
    np.testing.assert_array_equal(points[:, 1], samples[37:100])
    np.testing.assert_allclose(points[:, 0], np.arange(37, 100) / 48000, rtol=1e-7)
    for level in pyramid.levels:
        assert level.points[:, 1].min() == samples.min()
        assert level.points[:, 1].max() == samples.max()


def test_frame_accounting_handles_interleaved_qt_paints_and_duplicates() -> None:
    measurements = Measurements()
    measurements.paints = [
        (1, "wave", 0.0, 1, 0),
        (1, "wave", 0.001, 1, 0),
        (2, "evidence", 0.002, 1, 0),
        (3, "evidence", 0.010, 1, 0),
        (4, "wave", 0.012, 1, 0),
    ]
    result = measurements.result(120, {"wave", "evidence"})
    assert result["complete_logical_frames"] == 2
    assert result["frame_interval_ms"]["count"] == 1
    assert abs(result["frame_interval_ms"]["median"] - 10) < 1e-10


def test_streaming_rings_preserve_recent_samples_when_the_gui_falls_behind() -> None:
    from time import monotonic, sleep

    import pytest

    pytest.importorskip("vispy")
    from benchmarks.review_performance.streaming import RollingPCM, StreamingLineVisual

    samples = np.arange(37, dtype="float32")
    history = RollingPCM(samples, 1000, seconds=0.02)
    history.start()
    try:
        deadline = monotonic() + 2
        while history.written < 80 and monotonic() < deadline:
            sleep(0.002)
        assert history.written >= 80
        first, recent = history.read()
        assert len(recent) == history.capacity
        assert history.overwritten > 0
        np.testing.assert_array_equal(recent, samples[np.arange(first, first + len(recent)) % len(samples)])
    finally:
        history.close()
    assert not history.thread.is_alive()

    # Exercise GPU payload construction and deferred subrange writes without
    # requiring a desktop context. Every pair must span adjacent source samples.
    visual = StreamingLineVisual(capacity=32, rate=1000)
    for start, count in ((0, 20), (20, 17), (37, 19)):
        visual.append(start, np.arange(start, start + count, dtype="float32"))
    ordered = visual.points.reshape(32, 2, 2)[np.arange(24, 56) % 32]
    np.testing.assert_array_equal(ordered[:, 1, 1], np.arange(24, 56))
    np.testing.assert_array_equal(ordered[:, 0, 1], np.arange(23, 55))
    np.testing.assert_allclose(ordered[:, 1, 0], np.arange(24, 56) / 1000, rtol=1e-7)
