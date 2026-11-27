"""Capture preserves audio-time alignment and processes every window without GUI polling."""

import numpy as np
import pytest

from spectra.analysis import KickOnsetDetector, SpectrumAnalyzer
from spectra.audio import AudioSourceError, AudioSourceInfo
from spectra.capture import CaptureWorker


class RecordingSource:
    def __init__(self, samples: np.ndarray) -> None:
        self.info = AudioSourceInfo("memory", "Deterministic PCM")
        self.sample_rate = 48000
        self.channels = 2
        self.samples = samples
        self.position = 0
        self.stopped = False

    def start(self) -> None:
        self.position = 0

    def read(self, frames: int = 1024) -> np.ndarray:
        if self.position == len(self.samples):
            raise AudioSourceError("End of fixture")
        chunk = self.samples[self.position : self.position + frames]
        self.position += len(chunk)
        return chunk

    def stop(self) -> None:
        self.stopped = True

    def request_stop(self) -> None:
        self.position = len(self.samples)


def test_capture_history_and_kicks_share_first_frame_origin_without_gui_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mono = np.zeros(48000, dtype=np.float32)
    time = np.arange(6000) / 48000
    for start in (4800, 24000):
        mono[start : start + 6000] = (0.8 * np.sin(2 * np.pi * 70 * time) * np.exp(-time / 0.04)).astype(np.float32)
    stereo = np.column_stack((mono, mono))
    reference = SpectrumAnalyzer()
    detector = KickOnsetDetector(reference.band_edges)
    expected_results = reference.process(stereo)
    expected_kicks = [kick for result in expected_results if (kick := detector.process(result)) is not None]
    assert expected_kicks
    production_times = iter(range(1000, 1000 + len(expected_results)))
    monkeypatch.setattr("spectra.capture.monotonic", lambda: next(production_times))
    source = RecordingSource(stereo)
    published = []
    worker = CaptureWorker(source, on_result=lambda result, kick: published.append((result, kick)))
    worker.start()
    worker.join()
    assert source.stopped and len(published) == len(expected_results)
    assert [result.produced_at for result, _ in published] == list(range(1000, 1000 + len(expected_results)))
    latest, _, _ = worker.take_latest()
    assert latest is published[-1][0]  # Consuming does not replace the production timestamp.
    assert worker.waveform_kicks == tuple(expected_kicks)
    assert worker.history.written == len(stereo)
    scratch = np.empty(worker.history.capacity, dtype=np.float32)
    copied = worker.history.copy_since(0, scratch)
    assert copied is not None and copied.first == 0 and copied.stop == len(stereo)
    np.testing.assert_array_equal(scratch[: copied.stop], mono)
    for kick in worker.waveform_kicks:
        sample = kick.timestamp * source.sample_rate
        # Detector timestamps remain FFT window centers, not GUI receipt time or
        # a new estimate of the physical onset. No renderer-adjusted offset.
        assert sample >= reference.fft_size / 2
        assert (sample - reference.fft_size / 2) / reference.hop_size == round(
            (sample - reference.fft_size / 2) / reference.hop_size
        )
