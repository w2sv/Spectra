"""Single-writer PCM history with nonblocking, consistent reader snapshots."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spectra.audio import AudioSamples


@dataclass(frozen=True)
class HistoryRange:
    first: int
    stop: int
    skipped: int


class PCMHistory:
    """Preallocated arithmetic-mono float32 ring, indexed from capture sample zero.

    Capture is the only writer. The GUI copies into its own storage and checks a
    sequence counter before publishing the snapshot. NumPy can release the GIL
    during copies: a concurrent write invalidates the read, never blocks capture.
    A failed read is retried on the next GUI tick; pending data cannot accumulate.
    """

    def __init__(self, sample_rate: int, seconds: float = 10) -> None:
        if sample_rate <= 0 or seconds <= 0:
            raise ValueError("History requires a positive sample rate and duration.")
        self.sample_rate = sample_rate
        self.capacity = max(1, round(sample_rate * seconds))
        self.samples = np.zeros(self.capacity, dtype=np.float32)
        self.written = 0
        self._sequence = 0

    def append(self, stereo: AudioSamples) -> None:
        if stereo.dtype != np.float32 or stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError("Expected stereo float32 PCM.")
        count = len(stereo)
        if not count:
            return
        kept = min(count, self.capacity)
        first = self.written + count - kept
        source = stereo[count - kept :]
        start = first % self.capacity
        split = min(kept, self.capacity - start)
        self._sequence += 1
        for output, chunk in (
            (self.samples[start : start + split], source[:split]),
            (self.samples[: kept - split], source[split:]),
        ):
            np.add(chunk[:, 0], chunk[:, 1], out=output)
            output *= 0.5
        self.written += count
        self._sequence += 1

    def copy_since(self, first: int, destination: NDArray[np.float32]) -> HistoryRange | None:
        """Copy the available dirty range, or return None if capture overtook it.

        The returned range has at most capacity samples. Data in destination is
        valid only if a range was returned. The reader owns its cursor and storage.
        """
        if destination.ndim != 1 or destination.dtype != np.float32 or len(destination) < self.capacity:
            raise ValueError("History destination must be a float32 vector at least capacity long.")
        version = self._sequence
        if version % 2:
            return None
        stop = self.written
        actual = max(first, stop - self.capacity, 0)
        if actual > stop:
            raise ValueError("History cursor is ahead of capture.")
        count = stop - actual
        offset = actual % self.capacity
        split = min(count, self.capacity - offset)
        destination[:split] = self.samples[offset : offset + split]
        destination[split:count] = self.samples[: count - split]
        if version != self._sequence:
            return None
        return HistoryRange(actual, stop, actual - first)
