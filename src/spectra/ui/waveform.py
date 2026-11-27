"""Peak-preserving recording geometry; independent of Qt and OpenGL."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray


def extrema_indices(samples: NDArray[np.float32], block: int) -> NDArray[np.int64]:
    """Indices of both extrema in chronological order, including a final short bucket."""
    count = len(samples) // block
    full = samples[: count * block].reshape(count, block)
    minima, maxima = full.argmin(axis=1), full.argmax(axis=1)
    offsets = np.arange(count, dtype=np.int64) * block
    indices = np.column_stack((offsets + np.minimum(minima, maxima), offsets + np.maximum(minima, maxima))).ravel()
    if count * block < len(samples):
        tail = samples[count * block :]
        indices = np.append(indices, count * block + np.sort([tail.argmin(), tail.argmax()]))
    return indices


@dataclass(frozen=True)
class WaveformLevel:
    block: int
    indices: NDArray[np.int64]
    values: NDArray[np.float32]


class WaveformPyramid:
    def __init__(self, samples: NDArray[np.float32]) -> None:
        started = perf_counter()
        self.levels: list[WaveformLevel] = []
        # Smaller views use original samples. Starting at 16 avoids retaining the
        # expensive finest envelope levels for geometry already cheap to submit.
        block = 16
        while block < len(samples):
            indices = extrema_indices(samples, block)
            self.levels.append(WaveformLevel(block, indices, samples[indices]))
            block *= 4
        self.seconds = perf_counter() - started
        self.bytes = sum(level.indices.nbytes + level.values.nbytes for level in self.levels)

    def level(self, visible_samples: float, physical_pixels: int) -> WaveformLevel | None:
        eligible = [level for level in self.levels if level.block <= visible_samples / max(1, physical_pixels)]
        return eligible[-1] if eligible else None
