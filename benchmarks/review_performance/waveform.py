"""Peak-preserving waveform geometry with explicit CPU memory accounting."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

FloatPoints = NDArray[np.float32]


def pcm_points(samples: NDArray[np.float32], rate: int, first: int, last: int) -> FloatPoints:
    points = np.empty((last - first, 2), dtype=np.float32)
    points[:, 0] = np.arange(first, last, dtype=np.float64) / rate
    points[:, 1] = samples[first:last]
    return points


def envelope(samples: NDArray[np.float32], rate: int, block: int) -> FloatPoints:
    """Two extrema per bucket, retaining their original sample times and order.

    The incomplete final bucket is included. No averaging can erase transients.
    At coarse zoom the extrema span <= one physical display pixel.
    """
    count = len(samples) // block
    full = samples[: count * block].reshape(count, block)
    minima, maxima = full.argmin(axis=1), full.argmax(axis=1)
    offsets = np.arange(count, dtype=np.int64) * block
    indices = np.column_stack((offsets + np.minimum(minima, maxima), offsets + np.maximum(minima, maxima))).ravel()
    if count * block < len(samples):
        tail = samples[count * block :]
        indices = np.append(indices, count * block + np.sort([tail.argmin(), tail.argmax()]))
    points = np.empty((len(indices), 2), dtype=np.float32)
    points[:, 0] = indices / rate
    points[:, 1] = samples[indices]
    return points


@dataclass(frozen=True)
class Level:
    block: int
    points: FloatPoints


class WaveformPyramid:
    def __init__(self, samples: NDArray[np.float32], rate: int) -> None:
        started = perf_counter()
        self.levels: list[Level] = []
        block = 4
        while block < len(samples):
            self.levels.append(Level(block, envelope(samples, rate, block)))
            block *= 4
        self.seconds = perf_counter() - started
        self.bytes = sum(level.points.nbytes for level in self.levels)

    def level(self, visible_samples: float, physical_pixels: int) -> Level | None:
        samples_per_pixel = visible_samples / max(1, physical_pixels)
        # Raw samples once there are <= two samples per physical pixel.
        eligible = [level for level in self.levels if level.block <= samples_per_pixel]
        return eligible[-1] if eligible else None
