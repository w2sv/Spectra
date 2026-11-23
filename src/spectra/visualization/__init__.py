"""Transform spectral measurements into RGB frames, independently of Qt."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from spectra.analysis import KickEvent


class Visualization(Protocol):
    """Render on one caller-owned thread, independently of capture and output.

    Intensities are a float64 (bands,) array in [0, 1]. Each kick is supplied
    once, retaining its original audio-relative timestamp. Render time uses
    the same audio timeline and may advance without new intensities or kicks.
    Frames are uint8 RGB arrays of shape (led_count, 3); inputs stay unchanged.
    """

    def render(
        self,
        intensities: NDArray[np.float64],
        kicks: tuple[KickEvent, ...],
        timestamp: float,
    ) -> NDArray[np.uint8]: ...

    def reset(self) -> None:
        """Clear animation state for a new stream or visualization selection."""
        ...
