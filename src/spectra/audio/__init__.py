"""Backend-independent contracts for PCM audio capture."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

type AudioSamples = NDArray[np.float32]

DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_READ_FRAMES = 1024


@dataclass(frozen=True)
class AudioSourceInfo:
    """An opaque source identifier and a human-readable name."""

    identifier: str
    display_name: str


class AudioSourceError(RuntimeError):
    """An audio source could not start or continue capturing."""


class AudioSource(Protocol):
    """A caller-owned, synchronous stream; no worker thread is implied.

    Samples are native float32 arrays of shape (frames, channels), contiguous
    in frame order. Stereo columns are left, right. Full scale is nominally
    [-1, 1]; samples are not clipped. Start before reading and always stop in
    a finally block. Calls must be serialized by the owner, except for the
    thread-safe request_stop() method used to interrupt a blocked read.
    """

    @property
    def info(self) -> AudioSourceInfo: ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def channels(self) -> int: ...

    def start(self) -> None:
        """Start capture, or raise AudioSourceError."""
        ...

    def read(self, frames: int = DEFAULT_READ_FRAMES) -> AudioSamples:
        """Block for up to frames complete frames; raise AudioSourceError on EOF.

        A short final chunk may precede EOF. Returned samples remain valid
        across subsequent reads. Nonpositive frame counts raise ValueError.
        """
        ...

    def stop(self) -> None:
        """Release capture resources; safe to call again or before starting."""
        ...

    def request_stop(self) -> None:
        """Request cancellation from any thread without waiting for cleanup.

        A blocked read must promptly raise AudioSourceError. The owner still
        calls stop() to release resources. Starting a new capture clears this request.
        """
        ...
