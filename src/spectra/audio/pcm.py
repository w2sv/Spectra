"""Reading frame-aligned, little-endian float32 PCM."""

from io import Reader

import numpy as np

from spectra.audio import AudioSamples


def read_float32_frames(stream: Reader[bytes], frames: int, channels: int) -> AudioSamples:
    """Read a chunk, retaining byte and channel alignment across partial reads.

    Only EOF permits a short chunk (including an empty one). An incomplete
    final frame raises ValueError rather than silently discarding samples.
    The returned array owns its backing storage independently of later reads.
    """
    if frames <= 0 or channels <= 0:
        raise ValueError("Frame and channel counts must be positive.")

    bytes_per_frame = channels * np.dtype("<f4").itemsize
    requested_bytes = frames * bytes_per_frame
    data = bytearray()
    while len(data) < requested_bytes:
        part = stream.read(requested_bytes - len(data))
        if not part:
            break
        data.extend(part)

    if len(data) % bytes_per_frame:
        raise ValueError("PCM stream ended with an incomplete audio frame.")

    return np.frombuffer(data, dtype="<f4").reshape(-1, channels).astype(np.float32, copy=False)
