"""PCM data integrity without an audio server or capture subprocess."""

from io import BytesIO

import numpy as np
import pytest

from spectra.audio.pcm import read_float32_frames


class PartialReadStream(BytesIO):
    """A pipe-like stream that splits both float samples and stereo frames."""

    def read(self, size: int | None = -1) -> bytes:
        return super().read(min(size, 3) if size is not None and size >= 0 else 3)


def test_partial_reads_preserve_samples_channels_and_chunk_boundaries() -> None:
    expected = np.array([[0.125, -0.25], [0.5, -1.0], [1.25, 0.0]], dtype=np.float32)
    stream = PartialReadStream(expected.astype("<f4").tobytes())

    first = read_float32_frames(stream, frames=2, channels=2)
    final = read_float32_frames(stream, frames=2, channels=2)
    eof = read_float32_frames(stream, frames=2, channels=2)

    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    np.testing.assert_array_equal(first, expected[:2])
    np.testing.assert_array_equal(final, expected[2:])
    assert eof.shape == (0, 2)


@pytest.mark.parametrize("missing_bytes", [1, 4, 7])
def test_truncated_final_frame_is_rejected(missing_bytes: int) -> None:
    data = np.array([[0.25, -0.5], [1.0, -1.0]], dtype="<f4").tobytes()
    with pytest.raises(ValueError, match="incomplete audio frame"):
        read_float32_frames(PartialReadStream(data[:-missing_bytes]), frames=2, channels=2)
