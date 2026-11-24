"""DDP wire format and real loopback UDP, without WLED or audio services."""

import socket
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from spectra.output import ddp


@pytest.fixture
def receiver(monkeypatch: pytest.MonkeyPatch) -> Iterator[socket.socket]:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as incoming:
        incoming.bind(("127.0.0.1", 0))
        incoming.settimeout(1)
        # Use an ephemeral loopback port so tests cannot conflict with a service
        # listening on 4048. Production always uses the DDP standard port.
        monkeypatch.setattr(ddp, "DDP_PORT", incoming.getsockname()[1])
        yield incoming


def test_ddp_header_and_rgb_order_for_a_noncontiguous_frame() -> None:
    frame = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)[::-1]

    packet = ddp.encode_ddp_frame(frame, sequence=7)

    assert ddp.DDP_PORT == 4048
    # Version 1 + PUSH, sequence 7, RGB8, display 1, byte offset 0, length 9.
    assert packet[:10] == bytes.fromhex("41 07 0b 01 00000000 0009")
    assert packet[10:] == bytes([7, 8, 9, 4, 5, 6, 1, 2, 3])


def test_300_led_udp_frames_sequence_rollover_and_cleanup(receiver: socket.socket) -> None:
    frame = (np.arange(900, dtype=np.uint16) % 256).astype(np.uint8).reshape(300, 3)
    output = ddp.DDPOutput("127.0.0.1")
    try:
        for sequence in [*range(1, 16), 1]:
            assert output.send(frame)
            packet = receiver.recv(2048)
            assert len(packet) == 910
            assert packet[:10] == bytes([0x41, sequence, 0x0B, 1, 0, 0, 0, 0, 3, 0x84])
            np.testing.assert_array_equal(np.frombuffer(packet[10:], dtype=np.uint8).reshape(300, 3), frame)
    finally:
        assert output.close()
    black = receiver.recv(2048)
    assert len(black) == 910 and black[10:] == bytes(900)
    assert output.close()
    with pytest.raises(RuntimeError, match="closed"):
        output.send(frame)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((300, 3), dtype=np.float32),
        np.zeros((300, 4), dtype=np.uint8),
        np.zeros(900, dtype=np.uint8),
        np.zeros((0, 3), dtype=np.uint8),
        np.zeros((481, 3), dtype=np.uint8),
    ],
)
def test_invalid_rgb_frames_are_rejected(frame: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError):
        ddp.encode_ddp_frame(frame)


def test_invalid_destination_and_frame_length(receiver: socket.socket) -> None:
    with pytest.raises(ValueError):
        ddp.DDPOutput("not-an-ipv4-address")
    output = ddp.DDPOutput("127.0.0.1")
    try:
        with pytest.raises(ValueError):
            output.send(np.zeros((299, 3), dtype=np.uint8))
    finally:
        output.close()


def test_busy_socket_drops_frames_and_failed_blackout_still_closes(
    receiver: socket.socket, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.zeros((300, 3), dtype=np.uint8)
    output = ddp.DDPOutput("127.0.0.1")

    def busy_send(*_arguments: object) -> int:
        raise BlockingIOError("UDP buffer busy")

    def failed_send(*_arguments: object) -> int:
        raise OSError("Network unavailable")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(socket.socket, "sendto", busy_send)
            assert not output.send(frame)
        assert output.send(frame)
        assert receiver.recv(2048)[1] == 1
        with monkeypatch.context() as patch:
            patch.setattr(socket.socket, "sendto", failed_send)
            with pytest.raises(OSError, match="Network unavailable"):
                output.close()
        assert output.close()
        with pytest.raises(RuntimeError, match="closed"):
            output.send(frame)
    finally:
        output.close()
