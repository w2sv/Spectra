"""DDP v1 RGB output to one WLED controller over nonblocking UDP.

Protocol: https://www.3waylabs.com/ddp/
WLED: https://kno.wled.ge/interfaces/ddp/
"""

import socket
import struct
from ipaddress import IPv4Address

import numpy as np
from numpy.typing import NDArray

DDP_PORT = 4048
MAX_LEDS_PER_PACKET = 480  # WLED's 1440-byte RGB payload limit; no IP fragmentation at a 1500-byte MTU.


def encode_ddp_frame(frame: NDArray[np.uint8], sequence: int = 0) -> bytes:
    """Encode a complete RGB frame as one pushed DDP v1 display packet.

    Input is uint8 (led_count, 3), in pixel and RGB channel order, with
    1–480 LEDs. Noncontiguous views are supported. The ten-byte header uses
    network byte order: version/push, four-bit sequence, RGB8 type, display
    destination, byte offset, and payload length. Sequence zero disables
    receiver sequence checking; otherwise values run from 1 through 15.
    """
    if frame.ndim != 2 or frame.shape[1] != 3 or frame.dtype != np.uint8:
        raise ValueError("Expected a (led_count, 3) uint8 RGB framebuffer.")
    if not 1 <= len(frame) <= MAX_LEDS_PER_PACKET:
        raise ValueError(f"A single DDP packet supports 1–{MAX_LEDS_PER_PACKET} RGB LEDs.")
    if not 0 <= sequence <= 15:
        raise ValueError("DDP sequence must be between 0 and 15.")

    payload = frame.tobytes(order="C")
    header = struct.pack("!BBBBIH", 0x41, sequence, 0x0B, 1, 0, len(payload))
    return header + payload


class DDPOutput:
    """Single-caller socket lifecycle; no DNS, connection handshake, or queue.

    Construction opens the socket, so callers create this only when output
    is enabled. Successful sends mean local UDP acceptance, not delivery.
    Other than a busy buffer, socket errors propagate to the presentation
    layer. close() attempts a final black frame and always releases the socket.
    """

    def __init__(self, address: str, led_count: int = 300) -> None:
        ip = IPv4Address(address.strip())
        if ip.is_unspecified or ip.is_multicast or ip == IPv4Address("255.255.255.255"):
            raise ValueError("Enter the controller's unicast IPv4 address.")
        if not 1 <= led_count <= MAX_LEDS_PER_PACKET:
            raise ValueError(f"DDP output supports 1–{MAX_LEDS_PER_PACKET} RGB LEDs.")

        self._destination = (str(ip), DDP_PORT)
        self._led_count = led_count
        self._sequence = 1
        output = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            output.setblocking(False)
        except OSError:
            output.close()
            raise
        self._socket: socket.socket | None = output

    def send(self, frame: NDArray[np.uint8]) -> bool:
        """Send now; return False if the busy socket drops this frame, with no retry."""
        if self._socket is None:
            raise RuntimeError("DDP output is closed.")
        if frame.shape != (self._led_count, 3):
            raise ValueError(f"Expected a ({self._led_count}, 3) RGB framebuffer.")
        packet = encode_ddp_frame(frame, self._sequence)
        try:
            self._socket.sendto(packet, self._destination)
        except BlockingIOError:
            return False
        self._sequence = self._sequence % 15 + 1
        return True

    def close(self) -> bool:
        """Best-effort blackout and unconditional release; safe to call repeatedly.

        Return whether the blackout was accepted locally (or output was
        already closed). Socket errors still propagate after release.
        """
        output = self._socket
        if output is None:
            return True
        try:
            return self.send(np.zeros((self._led_count, 3), dtype=np.uint8))
        finally:
            self._socket = None
            output.close()
