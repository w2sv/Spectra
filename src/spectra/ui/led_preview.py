"""A framebuffer-only preview of an ordered RGB LED strip."""

import numpy as np
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QSizePolicy, QWidget
from numpy.typing import NDArray


class LedPreview(QWidget):
    """Display separate LED segments without interpreting their colors."""

    def __init__(self, led_count: int = 300, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if led_count <= 0:
            raise ValueError("LED count must be positive.")
        self.led_count = led_count
        self._frame = np.zeros((led_count, 3), dtype=np.uint8)
        self.setMinimumSize(led_count, 40)
        self.setMaximumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_frame(self, frame: NDArray[np.uint8]) -> None:
        """Copy an RGB framebuffer so later caller mutations cannot affect painting."""
        if frame.shape != (self.led_count, 3) or frame.dtype != np.uint8:
            raise ValueError(f"Expected a ({self.led_count}, 3) uint8 RGB framebuffer.")
        if not np.array_equal(frame, self._frame):
            self._frame = frame.copy()
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()))

        pixels_per_led = max(1, self.width() // self.led_count)
        strip_width = pixels_per_led * self.led_count
        start_x = (self.width() - strip_width) // 2
        gap = 1 if pixels_per_led > 1 else 0

        for index, rgb in enumerate(self._frame):
            color = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            painter.fillRect(
                start_x + index * pixels_per_led,
                0,
                pixels_per_led - gap,
                self.height(),
                color,
                )

        painter.end()
