"""A framebuffer-only preview of an ordered RGB LED strip."""

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QSizePolicy, QWidget


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
        segment_width = self.width() / self.led_count
        gap = min(1.0, segment_width * 0.2)
        for index, rgb in enumerate(self._frame):
            color = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            rectangle = QRectF(index * segment_width, 0, segment_width - gap, self.height())
            painter.fillRect(rectangle, color)
        painter.end()
