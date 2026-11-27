"""Shared plot styles and native Qt samples for the wrapping review legend."""

from dataclasses import dataclass
from functools import cache

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QImageWriter, QPainter, QPen
from PySide6.QtWidgets import QLabel


@dataclass(frozen=True)
class MarkerStyle:
    label: str
    color: str
    symbol: str
    size: int


@dataclass(frozen=True)
class LineStyle:
    label: str
    color: str
    width: int = 1
    dashed: bool = False

    def pen(self) -> QPen:
        pen = QPen(QColor(self.color))
        pen.setWidth(self.width)
        pen.setStyle(Qt.PenStyle.DashLine if self.dashed else Qt.PenStyle.SolidLine)
        return pen


AUTOMATIC_KICK = MarkerStyle("Automatic kick", "#2bad62", "o", 10)
HUMAN_KICK = MarkerStyle("Human kick", "#3184e0", "s", 10)
CANDIDATE = MarkerStyle("Kick candidate", "#93979d", "o", 5)
REJECTED = MarkerStyle("Rejected candidate", "#a96872", "x", 9)
UNCERTAINTY = MarkerStyle("Classification uncertainty", "#e7ae40", "o", 12)
TIMING = MarkerStyle("Timing discrepancy", "#dca4df", "o", 12)
SCORE = MarkerStyle("Similarity", "#66b283", "o", 5)
MARKER_STYLES = (AUTOMATIC_KICK, HUMAN_KICK, CANDIDATE, REJECTED, UNCERTAINTY, TIMING, SCORE)
PLAYBACK = LineStyle("Playback", "#00bbdb", 2)
KICK_FLASH = LineStyle("Kick flash", "#00ff00", 6)
SELECTION = LineStyle("Selection", "#ef72c1")
BEAT = LineStyle("Estimated beat", "#60656b", dashed=True)
BASS = LineStyle("Bass evidence", "#6595dc")
BROADBAND = LineStyle("Broadband evidence", "#c48aa8")


@cache
def _sample(style: MarkerStyle | LineStyle, ratio: float) -> str:
    image = QImage(round(26 * ratio), round(18 * ratio), QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(ratio)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if isinstance(style, MarkerStyle):
        painter.translate(13, 9)
        outline = style in (UNCERTAINTY, TIMING)
        painter.setPen(QPen(QColor(style.color), 2 if outline else 1))
        painter.setBrush(Qt.BrushStyle.NoBrush if outline else QColor(style.color))
        radius = style.size / 2
        if style.symbol == "s":
            painter.drawRect(QRectF(-radius, -radius, style.size, style.size))
        elif style.symbol == "x":
            painter.drawLine(QPointF(-radius, -radius), QPointF(radius, radius))
            painter.drawLine(QPointF(-radius, radius), QPointF(radius, -radius))
        else:
            painter.drawEllipse(QPointF(0, 0), radius, radius)
    else:
        painter.setPen(style.pen())
        painter.drawLine(QPointF(2, 9), QPointF(24, 9))
    painter.end()
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    QImageWriter(buffer, QByteArray(b"png")).write(image)
    return bytes(data.toBase64().data()).decode("ascii")


class ReviewLegend(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.entries: tuple[MarkerStyle | LineStyle, ...] = ()
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)

    def set_entries(self, entries: tuple[MarkerStyle | LineStyle, ...]) -> None:
        if entries == self.entries:
            return
        self.entries = entries
        self.setText(
            " &nbsp; ".join(
                f'<img width="26" height="18" src="data:image/png;base64,{_sample(style, self.devicePixelRatioF())}">'
                + "&nbsp;"
                + style.label.replace(" ", "&nbsp;")
                for style in entries
            )
        )
