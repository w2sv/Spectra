"""Scrollable Matplotlib controls, kept separate from audio plots and their legend."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.text import Text
from matplotlib.widgets import Button, Slider

if TYPE_CHECKING:
    from benchmarks.annotation_review import AnnotationReview


@dataclass
class ControlRow:
    axes: Axes
    label: Text
    shortcut: Text
    height: float
    action: str | None = None
    button: Button | None = None
    template_body: bool = False
    destructive: bool = False


class ReviewControlPanel:
    def __init__(self, review: AnnotationReview) -> None:
        self.review = review
        self.figure = review.figure
        self.rows: list[ControlRow] = []
        self.buttons: dict[str, Button] = {}
        self.changed_axes: list[Axes] = []
        self.templates_expanded = False
        self.title = self.figure.text(0, 0, "Review controls", fontsize=13, weight="bold", va="top")
        self.scroll_hint = self.figure.text(0, 0, "Scroll ↓", fontsize=11, va="top", ha="right", color="0.4")
        self.scroll_axes = self.figure.add_axes((0.98, 0.1, 0.01, 0.8))
        self.slider = Slider(self.scroll_axes, "", 0, 1, valinit=1, orientation="vertical")
        self.slider.valtext.set_visible(False)
        self.slider.on_changed(lambda _: self.layout())
        self.figure.canvas.mpl_connect("scroll_event", self.on_scroll)
        self._max_scroll = 0.0
        self._left = 0.0

        self.heading("Playback")
        self.action("play", "Play", "Space")
        self.action("restart", "Restart interval", "Home")
        self.hint("Seek to position", "Alt + Click")
        self.action("previous_interval", "Previous interval", "←")
        self.action("next_interval", "Next interval", "→")
        self.heading("Review")
        self.action("previous_uncertain", "Previous uncertain candidate", ",")
        self.action("next_uncertain", "Next uncertain candidate", ".")
        self.action("previous_unsupported", "Previous unsupported beat", "[")
        self.action("next_unsupported", "Next unsupported beat", "]")
        self.hint("Toggle event classification", "Click")
        self.action("toggle", "Toggle selected event", "T")
        self.action("undo", "Undo correction", "Ctrl + Z")
        self.action("save", "Save review", "Ctrl + S")
        self.heading("Classification")
        self.action("reclassify", "Reclassify using corrections", "R")
        self.action("group_kick", "Similar events: kicks", "")
        self.action("group_nonkick", "Similar events: non-kicks", "", destructive=True)
        self.heading("Progress")
        self.progress = self.row("", "", 105)
        self.progress.label.set_position((0.02, 0.98))
        self.progress.label.set_verticalalignment("top")
        template_heading = self.action("advanced", "Advanced ▸", "", height=34)
        template_heading.label.set_fontweight("bold")
        template_heading.label.set_fontsize(12)
        self.hint("Insert missing event", "Shift + Click", template=True)
        self.action("clear_override", "Remove selected correction", "Esc", template=True, destructive=True)
        self.action("previous_corrected", "Previous correction", "<", template=True)
        self.action("next_corrected", "Next correction", ">", template=True)
        self.action("grid", "Beat grid: primary", "G", template=True)
        self.action("nonkick_visibility", "Non-kicks: shown", "H", template=True)

    @property
    def width(self) -> float:
        return 360 * self.figure.dpi / 100

    def row(
        self,
        label: str,
        shortcut: str,
        height: float,
        *,
        action: str | None = None,
        template: bool = False,
        destructive: bool = False,
    ) -> ControlRow:
        axes = self.figure.add_axes((0.7, 0.5, 0.2, 0.03))
        axes.set_navigate(False)
        axes.set_xticks([])
        axes.set_yticks([])
        button = None
        if action is not None:
            button = Button(axes, label, color="#f3f6fa", hovercolor="#e1e9f3")
            button.on_clicked(lambda _, name=action: self.review.perform_action(name))
            self.buttons[action] = button
            text = button.label
            text.set_position((0.03, 0.5))
            text.set_horizontalalignment("left")
        else:
            axes.set_axis_off()
            text = axes.text(0.03, 0.5, label, va="center", ha="left")
        text.set_fontsize(11)
        keys = axes.text(0.97, 0.5, shortcut, fontsize=11, va="center", ha="right", color="#475569")
        result = ControlRow(axes, text, keys, height, action, button, template, destructive)
        self.rows.append(result)
        return result

    def heading(self, label: str) -> None:
        row = self.row(label, "", 36)
        row.label.set_fontsize(12)
        row.label.set_fontweight("bold")
        row.label.set_color("#26364b")

    def action(
        self,
        action: str,
        label: str,
        shortcut: str,
        *,
        height: float = 29,
        template: bool = False,
        destructive: bool = False,
    ) -> ControlRow:
        return self.row(label, shortcut, height, action=action, template=template, destructive=destructive)

    def hint(self, label: str, shortcut: str, *, template: bool = False, destructive: bool = False) -> None:
        row = self.row(label, shortcut, 27, template=template, destructive=destructive)
        row.label.set_color("#9f1239" if destructive else "#475569")

    def layout(self) -> None:
        width, height = self.figure.get_size_inches() * self.figure.dpi
        scale = self.figure.dpi / 100
        self._left = width - self.width - 24 * scale
        top, bottom = height - 58 * scale, 16 * scale
        self.title.set_position((self._left / width, 1 - 18 * scale / height))
        self.scroll_hint.set_position(((self._left + self.width) / width, 1 - 20 * scale / height))
        visible = [row for row in self.rows if not row.template_body or self.templates_expanded]
        total = sum(row.height + 3 for row in visible) * scale
        self._max_scroll = max(0.0, total - (top - bottom))
        self.scroll_hint.set_visible(self._max_scroll > 0)
        self.scroll_axes.set_visible(self._max_scroll > 0)
        self.scroll_axes.set_position(
            (
                (self._left + self.width + 8 * scale) / width,
                bottom / height,
                12 * scale / width,
                (top - bottom) / height,
            )
        )
        y = top + (1 - self.slider.val) * self._max_scroll
        for row in self.rows:
            if row.template_body and not self.templates_expanded:
                row.axes.set_visible(False)
                continue
            row_height = row.height * scale
            row.axes.set_position(
                (self._left / width, (y - row_height) / height, self.width / width, row_height / height)
            )
            row.axes.set_visible(y <= top + 0.1 and y - row_height >= bottom - 0.1)
            y -= row_height + 3 * scale
        self.refresh()

    def refresh(self) -> bool:
        changed = False
        for row in self.rows:
            if row.button is None or row.action is None:
                continue
            label = self.review.action_label(row.action)
            enabled = self.review.action_enabled(row.action) and row.axes.get_visible()
            color = "#fff1f2" if row.destructive else "#f3f6fa"
            if row.action == "advanced":
                label = f"Advanced {'▾' if self.templates_expanded else '▸'}"
                color = "#e8edf4"
            if row.label.get_text() != label or row.button.active != enabled:
                changed = True
                if row.axes not in self.changed_axes:
                    self.changed_axes.append(row.axes)
            row.label.set_text(label)
            row.button.set_active(enabled)
            row.button.color = color if enabled else "#f4f4f4"
            row.button.hovercolor = "#ffe1e5" if row.destructive else "#dce8f6"
            row.axes.set_facecolor(row.button.color)
            row.label.set_color(("#9f1239" if row.destructive else "#172a40") if enabled else "#89919b")
            row.shortcut.set_color("#475569" if enabled else "#89919b")
            for spine in row.axes.spines.values():
                spine.set_color("#e1b3bb" if row.destructive and enabled else "#d6dde5")
        progress = self.review.progress_text()
        if self.progress.label.get_text() != progress:
            changed = True
            if self.progress.axes not in self.changed_axes:
                self.changed_axes.append(self.progress.axes)
        self.progress.label.set_text(progress)
        return changed

    def on_scroll(self, event: MouseEvent) -> None:
        if event.x < self._left or self._max_scroll <= 0:
            return
        value = max(0.0, min(1.0, self.slider.val + event.step * 80 * self.figure.dpi / (100 * self._max_scroll)))
        self.slider.set_val(value)
        self.review.invalidate_background()
        self.figure.canvas.draw_idle()
