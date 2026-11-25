"""Correction-first Matplotlib review with retained artists and cursor blitting."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent, key_press_handler
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D

from benchmarks.annotation_analysis import PARAMETERS, AcousticAnalysis, AutomaticCandidate, Proposal
from benchmarks.annotation_automatic import OnsetLocalizer, classify_candidates, grid_distances
from benchmarks.annotation_corrections import (
    CorrectionEditor,
    ReviewDraft,
    centered_interval,
    effective_kick,
    effective_timestamp,
    load_review,
    navigate_candidate,
    new_review,
    proposal_checksum,
    save_review,
)
from benchmarks.annotation_diagnostics import diagnostic_sections
from benchmarks.review_controls import ReviewControlPanel
from benchmarks.review_playback import MpvPlayback

KEY_ACTIONS = {
    " ": "play",
    "space": "play",
    "home": "restart",
    "left": "previous_interval",
    "right": "next_interval",
    "[": "previous_unsupported",
    "]": "next_unsupported",
    ",": "previous_uncertain",
    ".": "next_uncertain",
    "<": "previous_corrected",
    ">": "next_corrected",
    "t": "toggle",
    "ctrl+z": "undo",
    "ctrl+s": "save",
    "r": "reclassify",
    "g": "grid",
    "h": "nonkick_visibility",
    "escape": "clear_override",
}
LABELS = {
    "play": "Play",
    "restart": "Restart interval",
    "previous_interval": "Previous interval",
    "next_interval": "Next interval",
    "previous_uncertain": "Previous uncertain candidate",
    "next_uncertain": "Next uncertain candidate",
    "previous_unsupported": "Previous unsupported beat",
    "next_unsupported": "Next unsupported beat",
    "previous_corrected": "Previous correction",
    "next_corrected": "Next correction",
    "toggle": "Toggle selected event",
    "undo": "Undo correction",
    "save": "Save review",
    "reclassify": "Reclassify using corrections",
    "group_kick": "Similar events: kicks",
    "group_nonkick": "Similar events: non-kicks",
    "clear_override": "Remove selected correction",
    "grid": "Beat grid: primary",
    "nonkick_visibility": "Non-kicks: shown",
    "advanced": "Advanced ▸",
}
STYLES = {
    "automatic_kick": ("#16703c", "o", True, "Automatic kick"),
    "automatic_nonkick": ("#777777", "v", False, "Automatic non-kick"),
    "human_kick": ("#135dba", "s", True, "Human-corrected kick"),
    "human_nonkick": ("#a12842", "X", False, "Human-corrected non-kick"),
    "uncertain": ("#c88410", "D", False, "Uncertain classification"),
}


def waveform_envelope(
    analysis: AcousticAnalysis, start: float, end: float, pixels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max buckets preserve short transients at display resolution."""
    rate = PARAMETERS.sample_rate
    first, last = max(0, int(start * rate)), min(len(analysis.mono), int(end * rate) + 1)
    signal = analysis.mono[first:last]
    bucket = max(1, int(np.ceil(len(signal) / max(1, pixels))))
    if bucket == 1:
        return np.arange(first, last) / rate, signal
    count = len(signal) // bucket
    if not count:
        return np.array([]), np.array([])
    blocks = signal[: count * bucket].reshape(count, bucket)
    x = np.repeat((first + np.arange(count) * bucket) / rate, 2)
    y = np.column_stack((blocks.min(axis=1), blocks.max(axis=1))).ravel()
    if count * bucket < len(signal):
        remainder = signal[count * bucket :]
        x = np.append(x, [(first + count * bucket) / rate] * 2)
        y = np.append(y, [remainder.min(), remainder.max()])
    return x, y


class AnnotationReview:
    def __init__(
        self,
        analysis: AcousticAnalysis,
        proposal: Proposal[AutomaticCandidate],
        draft: ReviewDraft,
        draft_path: Path,
        start: float,
        duration: float,
        *,
        audio_path: Path | None = None,
    ) -> None:
        self.analysis, self.proposal = analysis, proposal
        self.editor = CorrectionEditor(proposal, draft)
        self.draft_path = draft_path
        self.start, self.duration = start, duration
        self.dirty, self.show_nonkicks, self.alternative_grid = False, True, False
        self.selected: str | None = None
        self.cursor_time: float | None = None
        self.status = "Click an event to toggle kick / non-kick. Predictions are unreviewed."
        self.pending_group: tuple[str, bool, tuple[str, ...]] | None = None
        self.player: MpvPlayback | None = None
        self.displayed_playback_position: float | None = None
        self._background = None
        self._static_background = None
        self._updating_range = False
        self._visible_range: tuple[float, float, int] | None = None
        self._inspected_key: str | None = None
        self.audio_update_count = 0
        self.stale = draft.originating_proposal_sha256 != proposal_checksum(proposal)
        self.log_spectrogram = 10 * np.log10(analysis.low_spectral_power.T + 1e-12)
        self.grids = {
            "primary": np.asarray(proposal["beat_positions"]),
            "alternative": np.asarray(proposal.get("alternative_beat_positions", [])),
        }
        self.localizer: OnsetLocalizer | None = None
        self.figure, raw_axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
        self.axes = list(raw_axes)
        (self.waveform,) = self.axes[0].plot([], [], color=".25", linewidth=0.6)
        self.spectrogram = self.axes[1].imshow(
            np.zeros((len(analysis.frequencies), 2)),
            origin="lower",
            aspect="auto",
            extent=(0, 0.01, 20, 500),
            cmap="magma",
            vmin=-90,
            vmax=-15,
            interpolation="nearest",
        )
        (self.bass_line,) = self.axes[2].plot([], [], color="#47689d", linewidth=1)
        (self.broad_line,) = self.axes[2].plot([], [], color="#9c667c", linewidth=0.8)
        self.score_points = self.axes[2].scatter([], [], s=13, color="#196c49", marker=".")
        self.beat_lines = [
            axis.vlines([], 0, 1, transform=axis.get_xaxis_transform(), color=".65", linestyles="--", linewidth=0.6)
            for axis in self.axes
        ]
        self.overlays = {
            name: [
                axis.scatter(
                    [],
                    [],
                    transform=axis.get_xaxis_transform(),
                    marker=marker,
                    s=52 if name != "uncertain" else 85,
                    edgecolors=color,
                    facecolors=color if filled else "none",
                    linewidths=1.2,
                    zorder=5,
                )
                for axis in self.axes
            ]
            for name, (color, marker, filled, _) in STYLES.items()
        }
        self.grid_markers = [
            axis.scatter([], [], transform=axis.get_xaxis_transform(), marker="|", s=35, color=".15", zorder=5)
            for axis in self.axes
        ]
        self.offgrid_markers = [
            axis.scatter([], [], transform=axis.get_xaxis_transform(), marker="_", s=35, color=".15", zorder=5)
            for axis in self.axes
        ]
        self.selected_lines = [axis.axvline(0, color="#b52a87", linewidth=1.8, visible=False) for axis in self.axes]
        self.playback_lines = [
            axis.axvline(0, color="#00a5c8", linewidth=1.5, visible=False, animated=self.figure.canvas.supports_blit)
            for axis in self.axes
        ]
        for axis, label in zip(self.axes, ("PCM amplitude", "Frequency (Hz)", "Evidence / similarity"), strict=True):
            axis.set_ylabel(label, fontsize=11)
            axis.tick_params(labelsize=11)
        self.axes[1].set_ylim(20, 500)
        self.axes[2].set_xlabel("Recording time (seconds)", fontsize=11)
        self.title_text = self.figure.text(0.065, 0.97, "", fontsize=13, va="top")
        self.warning_text = self.figure.text(0.065, 0.935, "", fontsize=11, va="top", color="#8b2f45")
        self.inspection = self.figure.text(0.065, 0.255, "", fontsize=11, va="top")
        self.dynamic_artists = [
            self.score_points,
            self.title_text,
            self.warning_text,
            self.inspection,
            *self.selected_lines,
            *self.grid_markers,
            *self.offgrid_markers,
            *(artist for group in self.overlays.values() for artist in group),
        ]
        self.use_blitting = self.figure.canvas.supports_blit and isinstance(self.figure.canvas, FigureCanvasAgg)
        for artist in self.dynamic_artists + self.playback_lines:
            artist.set_animated(self.use_blitting)
        self.legend = self.figure.legend(
            handles=self.legend_handles(),
            loc="lower left",
            bbox_to_anchor=(0.045, 0.01),
            ncol=3,
            fontsize=11,
            frameon=False,
            columnspacing=1.3,
            handlelength=1.8,
        )
        self.controls = ReviewControlPanel(self)
        self.controls.progress.axes.set_animated(self.use_blitting)
        self.dynamic_artists.append(self.controls.progress.axes)
        self._ordered: list[AutomaticCandidate] = []
        self._times = np.array([], dtype=np.float64)
        self._kick_count, self._uncertain_count = 0, 0
        self._clusters: dict[int | None, list[AutomaticCandidate]] = {}
        self._queues: dict[str, tuple[list[str], np.ndarray]] = {}
        self._grid_alignment: dict[str, dict[str, bool]] = {}
        self.refresh_cache()
        self.layout()
        manager = self.figure.canvas.manager
        if manager is not None and manager.key_press_handler_id is not None:
            self.figure.canvas.mpl_disconnect(manager.key_press_handler_id)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.figure.canvas.mpl_connect("resize_event", self.on_resize)
        self.figure.canvas.mpl_connect("draw_event", self.on_draw)
        self.figure.canvas.mpl_connect("close_event", lambda _: self.close())
        self.axes[0].callbacks.connect("xlim_changed", lambda _: self.range_changed())
        self.timer = self.figure.canvas.new_timer(interval=50)
        self.timer.add_callback(self.update_playback)
        if audio_path is not None:
            try:
                self.player = MpvPlayback.start(audio_path)
                if self.player is None:
                    self.status = "Playback unavailable: sudo apt install mpv. Review remains available."
                else:
                    self.player.interval(start, min(start + duration, analysis.duration_seconds))
                    self.timer.start()
            except (OSError, RuntimeError) as error:
                self.status = f"Playback unavailable: {error}"
        self.navigate(start, duration)

    @property
    def draft(self) -> ReviewDraft:
        return self.editor.draft

    def legend_handles(self) -> list[Line2D]:
        handles = [
            Line2D([], [], marker=m, color=c, linestyle="none", markerfacecolor=c if filled else "none", label=label)
            for c, m, filled, label in STYLES.values()
        ]
        handles.extend(
            [
                Line2D([], [], color=".65", linestyle="--", label="Estimated beat"),
                Line2D([], [], color="#b52a87", label="Selected candidate"),
                Line2D([], [], marker="|", color=".15", linestyle="none", label="Grid aligned (independent)"),
                Line2D([], [], marker="_", color=".15", linestyle="none", label="Off grid (independent)"),
                Line2D([], [], color="#00a5c8", label="Playback position"),
                Line2D([], [], color="#47689d", label="Bass-onset evidence"),
                Line2D([], [], color="#9c667c", label="Broadband evidence"),
            ]
        )
        return handles

    def layout(self) -> None:
        width, _ = self.figure.get_size_inches() * self.figure.dpi
        right = (width - self.controls.width - 52) / width
        self.figure.subplots_adjust(left=0.065, right=right, top=0.88, bottom=0.33, hspace=0.16)
        self.controls.layout()
        self.invalidate_background()

    def on_resize(self, _: object) -> None:
        self._visible_range = None
        self.layout()
        self.draw()

    def invalidate_background(self) -> None:
        self._background = None
        self._static_background = None

    def refresh_cache(self) -> None:
        self._ordered = sorted(
            self.editor.candidates.values(), key=lambda c: (effective_timestamp(c, self.draft), c["candidate_id"])
        )
        self._times = np.asarray([effective_timestamp(c, self.draft) for c in self._ordered])
        self._kick_count = sum(effective_kick(c, self.draft) for c in self._ordered)
        self._uncertain_count = sum(
            bool(c["uncertainty_reasons"]) and c["candidate_id"] not in self.draft.overrides for c in self._ordered
        )
        self._clusters = {}
        for candidate in self._ordered:
            self._clusters.setdefault(candidate["cluster_id"], []).append(candidate)
        for name, beats in self.grids.items():
            distances, _ = grid_distances(self._times, beats.tolist())
            self._grid_alignment[name] = {
                c["candidate_id"]: abs(distance) <= 0.050 for c, distance in zip(self._ordered, distances, strict=True)
            }
        kicks = np.asarray(
            sorted(effective_timestamp(c, self.draft) for c in self._ordered if effective_kick(c, self.draft))
        )
        for kind in ("uncertain", "corrected", "unsupported_primary", "unsupported_alternative"):
            remaining = []
            for candidate in self._ordered:
                key = candidate["candidate_id"]
                corrected = key in self.draft.overrides
                eligible = (
                    corrected if kind == "corrected" else not corrected and bool(candidate["uncertainty_reasons"])
                )
                if kind.startswith("unsupported_"):
                    time = effective_timestamp(candidate, self.draft)
                    index = int(np.searchsorted(kicks, time))
                    covered = any(abs(time - kicks[i]) <= 0.050 for i in (index - 1, index) if 0 <= i < len(kicks))
                    eligible = (
                        not corrected
                        and not covered
                        and kind.removeprefix("unsupported_") in candidate["unsupported_grids"]
                    )
                if eligible:
                    remaining.append(candidate)
            self._queues[kind] = (
                [c["candidate_id"] for c in remaining],
                np.asarray([effective_timestamp(c, self.draft) for c in remaining]),
            )

    def visible_candidates(self, start: float, end: float) -> list[AutomaticCandidate]:
        first, last = np.searchsorted(self._times, [start, end])
        return self._ordered[int(first) : int(last)]

    def current_interval(self) -> tuple[float, float]:
        left, right = self.axes[0].get_xlim()
        return max(0.0, left), min(self.analysis.duration_seconds, right)

    def update_audio(self) -> None:
        start, end = self.current_interval()
        pixels = max(1, int(self.axes[0].bbox.width))
        key = (start, end, pixels)
        if key == self._visible_range:
            return
        self._visible_range = key
        self.audio_update_count += 1
        x, y = waveform_envelope(self.analysis, start, end, pixels)
        self.waveform.set_data(x, y)
        limit = max(0.01, float(np.max(np.abs(y)))) if len(y) else 1
        self.axes[0].set_ylim(-limit * 1.12, limit * 1.12)
        first, last = np.searchsorted(self.analysis.timestamps, [start, end])
        first, last = int(first), min(len(self.analysis.timestamps), max(int(last), int(first) + 1))
        times = self.analysis.timestamps[first:last]
        if len(times):
            self.spectrogram.set_data(self.log_spectrogram[:, first:last])
            self.spectrogram.set_extent((float(times[0]) - 0.0025, float(times[-1]) + 0.0025, 20, 500))
        self.bass_line.set_data(times, self.analysis.bass[first:last])
        self.broad_line.set_data(times, self.analysis.broadband[first:last])
        evidence = max(
            1.1,
            float(np.max(self.analysis.bass[first:last], initial=0)),
            float(np.max(self.analysis.broadband[first:last], initial=0)),
        )
        self.axes[2].set_ylim(0, evidence * 1.15)

    def update_overlays(self) -> None:
        start, end = self.current_interval()
        visible = self.visible_candidates(start, end)
        groups: dict[str, list[float]] = {key: [] for key in STYLES}
        aligned, offgrid, scores = [], [], []
        beats = self.grids["alternative" if self.alternative_grid else "primary"]
        for candidate in visible:
            kick = effective_kick(candidate, self.draft)
            key = candidate["candidate_id"]
            time = effective_timestamp(candidate, self.draft)
            if not kick and not self.show_nonkicks and key != self.selected:
                continue
            group = ("human_" if key in self.draft.overrides else "automatic_") + ("kick" if kick else "nonkick")
            groups[group].append(time)
            if candidate["uncertainty_reasons"] and key not in self.draft.overrides:
                groups["uncertain"].append(time)
            on_grid = self._grid_alignment["alternative" if self.alternative_grid else "primary"][key]
            (aligned if on_grid else offgrid).append(time)
            score = candidate.get("positive_similarity")
            if score is not None:
                scores.append((time, score))
        for name, times in groups.items():
            positions = np.column_stack((times, np.full(len(times), 0.92))) if times else np.empty((0, 2))
            for artist in self.overlays[name]:
                artist.set_offsets(positions)
        for collection, times in ((self.grid_markers, aligned), (self.offgrid_markers, offgrid)):
            for artist in collection:
                artist.set_offsets(np.column_stack((times, np.full(len(times), 0.82))) if times else np.empty((0, 2)))
        self.score_points.set_offsets(np.asarray(scores).reshape(-1, 2))
        first, last = np.searchsorted(beats, [start, end])
        for artist in self.beat_lines:
            artist.set_segments([[(t, 0), (t, 1)] for t in beats[first:last]])
        for line in self.selected_lines:
            line.set_visible(self.selected in self.editor.candidates)
            if self.selected in self.editor.candidates:
                line.set_xdata([effective_timestamp(self.editor.candidates[self.selected], self.draft)] * 2)

    def update_text(self) -> None:
        start, end = self.current_interval()
        self.title_text.set_text(
            f"{self.proposal['track_identifier']} · {start:.2f}–{end:.2f}s · BPM {self.proposal['bpm']}"
            + (" · unsaved" if self.dirty else "")
        )
        warning = "STALE REVIEW: matched human corrections preserved. " if self.stale else ""
        self.warning_text.set_text(warning + self.status)
        if self.selected is not None:
            self.inspect_candidate(self.selected, redraw=False)
        self.controls.refresh()

    def draw(self) -> None:
        previous_range = self._visible_range
        self.update_audio()
        self.update_overlays()
        self.update_text()
        if self._visible_range != previous_range:
            self.invalidate_background()
        self.render_dynamic()

    def range_changed(self) -> None:
        if not self._updating_range:
            self.draw()

    def navigate(self, start: float, duration: float) -> None:
        self.duration = min(max(0.005, duration), self.analysis.duration_seconds)
        self.start = min(max(0.0, start), self.analysis.duration_seconds - self.duration)
        self._updating_range = True
        self.axes[0].set_xlim(self.start, self.start + self.duration)
        self._updating_range = False
        if self.player is not None:
            self.player.interval(self.start, self.start + self.duration)
        self.draw()

    def nearest(self, timestamp: float) -> str | None:
        index = int(np.searchsorted(self._times, timestamp))
        choices = [i for i in (index - 1, index) if 0 <= i < len(self._times)]
        if not choices:
            return None
        best = min(choices, key=lambda i: abs(self._times[i] - timestamp))
        return self._ordered[best]["candidate_id"] if abs(self._times[best] - timestamp) <= 0.050 else None

    def inspect_candidate(self, key: str | None, *, redraw: bool = True) -> None:
        if key is None:
            return
        c = self.editor.candidates[key]
        members = self._clusters.get(c["cluster_id"], []) if c["cluster_id"] is not None else []
        cluster = (
            f"Cluster {c['cluster_id']}: {len(members)} similar, {sum(effective_kick(m, self.draft) for m in members)} kicks, {sum(m['candidate_id'] in self.draft.overrides for m in members)} corrected"
            if members
            else "No recurring cluster"
        )
        label = "Kick" if effective_kick(c, self.draft) else "Non-kick"
        text = f"Onset {effective_timestamp(c, self.draft):.4f}s · {label} · {c['timing_reason']}\n{cluster}"
        if self.controls.templates_expanded:
            text += f"\nNovelty {c['timestamp']:.4f}s · localized {c['localized_timestamp']:.4f}s · aligned {c.get('refined_timestamp')}"
        if text != self.inspection.get_text():
            self.inspection.set_text(text)
            if redraw:
                self.render_dynamic()

    def on_motion(self, event: MouseEvent) -> None:
        if event.inaxes in self.axes and event.xdata is not None:
            self.cursor_time = float(event.xdata)
            key = self.nearest(self.cursor_time)
            if key is not None and self._inspected_key != key:
                self._inspected_key = key
                self.inspect_candidate(key)

    def on_click(self, event: MouseEvent) -> None:
        if event.inaxes not in self.axes or event.xdata is None or event.button != 1:
            return
        if self.figure.canvas.toolbar is not None and self.figure.canvas.toolbar.mode:
            return
        time = float(event.xdata)
        if "alt" in (event.key or ""):
            if self.player is not None:
                self.player.seek(time)
        elif "shift" in (event.key or ""):
            self.insert_manual(time)
        elif (key := self.nearest(time)) is not None:
            self.selected = key
            self.perform_action("toggle")

    def on_key(self, event: KeyEvent) -> None:
        if (action := KEY_ACTIONS.get((event.key or "").lower())) is not None:
            self.perform_action(action)
        else:
            key_press_handler(event, self.figure.canvas, self.figure.canvas.toolbar)

    def action_label(self, action: str) -> str:
        if action == "play":
            return "Pause" if self.player is not None and not self.player.paused else "Play"
        if action == "grid":
            return f"Beat grid: {'alternative' if self.alternative_grid else 'primary'}"
        if action == "nonkick_visibility":
            return f"Non-kicks: {'shown' if self.show_nonkicks else 'hidden'}"
        if (
            self.pending_group is not None
            and action in ("group_kick", "group_nonkick")
            and self.pending_group[1] == (action == "group_kick")
        ):
            return f"Confirm {len(self.pending_group[2])} {'kicks' if self.pending_group[1] else 'non-kicks'}"
        return LABELS[action]

    def action_enabled(self, action: str) -> bool:
        if action.startswith(("previous_", "next_")) and not action.endswith("interval"):
            return self.navigation_target(action) is not None
        if action in ("play", "restart"):
            return self.player is not None
        if action in ("toggle", "group_kick", "group_nonkick", "clear_override"):
            if self.selected not in self.editor.candidates:
                return False
            if action == "clear_override":
                return self.selected in self.draft.overrides
            if action.startswith("group"):
                return bool(self.editor.group_targets(self.selected))
        if action == "undo":
            return bool(self.editor.history)
        if action == "grid":
            return bool(self.grids["alternative"].size)
        if action == "reclassify":
            return bool(self.draft.overrides)
        return True

    def navigation_target(self, action: str) -> str | None:
        direction, kind = action.split("_", 1)
        if kind == "unsupported":
            kind += "_alternative" if self.alternative_grid else "_primary"
        keys, times = self._queues[kind]
        selected = self.editor.candidates.get(self.selected) if self.selected is not None else None
        position = effective_timestamp(selected, self.draft) if selected is not None else self.current_interval()[0]
        index = int(
            np.searchsorted(times, position, side="right" if selected is not None and direction == "next" else "left")
        )
        if direction == "previous":
            index -= 1
        return keys[index] if 0 <= index < len(keys) else None

    def progress_text(self) -> str:
        return (
            f"{len(self._ordered)} candidates\n{self._kick_count} kicks · {len(self._ordered) - self._kick_count} non-kicks\n"
            f"{len(self.draft.overrides)} human corrections\n{self._uncertain_count} uncertain, uncorrected\nCorrection count is not review coverage."
        )

    def perform_action(self, action: str) -> None:
        if not self.action_enabled(action):
            if action.startswith(("previous_", "next_")) and not action.endswith("interval"):
                self.status = (
                    f"No {'further' if action.startswith('next') else 'earlier'} {action.split('_', 1)[1]} candidates."
                )
                self.update_text()
                self.render_dynamic()
            return
        try:
            if action == "play" and self.player is not None:
                self.player.toggle()
                return
            if action == "restart" and self.player is not None:
                self.player.interval(*self.current_interval())
                return
            if action in ("previous_interval", "next_interval"):
                left, right = self.current_interval()
                self.navigate(left + (right - left) * (1 if action.startswith("next") else -1), right - left)
                return
            if action.startswith(("previous_", "next_")):
                kind = action.split("_", 1)[1]
                key = navigate_candidate(
                    self._ordered,
                    self.draft,
                    kind,
                    1 if action.startswith("next") else -1,
                    self.selected,
                    self.current_interval()[0],
                    "alternative" if self.alternative_grid else "primary",
                )
                if key is None:
                    self.status = f"No {'further' if action.startswith('next') else 'earlier'} {kind} candidates."
                else:
                    self.selected = key
                    self.navigate(
                        *centered_interval(
                            effective_timestamp(self.editor.candidates[key], self.draft), self.analysis.duration_seconds
                        )
                    )
                    self.inspect_candidate(key)
                    return
            elif action == "toggle" and self.selected is not None:
                self.editor.toggle(self.selected)
                self.dirty = True
            elif action == "clear_override" and self.selected is not None:
                self.editor.clear_override(self.selected)
                self.dirty = True
            elif action == "undo":
                self.editor.undo()
                self.dirty = True
            elif action == "save":
                save_review(self.draft_path, self.draft, self.proposal)
                self.dirty, self.stale = False, False
                self.status = "Review saved. Full-recording confirmation still required to finalize."
            elif action == "reclassify":
                self.reclassify()
            elif action in ("group_kick", "group_nonkick") and self.selected is not None:
                targets = tuple(self.editor.group_targets(self.selected))
                request = (self.selected, action == "group_kick", targets)
                if self.pending_group == request:
                    count = self.editor.group_correct(self.selected, action == "group_kick")
                    self.pending_group = None
                    self.dirty = True
                    self.status = f"Corrected {count} similar events; existing explicit decisions preserved."
                else:
                    self.pending_group = request
                    self.status = (
                        f"Will change {len(targets)} uncorrected events. Click the same action again to confirm."
                    )
            elif action == "advanced":
                self.controls.templates_expanded = not self.controls.templates_expanded
                self.controls.layout()
                self.invalidate_background()
            elif action == "grid":
                self.alternative_grid = not self.alternative_grid
                self.invalidate_background()
            elif action == "nonkick_visibility":
                self.show_nonkicks = not self.show_nonkicks
            self.refresh_cache()
            self.draw()
        except (OSError, ValueError, RuntimeError) as error:
            self.status = str(error)
            self.update_text()
            self.figure.canvas.draw_idle()

    def insert_manual(self, time: float) -> None:
        if self.localizer is None:
            self.localizer = OnsetLocalizer(self.analysis)
        localized, reliable, reason, increase = self.localizer.localize(time)
        key = "manual-" + hashlib.sha256(f"{self.proposal['audio_sha256']}:{time:.9f}".encode()).hexdigest()[:20]
        candidate: AutomaticCandidate = {
            "candidate_id": key,
            "timestamp": time,
            "novelty_timestamp": time,
            "localized_timestamp": localized,
            "final_timestamp": localized,
            "localization_reliable": reliable,
            "timing_reason": reason,
            "low_frequency_increase": increase,
            "provenance": "manual",
            "bass_evidence": 0.0,
            "prominence": 0.0,
            "broadband_evidence": 0.0,
            "nearest_beat_distance_seconds": None,
            "grid_aligned": False,
            "unsupported_grids": [],
            "automatic_kick": False,
            "low_confidence": True,
            "cluster_id": None,
            "uncertainty_reasons": [],
            "classification": "unclassified",
        }
        if self.editor.insert(candidate):
            self.selected, self.dirty = key, True
            self.status = "Manually identified event; onset localized automatically."
            self.refresh_cache()
            self.draw()
        else:
            self.status = "Existing candidate within 30ms. Toggle it instead."
            self.update_text()
            self.figure.canvas.draw_idle()

    def reclassify(self) -> None:
        references = set(self.proposal.get("automatic_reference_ids", []))
        positives = [
            c["localized_timestamp"]
            for c in self.proposal["kick_candidates"]
            if c["candidate_id"] in references
            and (c["candidate_id"] not in self.draft.overrides or self.draft.overrides[c["candidate_id"]].kick)
        ]
        positives += [c.timestamp for c in self.draft.overrides.values() if c.kick]
        negatives = [c.timestamp for c in self.draft.overrides.values() if not c.kick]
        changed = self.editor.replace_predictions(
            classify_candidates(self.analysis, self.editor.predictions, positives, negatives)
        )
        self.dirty = True
        self.status = f"Changed {changed} automatic predictions. Human decisions and their timestamps preserved."

    def on_draw(self, _: object) -> None:
        if self.use_blitting and isinstance(self.figure.canvas, FigureCanvasAgg):
            self.controls.changed_axes.clear()
            self._static_background = self.figure.canvas.copy_from_bbox(self.figure.bbox)
            # draw_event may run inside Qt's paintEvent: never recursively blit here.
            self.render_dynamic(blit=False)

    def render_dynamic(self, *, blit: bool = True) -> None:
        canvas = self.figure.canvas
        if self.use_blitting and isinstance(canvas, FigureCanvasAgg) and self._static_background is not None:
            canvas.restore_region(self._static_background)
            for axes in self.controls.changed_axes:
                if axes.get_visible() and axes is not self.controls.progress.axes:
                    self.figure.draw_artist(axes)
            if self.controls.changed_axes:
                self._static_background = canvas.copy_from_bbox(self.figure.bbox)
                self.controls.changed_axes.clear()
            for artist in self.dynamic_artists:
                self.figure.draw_artist(artist)
            self._background = canvas.copy_from_bbox(self.figure.bbox)
            self.render_cursor(blit=blit)
        else:
            canvas.draw_idle()

    def render_cursor(self, *, blit: bool = True) -> None:
        canvas = self.figure.canvas
        if canvas.supports_blit and isinstance(canvas, FigureCanvasAgg) and self._background is not None:
            canvas.restore_region(self._background)
            for axis, line in zip(self.axes, self.playback_lines, strict=True):
                if line.get_visible():
                    axis.draw_artist(line)
            if blit:
                canvas.blit(self.figure.bbox)
        else:
            canvas.draw_idle()

    def update_playback(self) -> None:
        if self.player is None:
            return
        try:
            position = self.player.poll()
            if self.controls.refresh():
                self.render_dynamic()
            if position is None or position == self.displayed_playback_position:
                return
            self.displayed_playback_position = position
            for line in self.playback_lines:
                line.set_xdata([position] * 2)
                line.set_visible(True)
            self.render_cursor()
        except (OSError, RuntimeError) as error:
            self.status = f"Playback stopped: {error}"
            self.close()
            self.update_text()

    def close(self) -> None:
        self.timer.stop()
        if self.player is not None:
            self.player.close()
            self.player = None


def show_review(
    analysis: AcousticAnalysis,
    proposal: Proposal[AutomaticCandidate],
    draft_path: Path,
    start: float,
    duration: float,
    *,
    root: Path,
) -> None:
    draft = load_review(draft_path, proposal)
    view = AnnotationReview(
        analysis,
        proposal,
        draft,
        draft_path,
        start,
        duration,
        audio_path=root / "audio" / f"{proposal['track_identifier']}.wav",
    )
    try:
        if not draft_path.exists():
            save_review(draft_path, draft, proposal)
        plt.show()
    finally:
        view.close()
    if view.dirty:
        print("Unsaved corrections remain; use Ctrl+S before closing.")


def save_diagnostics(
    analysis: AcousticAnalysis, proposal: Proposal[AutomaticCandidate], output: Path
) -> dict[str, object]:
    selected = diagnostic_sections({**proposal, "kick_candidates": list(proposal["kick_candidates"])})
    output.mkdir(parents=True, exist_ok=True)
    for label in ("regular", "challenging", "phase_discrepancy"):
        interval = selected[label]
        if interval is not None:
            review = AnnotationReview(
                analysis, proposal, new_review(proposal), output / "unused.json", interval["start"], 10
            )
            review.figure.savefig(output / f"{proposal['track_identifier']}-{label}.png", dpi=110)
            review.close()
            plt.close(review.figure)
    clusters = sorted(proposal.get("clusters", []), key=lambda c: -int(str(c["member_count"])))[:6]
    if clusters:
        from benchmarks.annotation_templates import event_feature

        by_id = {c["candidate_id"]: c for c in proposal["kick_candidates"]}
        figure, axes = plt.subplots(len(clusters), 1, figsize=(10, 2 * len(clusters)), squeeze=False)
        for axis, cluster in zip(axes[:, 0], clusters, strict=True):
            candidate = by_id[str(cluster["medoid_candidate_id"])]
            feature = event_feature(analysis, candidate["localized_timestamp"])
            if feature is not None:
                axis.imshow(feature.reshape(-1, 12).T, aspect="auto", origin="lower", extent=(-20, 160, 0, 12))
            axis.set_title(
                f"Cluster {cluster['cluster_id']} · {cluster['member_count']} events · reference {candidate['localized_timestamp']:.3f}s"
            )
        figure.tight_layout()
        figure.savefig(output / f"{proposal['track_identifier']}-competing-signatures.png")
        plt.close(figure)
    return dict(selected)
