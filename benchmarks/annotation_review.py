"""Matplotlib template selection, reversible drafts, diagnostics, and optional mpv."""

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent, key_press_handler
from matplotlib.lines import Line2D

from benchmarks.annotation_analysis import PARAMETERS, AcousticAnalysis, Candidate, Proposal, associate_candidates
from benchmarks.annotation_editing import AnnotationDraft, DraftEditor, is_stale, load_draft, new_draft, save_draft
from benchmarks.annotation_templates import (
    TemplateDefinition,
    candidate_onset,
    load_templates,
    refine_proposal,
    save_templates,
    template_checksum,
)
from benchmarks.review_playback import MpvPlayback

SHORTCUTS = (
    "Space play/pause · Home start · Alt+click seek · ←/→ interval · Shift+click add · Ctrl+click remove\n"
    "K/N reference at cursor · Delete reference · R classify · A accept likely · U remove unlikely · C clear\n"
    "G grid · H hide unlikely · Ctrl+Z undo · Ctrl+S save draft"
)
STYLES = {
    "likely": ("green", "o"),
    "ambiguous": ("darkorange", "D"),
    "unlikely": ("gray", "x"),
    "unclassified": ("purple", "v"),
}


class AnnotationReview:
    def __init__(
        self,
        analysis: AcousticAnalysis,
        proposal: Proposal,
        draft: AnnotationDraft,
        draft_path: Path,
        start: float,
        duration: float,
        *,
        templates: TemplateDefinition | None = None,
        template_path: Path | None = None,
        proposal_path: Path | None = None,
        audio_path: Path | None = None,
    ) -> None:
        self.analysis = analysis
        self.proposal = proposal
        definition = (
            templates
            if templates is not None
            else TemplateDefinition(proposal["track_identifier"], proposal["audio_sha256"] or "")
        )
        self.editor = DraftEditor(draft, definition, proposal)
        self.draft_path = draft_path
        self.template_path = template_path
        self.proposal_path = proposal_path
        self.start = start
        self.duration = duration
        self.dirty = False
        self.show_unlikely = True
        self.alternative_grid = False
        self.cursor_time: float | None = None
        self.status = ""
        self.player: MpvPlayback | None = None
        self.displayed_playback_position: float | None = None
        self.playback_lines: list[Line2D] = []
        self.figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        self.axes = list(axes)
        self.figure.subplots_adjust(top=0.85, bottom=0.16, hspace=0.12)
        self.inspection = self.figure.text(
            0.08, 0.025, "Hover near a candidate to inspect original/refined times.", fontsize=9
        )
        self.figure.text(0.08, 0.07, SHORTCUTS, fontsize=9)
        manager = self.figure.canvas.manager
        handler_id = manager.key_press_handler_id if manager is not None else None
        if handler_id is not None:
            self.figure.canvas.mpl_disconnect(handler_id)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.figure.canvas.mpl_connect("close_event", lambda _: self.close())
        self.timer = self.figure.canvas.new_timer(interval=50)
        self.timer.add_callback(self.update_playback)
        if audio_path is not None:
            try:
                self.player = MpvPlayback.start(audio_path)
                if self.player is None:
                    self.status = "Playback unavailable: sudo apt install mpv. Annotation remains available."
                    print(self.status)
                else:
                    self.player.interval(start, min(start + duration, analysis.duration_seconds))
                    self.timer.start()
            except (OSError, RuntimeError) as error:
                self.status = f"Playback unavailable: {error}"
                self.close()
        self.draw()

    @property
    def draft(self) -> AnnotationDraft:
        return self.editor.draft

    def active_grid(self) -> tuple[list[float], list[float], list[Candidate]]:
        beats = (
            self.proposal.get("alternative_beat_positions", [])
            if self.alternative_grid
            else self.proposal["beat_positions"]
        )
        aligned, unsupported = associate_candidates(
            self.proposal["kick_candidates"], np.asarray(beats, dtype=np.float64)
        )
        return beats, unsupported, aligned

    def draw(self) -> None:
        end = min(self.start + self.duration, self.analysis.duration_seconds)
        for axis in self.axes:
            axis.clear()
        rate = PARAMETERS.sample_rate
        first, last = max(0, round(self.start * rate)), min(len(self.analysis.mono), round(end * rate))
        indices = np.arange(first, last, max(1, (last - first) // 30_000))
        self.axes[0].plot(indices / rate, self.analysis.mono[indices], color="0.25", linewidth=0.5)
        self.axes[0].set_ylabel("PCM amplitude")
        mask = (self.analysis.timestamps >= self.start) & (self.analysis.timestamps <= end)
        times = self.analysis.timestamps[mask]
        if len(times) >= 2:
            self.axes[1].pcolormesh(
                times,
                self.analysis.frequencies,
                np.log(self.analysis.low_spectral_power[mask].T + 1e-12),
                shading="auto",
                cmap="magma",
            )
        self.axes[2].plot(times, self.analysis.bass[mask], label="Bass evidence", color="purple")
        self.axes[2].plot(times, self.analysis.broadband[mask], label="Broadband evidence", color="teal", alpha=0.7)
        self.axes[1].set_ylim(20, 500)
        self.axes[1].set_ylabel("Frequency (Hz)")
        self.axes[2].set_ylabel("Evidence / cosine score")
        self.axes[2].set_xlabel("Original recording time (seconds)")
        beats, unsupported, candidates = self.active_grid()
        visible = [c for c in candidates if self.start <= candidate_onset(c) < end]
        for axis in self.axes:
            for beat in beats:
                if self.start <= beat < end:
                    axis.axvline(beat, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
            for label, (color, marker) in STYLES.items():
                if label == "unlikely" and not self.show_unlikely:
                    continue
                for aligned in (True, False):
                    group = [
                        c
                        for c in visible
                        if c.get("classification", "unclassified") == label and c["grid_aligned"] == aligned
                    ]
                    onsets = [candidate_onset(c) for c in group]
                    axis.plot(
                        onsets,
                        [0.95 if aligned else 0.87] * len(onsets),
                        linestyle="",
                        marker=marker,
                        color=color,
                        markerfacecolor=color if aligned else "none",
                        transform=axis.get_xaxis_transform(),
                    )
            missing = [b for b in unsupported if self.start <= b < end]
            axis.plot(missing, [0.70] * len(missing), "x", color="red", transform=axis.get_xaxis_transform())
            accepted = [time for time in self.draft.kick_onsets if self.start <= time < end]
            axis.plot(
                accepted, [0.79] * len(accepted), "+", color="blue", markersize=9, transform=axis.get_xaxis_transform()
            )
            for references, color, marker in (
                (self.editor.templates.positive_onsets, "green", "*"),
                (self.editor.templates.negative_onsets, "crimson", "s"),
            ):
                onsets = [time for time in references if self.start <= time < end]
                axis.plot(
                    onsets,
                    [0.60] * len(onsets),
                    marker=marker,
                    color=color,
                    linestyle="",
                    markersize=8,
                    transform=axis.get_xaxis_transform(),
                )
        for label, (color, marker) in STYLES.items():
            scored = [
                (c, score)
                for c in visible
                if (score := c.get("positive_similarity")) is not None
                and c.get("classification") == label
                and (label != "unlikely" or self.show_unlikely)
            ]
            if scored:
                self.axes[2].scatter(
                    [candidate_onset(c) for c, _ in scored],
                    [score for _, score in scored],
                    color=color,
                    marker=marker,
                    s=18,
                    label=f"{label} cosine",
                )
        handles = [
            Line2D([], [], color=color, marker=marker, linestyle="", label=label)
            for label, (color, marker) in STYLES.items()
        ]
        handles += [
            Line2D([], [], color="blue", marker="+", linestyle="", label="Accepted draft"),
            Line2D([], [], color="green", marker="*", linestyle="", label="Positive template"),
            Line2D([], [], color="crimson", marker="s", linestyle="", label="Negative template"),
            Line2D([], [], color="black", marker="o", linestyle="", label="Filled/high: on grid"),
            Line2D(
                [], [], color="black", marker="o", markerfacecolor="none", linestyle="", label="Hollow/low: off grid"
            ),
            Line2D([], [], color="red", marker="x", linestyle="", label="Unsupported beat"),
        ]
        self.axes[0].legend(handles=handles, loc="lower right", ncols=5, fontsize=7)
        self.axes[2].legend(loc="upper right", fontsize=8)
        self.axes[0].set_xlim(self.start, end)
        self.playback_lines = [axis.axvline(0, color="deepskyblue", linewidth=1.2, visible=False) for axis in self.axes]
        self.displayed_playback_position = None
        if self.cursor_time is not None:
            self.inspect_candidate(self.cursor_time)
        self.update_title()
        self.figure.canvas.draw_idle()

    def update_title(self) -> None:
        start, end = self.axes[0].get_xlim()
        warning = (
            "STALE DRAFT: originating proposals differ; edits preserved. "
            if is_stale(self.draft, self.proposal)
            else ""
        )
        if self.proposal.get("template_definition_sha256") != template_checksum(self.editor.templates):
            warning += "Templates changed: R to classify. "
        self.figure.suptitle(
            f"{self.proposal['track_identifier']} · {self.draft.confirmed_bpm:g} BPM · {start:.3f}–{end:.3f}s · "
            f"{'alternative' if self.alternative_grid else 'primary'} grid{' · unsaved draft' if self.dirty else ''}\n"
            f"{warning}\n{self.status}",
            fontsize=10,
        )

    def current_interval(self) -> tuple[float, float]:
        left, right = self.axes[0].get_xlim()
        return max(0.0, left), min(right, self.analysis.duration_seconds)

    def navigate(self, start: float, duration: float) -> None:
        self.duration = max(0.005, duration)
        self.start = max(0.0, min(start, max(0.0, self.analysis.duration_seconds - self.duration)))
        if self.player is not None:
            self.player.interval(self.start, min(self.start + self.duration, self.analysis.duration_seconds))
        self.draw()

    def persist_templates(self) -> None:
        if self.template_path is not None:
            save_templates(self.template_path, self.editor.templates, self.analysis.duration_seconds)

    def on_motion(self, event: MouseEvent) -> None:
        if event.inaxes not in self.axes or event.xdata is None:
            return
        timestamp = float(event.xdata)
        self.cursor_time = timestamp
        self.inspect_candidate(timestamp)
        self.figure.canvas.draw_idle()

    def inspect_candidate(self, timestamp: float) -> None:
        candidates = self.proposal["kick_candidates"]
        if candidates:
            nearest = min(
                candidates, key=lambda c: min(abs(c["timestamp"] - timestamp), abs(candidate_onset(c) - timestamp))
            )
            refined = nearest.get("refined_timestamp")
            self.inspection.set_text(
                f"Cursor {timestamp:.4f}s · novelty {nearest['timestamp']:.4f}s · "
                f"refined {f'{refined:.4f}s' if refined is not None else 'unavailable'} · "
                f"{nearest.get('classification', 'unclassified')} · cosine {nearest.get('positive_similarity')} · "
                f"search edge {nearest.get('alignment_at_search_boundary', False)}"
            )

    def on_key(self, event: KeyEvent) -> None:
        key = event.key or ""
        try:
            if key in ("right", "left"):
                start, end = self.current_interval()
                self.navigate(start + (end - start) * (1 if key == "right" else -1), end - start)
                return
            if key in (" ", "space"):
                if self.player is not None:
                    self.player.toggle()
                else:
                    self.status = "Playback unavailable: sudo apt install mpv."
            elif key == "home":
                if self.player is not None:
                    self.player.seek(self.current_interval()[0])
            elif key in ("ctrl+s", "control+s"):
                save_draft(self.draft_path, self.draft, self.analysis.duration_seconds)
                self.dirty = False
                self.status = f"Saved draft: {self.draft_path.name}"
            elif key in ("ctrl+z", "control+z"):
                if self.editor.undo():
                    self.persist_templates()
                    self.dirty = True
            elif key.lower() in ("a", "u", "c"):
                self.editor.bulk({"a": "accept", "u": "unlikely", "c": "clear"}[key.lower()], *self.current_interval())
                self.dirty = True
            elif key.lower() in ("k", "n"):
                if self.cursor_time is None:
                    return
                timestamp = self.cursor_time
                references = self.editor.templates.positive_onsets + self.editor.templates.negative_onsets
                selectable = [c["timestamp"] for c in self.proposal["kick_candidates"]] + references
                if not selectable or min(abs(time - timestamp) for time in selectable) > 0.05:
                    self.status = (
                        "Select within 50 ms of a candidate/reference; zoom the waveform for precise placement."
                    )
                else:
                    nearby = min(references, key=lambda time: abs(time - timestamp)) if references else None
                    old = nearby if nearby is not None and abs(nearby - timestamp) <= 0.05 else None
                    self.editor.set_template(timestamp, key.lower() == "k", old)
                    self.persist_templates()
                    self.status = f"Saved reference at {timestamp:.4f}s; R to classify."
            elif key == "delete":
                if self.cursor_time is not None and self.editor.remove_template(self.cursor_time):
                    self.persist_templates()
            elif key.lower() == "r":
                if self.player is not None:
                    self.player.pause()
                self.persist_templates()
                self.status = "Reclassifying acoustic candidates…"
                self.update_title()
                self.figure.canvas.draw()
                self.proposal = refine_proposal(self.analysis, self.proposal, self.editor.templates)
                self.editor.proposal = self.proposal
                if self.proposal_path is not None:
                    from benchmarks.annotate import proposal_summary, write_json

                    write_json(self.proposal_path, self.proposal)
                    write_json(
                        self.proposal_path.with_name(f"{self.proposal['track_identifier']}-summary.json"),
                        proposal_summary(self.proposal),
                    )
                self.status = "Reclassified suggestions; editable draft preserved. A accepts visible likely kicks."
                invalid = self.proposal.get("invalid_positive_templates", []) + self.proposal.get(
                    "invalid_negative_templates", []
                )
                if invalid:
                    self.status += f" Invalid reference signatures at {invalid}."
            elif key.lower() == "g":
                self.alternative_grid = not self.alternative_grid
            elif key.lower() == "h":
                self.show_unlikely = not self.show_unlikely
            else:
                key_press_handler(event, self.figure.canvas, self.figure.canvas.toolbar)
                return
            self.start, end = self.current_interval()
            self.duration = end - self.start
            self.draw()
        except (OSError, ValueError, RuntimeError) as error:
            self.status = str(error)
            self.update_title()
            self.figure.canvas.draw_idle()

    def on_click(self, event: MouseEvent) -> None:
        if event.button != 1 or event.inaxes not in self.axes or event.xdata is None:
            return
        timestamp = float(event.xdata)
        if not 0 <= timestamp < self.analysis.duration_seconds:
            return
        if event.key == "alt":
            if self.player is not None:
                self.player.seek(timestamp)
            return
        if event.key == "shift":
            self.editor.add(timestamp)
        elif event.key in ("control", "ctrl"):
            if not self.editor.remove(timestamp):
                return
        else:
            return
        self.dirty = True
        self.start, end = self.current_interval()
        self.duration = end - self.start
        self.draw()

    def on_release(self, event: MouseEvent) -> None:
        toolbar = self.figure.canvas.toolbar
        if event.inaxes in self.axes and toolbar is not None and toolbar.mode:
            start, end = self.axes[0].get_xlim()
            self.navigate(start, end - start)

    def update_playback(self) -> None:
        if self.player is None:
            return
        try:
            position = self.player.poll()
            if position == self.displayed_playback_position:
                return
            self.displayed_playback_position = position
            for line in self.playback_lines:
                line.set_visible(position is not None)
                if position is not None:
                    line.set_xdata([position, position])
            self.figure.canvas.draw_idle()
        except (OSError, RuntimeError, ValueError) as error:
            self.status = f"Playback stopped: {error}"
            self.close()
            self.update_title()

    def close(self) -> None:
        self.timer.stop()
        if self.player is not None:
            self.player.close()
            self.player = None


def show_review(
    analysis: AcousticAnalysis, proposal: Proposal, draft_path: Path, start: float, duration: float, *, root: Path
) -> None:
    identifier, checksum = proposal["track_identifier"], proposal["audio_sha256"]
    if checksum is None:
        raise ValueError("Review requires an audio checksum.")
    template_path = root / "templates" / f"{identifier}.json"
    templates = load_templates(template_path, identifier, checksum, analysis.duration_seconds)
    draft = load_draft(draft_path, proposal)
    if is_stale(draft, proposal):
        print("STALE DRAFT: originating proposals differ; edits preserved. --reset-draft explicitly discards edits.")
    review = AnnotationReview(
        analysis,
        proposal,
        draft,
        draft_path,
        start,
        duration,
        templates=templates,
        template_path=template_path,
        proposal_path=root / "proposals" / f"{identifier}.json",
        audio_path=root / "audio" / f"{identifier}.wav",
    )
    try:
        if not draft_path.exists():
            save_draft(draft_path, draft, analysis.duration_seconds)
        plt.show()
    finally:
        review.close()
    if review.dirty:
        print("Unsaved draft edits were not saved. Use Ctrl+S before closing to preserve changes.")


class DiagnosticInterval(TypedDict):
    start: float
    end: float
    beats: int
    support: float
    challenge: int
    classification_counts: dict[str, int]
    off_grid_likely: int
    unsupported_beats: int


class DiagnosticSelection(TypedDict):
    regular: DiagnosticInterval | None
    challenging: DiagnosticInterval | None
    phase_discrepancy: DiagnosticInterval | None
    criteria: dict[str, str | bool]
    ranked_intervals: list[DiagnosticInterval]


def diagnostic_sections(proposal: Proposal) -> DiagnosticSelection:
    classified = proposal.get("positive_template_count", 0) > 0
    intervals: list[DiagnosticInterval] = []
    for raw_start in np.arange(0, proposal["duration_seconds"], 10.0):
        start = float(raw_start)
        end = min(start + 10, proposal["duration_seconds"])
        beats = [beat for beat in proposal["beat_positions"] if start <= beat < end]
        candidates = [c for c in proposal["kick_candidates"] if start <= candidate_onset(c) < end]
        unsupported = sum(start <= beat < end for beat in proposal["unsupported_beat_positions"])
        likely = [c for c in candidates if c.get("classification") == "likely"]
        support_candidates = []
        for candidate in proposal["kick_candidates"]:
            if classified and candidate.get("classification") != "likely":
                continue
            copied = candidate.copy()
            copied["timestamp"] = candidate_onset(candidate) if classified else candidate["timestamp"]
            support_candidates.append(copied)
        associated, _ = associate_candidates(support_candidates, np.asarray(beats, dtype=np.float64))
        aligned = sum(c["grid_aligned"] for c in associated)
        ambiguous = sum(c.get("classification") == "ambiguous" for c in candidates)
        offgrid = (
            sum(not c["grid_aligned"] for c in likely) if classified else sum(not c["grid_aligned"] for c in candidates)
        )
        intervals.append(
            {
                "start": float(start),
                "end": end,
                "beats": len(beats),
                "support": aligned / len(beats) if beats else 0,
                "challenge": ambiguous + offgrid + unsupported,
                "classification_counts": {
                    label: sum(c.get("classification", "unclassified") == label for c in candidates) for label in STYLES
                },
                "off_grid_likely": sum(not c["grid_aligned"] for c in likely),
                "unsupported_beats": unsupported,
            }
        )
    eligible = [interval for interval in intervals if interval["beats"] >= 8]
    regular = max(eligible, key=lambda item: (item["support"], -item["start"])) if eligible else None
    remaining = [
        item
        for item in intervals
        if regular is None or item["start"] >= regular["end"] or item["end"] <= regular["start"]
    ]
    challenging = max(remaining, key=lambda item: (item["challenge"], -item["start"])) if remaining else None
    phase_ranked: list[tuple[float, DiagnosticInterval]] = []
    for interval in intervals:
        section = next(
            (
                section
                for section in proposal["beat_grid_validation"]
                if section["start_seconds"] <= interval["start"] < section["end_seconds"]
            ),
            None,
        )
        if section is not None:
            phase_ranked.append((abs(section["phase_difference_seconds"]), interval))
    largest_phase = max(phase_ranked, key=lambda item: (item[0], -item[1]["start"]))[1] if phase_ranked else None
    phase = (
        largest_phase
        if largest_phase is not None
        and all(other is None or largest_phase["start"] != other["start"] for other in (regular, challenging))
        else None
    )
    return {
        "regular": regular,
        "challenging": challenging,
        "phase_discrepancy": phase,
        "criteria": {
            "regular": "highest likely-kick beat support; >=8 beats; earliest tie",
            "challenging": "ambiguous + off-grid likely + unsupported; nonoverlapping regular; earliest tie",
            "phase_discrepancy": "largest absolute 30s local discrepancy, earliest distinct 10s interval",
            "fallback_without_templates": not classified,
        },
        "ranked_intervals": intervals,
    }


def save_diagnostics(
    analysis: AcousticAnalysis, proposal: Proposal, output: Path, templates: TemplateDefinition | None = None
) -> DiagnosticSelection:
    selected = diagnostic_sections(proposal)
    output.mkdir(parents=True, exist_ok=True)
    for label in ("regular", "challenging", "phase_discrepancy"):
        interval = selected[label]
        if interval is None:
            print(f"{proposal['track_identifier']}: no distinct qualifying {label} diagnostic interval")
            continue
        draft = new_draft(proposal)
        review = AnnotationReview(
            analysis, proposal, draft, output / "unused-draft.json", interval["start"], 10, templates=templates
        )
        review.figure.savefig(output / f"{proposal['track_identifier']}-{label}.png", dpi=130)
        review.close()
        plt.close(review.figure)
    return selected
