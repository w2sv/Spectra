"""Matplotlib review and editable drafts, separate from proposals and ground truth."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent, key_press_handler
from matplotlib.lines import Line2D

from benchmarks.annotation_analysis import PARAMETERS, AcousticAnalysis, Proposal
from benchmarks.tracks import validate_timestamps


@dataclass
class AnnotationDraft:
    track_identifier: str
    audio_sha256: str
    annotation_algorithm_version: int
    confirmed_bpm: float
    kick_onsets: list[float]
    manually_confirmed_kick_onsets: list[float]
    review_status: str = "draft"


def load_draft(path: Path, proposal: Proposal) -> AnnotationDraft:
    if path.exists():
        draft = AnnotationDraft(**json.loads(path.read_text(encoding="utf-8")))
        if draft.track_identifier != proposal["track_identifier"] or draft.audio_sha256 != proposal["audio_sha256"]:
            raise ValueError("Draft does not refer to this recording and checksum.")
        validate_timestamps(draft.kick_onsets, proposal["duration_seconds"])
        validate_timestamps(draft.manually_confirmed_kick_onsets, proposal["duration_seconds"])
        return draft
    if proposal["audio_sha256"] is None or proposal["bpm"] is None:
        raise ValueError("Real recording drafts require a checksum and BPM.")
    return AnnotationDraft(
        proposal["track_identifier"],
        proposal["audio_sha256"],
        proposal["annotation_algorithm_version"],
        proposal["bpm"],
        [candidate["timestamp"] for candidate in proposal["kick_candidates"]],
        [],
    )


def save_draft(path: Path, draft: AnnotationDraft, duration: float) -> None:
    validate_timestamps(draft.kick_onsets, duration)
    validate_timestamps(draft.manually_confirmed_kick_onsets, duration)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(asdict(draft), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class AnnotationReview:
    def __init__(
        self,
        analysis: AcousticAnalysis,
        proposal: Proposal,
        draft: AnnotationDraft,
        draft_path: Path,
        start: float,
        duration: float,
    ) -> None:
        self.analysis = analysis
        self.proposal = proposal
        self.draft = draft
        self.draft_path = draft_path
        self.start = start
        self.duration = duration
        self.figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, layout="constrained")
        self.axes = list(axes)
        self.dirty = False
        manager = self.figure.canvas.manager
        handler_id = manager.key_press_handler_id if manager is not None else None
        if handler_id is not None:
            self.figure.canvas.mpl_disconnect(handler_id)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.draw()

    def draw(self) -> None:
        end = min(self.start + self.duration, self.analysis.duration_seconds)
        for axis in self.axes:
            axis.clear()
        rate = PARAMETERS.sample_rate
        first, last = round(self.start * rate), min(len(self.analysis.mono), round(end * rate))
        # Bound drawing cost at long zoom ranges; amplitude values remain unchanged.
        stride = max(1, (last - first) // 30_000)
        indices = np.arange(first, last, stride)
        self.axes[0].plot(indices / rate, self.analysis.mono[indices], color="0.25", linewidth=0.5)
        self.axes[0].set_ylabel("PCM amplitude")
        mask = (self.analysis.timestamps >= self.start) & (self.analysis.timestamps <= end)
        times = self.analysis.timestamps[mask]
        if len(times):
            self.axes[1].pcolormesh(
                times,
                self.analysis.frequencies,
                np.log(self.analysis.low_spectral_power[mask].T + 1e-12),
                shading="auto",
                cmap="magma",
            )
            self.axes[2].plot(times, self.analysis.bass[mask], label="Bass evidence", color="purple")
            self.axes[2].plot(times, self.analysis.broadband[mask], label="Broadband evidence", color="teal", alpha=0.8)
        self.axes[1].set_ylim(20, 500)
        self.axes[1].set_ylabel("Frequency (Hz)")
        self.axes[2].set_ylabel("Novelty evidence")
        self.axes[2].set_xlabel("Original recording time (seconds)")
        beats = [beat for beat in self.proposal["beat_positions"] if self.start <= beat <= end]
        unsupported = [beat for beat in self.proposal["unsupported_beat_positions"] if self.start <= beat <= end]
        for axis in self.axes:
            for beat in beats:
                axis.axvline(beat, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
            for aligned, color in ((True, "green"), (False, "orange")):
                original = [
                    c["timestamp"]
                    for c in self.proposal["kick_candidates"]
                    if c["grid_aligned"] == aligned and self.start <= c["timestamp"] <= end
                ]
                pending = [
                    time
                    for time in original
                    if time in self.draft.kick_onsets and time not in self.draft.manually_confirmed_kick_onsets
                ]
                axis.plot(
                    original, [0.95] * len(original), "v", color=color, alpha=0.2, transform=axis.get_xaxis_transform()
                )
                axis.plot(pending, [0.95] * len(pending), "v", color=color, transform=axis.get_xaxis_transform())
            axis.plot(unsupported, [0.88] * len(unsupported), "x", color="red", transform=axis.get_xaxis_transform())
            manual = [time for time in self.draft.manually_confirmed_kick_onsets if self.start <= time <= end]
            axis.plot(manual, [0.8] * len(manual), "o", color="blue", transform=axis.get_xaxis_transform())
        handles = [
            Line2D([], [], color="gray", linestyle="--", label="Estimated beat"),
            Line2D([], [], color="green", marker="v", linestyle="", label="Grid proposal"),
            Line2D([], [], color="orange", marker="v", linestyle="", label="Off-grid proposal"),
            Line2D([], [], color="red", marker="x", linestyle="", label="Unsupported beat"),
            Line2D([], [], color="blue", marker="o", linestyle="", label="Manually confirmed"),
        ]
        self.axes[0].legend(handles=handles, loc="lower right", ncols=3, fontsize="small")
        self.axes[2].legend(loc="upper right")
        self.axes[0].set_xlim(self.start, end)
        self.update_title(self.start, end)
        self.figure.canvas.draw_idle()

    def update_title(self, start: float, end: float) -> None:
        self.figure.suptitle(
            f"{self.proposal['track_identifier']} · {self.draft.confirmed_bpm:g} BPM · {start:.2f}–{end:.2f}s"
            f"{' · unsaved draft' if self.dirty else ''}\n"
            "←/→ interval · Shift+click add · Ctrl+click remove · Ctrl+S save draft"
        )

    def on_key(self, event: KeyEvent) -> None:
        if event.key in ("right", "left"):
            # Respect the current zoomed/panned time range when advancing.
            left, right = self.axes[0].get_xlim()
            self.duration = right - left
            step = self.duration if event.key == "right" else -self.duration
            self.start = max(0.0, min(left + step, max(0.0, self.analysis.duration_seconds - self.duration)))
            self.draw()
        elif event.key in ("ctrl+s", "control+s"):
            save_draft(self.draft_path, self.draft, self.analysis.duration_seconds)
            self.dirty = False
            self.update_title(*self.axes[0].get_xlim())
            self.figure.canvas.draw_idle()
            print(f"Saved draft: {self.draft_path}")
        else:
            key_press_handler(event, self.figure.canvas, self.figure.canvas.toolbar)

    def on_release(self, event: MouseEvent) -> None:
        toolbar = self.figure.canvas.toolbar
        if event.inaxes in self.axes and toolbar is not None and toolbar.mode:
            left, right = self.axes[0].get_xlim()
            self.start = max(0.0, min(left, self.analysis.duration_seconds - 0.005))
            self.duration = max(0.005, right - left)
            self.draw()

    def on_click(self, event: MouseEvent) -> None:
        if event.button != 1 or event.inaxes not in self.axes or event.xdata is None:
            return
        timestamp = float(event.xdata)
        if not 0 <= timestamp < self.analysis.duration_seconds:
            return
        if event.key == "shift":
            self.draft.kick_onsets = sorted(set([*self.draft.kick_onsets, timestamp]))
            self.draft.manually_confirmed_kick_onsets = sorted(
                set([*self.draft.manually_confirmed_kick_onsets, timestamp])
            )
        elif event.key in ("control", "ctrl"):
            if not self.draft.kick_onsets:
                return
            nearest = min(self.draft.kick_onsets, key=lambda time: abs(time - timestamp))
            if abs(nearest - timestamp) > 0.05:
                return
            self.draft.kick_onsets.remove(nearest)
            if nearest in self.draft.manually_confirmed_kick_onsets:
                self.draft.manually_confirmed_kick_onsets.remove(nearest)
        else:
            return
        self.dirty = True
        # Keep the current zoom/pan window when changing an annotation.
        left, right = self.axes[0].get_xlim()
        self.start, self.duration = left, right - left
        self.draw()


def show_review(
    analysis: AcousticAnalysis, proposal: Proposal, draft_path: Path, start: float, duration: float
) -> None:
    draft = load_draft(draft_path, proposal)
    review = AnnotationReview(analysis, proposal, draft, draft_path, start, duration)
    if not draft_path.exists():
        save_draft(draft_path, draft, analysis.duration_seconds)
    # Keep callbacks alive for the lifetime of the blocking figure window.
    plt.show()
    if review.dirty:
        print("Unsaved draft edits remain; use Ctrl+S before closing to preserve changes.")


def diagnostic_sections(proposal: Proposal) -> dict[str, float | None]:
    """First qualifying complete/partial 10s interval; criteria are explicit heuristics."""
    selected: dict[str, float | None] = {"regular": None, "challenging": None}
    for start in np.arange(0, proposal["duration_seconds"], 10.0):
        end = min(float(start) + 10, proposal["duration_seconds"])
        beats = [b for b in proposal["beat_positions"] if start <= b < end]
        unsupported = [b for b in proposal["unsupported_beat_positions"] if start <= b < end]
        candidates = [c for c in proposal["kick_candidates"] if start <= c["timestamp"] < end]
        aligned = sum(c["grid_aligned"] for c in candidates)
        offgrid = len(candidates) - aligned
        # "Regular" means supported estimated beats, not verified kick presence.
        if selected["regular"] is None and len(beats) >= 8 and aligned / len(beats) >= 0.9:
            selected["regular"] = float(start)
        if selected["challenging"] is None and (offgrid >= 5 or len(unsupported) >= 3):
            selected["challenging"] = float(start)
    return selected


def save_diagnostics(analysis: AcousticAnalysis, proposal: Proposal, output: Path) -> dict[str, float | None]:
    selected = diagnostic_sections(proposal)
    output.mkdir(parents=True, exist_ok=True)
    for label, start in selected.items():
        if start is None:
            print(f"{proposal['track_identifier']}: no {label} diagnostic interval satisfies the criterion")
            continue
        draft = load_draft(output / "unused-draft.json", proposal)
        review = AnnotationReview(analysis, proposal, draft, output / "unused-draft.json", start, 10)
        review.figure.savefig(output / f"{proposal['track_identifier']}-{label}.png", dpi=130)
        plt.close(review.figure)
    return selected
