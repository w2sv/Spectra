"""Draft provenance and reversible editing, without GUI or playback dependencies."""

import copy
import hashlib
import json
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from benchmarks.annotation_analysis import PARAMETERS, Proposal
from benchmarks.annotation_templates import TEMPLATE_PARAMETERS, TemplateDefinition, candidate_onset
from benchmarks.tracks import validate_timestamps


def proposal_checksum(proposal: Proposal) -> str:
    # Informational execution time must not make an otherwise identical draft stale.
    content = {key: value for key, value in proposal.items() if key != "informational"}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass
class AnnotationDraft:
    track_identifier: str
    audio_sha256: str
    annotation_algorithm_version: int
    confirmed_bpm: float
    kick_onsets: list[float]
    manually_confirmed_kick_onsets: list[float]
    review_status: str = "draft"
    originating_proposal_sha256: str | None = None
    # Keys identify the original acoustic candidate; values retain the accepted time.
    automatic_events: dict[str, float] = field(default_factory=dict)
    # Grid identity includes its role, tempo, phase, and complete beat positions.
    unsupported_beat_reviews: dict[str, dict[str, Literal["kick", "gap"]]] = field(default_factory=dict)


def beat_grid_identity(proposal: Proposal, alternative: bool) -> str:
    role = "alternative" if alternative else "primary"
    phase = proposal.get("alternative_phase_seconds") if alternative else proposal["beat_grid_phase_seconds"]
    content = {
        "bpm": float(proposal["bpm"]) if proposal["bpm"] is not None else None,
        "phase": float(phase) if phase is not None else None,
        "beats": proposal.get("alternative_beat_positions", []) if alternative else proposal["beat_positions"],
    }
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"{role}:{digest}"


def covered_by_annotation(timestamp: float, onsets: list[float]) -> bool:
    tolerance = PARAMETERS.association_radius_seconds + 1e-10
    index = bisect_left(onsets, timestamp - tolerance)
    return index < len(onsets) and onsets[index] <= timestamp + tolerance


class UnsupportedBeatNavigator:
    """Keep an event anchor across edits and decisions, independently of its viewport."""

    def __init__(self) -> None:
        self.grid_identity: str | None = None
        self.selected: float | None = None

    def use_grid(self, identity: str) -> None:
        if identity != self.grid_identity:
            self.grid_identity = identity
            self.selected = None

    def remaining(self, unsupported: list[float], draft: AnnotationDraft) -> list[float]:
        decisions = draft.unsupported_beat_reviews.get(self.grid_identity or "", {})
        return [
            beat
            for beat in unsupported
            if str(beat) not in decisions and not covered_by_annotation(beat, draft.kick_onsets)
        ]

    def destination(self, direction: int, remaining: list[float], view_start: float) -> float | None:
        if direction == 1:
            index = (
                bisect_right(remaining, self.selected)
                if self.selected is not None
                else bisect_left(remaining, view_start)
            )
        elif direction == -1:
            index = bisect_left(remaining, self.selected if self.selected is not None else view_start) - 1
        else:
            raise ValueError("Unsupported-beat navigation requires direction -1 or +1.")
        return remaining[index] if 0 <= index < len(remaining) else None

    def jump(
        self, direction: int, unsupported: list[float], draft: AnnotationDraft, view_start: float, duration: float
    ) -> tuple[float, float] | None:
        destination = self.destination(direction, self.remaining(unsupported, draft), view_start)
        if destination is None:
            return None
        self.selected = destination
        width = min(2.0, duration)
        return max(0.0, min(destination - width / 2, duration - width)), width


def validate_unsupported_reviews(draft: AnnotationDraft, duration: float) -> None:
    for decisions in draft.unsupported_beat_reviews.values():
        validate_timestamps(sorted(float(timestamp) for timestamp in decisions), duration)
        if any(value not in ("kick", "gap") for value in decisions.values()):
            raise ValueError("Unsupported-beat decisions must be 'kick' or 'gap'.")


def new_draft(proposal: Proposal) -> AnnotationDraft:
    checksum, bpm = proposal["audio_sha256"], proposal["bpm"]
    if checksum is None or bpm is None:
        raise ValueError("Real recording drafts require a checksum and BPM.")
    classified = proposal.get("positive_template_count", 0) > 0
    draft = AnnotationDraft(
        proposal["track_identifier"],
        checksum,
        proposal["annotation_algorithm_version"],
        bpm,
        [],
        [],
        originating_proposal_sha256=proposal_checksum(proposal),
    )
    for candidate in proposal["kick_candidates"]:
        accept = candidate.get("classification") == "likely" if classified else candidate["grid_aligned"]
        if accept:
            onset = candidate_onset(candidate) if classified else candidate["timestamp"]
            if all(abs(time - onset) > TEMPLATE_PARAMETERS.duplicate_tolerance_seconds for time in draft.kick_onsets):
                draft.kick_onsets.append(onset)
                draft.automatic_events[str(candidate["timestamp"])] = onset
    draft.kick_onsets.sort()
    return draft


def load_draft(path: Path, proposal: Proposal) -> AnnotationDraft:
    if not path.exists():
        return new_draft(proposal)
    # Older drafts are preserved and flagged stale rather than silently migrated/reseeded.
    draft = AnnotationDraft(**json.loads(path.read_text(encoding="utf-8")))
    if draft.track_identifier != proposal["track_identifier"] or draft.audio_sha256 != proposal["audio_sha256"]:
        raise ValueError("Draft does not refer to this recording and checksum.")
    validate_timestamps(draft.kick_onsets, proposal["duration_seconds"])
    validate_timestamps(draft.manually_confirmed_kick_onsets, proposal["duration_seconds"])
    validate_unsupported_reviews(draft, proposal["duration_seconds"])
    return draft


def is_stale(draft: AnnotationDraft, proposal: Proposal) -> bool:
    return draft.originating_proposal_sha256 != proposal_checksum(proposal)


def save_draft(path: Path, draft: AnnotationDraft, duration: float) -> None:
    validate_timestamps(draft.kick_onsets, duration)
    validate_timestamps(draft.manually_confirmed_kick_onsets, duration)
    validate_unsupported_reviews(draft, duration)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(asdict(draft), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def reset_draft(path: Path, proposal: Proposal) -> AnnotationDraft:
    if path.exists():
        required = f"RESET {proposal['track_identifier']} DRAFT"
        if input(f"Discard existing draft edits? Type '{required}': ") != required:
            raise ValueError("Draft reset was not confirmed; existing edits preserved.")
    draft = new_draft(proposal)
    save_draft(path, draft, proposal["duration_seconds"])
    return draft


class DraftEditor:
    def __init__(self, draft: AnnotationDraft, templates: TemplateDefinition, proposal: Proposal) -> None:
        self.draft = draft
        self.templates = templates
        self.proposal = proposal
        self._undo: list[tuple[AnnotationDraft, TemplateDefinition]] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    def review_unsupported_beat(self, identity: str, timestamp: float, kick: bool) -> None:
        """Confirm presence/absence; a beat-time kick remains provisionally localized."""
        validate_timestamps([timestamp], self.proposal["duration_seconds"])
        self.remember()
        decisions = self.draft.unsupported_beat_reviews.setdefault(identity, {})
        decisions[str(timestamp)] = "kick" if kick else "gap"
        if kick and not covered_by_annotation(timestamp, self.draft.kick_onsets):
            self.draft.kick_onsets = sorted([*self.draft.kick_onsets, timestamp])
            self.draft.manually_confirmed_kick_onsets = sorted(
                set([*self.draft.manually_confirmed_kick_onsets, timestamp])
            )

    def remember(self) -> None:
        self._undo.append((copy.deepcopy(self.draft), copy.deepcopy(self.templates)))
        del self._undo[:-100]

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.draft, self.templates = self._undo.pop()
        return True

    def add(self, timestamp: float) -> None:
        self.remember()
        self.draft.kick_onsets = sorted(set([*self.draft.kick_onsets, timestamp]))
        self.draft.manually_confirmed_kick_onsets = sorted(set([*self.draft.manually_confirmed_kick_onsets, timestamp]))

    def remove(self, timestamp: float) -> bool:
        if not self.draft.kick_onsets:
            return False
        nearest = min(self.draft.kick_onsets, key=lambda time: abs(time - timestamp))
        if abs(nearest - timestamp) > 0.05:
            return False
        self.remember()
        self.draft.kick_onsets.remove(nearest)
        self.draft.manually_confirmed_kick_onsets = [
            time for time in self.draft.manually_confirmed_kick_onsets if time != nearest
        ]
        self.draft.automatic_events = {
            key: time for key, time in self.draft.automatic_events.items() if time != nearest
        }
        return True

    def bulk(self, action: str, start: float, end: float) -> None:
        self.remember()
        if action == "accept":
            for candidate in self.proposal["kick_candidates"]:
                onset = candidate_onset(candidate)
                if candidate.get("classification") == "likely" and start <= onset < end:
                    previous = self.draft.automatic_events.get(str(candidate["timestamp"]))
                    if previous is not None and previous in self.draft.kick_onsets:
                        continue
                    if all(
                        abs(time - onset) > TEMPLATE_PARAMETERS.duplicate_tolerance_seconds + 1e-10
                        for time in self.draft.kick_onsets
                    ):
                        self.draft.kick_onsets.append(onset)
                        self.draft.automatic_events[str(candidate["timestamp"])] = onset
        elif action == "unlikely":
            for candidate in self.proposal["kick_candidates"]:
                key = str(candidate["timestamp"])
                onset = self.draft.automatic_events.get(key)
                if candidate.get("classification") == "unlikely" and onset is not None and start <= onset < end:
                    if onset not in self.draft.manually_confirmed_kick_onsets:
                        self.draft.kick_onsets = [time for time in self.draft.kick_onsets if time != onset]
                        self.draft.automatic_events.pop(key)
        elif action == "clear":
            self.draft.kick_onsets = [time for time in self.draft.kick_onsets if not start <= time < end]
            self.draft.manually_confirmed_kick_onsets = [
                time for time in self.draft.manually_confirmed_kick_onsets if not start <= time < end
            ]
            self.draft.automatic_events = {
                key: time for key, time in self.draft.automatic_events.items() if not start <= time < end
            }
        else:
            raise ValueError(f"Unknown bulk action: {action}")
        self.draft.kick_onsets.sort()

    def set_template(self, timestamp: float, positive: bool, replace_timestamp: float | None = None) -> None:
        self.remember()
        if replace_timestamp is not None:
            self.templates.positive_onsets = [
                time for time in self.templates.positive_onsets if time != replace_timestamp
            ]
            self.templates.negative_onsets = [
                time for time in self.templates.negative_onsets if time != replace_timestamp
            ]
        if positive:
            self.templates.negative_onsets = [time for time in self.templates.negative_onsets if time != timestamp]
        else:
            self.templates.positive_onsets = [time for time in self.templates.positive_onsets if time != timestamp]
        target = self.templates.positive_onsets if positive else self.templates.negative_onsets
        target[:] = sorted(set([*target, timestamp]))

    def remove_template(self, timestamp: float) -> bool:
        times = self.templates.positive_onsets + self.templates.negative_onsets
        if not times:
            return False
        nearest = min(times, key=lambda time: abs(time - timestamp))
        if abs(nearest - timestamp) > 0.05:
            return False
        self.remember()
        self.templates.positive_onsets = [time for time in self.templates.positive_onsets if time != nearest]
        self.templates.negative_onsets = [time for time in self.templates.negative_onsets if time != nearest]
        return True
