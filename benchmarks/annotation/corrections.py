"""Identity-based review decisions; effective annotations are always derived."""

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from benchmarks.annotation.analysis import AutomaticCandidate, Proposal
from benchmarks.annotation.templates import TEMPLATE_PARAMETERS
from benchmarks.tracks import validate_timestamps


def proposal_checksum(proposal: Proposal[AutomaticCandidate]) -> str:
    content = {key: value for key, value in proposal.items() if key != "informational"}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass
class Correction:
    kick: bool
    timestamp: float
    original_timestamp: float
    provenance: str = "individual"


@dataclass
class ReviewDraft:
    track_identifier: str
    audio_sha256: str | None
    annotation_algorithm_version: int
    originating_proposal_sha256: str
    confirmed_bpm: float | None
    candidate_ids: list[str]
    overrides: dict[str, Correction] = field(default_factory=dict)
    manual_events: list[AutomaticCandidate] = field(default_factory=list)
    prediction_updates: list[AutomaticCandidate] = field(default_factory=list)
    review_status: str = "draft"


def new_review(proposal: Proposal[AutomaticCandidate]) -> ReviewDraft:
    return ReviewDraft(
        proposal["track_identifier"],
        proposal["audio_sha256"],
        proposal["annotation_algorithm_version"],
        proposal_checksum(proposal),
        proposal["bpm"],
        [c["candidate_id"] for c in proposal["kick_candidates"]],
    )


def has_review_state(draft: ReviewDraft, proposal: Proposal[AutomaticCandidate]) -> bool:
    """Protect explicit decisions and nondefault review metadata, not derived predictions."""
    return bool(
        draft.overrides
        or draft.manual_events
        or draft.confirmed_bpm != proposal["bpm"]
        or draft.review_status != "draft"
    )


def effective_kick(candidate: AutomaticCandidate, draft: ReviewDraft) -> bool:
    correction = draft.overrides.get(candidate["candidate_id"])
    return correction.kick if correction is not None else candidate["automatic_kick"]


def effective_timestamp(candidate: AutomaticCandidate, draft: ReviewDraft) -> float:
    correction = draft.overrides.get(candidate["candidate_id"])
    return correction.timestamp if correction is not None else candidate["final_timestamp"]


def effective_onsets(proposal: Proposal[AutomaticCandidate], draft: ReviewDraft) -> list[float]:
    return sorted(
        set(
            effective_timestamp(c, draft)
            for c in (draft.prediction_updates or proposal["kick_candidates"]) + draft.manual_events
            if effective_kick(c, draft)
        )
    )


def reconcile_draft(draft: ReviewDraft, proposal: Proposal[AutomaticCandidate]) -> list[str]:
    """Only an identical source identity AND anchor can carry a correction forward."""
    candidates = {c["candidate_id"]: c for c in proposal["kick_candidates"] + draft.manual_events}
    return [
        key
        for key, correction in draft.overrides.items()
        if key not in candidates or candidates[key]["timestamp"] != correction.original_timestamp
    ]


def load_review(path: Path, proposal: Proposal[AutomaticCandidate]) -> ReviewDraft:
    if not path.exists():
        return new_review(proposal)
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        data["overrides"] = {key: Correction(**value) for key, value in data.get("overrides", {}).items()}
        draft = ReviewDraft(**data)
    except (TypeError, KeyError) as error:
        raise ValueError(f"Unsupported review draft in {path}; file preserved. Deliberate reset required.") from error
    if draft.track_identifier != proposal["track_identifier"] or draft.audio_sha256 != proposal["audio_sha256"]:
        raise ValueError("Review draft does not match the recording checksum.")
    if draft.originating_proposal_sha256 != proposal_checksum(proposal):
        # Source identities can survive numerical changes. Reclassification is
        # reproducible; only explicit corrections carry their frozen onsets over.
        draft.prediction_updates = []
        draft.candidate_ids = [c["candidate_id"] for c in proposal["kick_candidates"]]
    if unmatched := reconcile_draft(draft, proposal):
        raise ValueError(
            f"Unmatched human corrections retained in {path}: {', '.join(unmatched)}. Resolve the corrections or deliberately reset the draft."
        )
    validate_review(draft, proposal["duration_seconds"])
    return draft


def validate_review(draft: ReviewDraft, duration: float) -> None:
    ids = draft.candidate_ids + [c["candidate_id"] for c in draft.manual_events]
    if len(ids) != len(set(ids)) or any(key not in ids for key in draft.overrides):
        raise ValueError("Duplicate candidate identities or orphan human overrides in review draft.")
    for correction in draft.overrides.values():
        if type(correction.kick) is not bool:
            raise ValueError("Human classifications must be explicit booleans.")
        validate_timestamps([correction.timestamp], duration)
        validate_timestamps([correction.original_timestamp], duration)
    for candidate in draft.manual_events:
        validate_timestamps([candidate["final_timestamp"]], duration)
    if draft.prediction_updates:
        if {c["candidate_id"] for c in draft.prediction_updates} != set(draft.candidate_ids):
            raise ValueError("Saved reclassification has inconsistent candidate identities.")
        for candidate in draft.prediction_updates:
            validate_timestamps([candidate["final_timestamp"]], duration)


def save_review(path: Path, draft: ReviewDraft, proposal: Proposal[AutomaticCandidate]) -> None:
    validate_review(draft, proposal["duration_seconds"])
    if unmatched := reconcile_draft(draft, proposal):
        raise ValueError(f"Cannot save unmatched corrections: {unmatched}")
    saved = copy.copy(draft)
    saved.originating_proposal_sha256 = proposal_checksum(proposal)
    saved.candidate_ids = [c["candidate_id"] for c in proposal["kick_candidates"]]
    saved.annotation_algorithm_version = proposal["annotation_algorithm_version"]
    validate_review(saved, proposal["duration_seconds"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(asdict(saved), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    draft.originating_proposal_sha256 = saved.originating_proposal_sha256
    draft.candidate_ids = saved.candidate_ids
    draft.annotation_algorithm_version = saved.annotation_algorithm_version


class CorrectionEditor:
    def __init__(self, proposal: Proposal[AutomaticCandidate], draft: ReviewDraft) -> None:
        self.proposal = proposal
        self.draft = draft
        self.history: list[ReviewDraft] = []
        self.redo_history: list[ReviewDraft] = []
        self.saved_draft = copy.deepcopy(draft)
        self.candidates = {c["candidate_id"]: c for c in self.predictions + draft.manual_events}

    @property
    def dirty(self) -> bool:
        return self.draft != self.saved_draft

    def mark_saved(self) -> None:
        self.saved_draft = copy.deepcopy(self.draft)
        self.history.clear()
        self.redo_history.clear()

    @property
    def predictions(self) -> list[AutomaticCandidate]:
        return self.draft.prediction_updates or self.proposal["kick_candidates"]

    def checkpoint(self) -> None:
        self.redo_history.clear()
        self.history.append(copy.deepcopy(self.draft))
        self.history = self.history[-100:]

    def set_classification(self, key: str, kick: bool, provenance: str = "individual") -> None:
        candidate = self.candidates[key]
        self.draft.overrides[key] = Correction(
            kick, effective_timestamp(candidate, self.draft), candidate["timestamp"], provenance
        )

    def _remove_manual_event(self, key: str) -> None:
        self.draft.manual_events = [c for c in self.draft.manual_events if c["candidate_id"] != key]
        self.draft.overrides.pop(key, None)
        del self.candidates[key]

    def toggle(self, key: str) -> None:
        self.checkpoint()
        if effective_kick(self.candidates[key], self.draft) and any(
            c["candidate_id"] == key for c in self.draft.manual_events
        ):
            self._remove_manual_event(key)
            return
        self.set_classification(key, not effective_kick(self.candidates[key], self.draft))

    def clear_override(self, key: str) -> None:
        manual = any(c["candidate_id"] == key for c in self.draft.manual_events)
        if manual or key in self.draft.overrides:
            self.checkpoint()
            if manual:
                self._remove_manual_event(key)
            else:
                del self.draft.overrides[key]

    def group_targets(self, key: str) -> list[str]:
        cluster = self.candidates[key].get("cluster_id")
        return [
            c["candidate_id"]
            for c in self.candidates.values()
            if cluster is not None and c.get("cluster_id") == cluster and c["candidate_id"] not in self.draft.overrides
        ]

    def group_correct(self, key: str, kick: bool) -> int:
        targets = self.group_targets(key)
        if targets:
            self.checkpoint()
            for target in targets:
                self.set_classification(target, kick, f"cluster:{self.candidates[key]['cluster_id']}")
        return len(targets)

    def insert(self, candidate: AutomaticCandidate) -> bool:
        if any(
            abs(effective_timestamp(c, self.draft) - candidate["final_timestamp"]) <= 0.030
            for c in self.candidates.values()
        ):
            return False
        self.checkpoint()
        self.draft.manual_events.append(candidate)
        self.candidates[candidate["candidate_id"]] = candidate
        self.set_classification(candidate["candidate_id"], True, "manual insertion")
        return True

    def replace_predictions(self, candidates: list[AutomaticCandidate]) -> int:
        old = self.candidates
        if {c["candidate_id"] for c in candidates} != {c["candidate_id"] for c in self.proposal["kick_candidates"]}:
            raise ValueError("Reclassification must preserve candidate identities.")
        self.checkpoint()
        for index, candidate in enumerate(candidates):
            key = candidate["candidate_id"]
            if key in self.draft.overrides:
                candidates[index] = old[key].copy()
        changed = sum(
            c["candidate_id"] not in self.draft.overrides
            and c["automatic_kick"] != old[c["candidate_id"]]["automatic_kick"]
            for c in candidates
        )
        self.draft.prediction_updates = candidates
        self.candidates = {c["candidate_id"]: c for c in candidates + self.draft.manual_events}
        return changed

    def undo(self) -> bool:
        if not self.history:
            return False
        self.redo_history.append(copy.deepcopy(self.draft))
        self.draft = self.history.pop()
        self.candidates = {c["candidate_id"]: c for c in self.predictions + self.draft.manual_events}
        return True

    def redo(self) -> bool:
        if not self.redo_history:
            return False
        self.history.append(copy.deepcopy(self.draft))
        self.draft = self.redo_history.pop()
        self.candidates = {c["candidate_id"]: c for c in self.predictions + self.draft.manual_events}
        return True


def centered_interval(timestamp: float, recording_duration: float, width: float = 2) -> tuple[float, float]:
    duration = min(width, recording_duration)
    return min(max(timestamp - duration / 2, 0), recording_duration - duration), duration


def uncertain_classification(candidate: AutomaticCandidate, draft: ReviewDraft) -> bool:
    """Review categories use existing metadata without changing predictions."""
    if candidate["candidate_id"] in draft.overrides:
        return False
    reasons = candidate["uncertainty_reasons"]
    positive = candidate.get("positive_similarity")
    negative = candidate.get("negative_similarity")
    parameters = TEMPLATE_PARAMETERS
    return (
        candidate.get("classification", "unclassified") in ("ambiguous", "unclassified")
        or "classification uncertainty" in reasons
        or "conflicting references" in reasons
        or positive is not None
        and (
            min(abs(positive - parameters.weak_similarity), abs(positive - parameters.strong_similarity)) <= 0.03
            or negative is not None
            and positive - negative < parameters.negative_margin
        )
    )


def nearby_hidden_candidates(editor: CorrectionEditor, timestamp: float, displayed: set[str]) -> list[str]:
    """Do not recover a grid-only expectation as though it were acoustic evidence.

    Return ambiguity to the caller for explicit clarification. The 50 ms window
    only narrows the search; the existing localizer supplies the actual onset.
    """
    return [
        candidate["candidate_id"]
        for candidate in sorted(
            editor.candidates.values(), key=lambda c: abs(effective_timestamp(c, editor.draft) - timestamp)
        )
        if candidate["candidate_id"] not in displayed
        and not effective_kick(candidate, editor.draft)
        and candidate["provenance"] != "beat_grid_only"
        and abs(effective_timestamp(candidate, editor.draft) - timestamp) <= 0.050
    ]


def navigate_candidate(
    candidates: list[AutomaticCandidate],
    draft: ReviewDraft,
    kind: str,
    direction: int,
    selected: str | None,
    position: float,
    grid: str = "primary",
) -> str | None:
    ordered = sorted(candidates, key=lambda c: (effective_timestamp(c, draft), c["candidate_id"]))
    covered = np.asarray([effective_timestamp(c, draft) for c in candidates if effective_kick(c, draft)])

    def eligible(candidate: AutomaticCandidate) -> bool:
        key = candidate["candidate_id"]
        if kind == "corrected":
            return key in draft.overrides
        if key in draft.overrides:
            return False
        if kind == "unsupported":
            return grid in candidate.get("unsupported_grids", []) and not np.any(
                np.abs(covered - effective_timestamp(candidate, draft)) <= 0.050
            )
        return uncertain_classification(candidate, draft)

    anchor = next((i for i, c in enumerate(ordered) if c["candidate_id"] == selected), None)
    sequence = (
        range(anchor + direction, len(ordered), 1)
        if anchor is not None and direction > 0
        else range(anchor - 1, -1, -1)
        if anchor is not None
        else range(len(ordered))
        if direction > 0
        else range(len(ordered) - 1, -1, -1)
    )
    for i in sequence:
        candidate = ordered[i]
        if anchor is None and (effective_timestamp(candidate, draft) - position) * direction < 0:
            continue
        if eligible(candidate):
            return candidate["candidate_id"]
    return None
