"""Read-only recording preparation shared by both renderers."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from benchmarks.annotation.analysis import AcousticAnalysis, AutomaticCandidate, Proposal, analyze_audio
from benchmarks.annotation.corrections import ReviewDraft, load_review
from benchmarks.fixtures import verify_checksum
from benchmarks.tracks import TECHNO, read_catalog, verified_audio


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Recording:
    analysis: AcousticAnalysis
    proposal: Proposal[AutomaticCandidate]
    draft: ReviewDraft
    audio_path: Path
    preparation_seconds: float
    protected_hashes: dict[str, str]


def load_recording(identifier: str) -> Recording:
    started = perf_counter()
    track = next(track for track in read_catalog() if track.identifier == identifier)
    proposal_path = TECHNO / "proposals" / f"{identifier}.json"
    draft_path = TECHNO / "drafts" / f"{identifier}.json"
    audio_path = TECHNO / "audio" / track.filename
    protected = {str(path): checksum(path) for path in (proposal_path, draft_path) if path.exists()}
    proposal: Proposal[AutomaticCandidate] = json.loads(proposal_path.read_text())
    if proposal["audio_sha256"] != track.sha256 or proposal["track_identifier"] != identifier:
        raise ValueError("Cached proposal does not identify the catalog recording.")
    rate, samples = verified_audio(track)
    analysis = analyze_audio(samples, rate)
    draft = load_review(draft_path, proposal)
    verify_checksum(audio_path, track.sha256)
    return Recording(analysis, proposal, draft, audio_path, perf_counter() - started, protected)
