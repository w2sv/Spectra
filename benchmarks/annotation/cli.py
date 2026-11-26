"""Independent acoustic kick proposals, review visualization, and explicit finalization."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Literal

from benchmarks.annotation.analysis import (
    AutomaticCandidate,
    Proposal,
    analyze_audio,
    generate_proposal,
)
from benchmarks.annotation.automatic import AUTOMATIC_ALGORITHM_VERSION, automatic_proposal
from benchmarks.annotation.corrections import (
    effective_onsets,
    has_review_state,
    load_review,
    new_review,
    proposal_checksum,
    save_review,
)
from benchmarks.annotation.review import DEFAULT_VIEW_DURATION
from benchmarks.annotation.validation import ValidationStrategy, validate
from benchmarks.fixtures import verify_checksum
from benchmarks.tracks import TECHNO, Track, read_catalog, validate_timestamps, verified_audio


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def proposal_summary(proposal: Proposal[AutomaticCandidate]) -> dict[str, object]:
    aligned = sum(candidate["grid_aligned"] for candidate in proposal["kick_candidates"])
    return {
        "track_identifier": proposal["track_identifier"],
        "audio_sha256": proposal["audio_sha256"],
        "duration_seconds": proposal["duration_seconds"],
        "source_sample_rate": proposal["source_sample_rate"],
        "bpm": proposal["bpm"],
        "beat_grid_phase_seconds": proposal["beat_grid_phase_seconds"],
        "estimated_beats": len(proposal["beat_positions"]),
        "grid_aligned_candidates": aligned,
        "off_grid_candidates": len(proposal["kick_candidates"]) - aligned,
        "unsupported_beats": len(proposal["unsupported_beat_positions"]),
        "drift_sections": [section for section in proposal["beat_grid_validation"] if section["drift_flagged"]],
        "informational": proposal["informational"],
        "annotation_algorithm_version": proposal["annotation_algorithm_version"],
        "classification_counts": {
            label: sum(
                candidate.get("classification", "unclassified") == label for candidate in proposal["kick_candidates"]
            )
            for label in ("unclassified", "likely", "ambiguous", "unlikely")
        },
        "grid_comparisons": proposal.get("grid_comparisons", []),
        "positive_template_count": proposal.get("positive_template_count", 0),
        "invalid_positive_templates": proposal.get("invalid_positive_templates", []),
        "invalid_negative_templates": proposal.get("invalid_negative_templates", []),
        "clusters": proposal.get("clusters", []),
        "automatic_reference_ids": proposal.get("automatic_reference_ids", []),
        "selected_grid": proposal.get("selected_grid"),
        "automatic_confidence": proposal.get("automatic_confidence"),
        "candidates": len(proposal["kick_candidates"]),
        "predicted_kicks": sum(c["automatic_kick"] for c in proposal["kick_candidates"]),
        "predicted_non_kicks": sum(not c["automatic_kick"] for c in proposal["kick_candidates"]),
        "uncertain_candidates": sum(bool(c["uncertainty_reasons"]) for c in proposal["kick_candidates"]),
        "supplementary_candidates": sum(c["provenance"] != "acoustic" for c in proposal["kick_candidates"]),
        "unreliable_onsets": sum(not c["localization_reliable"] for c in proposal["kick_candidates"]),
    }


def annotate_track(
    track: Track,
    bpm: float | None = None,
    phase: float | None = None,
    *,
    regenerate: bool = False,
    mode: Literal["review", "prepare", "diagnostics"] = "review",
    start: float = 0,
    duration: float = DEFAULT_VIEW_DURATION,
    reset: bool = False,
    root: Path = TECHNO,
) -> dict[str, object]:
    if track.partition == "evaluation":
        raise ValueError("Annotation pilot excludes held-out evaluation tracks.")
    if mode == "review" and (not math.isfinite(start) or start < 0 or not math.isfinite(duration) or duration <= 0):
        raise ValueError("Review start must be nonnegative and duration positive; both must be finite.")
    if track.sha256 is None:
        raise ValueError("Annotation requires a catalog audio checksum.")
    path = root / "proposals" / f"{track.identifier}.json"
    draft_path = root / "drafts" / f"{track.identifier}.json"
    cached: Proposal[AutomaticCandidate] | None = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    )
    verify_checksum(root / "audio" / track.filename, track.sha256)
    backup = draft_path.with_suffix(".before-reset.json")
    if reset and draft_path.exists():
        required = f"RESET {track.identifier} DRAFT"
        if input(f"Discard existing edits? Type '{required}': ") != required:
            raise ValueError("Draft reset cancelled; existing draft preserved.")
        if backup.exists():
            raise ValueError(f"Backup already exists: {backup}")
        backup.write_bytes(draft_path.read_bytes())
    review = load_review(draft_path, cached) if draft_path.exists() and cached is not None and not reset else None
    meaningful = draft_path.exists() and not reset
    if review is not None and cached is not None:
        meaningful = has_review_state(review, cached)
    reusable = cached is not None and (
        cached["track_identifier"] == track.identifier
        and cached["audio_sha256"] == track.sha256
        and cached["annotation_algorithm_version"] == AUTOMATIC_ALGORITHM_VERSION
    )
    if meaningful and not reusable and not regenerate:
        raise ValueError(
            "Cached proposal is outdated or missing and review state exists. Use --regenerate explicitly; corrections will be reconciled before replacement."
        )
    if reusable and not regenerate and (bpm is not None or phase is not None):
        raise ValueError("Use --regenerate to replace proposals with BPM/phase overrides.")
    analysis = None
    if reusable and not regenerate:
        assert cached is not None
        proposal = cached
    else:
        bpm = track.bpm if bpm is None else bpm
        if bpm is None:
            raise ValueError(f"{track.identifier}: catalog BPM unknown; provide --bpm.")
        started = perf_counter()
        rate, samples = verified_audio(track, root)
        analysis = analyze_audio(samples, rate)
        del samples
        acoustic = generate_proposal(analysis, track.identifier, track.sha256, rate, bpm, phase)
        proposal = automatic_proposal(analysis, acoustic)
        proposal["informational"]["annotation_processing_seconds"] = perf_counter() - started
        if draft_path.exists() and not reset:
            # Reconcile explicit corrections before replacing cached proposal bytes.
            load_review(draft_path, proposal)
        if mode == "review" and start >= proposal["duration_seconds"]:
            raise ValueError("Review start must be within the recording.")
        write_json(path, proposal)
    if reset:
        save_review(draft_path, new_review(proposal), proposal)
    summary = proposal_summary(proposal)
    if mode != "prepare":
        if mode == "review" and start >= proposal["duration_seconds"]:
            raise ValueError("Review start must be within the recording.")
        if analysis is None:
            rate, samples = verified_audio(track, root)
            analysis = analyze_audio(samples, rate)
            del samples
        if mode == "diagnostics":
            from benchmarks.annotation.diagnostics import save_diagnostics

            summary["diagnostic_intervals"] = save_diagnostics(analysis, proposal, root / "proposals" / "diagnostics")
        else:
            from benchmarks.annotation.review.window import show_review

            show_review(analysis, proposal, draft_path, start, duration, root=root)
    write_json(root / "proposals" / f"{track.identifier}-summary.json", summary)
    return summary


def finalize_annotation(track: Track, root: Path = TECHNO) -> None:
    draft_path = root / "drafts" / f"{track.identifier}.json"
    if not draft_path.is_file():
        raise ValueError("No editable draft exists; open the review and save your corrections first.")
    proposal: Proposal[AutomaticCandidate] = json.loads((root / "proposals" / f"{track.identifier}.json").read_text())
    rate, samples = verified_audio(track, root)
    duration = len(samples) / rate
    del samples
    if proposal["track_identifier"] != track.identifier or proposal["audio_sha256"] != track.sha256:
        raise ValueError("Proposal audio checksum or track identifier disagrees with the recording.")
    if proposal["source_sample_rate"] != rate or proposal["duration_seconds"] != duration:
        raise ValueError("Proposal acquisition metadata disagrees with the recording.")
    review = load_review(draft_path, proposal)
    if review.originating_proposal_sha256 != proposal_checksum(proposal):
        raise ValueError("Stale review: inspect the regenerated predictions and explicitly save before finalizing.")
    if review.annotation_algorithm_version != proposal["annotation_algorithm_version"]:
        raise ValueError("Review algorithm version disagrees with its proposal; explicitly save before finalizing.")
    timestamps = effective_onsets(proposal, review)
    validate_timestamps(timestamps, duration)
    bpm = review.confirmed_bpm
    if bpm is None or type(bpm) not in (int, float) or not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("Confirmed BPM must be finite and positive; edit the draft if needed.")
    version = proposal["annotation_algorithm_version"]
    if type(version) is not int or version < 1:
        raise ValueError("Proposal annotation algorithm version must be a positive integer.")
    required = f"REVIEWED {track.identifier} ENTIRE RECORDING"
    response = input(
        "Confirm that you reviewed the ENTIRE recording, including sections with no proposals, "
        "and that every remaining draft timestamp is a genuine kick.\n"
        f"Type '{required}': "
    )
    if response != required:
        raise ValueError("Full-recording review was not explicitly confirmed; nothing finalized.")
    path = root / "annotations.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records = [record for record in records if record["track_identifier"] != track.identifier]
    records.append(
        {
            "track_identifier": track.identifier,
            "audio_sha256": track.sha256,
            "annotation_algorithm_version": version,
            "confirmed_kick_onsets": timestamps,
            "confirmed_bpm": bpm,
            "review_status": "reviewed",
        }
    )
    write_json(path, sorted(records, key=lambda record: record["track_identifier"]))
    print(f"Finalized fully reviewed recording: {track.identifier}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--track")
    selection.add_argument("--validate", action="store_true", help="Run numerical synthetic validation.")
    parser.add_argument(
        "--strategy",
        choices=(*ValidationStrategy, "all"),
        help="Validation strategy (default: automatic-template-matching).",
    )
    parser.add_argument("--bpm", type=float)
    parser.add_argument("--phase", type=float)
    parser.add_argument("--regenerate", action="store_true", help="Explicitly regenerate and reconcile proposals.")
    parser.add_argument(
        "--reset-draft", action="store_true", help="Deliberately discard a review, with confirmation and backup."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", dest="mode", action="store_const", const="prepare")
    modes.add_argument("--diagnostics", dest="mode", action="store_const", const="diagnostics")
    modes.add_argument("--finalize", dest="mode", action="store_const", const="finalize")
    parser.set_defaults(mode="review")
    parser.add_argument("--start", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--json", type=Path, help="Export the selected validation report; no implicit output path.")
    arguments = parser.parse_args(argv)
    track_modifiers = {"--bpm", "--phase", "--regenerate", "--reset-draft"}
    allowed = {
        "validation": {"--strategy", "--json"},
        "review": track_modifiers | {"--start", "--duration"},
        "prepare": track_modifiers,
        "diagnostics": track_modifiers,
        "finalize": set(),
    }
    provided = {
        name
        for name, value in {
            "--strategy": arguments.strategy,
            "--json": arguments.json,
            "--bpm": arguments.bpm,
            "--phase": arguments.phase,
            "--regenerate": arguments.regenerate,
            "--reset-draft": arguments.reset_draft,
            "--start": arguments.start,
            "--duration": arguments.duration,
        }.items()
        if value is not None and value is not False
    }
    if arguments.validate and arguments.mode != "review":
        parser.error("Track execution modes require --track.")
    mode: Literal["review", "prepare", "diagnostics", "finalize"] = arguments.mode
    execution = "validation" if arguments.validate else mode
    if invalid := provided - allowed[execution]:
        parser.error(f"Not supported for {execution}: {', '.join(sorted(invalid))}")
    try:
        if arguments.validate:
            strategy = (
                "all"
                if arguments.strategy == "all"
                else ValidationStrategy(arguments.strategy or ValidationStrategy.AUTOMATIC_TEMPLATE_MATCHING)
            )
            report = validate(strategy)
            if arguments.json is not None:
                write_json(arguments.json, report)
            print(json.dumps(report, indent=2, allow_nan=False))
            return
        tracks = {track.identifier: track for track in read_catalog()}
        if arguments.track not in tracks:
            raise ValueError("Unknown catalog track identifier.")
        track = tracks[arguments.track]
        if mode == "finalize":
            finalize_annotation(track)
        else:
            summary = annotate_track(
                track,
                arguments.bpm,
                arguments.phase,
                regenerate=arguments.regenerate,
                mode=mode,
                start=arguments.start if arguments.start is not None else 0,
                duration=arguments.duration if arguments.duration is not None else DEFAULT_VIEW_DURATION,
                reset=arguments.reset_draft,
            )
            print(json.dumps(summary, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        parser.exit(2, f"Annotation error: {error}\n")


if __name__ == "__main__":
    main()
