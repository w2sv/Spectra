"""Independent acoustic kick proposals, review visualization, and explicit finalization."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

from benchmarks.annotation.analysis import (
    AutomaticCandidate,
    Proposal,
    analyze_audio,
    generate_proposal,
)
from benchmarks.annotation.automatic import AUTOMATIC_ALGORITHM_VERSION, automatic_proposal
from benchmarks.annotation.corrections import (
    effective_onsets,
    load_review,
    migrate_legacy,
    new_review,
    proposal_checksum,
    reconcile_draft,
    save_review,
)
from benchmarks.annotation.validation.acoustic import validate_synthetics
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
        "template_definition_sha256": proposal.get("template_definition_sha256"),
        "classification_counts": {
            label: sum(
                candidate.get("classification", "unclassified") == label for candidate in proposal["kick_candidates"]
            )
            for label in ("unclassified", "likely", "ambiguous", "unlikely")
        },
        "grid_comparisons": proposal.get("grid_comparisons", []),
        "reference_alignment_checks": proposal.get("reference_alignment_checks", []),
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
    plot: bool = False,
    start: float = 0,
    duration: float = 10,
    diagnostics: bool = False,
    reset: bool = False,
    migrate: bool = False,
    root: Path = TECHNO,
) -> dict[str, object]:
    if track.partition == "evaluation":
        raise ValueError("Annotation pilot excludes held-out evaluation tracks.")
    path = root / "proposals" / f"{track.identifier}.json"
    proposal: Proposal[AutomaticCandidate]
    if path.exists() and not regenerate:
        if bpm is not None or phase is not None:
            raise ValueError("Use --regenerate to replace proposals with BPM/phase overrides.")
        proposal = json.loads(path.read_text(encoding="utf-8"))
        verify_checksum(root / "audio" / track.filename, track.sha256)
        if proposal["track_identifier"] != track.identifier or proposal["audio_sha256"] != track.sha256:
            raise ValueError("Existing proposals do not match this recording.")
        if proposal["annotation_algorithm_version"] != AUTOMATIC_ALGORITHM_VERSION:
            raise ValueError("Existing proposals use another algorithm version; explicitly --regenerate them.")
        if track.sha256 is None:
            raise ValueError("Template analysis requires a catalog audio checksum.")
        analysis = None
    else:
        bpm = track.bpm if bpm is None else bpm
        if bpm is None:
            raise ValueError(f"{track.identifier}: catalog BPM unknown; provide --bpm.")
        started = perf_counter()
        rate, samples = verified_audio(track, root)
        analysis = analyze_audio(samples, rate)
        del samples
        acoustic_proposal = generate_proposal(analysis, track.identifier, track.sha256, rate, bpm, phase)
        if track.sha256 is None:
            raise ValueError("Template analysis requires a catalog audio checksum.")
        proposal = automatic_proposal(analysis, acoustic_proposal)
        proposal["informational"]["annotation_processing_seconds"] = perf_counter() - started
        draft_path = root / "drafts" / f"{track.identifier}.json"
        if draft_path.exists() and "candidate_ids" in json.loads(draft_path.read_text()):
            # Validate before replacing proposals. Unmatched decisions remain recoverable.
            load_review(draft_path, proposal)
        write_json(path, proposal)
    draft_path = root / "drafts" / f"{track.identifier}.json"
    if reset:
        if draft_path.exists():
            required = f"RESET {track.identifier} DRAFT"
            if input(f"Discard existing edits? Type '{required}': ") != required:
                raise ValueError("Draft reset cancelled; existing draft preserved.")
            backup = draft_path.with_suffix(".before-reset.json")
            if backup.exists():
                raise ValueError(f"Backup already exists: {backup}")
            backup.write_bytes(draft_path.read_bytes())
        save_review(draft_path, new_review(proposal), proposal)
    if migrate:
        migrate_legacy(draft_path, proposal)
    summary = proposal_summary(proposal)
    if plot or diagnostics:
        if (
            not math.isfinite(start)
            or not math.isfinite(duration)
            or start < 0
            or duration <= 0
            or start >= proposal["duration_seconds"]
        ):
            raise ValueError("Plot start must be within the recording and duration must be positive.")
        if analysis is None:
            rate, samples = verified_audio(track, root)
            analysis = analyze_audio(samples, rate)
            del samples
        if diagnostics:
            from benchmarks.annotation.diagnostics import save_diagnostics

            summary["diagnostic_intervals"] = save_diagnostics(
                analysis, proposal, root / "proposals" / "diagnostics_v3"
            )
        if plot:
            from benchmarks.annotation.review.window import show_review

            show_review(analysis, proposal, root / "drafts" / f"{track.identifier}.json", start, duration, root=root)
    write_json(root / "proposals" / f"{track.identifier}-summary.json", summary)
    return summary


def finalize_annotation(track: Track, root: Path = TECHNO) -> None:
    draft_path = root / "drafts" / f"{track.identifier}.json"
    if not draft_path.is_file():
        raise ValueError("No editable draft exists; open --plot and save your reviewed draft first.")
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    rate, samples = verified_audio(track, root)
    duration = len(samples) / rate
    del samples
    if draft["track_identifier"] != track.identifier or draft["audio_sha256"] != track.sha256:
        raise ValueError("Draft audio checksum or track identifier disagrees with the recording.")
    if "candidate_ids" in draft:
        proposal: Proposal[AutomaticCandidate] = json.loads(
            (root / "proposals" / f"{track.identifier}.json").read_text()
        )
        review = load_review(draft_path, proposal)
        if review.originating_proposal_sha256 != proposal_checksum(proposal):
            raise ValueError("Stale review: inspect the regenerated predictions and explicitly save before finalizing.")
        if reconcile_draft(review, proposal):
            raise ValueError("Unmatched corrections must be resolved before finalization.")
        timestamps = effective_onsets(proposal, review)
    else:
        timestamps = draft["kick_onsets"]
    if not isinstance(timestamps, list):
        raise ValueError("Draft kick_onsets must be an explicit list of timestamps.")
    validate_timestamps(timestamps, duration)
    bpm = draft["confirmed_bpm"]
    if type(bpm) not in (int, float) or not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("Confirmed BPM must be finite and positive; edit the draft if needed.")
    if type(draft["annotation_algorithm_version"]) is not int or draft["annotation_algorithm_version"] < 1:
        raise ValueError("Draft annotation algorithm version must be a positive integer.")
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
            "annotation_algorithm_version": draft["annotation_algorithm_version"],
            "confirmed_kick_onsets": timestamps,
            "confirmed_bpm": bpm,
            "review_status": "reviewed",
        }
    )
    write_json(path, sorted(records, key=lambda record: record["track_identifier"]))
    print(f"Finalized fully reviewed recording: {track.identifier}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--track")
    selection.add_argument("--synthetic", action="store_true", help="Validate proposals on all 15 synthetic fixtures.")
    selection.add_argument(
        "--refined-synthetic", action="store_true", help="Validate template refinement, excluding reference events."
    )
    selection.add_argument(
        "--automatic-synthetic",
        action="store_true",
        help="Validate automatic discovery without ground-truth references.",
    )
    parser.add_argument("--bpm", type=float)
    parser.add_argument("--phase", type=float)
    parser.add_argument("--regenerate", action="store_true", help="Explicitly replace automatic proposals only.")
    parser.add_argument(
        "--migrate-draft",
        action="store_true",
        help="Explicitly migrate an old accepted-event draft, retaining a backup.",
    )
    parser.add_argument(
        "--reset-draft",
        action="store_true",
        help="Recreate a draft from current proposals; confirm before discarding edits.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--diagnostics", action="store_true", help="Save the first qualifying 10s diagnostic plots.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--json", type=Path, help="Export synthetic validation metrics.")
    arguments = parser.parse_args()
    try:
        if arguments.migrate_draft and (
            arguments.reset_draft
            or arguments.finalize
            or arguments.synthetic
            or arguments.refined_synthetic
            or arguments.automatic_synthetic
        ):
            raise ValueError(
                "Draft migration is a separate explicit operation; do not combine it with reset, finalization, or synthetic validation."
            )
        if arguments.synthetic or arguments.refined_synthetic or arguments.automatic_synthetic:
            if (
                arguments.plot
                or arguments.finalize
                or arguments.diagnostics
                or arguments.bpm is not None
                or arguments.phase is not None
                or arguments.reset_draft
            ):
                raise ValueError("Synthetic validation uses fixture tempos and does not edit real annotations.")
            if arguments.automatic_synthetic:
                from benchmarks.annotation.validation.automatic import validate_automatic

                report = validate_automatic()
            elif arguments.refined_synthetic:
                from benchmarks.annotation.validation.refinement import validate_refinement

                report = validate_refinement()
            else:
                report = validate_synthetics()
            if arguments.json is not None:
                write_json(arguments.json, report)
            print(json.dumps(report, indent=2, allow_nan=False))
            return
        tracks = {track.identifier: track for track in read_catalog()}
        if arguments.track not in tracks:
            raise ValueError("Unknown catalog track identifier.")
        track = tracks[arguments.track]
        if arguments.finalize:
            if (
                arguments.regenerate
                or arguments.reset_draft
                or arguments.bpm is not None
                or arguments.phase is not None
            ):
                raise ValueError("Finalization consumes the saved draft; edit its confirmed_bpm before finalizing.")
            finalize_annotation(track)
        else:
            summary = annotate_track(
                track,
                arguments.bpm,
                arguments.phase,
                regenerate=arguments.regenerate,
                plot=arguments.plot,
                start=arguments.start,
                duration=arguments.duration,
                diagnostics=arguments.diagnostics,
                reset=arguments.reset_draft,
                migrate=arguments.migrate_draft,
            )
            print(json.dumps(summary, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        parser.exit(2, f"Annotation error: {error}\n")


if __name__ == "__main__":
    main()
