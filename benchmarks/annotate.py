"""Independent acoustic kick proposals, review visualization, and explicit finalization."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from benchmarks.annotation_analysis import (
    ALGORITHM_VERSION,
    PARAMETERS,
    Proposal,
    analyze_audio,
    generate_proposal,
)
from benchmarks.fixtures import synthetic_fixtures, verify_checksum
from benchmarks.metrics import TOLERANCES, Evaluation, aggregate, evaluate
from benchmarks.tracks import TECHNO, Track, read_catalog, validate_timestamps, verified_audio


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def proposal_summary(proposal: Proposal) -> dict[str, object]:
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
    root: Path = TECHNO,
) -> dict[str, object]:
    if track.partition == "evaluation":
        raise ValueError("Annotation pilot excludes held-out evaluation tracks.")
    path = root / "proposals" / f"{track.identifier}.json"
    proposal: Proposal
    if path.exists() and not regenerate:
        if bpm is not None or phase is not None:
            raise ValueError("Use --regenerate to replace proposals with BPM/phase overrides.")
        proposal = json.loads(path.read_text(encoding="utf-8"))
        verify_checksum(root / "audio" / track.filename, track.sha256)
        if proposal["track_identifier"] != track.identifier or proposal["audio_sha256"] != track.sha256:
            raise ValueError("Existing proposals do not match this recording.")
        if proposal["annotation_algorithm_version"] != ALGORITHM_VERSION:
            raise ValueError("Existing proposals use another algorithm version; explicitly --regenerate them.")
        analysis = None
    else:
        bpm = track.bpm if bpm is None else bpm
        if bpm is None:
            raise ValueError(f"{track.identifier}: catalog BPM unknown; provide --bpm.")
        started = perf_counter()
        rate, samples = verified_audio(track, root)
        analysis = analyze_audio(samples, rate)
        del samples
        proposal = generate_proposal(analysis, track.identifier, track.sha256, rate, bpm, phase)
        proposal["informational"]["annotation_processing_seconds"] = perf_counter() - started
        write_json(path, proposal)
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
            import matplotlib

            matplotlib.use("Agg")
            from benchmarks.annotation_review import save_diagnostics

            summary["diagnostic_intervals"] = save_diagnostics(analysis, proposal, root / "proposals" / "diagnostics")
        if plot:
            from benchmarks.annotation_review import show_review

            show_review(analysis, proposal, root / "drafts" / f"{track.identifier}.json", start, duration)
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


def validate_synthetics() -> dict[str, object]:
    fixtures = []
    pooled_all: dict[str, list[Evaluation]] = {str(round(t * 1000)): [] for t in TOLERANCES}
    pooled_grid: dict[str, list[Evaluation]] = {key: [] for key in pooled_all}
    for fixture in synthetic_fixtures():
        analysis = analyze_audio(fixture.load_audio(), fixture.sample_rate)
        tempo = fixture.generation_parameters["tempo_bpm"]
        bpm = float(tempo) if isinstance(tempo, (int, float)) else None
        proposal = generate_proposal(analysis, fixture.identifier, None, fixture.sample_rate, bpm)
        all_events = [candidate["timestamp"] for candidate in proposal["kick_candidates"]]
        grid_events = [candidate["timestamp"] for candidate in proposal["kick_candidates"] if candidate["grid_aligned"]]
        all_metrics, grid_metrics = {}, {}
        for tolerance in TOLERANCES:
            key = str(round(tolerance * 1000))
            all_result = evaluate(fixture.kick_onsets, all_events, analysis.duration_seconds, tolerance)
            all_metrics[key] = all_result.metrics()
            pooled_all[key].append(all_result)
            if bpm is not None:
                grid_result = evaluate(fixture.kick_onsets, grid_events, analysis.duration_seconds, tolerance)
                grid_metrics[key] = grid_result.metrics()
                pooled_grid[key].append(grid_result)
        fixtures.append(
            {
                "identifier": fixture.identifier,
                "category": fixture.category,
                "bpm": bpm,
                "ground_truth_kick_onsets": fixture.kick_onsets,
                "all_candidate_timestamps": all_events,
                "grid_aligned_timestamps": grid_events if bpm is not None else None,
                "all_candidates_metrics_by_tolerance_ms": all_metrics,
                "grid_aligned_metrics_by_tolerance_ms": grid_metrics if bpm is not None else None,
            }
        )
    return {
        "annotation_algorithm_version": ALGORITHM_VERSION,
        "analysis_parameters": asdict(PARAMETERS),
        "note": "Independent annotation sanity check; grid metrics only include fixtures with tempo metadata.",
        "fixtures": fixtures,
        "aggregate_all_candidates_by_tolerance_ms": {
            key: aggregate(values).metrics() for key, values in pooled_all.items()
        },
        "aggregate_grid_aligned_by_tolerance_ms": {
            key: aggregate(values).metrics() for key, values in pooled_grid.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--track")
    selection.add_argument("--synthetic", action="store_true", help="Validate proposals on all 15 synthetic fixtures.")
    parser.add_argument("--bpm", type=float)
    parser.add_argument("--phase", type=float)
    parser.add_argument("--regenerate", action="store_true", help="Explicitly replace automatic proposals only.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--diagnostics", action="store_true", help="Save the first qualifying 10s diagnostic plots.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--json", type=Path, help="Export synthetic validation metrics.")
    arguments = parser.parse_args()
    try:
        if arguments.synthetic:
            if (
                arguments.plot
                or arguments.finalize
                or arguments.diagnostics
                or arguments.bpm is not None
                or arguments.phase is not None
            ):
                raise ValueError("Synthetic validation uses fixture tempos and does not edit real annotations.")
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
            if arguments.regenerate or arguments.bpm is not None or arguments.phase is not None:
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
            )
            print(json.dumps(summary, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        parser.exit(2, f"Annotation error: {error}\n")


if __name__ == "__main__":
    main()
