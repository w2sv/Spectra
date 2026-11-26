"""Ground truth is used only after automatic proposal generation, for evaluation."""

import json
from dataclasses import asdict

import numpy as np
from mir_eval.util import match_events

from benchmarks.annotation.analysis import PARAMETERS, analyze_audio, generate_proposal
from benchmarks.annotation.automatic import AUTOMATIC_ALGORITHM_VERSION, AUTOMATIC_PARAMETERS, automatic_proposal
from benchmarks.annotation.validation.evaluation import (
    AcousticPopulation,
    compare_acoustic_populations,
    exclude_references,
)
from benchmarks.fixtures import synthetic_fixtures
from benchmarks.metrics import TOLERANCES, Evaluation, aggregate, evaluate
from benchmarks.result_paths import GUIDED_BASELINE


def validate_automatic_template_matching() -> dict[str, object]:
    stages = (
        "original_acoustic",
        "population_original",
        "population_localized",
        "population_refined",
        "automatic_original",
        "automatic_localized",
        "automatic_refined",
    )
    pooled: dict[str, dict[str, list[Evaluation]]] = {
        stage: {str(round(t * 1000)): [] for t in TOLERANCES} for stage in stages
    }
    partitions: dict[str, list[Evaluation]] = {"with_tempo": [], "gridless": []}
    previous_guided = json.loads(GUIDED_BASELINE.read_text(encoding="utf-8"))
    guided = {fixture["identifier"]: fixture for fixture in previous_guided["cases"]["positive_only"]["fixtures"]}
    populations: list[AcousticPopulation] = []
    results = []
    for fixture in synthetic_fixtures():
        analysis = analyze_audio(fixture.load_audio(), fixture.sample_rate)
        tempo = fixture.generation_parameters["tempo_bpm"]
        bpm = float(tempo) if isinstance(tempo, (int, float)) else None
        acoustic = generate_proposal(analysis, fixture.identifier, None, fixture.sample_rate, bpm)
        # No ground truth, known onsets, or manual templates enter this call.
        proposal = automatic_proposal(analysis, acoustic)
        candidates = proposal["kick_candidates"]
        predicted = [c for c in candidates if c["automatic_kick"]]
        events = {"original_acoustic": [c["timestamp"] for c in acoustic["kick_candidates"]]}
        for prefix, population in (("population", candidates), ("automatic", predicted)):
            for suffix, field in (
                ("original", "timestamp"),
                ("localized", "localized_timestamp"),
                ("refined", "final_timestamp"),
            ):
                events[f"{prefix}_{suffix}"] = sorted(float(c.get(field, c["timestamp"])) for c in population)
        metrics = {}
        for stage, timestamps in events.items():
            metrics[stage] = {}
            for tolerance in TOLERANCES:
                key = str(round(tolerance * 1000))
                result = evaluate(fixture.kick_onsets, timestamps, analysis.duration_seconds, tolerance)
                metrics[stage][key] = result.metrics()
                pooled[stage][key].append(result)
        partitions["with_tempo" if bpm is not None else "gridless"].append(
            evaluate(fixture.kick_onsets, events["automatic_refined"], analysis.duration_seconds, 0.050)
        )
        # A single assignment on the SAME population separates label changes from
        # missing events. Do not subtract independently matched TP counts.
        ordered = sorted(candidates, key=lambda c: c["final_timestamp"])
        pairs = (
            match_events(np.asarray(fixture.kick_onsets), np.asarray([c["final_timestamp"] for c in ordered]), 0.050)
            if fixture.kick_onsets and ordered
            else []
        )
        matched = {index for _, index in pairs}
        corrections = sum(c["automatic_kick"] != (i in matched) for i, c in enumerate(ordered))
        old = guided.get(fixture.identifier)
        compatible = old is not None and sorted(
            old["ground_truth_non_reference_onsets"] + old["excluded_positive_onsets"]
        ) == list(fixture.kick_onsets)
        if old is not None and compatible:
            references = old["excluded_positive_onsets"]
            remaining = exclude_references(candidates, references)
            comparison = {
                str(round(t * 1000)): evaluate(
                    old["ground_truth_non_reference_onsets"],
                    sorted(c["final_timestamp"] for c in remaining if c["automatic_kick"]),
                    analysis.duration_seconds,
                    t,
                ).metrics()
                for t in TOLERANCES
            }
            comparison_report = {
                "compatible": True,
                "reason": None,
                "automatic": comparison,
                "historical_guided": old["metrics_by_stage_and_tolerance_ms"],
                "excluded_reference_onsets": references,
            }
        else:
            comparison_report = {
                "compatible": False,
                "reason": "Historical guided reference/evaluation population is unavailable or changed.",
                "automatic": None,
                "historical_guided": None,
                "excluded_reference_onsets": [],
            }
        populations.append(
            (fixture.identifier, fixture.kick_onsets, events["original_acoustic"], analysis.duration_seconds)
        )
        results.append(
            {
                "identifier": fixture.identifier,
                "bpm": bpm,
                "generation_parameters": fixture.generation_parameters,
                "ground_truth_onsets": fixture.kick_onsets,
                "candidate_count": len(candidates),
                "predicted_kicks": len(predicted),
                "cluster_count": len(proposal.get("clusters", [])),
                "reference_count": len(proposal.get("automatic_reference_ids", [])),
                "automatic_confidence": proposal.get("automatic_confidence"),
                "selected_grid": proposal.get("selected_grid"),
                "binary_classification_corrections_at_50ms": corrections,
                "manual_insertions_at_50ms": len(fixture.kick_onsets) - len(pairs),
                "candidate_classifications": candidates,
                "event_timestamps_by_stage": events,
                "metrics_by_stage_and_tolerance_ms": metrics,
                "comparison_on_guided_non_reference_population": comparison_report,
            }
        )
    return {
        "strategy": "automatic-template-matching",
        "historical_comparison": compare_acoustic_populations(populations),
        "annotation_algorithm_version": AUTOMATIC_ALGORITHM_VERSION,
        "automatic_parameters": asdict(AUTOMATIC_PARAMETERS),
        "acoustic_parameters": asdict(PARAMETERS),
        "fixtures": results,
        "aggregate_by_stage_and_tolerance_ms": {
            stage: {key: aggregate(values).metrics() for key, values in windows.items()}
            for stage, windows in pooled.items()
        },
        "automatic_by_tempo_availability_at_50ms": {
            key: aggregate(values).metrics() for key, values in partitions.items()
        },
        "comparison_notes": "Primary automatic metrics use the full current fixture population without supplied reference templates. The separately labeled historical guided comparison excludes its references only when ground truth is compatible. Acoustic changes are descriptive historical comparisons.",
        "total_binary_corrections_at_50ms": sum(item["binary_classification_corrections_at_50ms"] for item in results),
        "total_manual_insertions_at_50ms": sum(item["manual_insertions_at_50ms"] for item in results),
    }
