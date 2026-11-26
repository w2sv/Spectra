"""Reference-excluded synthetic validation of independent template classification."""

from dataclasses import asdict

from benchmarks.annotation.analysis import (
    ALGORITHM_VERSION,
    PARAMETERS,
    TEMPLATE_BANDS,
    AcousticAnalysis,
    Candidate,
    analyze_audio,
    generate_proposal,
)
from benchmarks.annotation.templates import (
    REFINED_ALGORITHM_VERSION,
    TEMPLATE_PARAMETERS,
    candidate_onset,
    compare_signatures,
    event_feature,
)
from benchmarks.annotation.validation.evaluation import (
    AcousticPopulation,
    compare_acoustic_populations,
    exclude_references,
    reference_onsets,
)
from benchmarks.fixtures import synthetic_fixtures
from benchmarks.metrics import TOLERANCES, Evaluation, aggregate, evaluate

STAGES = ("acoustic", "grid_aligned", "likely_original", "likely_refined", "likely_and_ambiguous")


def stage_events(candidates: list[Candidate], grid_available: bool) -> dict[str, list[float] | None]:
    return {
        "acoustic": [c["timestamp"] for c in candidates],
        "grid_aligned": [c["timestamp"] for c in candidates if c["grid_aligned"]] if grid_available else None,
        "likely_original": [c["timestamp"] for c in candidates if c.get("classification") == "likely"],
        "likely_refined": sorted(candidate_onset(c) for c in candidates if c.get("classification") == "likely"),
        "likely_and_ambiguous": sorted(
            candidate_onset(c) for c in candidates if c.get("classification") in ("likely", "ambiguous")
        ),
    }


def validate_guided_template_matching() -> dict[str, object]:
    fixtures = synthetic_fixtures()
    analyses: dict[str, AcousticAnalysis] = {
        fixture.identifier: analyze_audio(fixture.load_audio(), fixture.sample_rate) for fixture in fixtures
    }
    clean = next(fixture for fixture in fixtures if fixture.identifier == "four_on_floor")
    clean_references = reference_onsets(clean.kick_onsets)
    negative_source = analyses["offbeat_bass"]
    negative_reference = 0.75  # Generator's independently known first offbeat note onset.
    populations: list[AcousticPopulation] = []
    cases = {}
    for case, use_negative in (("positive_only", False), ("with_offbeat_negative", True)):
        pooled: dict[str, dict[str, list[Evaluation]]] = {
            stage: {str(round(t * 1000)): [] for t in TOLERANCES} for stage in STAGES
        }
        results = []
        for fixture in fixtures:
            analysis = analyses[fixture.identifier]
            local = reference_onsets(fixture.kick_onsets)
            positives = local if local else clean_references
            source = analysis if local else analyses["four_on_floor"]
            positive_features = [(time, event_feature(source, time)) for time in positives]
            negative_features = (
                [(negative_reference, event_feature(negative_source, negative_reference))] if use_negative else []
            )
            tempo = fixture.generation_parameters["tempo_bpm"]
            bpm = float(tempo) if isinstance(tempo, (int, float)) else None
            proposal = generate_proposal(analysis, fixture.identifier, None, fixture.sample_rate, bpm)
            candidates = proposal["kick_candidates"]
            if not use_negative:
                populations.append(
                    (
                        fixture.identifier,
                        fixture.kick_onsets,
                        [c["timestamp"] for c in candidates],
                        analysis.duration_seconds,
                    )
                )
            references_to_exclude = local + (
                [negative_reference] if use_negative and fixture.identifier == "offbeat_bass" else []
            )
            remaining = exclude_references(candidates, references_to_exclude)
            classified = compare_signatures(analysis, remaining, positive_features, negative_features)
            truth = [onset for onset in fixture.kick_onsets if onset not in local]
            events = stage_events(classified, bpm is not None)
            metrics = {}
            for stage, times in events.items():
                if times is None:
                    metrics[stage] = None
                    continue
                stage_metrics = {}
                for tolerance in TOLERANCES:
                    key = str(round(tolerance * 1000))
                    result = evaluate(truth, times, analysis.duration_seconds, tolerance)
                    stage_metrics[key] = result.metrics()
                    pooled[stage][key].append(result)
                metrics[stage] = stage_metrics
            results.append(
                {
                    "identifier": fixture.identifier,
                    "category": fixture.category,
                    "positive_template_source": fixture.identifier if local else "four_on_floor",
                    "positive_reference_onsets": positives,
                    "excluded_positive_onsets": local,
                    "negative_template_source": "offbeat_bass" if use_negative else None,
                    "negative_reference_onset": negative_reference if use_negative else None,
                    "excluded_candidate_timestamps": [c["timestamp"] for c in candidates if c not in remaining],
                    "ground_truth_non_reference_onsets": truth,
                    "stage_event_timestamps": events,
                    "classification_counts": {
                        label: sum(c.get("classification") == label for c in classified)
                        for label in ("likely", "ambiguous", "unlikely")
                    },
                    "timing_search_boundary_count": sum(
                        c.get("alignment_at_search_boundary", False) for c in classified
                    ),
                    "candidate_classifications": classified,
                    "metrics_by_stage_and_tolerance_ms": metrics,
                }
            )
        cases[case] = {
            "fixtures": results,
            "aggregate_by_stage_and_tolerance_ms": {
                stage: {key: aggregate(values).metrics() for key, values in windows.items()}
                for stage, windows in pooled.items()
            },
        }
    return {
        "strategy": "guided-template-matching",
        "historical_comparison": compare_acoustic_populations(populations),
        "annotation_algorithm_version": REFINED_ALGORITHM_VERSION,
        "acoustic_algorithm_version": ALGORITHM_VERSION,
        "acoustic_parameters": asdict(PARAMETERS),
        "template_parameters": asdict(TEMPLATE_PARAMETERS)
        | {"bands_hz": TEMPLATE_BANDS, "frame_interval_seconds": 0.005},
        "reference_exclusion": "First/middle/last kick (unique); one-to-one acoustic matching at 100ms removes reference candidates from every stage. One-kick fixture has no non-reference positives.",
        "negative_evaluation": "Cross-fixture stress test using known first offbeat note at 0.75s; exclude that candidate in its source fixture. Not the intended track-specific workflow.",
        "timing": "5ms feature frames; nearest frame context; offset-only refinement with no global correction; combined likely/ambiguous uses refined timestamps.",
        "baseline_comparison": "All current stages use the same non-reference candidate and ground-truth population. Historical acoustic differences are reported separately.",
        "cases": cases,
    }
