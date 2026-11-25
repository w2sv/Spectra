"""Independent acoustic proposal validation on the original synthetic fixtures."""

from dataclasses import asdict

from benchmarks.annotation.analysis import ALGORITHM_VERSION, PARAMETERS, analyze_audio, generate_proposal
from benchmarks.fixtures import synthetic_fixtures
from benchmarks.metrics import TOLERANCES, Evaluation, aggregate, evaluate


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
