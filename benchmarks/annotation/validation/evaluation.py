"""Shared reference exclusion and descriptive historical population comparisons."""

import json
from collections import Counter
from dataclasses import asdict
from typing import TypedDict

import numpy as np
from mir_eval.util import match_events

from benchmarks.annotation.analysis import ALGORITHM_VERSION, PARAMETERS, Candidate
from benchmarks.metrics import TOLERANCES, evaluate
from benchmarks.result_paths import ACOUSTIC_BASELINE


class AcousticFixture(TypedDict):
    identifier: str
    ground_truth_kick_onsets: list[float]
    all_candidate_timestamps: list[float]
    all_candidates_metrics_by_tolerance_ms: dict[str, dict[str, int | float | None]]


class AcousticBaseline(TypedDict):
    annotation_algorithm_version: int
    analysis_parameters: dict[str, object]
    fixtures: list[AcousticFixture]


def reference_onsets(onsets: tuple[float, ...]) -> list[float]:
    return sorted({onsets[0], onsets[len(onsets) // 2], onsets[-1]}) if onsets else []


def exclude_references[T: Candidate](candidates: list[T], references: list[float]) -> list[T]:
    # Match once on original acoustic anchors, independently of later labels or
    # timing refinement. Duplicates remain evaluation errors.
    pairs = (
        match_events(np.asarray(references), np.asarray([c["timestamp"] for c in candidates]), max(TOLERANCES))
        if references and candidates
        else []
    )
    excluded = {index for _, index in pairs}
    return [candidate for index, candidate in enumerate(candidates) if index not in excluded]


type AcousticPopulation = tuple[str, tuple[float, ...], list[float], float]


def compare_acoustic_populations(populations: list[AcousticPopulation]) -> dict[str, object]:
    """Describe changes to V1 discovery without constraining current experiments."""
    baseline: AcousticBaseline = json.loads(ACOUSTIC_BASELINE.read_text(encoding="utf-8"))
    recorded = {fixture["identifier"]: fixture for fixture in baseline["fixtures"]}
    fixtures: list[dict[str, object]] = []
    for identifier, truth, times, duration in populations:
        historical = recorded.get(identifier)
        if historical is None:
            fixtures.append({"identifier": identifier, "compatible": False, "reason": "No historical fixture."})
            continue
        old = historical["all_candidate_timestamps"]
        compatible = list(truth) == historical["ground_truth_kick_onsets"]
        fixtures.append(
            {
                "identifier": identifier,
                "compatible": compatible,
                "reason": None if compatible else "Ground-truth population changed; metrics are not comparable.",
                "candidate_population_changed": times != old,
                "historical_candidate_count": len(old),
                "current_candidate_count": len(times),
                "added_timestamps": sorted((Counter(times) - Counter(old)).elements()),
                "removed_timestamps": sorted((Counter(old) - Counter(times)).elements()),
                "current_acoustic_metrics_at_50ms": evaluate(truth, times, duration, 0.050).metrics(),
                "historical_acoustic_metrics_at_50ms": historical["all_candidates_metrics_by_tolerance_ms"]["50"]
                if compatible
                else None,
            }
        )
    return {
        "baseline": str(ACOUSTIC_BASELINE),
        "historical_algorithm_version": baseline["annotation_algorithm_version"],
        "current_algorithm_version": ALGORITHM_VERSION,
        "parameters_changed": baseline["analysis_parameters"] != json.loads(json.dumps(asdict(PARAMETERS))),
        "historical_only_fixtures": sorted(recorded.keys() - {item[0] for item in populations}),
        "fixtures": fixtures,
        "note": "Acoustic candidate discovery only; not a ranking of guided or automatic classification.",
    }
