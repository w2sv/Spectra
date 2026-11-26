"""Shared feature representation, supplied-reference matching, and grid diagnostics."""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from benchmarks.annotation.analysis import (
    PARAMETERS,
    TEMPLATE_BANDS,
    AcousticAnalysis,
    Candidate,
    Proposal,
    associate_candidates,
    beat_positions,
)

REFINED_ALGORITHM_VERSION = 2


@dataclass(frozen=True)
class TemplateParameters:
    context_start_seconds: float = -0.020
    context_end_seconds: float = 0.160
    background_start_seconds: float = -0.100
    background_end_seconds: float = -0.025
    search_radius_seconds: float = 0.030
    strong_similarity: float = 0.85
    weak_similarity: float = 0.70
    negative_margin: float = 0.10
    duplicate_tolerance_seconds: float = 0.030

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if not np.all(np.isfinite(values)) or not 0 <= self.weak_similarity <= self.strong_similarity <= 1:
            raise ValueError("Use finite template parameters and 0 <= weak <= strong <= 1.")
        if (
            not 0 <= self.negative_margin <= 1
            or self.search_radius_seconds <= 0
            or self.duplicate_tolerance_seconds < 0
        ):
            raise ValueError(
                "Use a valid similarity margin, positive search radius, and nonnegative duplicate tolerance."
            )

        if (
            not self.background_start_seconds
            < self.background_end_seconds
            <= self.context_start_seconds
            <= 0
            < self.context_end_seconds
        ):
            raise ValueError("Background must precede an event context that includes its onset.")
        if self.search_radius_seconds < PARAMETERS.hop_size / PARAMETERS.sample_rate:
            raise ValueError("The search radius must include at least one analysis frame.")


TEMPLATE_PARAMETERS = TemplateParameters()


def event_feature(
    analysis: AcousticAnalysis, timestamp: float, parameters: TemplateParameters = TEMPLATE_PARAMETERS
) -> NDArray[np.float64] | None:
    if not np.isfinite(timestamp) or not 0 <= timestamp < analysis.duration_seconds:
        return None
    step = PARAMETERS.hop_size / PARAMETERS.sample_rate
    center = round(timestamp / step)

    def context(first: float, last: float) -> NDArray[np.float64]:
        indices = center + np.arange(round(first / step), round(last / step) + 1)
        powers = np.zeros((len(indices), len(TEMPLATE_BANDS)), dtype=np.float64)
        available = (indices >= 0) & (indices < len(analysis.template_band_power))
        powers[available] = analysis.template_band_power[indices[available]]
        return powers

    power = context(parameters.context_start_seconds, parameters.context_end_seconds)
    background = np.median(context(parameters.background_start_seconds, parameters.background_end_seconds), axis=0)
    feature = np.maximum(np.sqrt(power) - np.sqrt(background), 0).ravel()
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 0 else None


def classification(
    positive: float | None,
    negative: float | None,
    has_negatives: bool,
    parameters: TemplateParameters = TEMPLATE_PARAMETERS,
) -> Literal["likely", "ambiguous", "unlikely"]:
    if positive is None or positive < parameters.weak_similarity:
        return "unlikely"
    if positive >= parameters.strong_similarity and (
        not has_negatives or (negative is not None and positive >= negative + parameters.negative_margin)
    ):
        return "likely"
    return "ambiguous"


def compare_signatures(
    analysis: AcousticAnalysis,
    candidates: list[Candidate],
    positive: list[tuple[float, NDArray[np.float64] | None]],
    negative: list[tuple[float, NDArray[np.float64] | None]],
    parameters: TemplateParameters = TEMPLATE_PARAMETERS,
) -> list[Candidate]:
    """Compare pre-extracted references; cross-fixture use is confined to synthetic validation."""
    valid_positive = [(time, feature) for time, feature in positive if feature is not None]
    valid_negative = [feature for _, feature in negative if feature is not None]
    step = PARAMETERS.hop_size / PARAMETERS.sample_rate
    radius = round(parameters.search_radius_seconds / step)
    # Exact ties prefer the smallest shift; no systematic timing correction.
    offsets = sorted((i * step for i in range(-radius, radius + 1)), key=lambda value: (abs(value), value))
    results: list[Candidate] = []
    for candidate in candidates:
        result = candidate.copy()
        result.update(
            novelty_timestamp=candidate["timestamp"],
            refined_timestamp=None,
            positive_similarity=None,
            negative_similarity=None,
            best_positive_template_timestamp=None,
            alignment_offset_seconds=None,
            classification="unclassified",
            alignment_at_search_boundary=False,
        )
        if valid_positive or valid_negative:
            best_positive: float | None = None
            best_negative: float | None = None
            best_time: float | None = None
            best_offset: float | None = None
            for offset in offsets:
                feature = event_feature(analysis, candidate["timestamp"] + offset, parameters)
                if feature is None:
                    continue
                for reference_time, reference in valid_positive:
                    score = float(np.clip(feature @ reference, -1, 1))
                    if best_positive is None or score > best_positive:
                        best_positive, best_time, best_offset = score, reference_time, offset
                for reference in valid_negative:
                    score = float(np.clip(feature @ reference, -1, 1))
                    if best_negative is None or score > best_negative:
                        best_negative = score
            result["positive_similarity"] = best_positive
            result["negative_similarity"] = best_negative
            result["best_positive_template_timestamp"] = best_time
            result["alignment_offset_seconds"] = best_offset
            result["refined_timestamp"] = candidate["timestamp"] + best_offset if best_offset is not None else None
            # An invalid negative reference cannot silently remove the margin requirement.
            if valid_positive:
                result["classification"] = classification(best_positive, best_negative, bool(negative), parameters)
            result["alignment_at_search_boundary"] = (
                best_offset is not None and abs(best_offset) >= radius * step - 1e-10
            )
        results.append(result)
    return results


def candidate_onset(candidate: Candidate) -> float:
    refined = candidate.get("refined_timestamp")
    return candidate["timestamp"] if refined is None else refined


def alternative_phase(phase: float, bpm: float) -> float:
    return (phase + 30 / bpm) % (60 / bpm)


def grid_comparisons(proposal: Proposal) -> list[dict[str, object]]:
    bpm, phase = proposal["bpm"], proposal["beat_grid_phase_seconds"]
    if bpm is None or phase is None:
        return []
    comparisons = []
    for section in proposal["beat_grid_validation"]:
        start, end = section["start_seconds"], section["end_seconds"]
        candidates = [candidate for candidate in proposal["kick_candidates"] if start <= candidate["timestamp"] < end]
        grids = {}
        for name, grid_phase in (("primary", phase), ("alternative", alternative_phase(phase, bpm))):
            beats = beat_positions(bpm, grid_phase, start, end)
            aligned, unsupported = associate_candidates(candidates, beats)
            supported = [candidate for candidate in aligned if candidate["grid_aligned"]]
            scores = [score for candidate in supported if (score := candidate.get("positive_similarity")) is not None]
            grids[name] = {
                "aligned_candidates": len(supported),
                "unsupported_beats": len(unsupported),
                "mean_positive_similarity": float(np.mean(scores)) if scores else None,
                "likely_kicks": sum(candidate.get("classification") == "likely" for candidate in supported),
            }
        comparisons.append({"acoustic_phase_evidence": dict(section), **grids})
    return comparisons
