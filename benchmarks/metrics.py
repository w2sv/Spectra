"""One-to-one event evaluation with pooled counts and matched timing errors."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from mir_eval.util import match_events

TOLERANCES = (0.025, 0.050, 0.100)


@dataclass(frozen=True)
class Evaluation:
    true_positives: int
    false_positives: int
    false_negatives: int
    duration_seconds: float
    signed_errors: tuple[float, ...]

    def metrics(self) -> dict[str, int | float | None]:
        tp, fp, fn = self.true_positives, self.false_positives, self.false_negatives
        absolute = np.abs(self.signed_errors)
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "false_positives_per_minute": fp * 60 / self.duration_seconds,
            "median_signed_error_seconds": float(np.median(self.signed_errors)) if self.signed_errors else None,
            "median_absolute_error_seconds": float(np.median(absolute)) if self.signed_errors else None,
            "p95_absolute_error_seconds": float(np.percentile(absolute, 95)) if self.signed_errors else None,
        }


def evaluate(
    annotated: Sequence[float], detected: Sequence[float], duration_seconds: float, tolerance: float
) -> Evaluation:
    if not np.isfinite(duration_seconds) or duration_seconds <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Duration and tolerance must be finite and positive.")
    reference = np.asarray(annotated, dtype=np.float64)
    estimated = np.asarray(detected, dtype=np.float64)
    for events in (reference, estimated):
        if events.ndim != 1 or not np.all(np.isfinite(events)) or np.any(events < 0) or np.any(np.diff(events) < 0):
            raise ValueError("Events must be finite, nonnegative, and sorted; duplicates are allowed.")
    # Avoid mir_eval's nonempty-input assumptions; empty cases still contribute counts.
    pairs = match_events(reference, estimated, tolerance) if len(reference) and len(estimated) else []
    return Evaluation(
        len(pairs),
        len(estimated) - len(pairs),
        len(reference) - len(pairs),
        duration_seconds,
        tuple(float(estimated[j] - reference[i]) for i, j in pairs),
    )


def aggregate(evaluations: Sequence[Evaluation]) -> Evaluation:
    return Evaluation(
        sum(result.true_positives for result in evaluations),
        sum(result.false_positives for result in evaluations),
        sum(result.false_negatives for result in evaluations),
        sum(result.duration_seconds for result in evaluations),
        tuple(error for result in evaluations for error in result.signed_errors),
    )
