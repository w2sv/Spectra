"""One-to-one event accounting, timing metrics and grouped benchmark reports."""

import numpy as np
import pytest

from benchmarks.fixtures import AudioFixture
from benchmarks.kick_detection import grouped_aggregates
from benchmarks.metrics import aggregate, evaluate


def test_matching_is_one_to_one_including_duplicate_detections() -> None:
    result = evaluate([1.0], [1.01, 1.01], 2.0, 0.05)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 0)
    assert result.signed_errors == pytest.approx((0.01,))
    # A maximum matching must retain both events when tolerance windows overlap.
    overlapping = evaluate([1.0, 1.08], [0.96, 1.04], 2.0, 0.05)
    assert overlapping.true_positives == 2


def test_missed_extra_detections_timing_and_pooled_counts() -> None:
    result = evaluate([1.0, 2.0, 3.0], [1.02, 1.5, 3.04], 60.0, 0.05)
    metrics = result.metrics()
    assert (result.true_positives, result.false_positives, result.false_negatives) == (2, 1, 1)
    for name in ("precision", "recall", "f1"):
        assert metrics[name] == pytest.approx(2 / 3)
    assert metrics["false_positives_per_minute"] == 1.0
    assert metrics["median_signed_error_seconds"] == pytest.approx(0.03)
    assert metrics["median_absolute_error_seconds"] == pytest.approx(0.03)
    assert metrics["p95_absolute_error_seconds"] == pytest.approx(0.039)
    pooled = aggregate([result, evaluate([], [1.0, 2.0], 30.0, 0.05)]).metrics()
    assert pooled["precision"] == pytest.approx(2 / 5)
    assert pooled["recall"] == pytest.approx(2 / 3)
    assert pooled["f1"] == 0.5
    assert pooled["false_positives_per_minute"] == 2.0
    assert evaluate([1.0], [0.98], 2.0, 0.05).metrics()["median_signed_error_seconds"] == pytest.approx(-0.02)


@pytest.mark.parametrize(
    ("annotated", "detected", "expected"),
    [([], [], (None, None, None)), ([1.0], [], (None, 0.0, 0.0)), ([], [1.0], (0.0, None, 0.0))],
)
def test_empty_annotations_and_detections(
    annotated: list[float], detected: list[float], expected: tuple[float | None, ...]
) -> None:
    result = evaluate(annotated, detected, 60.0, 0.05)
    metrics = result.metrics()
    assert result.true_positives == 0
    assert result.false_positives == len(detected)
    assert result.false_negatives == len(annotated)
    assert tuple(metrics[name] for name in ("precision", "recall", "f1")) == expected
    assert metrics["median_signed_error_seconds"] is None
    assert metrics["median_absolute_error_seconds"] is None
    assert metrics["p95_absolute_error_seconds"] is None
    assert metrics["false_positives_per_minute"] == len(detected)


def test_group_aggregates_pool_counts_and_keep_category_dataset_boundaries() -> None:
    audio = np.zeros((48_000, 2), dtype=np.float32)
    fixtures = [
        AudioFixture("one", "shared", audio, 48_000, (0.5,)),
        AudioFixture("two", "shared", audio, 48_000, (0.5,), dataset="techno", partition="development"),
        AudioFixture("three", "shared", audio, 48_000, (), dataset="techno", partition="development"),
    ]
    evaluations = [
        {"50": evaluate([0.5], [0.51], 1, 0.05)},
        {"50": evaluate([0.5], [], 1, 0.05)},
        {"50": evaluate([], [0.5, 0.6], 1, 0.05)},
    ]
    datasets, categories, partitions = grouped_aggregates(fixtures, evaluations)
    assert datasets["techno"]["fixture_count"] == 2
    assert datasets["techno"]["total_duration_seconds"] == 2
    assert datasets["techno"]["ground_truth_kick_count"] == 1
    assert datasets["techno"]["metrics_by_tolerance_ms"]["50"]["false_positives"] == 2
    assert datasets["techno"]["metrics_by_tolerance_ms"]["50"]["false_negatives"] == 1
    assert categories["synthetic"]["shared"]["fixture_count"] == 1
    assert categories["techno"]["shared"]["fixture_count"] == 2
    assert partitions["development"] == datasets["techno"]
