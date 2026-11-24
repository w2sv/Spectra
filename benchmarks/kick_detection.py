"""Benchmark the production kick pipeline against annotated audio, without capture or UI."""

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, fields
from importlib.metadata import version
from math import gcd
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import numpy as np
from scipy.signal import firwin, upfirdn

from benchmarks.fixtures import AudioFixture, load_manifest, synthetic_fixtures
from benchmarks.metrics import TOLERANCES, Evaluation, aggregate, evaluate
from benchmarks.tracks import CollectionStatus, collection_status, read_catalog, reviewed_fixtures
from spectra.analysis import KickDetectionConfig, KickOnsetDetector, SpectrumAnalyzer
from spectra.audio import DEFAULT_READ_FRAMES, AudioSamples

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 3


class FixtureReport(TypedDict):
    identifier: str
    category: str
    dataset: str
    partition: str
    audio_file: str | None
    source_sample_rate: int
    source_sample_count: int
    source_stereo_float32_sha256: str
    duration_seconds: float
    generation_parameters: dict[str, object]
    resampling_delay_seconds: float
    processed_sample_count: int
    ground_truth_kick_timestamps: tuple[float, ...]
    detected_kick_timestamps: tuple[float, ...]
    detections: list[dict[str, object]]
    metrics_by_tolerance_ms: dict[str, dict[str, int | float | None]]
    informational: dict[str, float]


class AggregateReport(TypedDict):
    fixture_count: int
    total_duration_seconds: float
    ground_truth_kick_count: int
    metrics_by_tolerance_ms: dict[str, dict[str, int | float | None]]


class BenchmarkReport(TypedDict):
    schema_version: int
    git: dict[str, str | bool | None]
    implementation_sha256: dict[str, str]
    environment: dict[str, str]
    detector_configuration: dict[str, object]
    dsp_configuration: dict[str, object]
    primary_tolerance_ms: int
    total_duration_seconds: float
    fixture_count: int
    ground_truth_kick_count: int
    fixtures: list[FixtureReport]
    aggregate_metrics_by_tolerance_ms: dict[str, dict[str, int | float | None]]
    aggregates_by_dataset: dict[str, AggregateReport]
    aggregates_by_category: dict[str, dict[str, AggregateReport]]
    aggregates_by_partition: dict[str, AggregateReport]
    collection_completeness: CollectionStatus
    result_scope: str
    benchmark_executed: bool


@dataclass(frozen=True)
class Detection:
    timestamp: float
    spectrum_center_timestamp: float
    strength: float


@dataclass(frozen=True)
class DetectionRun:
    detections: tuple[Detection, ...]
    resampling_delay_seconds: float
    processed_sample_count: int
    processing_seconds: float


def resample_causally(samples: AudioSamples, source_rate: int, target_rate: int) -> tuple[AudioSamples, float]:
    """Polyphase FIR with no centering, delay compensation, or end-of-file flush.

    Output sample m uses input samples only through floor(m * down / up).
    The linear-phase filter's group delay stays in the signal, so measured
    event latency includes conversion delay. No future input is exposed.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if source_rate == target_rate:
        return samples, 0.0
    divisor = gcd(source_rate, target_rate)
    up, down = target_rate // divisor, source_rate // divisor
    half_length = 10 * max(up, down)
    coefficients = firwin(2 * half_length + 1, 1 / max(up, down)) * up
    count = (len(samples) * up + down - 1) // down
    converted = upfirdn(coefficients, samples, up=up, down=down, axis=0)[:count].astype(np.float32)
    return converted, half_length / (source_rate * up)


def detect_offline(samples: AudioSamples, sample_rate: int, config: KickDetectionConfig | None = None) -> DetectionRun:
    """Run fresh production state, paced by samples rather than execution time.

    Like CaptureWorker, consume every spectrum returned by each source read.
    Keep the detector's center timestamps for adaptation/refractory behavior,
    but publish availability at the read end, after the whole FFT window and
    its enclosing live capture chunk have arrived. No partial FFT is padded.
    """
    started = perf_counter()
    analyzer = SpectrumAnalyzer()
    detector = KickOnsetDetector(analyzer.band_edges, KickDetectionConfig() if config is None else config)
    samples, delay = resample_causally(samples, sample_rate, analyzer.sample_rate)
    detections = []
    for offset in range(0, len(samples), DEFAULT_READ_FRAMES):
        chunk = samples[offset : offset + DEFAULT_READ_FRAMES]
        available_at = (offset + len(chunk)) / analyzer.sample_rate
        for spectrum in analyzer.process(chunk):
            event = detector.process(spectrum)
            if event is not None:
                detections.append(Detection(available_at, event.timestamp, event.strength))
    return DetectionRun(tuple(detections), delay, len(samples), perf_counter() - started)


def dsp_configuration() -> dict[str, object]:
    analyzer = SpectrumAnalyzer()
    return {
        "sample_rate": analyzer.sample_rate,
        "channels": 2,
        "pcm_dtype": "float32",
        "channel_mix": "arithmetic mean, accumulated in float64",
        "read_frames": DEFAULT_READ_FRAMES,
        "fft_size": analyzer.fft_size,
        "hop_size": analyzer.hop_size,
        "window": "periodic Hann",
        "band_edges_hz": analyzer.band_edges.tolist(),
        "power": "one-sided integrated mean-square, fractional bin-cell overlaps",
        "partial_windows": "discarded; no padding or tail flush",
        "resampling": "causal polyphase FIR; Hamming window; 20*max(up,down)+1 taps; no delay compensation",
        "detector_time": "audio-relative FFT window center",
        "reported_time": "audio-relative end of capture read containing the completed FFT window",
        "timing_excludes": "OS/backend scheduling, DSP execution time, presentation and hardware output",
    }


def git_metadata() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except OSError, subprocess.CalledProcessError:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def implementation_hashes() -> dict[str, str]:
    paths = [
        "src/spectra/analysis/kick.py",
        "src/spectra/analysis/spectrum.py",
        "src/spectra/audio/__init__.py",
        "benchmarks/fixtures.py",
        "benchmarks/tracks.py",
        "benchmarks/metrics.py",
        "benchmarks/kick_detection.py",
        "uv.lock",
    ]
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def grouped_aggregates(
    fixtures: list[AudioFixture], evaluations: list[dict[str, Evaluation]]
) -> tuple[dict[str, AggregateReport], dict[str, dict[str, AggregateReport]], dict[str, AggregateReport]]:
    def summarize(indices: list[int]) -> AggregateReport:
        return {
            "fixture_count": len(indices),
            "total_duration_seconds": sum(evaluations[i]["50"].duration_seconds for i in indices),
            "ground_truth_kick_count": sum(len(fixtures[i].kick_onsets) for i in indices),
            "metrics_by_tolerance_ms": {
                key: aggregate([evaluations[i][key] for i in indices]).metrics() for key in evaluations[0]
            },
        }

    datasets = {
        dataset: summarize([i for i, fixture in enumerate(fixtures) if fixture.dataset == dataset])
        for dataset in sorted({fixture.dataset for fixture in fixtures})
    }
    categories = {
        dataset: {
            category: summarize(
                [i for i, fixture in enumerate(fixtures) if fixture.dataset == dataset and fixture.category == category]
            )
            for category in sorted({fixture.category for fixture in fixtures if fixture.dataset == dataset})
        }
        for dataset in datasets
    }
    partitions = {
        partition: summarize([i for i, fixture in enumerate(fixtures) if fixture.partition == partition])
        for partition in sorted({fixture.partition for fixture in fixtures})
    }
    return datasets, categories, partitions


def benchmark(
    fixtures: list[AudioFixture], config: KickDetectionConfig, collection: CollectionStatus | None = None
) -> BenchmarkReport:
    if not fixtures and (collection is None or collection["complete"]):
        raise ValueError("Select at least one fixture.")
    pooled: dict[str, list[Evaluation]] = {str(round(window * 1000)): [] for window in TOLERANCES}
    results: list[FixtureReport] = []
    evaluations: list[dict[str, Evaluation]] = []
    for fixture in fixtures:
        samples = fixture.load_audio()
        run = detect_offline(samples, fixture.sample_rate, config)
        detected = tuple(event.timestamp for event in run.detections)
        duration = len(samples) / fixture.sample_rate
        metrics = {}
        fixture_evaluations = {}
        for tolerance in TOLERANCES:
            key = str(round(tolerance * 1000))
            evaluation = evaluate(fixture.kick_onsets, detected, duration, tolerance)
            pooled[key].append(evaluation)
            fixture_evaluations[key] = evaluation
            metrics[key] = evaluation.metrics()
        results.append(
            {
                "identifier": fixture.identifier,
                "category": fixture.category,
                "dataset": fixture.dataset,
                "partition": fixture.partition,
                "audio_file": str(fixture.audio) if isinstance(fixture.audio, Path) else None,
                "source_sample_rate": fixture.sample_rate,
                "source_sample_count": len(samples),
                "source_stereo_float32_sha256": hashlib.sha256(samples.astype("<f4").tobytes()).hexdigest(),
                "duration_seconds": duration,
                "generation_parameters": fixture.generation_parameters,
                "resampling_delay_seconds": run.resampling_delay_seconds,
                "processed_sample_count": run.processed_sample_count,
                "ground_truth_kick_timestamps": fixture.kick_onsets,
                "detected_kick_timestamps": detected,
                "detections": [asdict(event) for event in run.detections],
                "metrics_by_tolerance_ms": metrics,
                "informational": {
                    "processing_seconds": run.processing_seconds,
                    "processing_seconds_per_audio_second": run.processing_seconds / duration,
                },
            }
        )
        evaluations.append(fixture_evaluations)
    datasets, categories, partitions = grouped_aggregates(fixtures, evaluations)
    return {
        "schema_version": SCHEMA_VERSION,
        "git": git_metadata(),
        "implementation_sha256": implementation_hashes(),
        "environment": {
            "python": platform.python_version(),
            **{package: version(package) for package in ("numpy", "scipy", "mir_eval")},
        },
        "detector_configuration": asdict(config),
        "dsp_configuration": dsp_configuration(),
        "primary_tolerance_ms": 50,
        "total_duration_seconds": sum(result.duration_seconds for result in pooled["50"]),
        "fixture_count": len(fixtures),
        "ground_truth_kick_count": sum(len(fixture.kick_onsets) for fixture in fixtures),
        "fixtures": results,
        "aggregate_metrics_by_tolerance_ms": {key: aggregate(values).metrics() for key, values in pooled.items()},
        "aggregates_by_dataset": datasets,
        "aggregates_by_category": categories,
        "aggregates_by_partition": partitions,
        "collection_completeness": collection
        if collection is not None
        else {
            "requested_track_identifiers": [],
            "evaluated_track_identifiers": [],
            "missing_audio": [],
            "missing_reviewed_annotations": [],
            "complete": True,
        },
        "result_scope": "partial" if collection is not None and not collection["complete"] else "complete",
        "benchmark_executed": bool(fixtures),
    }


def print_summary(report: BenchmarkReport) -> None:
    # Use the serializable report as the single source for terminal and JSON results.
    def row(label: str, metrics: dict[str, int | float | None]) -> None:
        def number(key: str, scale: float = 1.0) -> str:
            value = metrics[key]
            return "n/a" if value is None else f"{value * scale:.3f}"

        print(
            f"{label:24} {metrics['true_positives']:>4} {metrics['false_positives']:>4} "
            f"{metrics['false_negatives']:>4} {number('precision'):>7} {number('recall'):>7} {number('f1'):>7} "
            f"{number('false_positives_per_minute'):>8} {number('median_signed_error_seconds', 1000):>9} "
            f"{number('median_absolute_error_seconds', 1000):>9} {number('p95_absolute_error_seconds', 1000):>9}"
        )

    status = report["collection_completeness"]
    if status["requested_track_identifiers"]:
        print(
            f"{'COMPLETE' if status['complete'] else 'PARTIAL'} COLLECTION: "
            f"{len(status['evaluated_track_identifiers'])}/{len(status['requested_track_identifiers'])} requested tracks evaluated"
        )
        print(f"Requested: {', '.join(status['requested_track_identifiers'])}")
        print(f"Evaluated: {', '.join(status['evaluated_track_identifiers']) or 'none'}")
        print(f"Missing audio: {', '.join(status['missing_audio']) or 'none'}")
        print(f"Missing reviewed annotations: {', '.join(status['missing_reviewed_annotations']) or 'none'}")
    print("Primary tolerance: +/-50 ms; positive timing error = late; n/a = undefined")
    print(f"{'fixture / aggregate':24}   TP   FP   FN    prec  recall      F1   FP/min signed ms    abs ms P95abs ms")
    for tolerance, metrics in report["aggregate_metrics_by_tolerance_ms"].items():
        row(f"ALL +/-{tolerance} ms", metrics)
    for fixture in report["fixtures"]:
        row(fixture["identifier"], fixture["metrics_by_tolerance_ms"]["50"])
    groups = list(report["aggregates_by_dataset"].items()) + list(report["aggregates_by_partition"].items())
    groups += [
        (f"{dataset}/{category}", value)
        for dataset, categories in report["aggregates_by_category"].items()
        for category, value in categories.items()
    ]
    print("Groups at +/-50 ms (fixture count, seconds, ground-truth kicks):")
    for label, group in groups:
        print(
            f"{label}: {group['fixture_count']} fixtures, {group['total_duration_seconds']:.3f}s, "
            f"{group['ground_truth_kick_count']} kicks"
        )
        row(label, group["metrics_by_tolerance_ms"]["50"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="append", default=[], help="Select an identifier; repeatable.")
    parser.add_argument("--category", action="append", default=[], help="Select a category; repeatable.")
    parser.add_argument("--dataset", choices=("synthetic", "techno"), action="append", default=[])
    parser.add_argument("--partition", choices=("regression", "development", "evaluation"), action="append", default=[])
    parser.add_argument(
        "--manifest", type=Path, action="append", default=[], help="Add annotated WAV fixtures from JSON."
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE", help="Config override."
    )
    parser.add_argument("--json", type=Path, help="Write a JSON report (replaces an existing file).")
    parser.add_argument("--list", action="store_true", help="List fixture identifiers and categories, then exit.")
    parser.add_argument(
        "--allow-partial", action="store_true", help="Explicitly permit incomplete requested techno collections."
    )
    arguments = parser.parse_args()
    try:
        datasets = arguments.dataset or ["synthetic"]
        collection: CollectionStatus | None = None
        fixtures = synthetic_fixtures() if "synthetic" in datasets else []
        for manifest in arguments.manifest:
            fixtures.extend(load_manifest(manifest))
        available_identifiers = {fixture.identifier for fixture in fixtures}
        available_categories = {fixture.category for fixture in fixtures}
        if "techno" in datasets:
            tracks = read_catalog()
            available_identifiers.update(track.identifier for track in tracks)
            available_categories.update(track.category for track in tracks)
            if arguments.fixture:
                tracks = [track for track in tracks if track.identifier in arguments.fixture]
            if any(track.partition == "evaluation" for track in tracks) and "evaluation" not in arguments.partition:
                if arguments.fixture and any(track.partition == "evaluation" for track in tracks):
                    raise ValueError("Held-out tracks require explicit --partition evaluation.")
                tracks = [track for track in tracks if track.partition != "evaluation"]
            if arguments.partition:
                tracks = [track for track in tracks if track.partition in arguments.partition]
            if arguments.category:
                tracks = [track for track in tracks if track.category in arguments.category]
            real_fixtures, missing = reviewed_fixtures(tracks)
            collection = collection_status(tracks, real_fixtures)
            for message in missing:
                print(message)
            fixtures.extend(real_fixtures)
        identifiers = [fixture.identifier for fixture in fixtures]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Fixture identifiers must be unique across synthetic fixtures and manifests.")
        if arguments.list:
            for fixture in fixtures:
                print(f"{fixture.identifier:24} {fixture.dataset:10} {fixture.partition:12} {fixture.category}")
            return
        if set(arguments.fixture) - available_identifiers:
            raise ValueError("Unknown fixture identifier; use --list.")
        if set(arguments.category) - available_categories:
            raise ValueError("Unknown fixture category; use --list.")
        if collection is not None and not collection["complete"] and not arguments.allow_partial:
            raise ValueError(
                f"Incomplete requested collection: evaluated {len(collection['evaluated_track_identifiers'])}/"
                f"{len(collection['requested_track_identifiers'])} tracks; missing audio: {collection['missing_audio']}; "
                f"missing reviewed annotations: {collection['missing_reviewed_annotations']}. "
                "Use --allow-partial to report an explicitly partial result."
            )
        fixtures = [
            fixture
            for fixture in fixtures
            if (not arguments.fixture or fixture.identifier in arguments.fixture)
            and (not arguments.category or fixture.category in arguments.category)
            and fixture.dataset in datasets
            and (not arguments.partition or fixture.partition in arguments.partition)
        ]
        if not fixtures and not (arguments.allow_partial and collection is not None and not collection["complete"]):
            raise ValueError(
                "No benchmark fixtures satisfy this selection; techno requires audio and fully reviewed annotations."
            )
        overrides = {}
        for override in arguments.overrides:
            name, separator, value = override.partition("=")
            if not separator or name not in {field.name for field in fields(KickDetectionConfig)}:
                raise ValueError("Use --set NAME=VALUE with a KickDetectionConfig field name.")
            overrides[name] = float(value)
        report = benchmark(fixtures, KickDetectionConfig(**overrides), collection)
        print_summary(report)
        if arguments.json is not None:
            arguments.json.parent.mkdir(parents=True, exist_ok=True)
            arguments.json.write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(f"JSON: {arguments.json}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Benchmark error: {error}\n")


if __name__ == "__main__":
    main()
