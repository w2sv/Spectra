"""Summarize retained runs and plot empirical software frame-time distributions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks.annotation.cli import write_json


def aggregate(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        p = run["parameters"]
        m = run["measurements"]
        key = (
            m["recording"]["track"],
            p["scenario"],
            p["target_hz"],
            p["backend"],
            p["mode"],
            p["waveform"],
            p["axes"],
            p["profiled"],
            p.get("gpu_timing", False),
            p["stream_full_upload"],
        )
        groups[key].append(run)
    summaries = []
    for key, group in sorted(groups.items()):
        measurements = [run["measurements"] for run in group]
        values = {
            "track": key[0],
            "scenario": key[1],
            "target_hz": key[2],
            "backend": key[3],
            "mode": key[4],
            "waveform": key[5],
            "axes": key[6],
            "profiled": key[7],
            "gpu_timing": key[8],
            "stream_full_upload": key[9],
            "repetitions": len(group),
            "duration_seconds": sorted({run["parameters"]["duration_seconds"] for run in group}),
            "quality_flags": sorted({flag for run in group for flag in run.get("quality_flags", [])}),
        }
        for metric in (
            "logical_draw_hz",
            "update_hz",
            "cpu_percent_one_core",
            "initialization_seconds",
            "waveform_preprocessing_seconds",
            "waveform_cpu_geometry_bytes",
            "buffer_upload_bytes_per_second",
            "initial_and_warmup_upload_bytes",
            "waveform_rebuilds",
            "waveform_prepared_bytes",
            "deadline_misses",
            "deadline_slots_missed",
        ):
            observed = [m[metric] for m in measurements if m.get(metric) is not None]
            values[metric] = median(observed) if observed else None
        for metric in (
            "frame_interval_ms",
            "paint_submission_ms",
            "update_cpu_ms",
            "snapshot_to_paint_ms",
            "cursor_gap_recording_ms",
            "callback_ms",
            "rss_bytes",
        ):
            values[metric] = {
                stat: median(m[metric][stat] for m in measurements if m[metric][stat] is not None)
                if any(m[metric][stat] is not None for m in measurements)
                else None
                for stat in ("median", "p95", "p99", "max")
            }
        values["frame_p95_run_range_ms"] = [
            min(
                (m["frame_interval_ms"]["p95"] for m in measurements if m["frame_interval_ms"]["p95"] is not None),
                default=None,
            ),
            max(
                (m["frame_interval_ms"]["p95"] for m in measurements if m["frame_interval_ms"]["p95"] is not None),
                default=None,
            ),
        ]
        values["measured_underruns_total"] = sum(m["audio"]["underruns"] or 0 for m in measurements)
        values["warmup_underruns_total"] = sum(m["audio"].get("warmup_underruns") or 0 for m in measurements)
        values["protected_files_unchanged"] = all(m["recording"]["protected_files_unchanged"] for m in measurements)
        summaries.append(values)
    return summaries


def figures(runs: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [
        run
        for run in runs
        if run["parameters"]["mode"] == "rendering"
        and not run["parameters"]["profiled"]
        and run["parameters"]["scenario"].startswith("follow_")
    ]
    target = max(run["parameters"]["target_hz"] for run in primary)
    from collections import Counter

    counts = Counter(
        (run["parameters"]["scenario"], run["parameters"]["backend"])
        for run in primary
        if run["parameters"]["target_hz"] == target
    )
    complete = [
        scenario
        for scenario in ("follow_2.5", "follow_10", "follow_40", "follow_120", "follow_full")
        if all(counts[scenario, backend] >= 3 for backend in ("pyqtgraph", "vispy"))
    ]
    preferred = ["follow_10", "follow_120", "follow_full"]
    scenarios = preferred if all(scenario in complete for scenario in preferred) else complete[:3]
    if not scenarios:
        return
    figure, axes = plt.subplots(1, len(scenarios), figsize=(12, 3.4), layout="constrained", squeeze=False)
    for axis, scenario in zip(axes[0], scenarios, strict=True):
        for track, color in (("universal", "#2465aa"), ("no_trust", "#b84936")):
            for backend, style in (("pyqtgraph", "--"), ("vispy", "-")):
                values = [
                    value
                    for run in runs
                    if run["parameters"]["scenario"] == scenario
                    and run["parameters"]["target_hz"] == target
                    and run["parameters"]["backend"] == backend
                    and run["parameters"]["mode"] == "rendering"
                    and not run["parameters"]["profiled"]
                    and run["parameters"]["waveform"] == "lod"
                    and run["measurements"]["recording"]["track"] == track
                    for value in run["measurements"]["raw"]["frame_intervals_ms"]
                ]
                if values:
                    ordered = np.sort(values)
                    axis.plot(
                        ordered,
                        np.arange(1, len(ordered) + 1) / len(ordered),
                        color=color,
                        linestyle=style,
                        label=f"{track} {backend}",
                    )
        axis.axvline(1000 / 60, color="gray", linewidth=0.7)
        axis.axvline(1000 / 120, color="gray", linewidth=0.7, linestyle=":")
        axis.set(
            title=scenario.removeprefix("follow_") + " s view",
            xlabel="Complete software frame interval (ms)",
            xlim=(0, 150),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Empirical cumulative probability")
    axes[0, -1].legend(fontsize=7, loc="lower right")
    figure.suptitle(f"Software draw completion intervals · {target} Hz requested")
    figure.savefig(output.with_suffix(".svg"))
    figure.savefig(output.with_suffix(".png"), dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/review_performance/vispy_summary.json"))
    args = parser.parse_args()
    runs = [run for path in args.inputs for run in json.loads(path.read_text())["runs"]]
    summary = aggregate(runs)
    write_json(
        args.output,
        {
            "aggregation": "Median of per-run statistics; repetitions are independent fresh processes. Raw intervals remain in input artifacts.",
            "sources": list(map(str, args.inputs)),
            "coverage_notes": {str(path): json.loads(path.read_text()).get("coverage_note") for path in args.inputs},
            "groups": summary,
        },
    )
    figures(runs, args.output)
    lines = [
        "# Comparative measurements",
        "",
        "P95 values are medians of per-run P95s. CPU is percent of one core. Draw Hz is software draw completion, not presentation FPS.",
        "",
        "| Track | Target | Scenario | Mode | Qt P95 ms | VisPy P95 ms | Reduction | Qt / VisPy CPU % | Evidence |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for baseline in summary:
        if baseline["backend"] != "pyqtgraph" or baseline["profiled"]:
            continue
        other = next(
            (
                row
                for row in summary
                if row["backend"] == "vispy"
                and all(
                    row[k] == baseline[k]
                    for k in ("track", "scenario", "target_hz", "mode", "waveform", "axes", "profiled", "gpu_timing")
                )
            ),
            None,
        )
        if other is None:
            continue
        a, b = baseline["frame_interval_ms"]["p95"], other["frame_interval_ms"]["p95"]
        if a is None or b is None:
            continue
        evidence = f"{baseline['repetitions']} x {baseline['duration_seconds'][0]:g} s"
        if baseline["quality_flags"] or other["quality_flags"]:
            evidence += "; draw gaps"
        elif baseline["repetitions"] < 3:
            evidence += "; exploratory"
        lines.append(
            f"| {baseline['track']} | {baseline['target_hz']} | {baseline['scenario']} | {baseline['mode']} | {a:.2f} | {b:.2f} | {100 * (1 - b / a):.1f}% | {baseline['cpu_percent_one_core']:.1f} / {other['cpu_percent_one_core']:.1f} | {evidence} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
