"""Deterministic section ranking, independent of interactive review state."""

from pathlib import Path
from typing import TypedDict

import numpy as np

from benchmarks.annotation.analysis import AcousticAnalysis, AutomaticCandidate, Proposal, associate_candidates
from benchmarks.annotation.templates import candidate_onset

STYLES = {label: label for label in ("likely", "ambiguous", "unlikely", "unclassified")}


class DiagnosticInterval(TypedDict):
    start: float
    end: float
    beats: int
    support: float
    challenge: int
    classification_counts: dict[str, int]
    off_grid_likely: int
    unsupported_beats: int


class DiagnosticSelection(TypedDict):
    regular: DiagnosticInterval | None
    challenging: DiagnosticInterval | None
    phase_discrepancy: DiagnosticInterval | None
    criteria: dict[str, str | bool]
    ranked_intervals: list[DiagnosticInterval]


def diagnostic_sections(proposal: Proposal) -> DiagnosticSelection:
    classified = proposal.get("positive_template_count", 0) > 0
    intervals: list[DiagnosticInterval] = []
    for raw_start in np.arange(0, proposal["duration_seconds"], 10.0):
        start = float(raw_start)
        end = min(start + 10, proposal["duration_seconds"])
        beats = [beat for beat in proposal["beat_positions"] if start <= beat < end]
        candidates = [c for c in proposal["kick_candidates"] if start <= candidate_onset(c) < end]
        unsupported = sum(start <= beat < end for beat in proposal["unsupported_beat_positions"])
        likely = [c for c in candidates if c.get("classification") == "likely"]
        support_candidates = []
        for candidate in proposal["kick_candidates"]:
            if classified and candidate.get("classification") != "likely":
                continue
            copied = candidate.copy()
            copied["timestamp"] = candidate_onset(candidate) if classified else candidate["timestamp"]
            support_candidates.append(copied)
        associated, _ = associate_candidates(support_candidates, np.asarray(beats, dtype=np.float64))
        aligned = sum(c["grid_aligned"] for c in associated)
        ambiguous = sum(c.get("classification") == "ambiguous" for c in candidates)
        offgrid = (
            sum(not c["grid_aligned"] for c in likely) if classified else sum(not c["grid_aligned"] for c in candidates)
        )
        intervals.append(
            {
                "start": float(start),
                "end": end,
                "beats": len(beats),
                "support": aligned / len(beats) if beats else 0,
                "challenge": ambiguous + offgrid + unsupported,
                "classification_counts": {
                    label: sum(c.get("classification", "unclassified") == label for c in candidates) for label in STYLES
                },
                "off_grid_likely": sum(not c["grid_aligned"] for c in likely),
                "unsupported_beats": unsupported,
            }
        )
    eligible = [interval for interval in intervals if interval["beats"] >= 8]
    regular = max(eligible, key=lambda item: (item["support"], -item["start"])) if eligible else None
    remaining = [
        item
        for item in intervals
        if regular is None or item["start"] >= regular["end"] or item["end"] <= regular["start"]
    ]
    challenging = max(remaining, key=lambda item: (item["challenge"], -item["start"])) if remaining else None
    phase_ranked: list[tuple[float, DiagnosticInterval]] = []
    for interval in intervals:
        section = next(
            (
                section
                for section in proposal["beat_grid_validation"]
                if section["start_seconds"] <= interval["start"] < section["end_seconds"]
            ),
            None,
        )
        if section is not None:
            phase_ranked.append((abs(section["phase_difference_seconds"]), interval))
    largest_phase = max(phase_ranked, key=lambda item: (item[0], -item[1]["start"]))[1] if phase_ranked else None
    phase = (
        largest_phase
        if largest_phase is not None
        and all(other is None or largest_phase["start"] != other["start"] for other in (regular, challenging))
        else None
    )
    return {
        "regular": regular,
        "challenging": challenging,
        "phase_discrepancy": phase,
        "criteria": {
            "regular": "highest likely-kick beat support; >=8 beats; earliest tie",
            "challenging": "ambiguous + off-grid likely + unsupported; nonoverlapping regular; earliest tie",
            "phase_discrepancy": "largest absolute 30s local discrepancy, earliest distinct 10s interval",
            "fallback_without_templates": not classified,
        },
        "ranked_intervals": intervals,
    }


def save_diagnostics(
    analysis: AcousticAnalysis, proposal: Proposal[AutomaticCandidate], output: Path
) -> dict[str, object]:
    """Static three-panel diagnostics retain spectral analysis without a desktop."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks.annotation.analysis import PARAMETERS
    from benchmarks.annotation.templates import event_feature

    selected = diagnostic_sections({**proposal, "kick_candidates": list(proposal["kick_candidates"])})
    output.mkdir(parents=True, exist_ok=True)
    for label in ("regular", "challenging", "phase_discrepancy"):
        interval = selected[label]
        if interval is None:
            continue
        start, end = interval["start"], interval["end"]
        figure, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        first, last = int(start * PARAMETERS.sample_rate), int(end * PARAMETERS.sample_rate)
        axes[0].plot(
            np.arange(first, last) / PARAMETERS.sample_rate, analysis.mono[first:last], linewidth=0.5, color=".3"
        )
        frames = (analysis.timestamps >= start) & (analysis.timestamps < end)
        times = analysis.timestamps[frames]
        if len(times):
            axes[1].imshow(
                10 * np.log10(analysis.low_spectral_power[frames].T + 1e-12),
                origin="lower",
                aspect="auto",
                extent=(times[0] - 0.0025, times[-1] + 0.0025, 20, 500),
                cmap="magma",
                vmin=-90,
                vmax=-15,
            )
        axes[2].plot(times, analysis.bass[frames], label="Bass-onset evidence")
        axes[2].plot(times, analysis.broadband[frames], label="Broadband evidence")
        candidates = [c for c in proposal["kick_candidates"] if start <= c["final_timestamp"] < end]
        for candidate in candidates:
            if candidate["automatic_kick"]:
                axes[0].axvline(candidate["final_timestamp"], color="#16703c", linewidth=0.6)
        axes[0].set_ylabel("PCM amplitude")
        axes[1].set_ylabel("Frequency (Hz)")
        axes[2].set_ylabel("Evidence")
        axes[2].set_xlabel("Recording time (seconds)")
        axes[2].legend()
        axes[0].set_xlim(start, end)
        figure.suptitle(f"{proposal['track_identifier']} · {label} · {start:g}–{end:g} s")
        figure.tight_layout()
        figure.savefig(output / f"{proposal['track_identifier']}-{label}.png", dpi=110)
        plt.close(figure)
    clusters = sorted(proposal.get("clusters", []), key=lambda c: -int(str(c["member_count"])))[:6]
    if clusters:
        by_id = {c["candidate_id"]: c for c in proposal["kick_candidates"]}
        figure, axes = plt.subplots(len(clusters), 1, figsize=(10, 2 * len(clusters)), squeeze=False)
        for axis, cluster in zip(axes[:, 0], clusters, strict=True):
            candidate = by_id[str(cluster["medoid_candidate_id"])]
            feature = event_feature(analysis, candidate["localized_timestamp"])
            if feature is not None:
                axis.imshow(feature.reshape(-1, 12).T, aspect="auto", origin="lower", extent=(-20, 160, 0, 12))
            axis.set_title(
                f"Cluster {cluster['cluster_id']} · {cluster['member_count']} events · reference {candidate['localized_timestamp']:.3f}s"
            )
        figure.tight_layout()
        figure.savefig(output / f"{proposal['track_identifier']}-competing-signatures.png")
        plt.close(figure)
    return dict(selected)
