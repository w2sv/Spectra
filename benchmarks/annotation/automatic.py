"""Automatic references and onset localization after the unchanged acoustic stage.

All processing here is offline and independent of production analysis. Energy
timestamps denote the end of trailing windows; zero-phase filtering has no
group delay. The raw-audio energy gate prevents filter pre-ringing alone from
establishing an onset.
"""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.spatial.distance import pdist

from benchmarks.annotation.analysis import (
    PARAMETERS,
    AcousticAnalysis,
    AutomaticCandidate,
    Candidate,
    Proposal,
    beat_positions,
)
from benchmarks.annotation.templates import (
    TEMPLATE_PARAMETERS,
    alternative_phase,
    classification,
    event_feature,
    grid_comparisons,
)

AUTOMATIC_ALGORITHM_VERSION = 3


@dataclass(frozen=True)
class AutomaticParameters:
    cluster_cosine_distance: float = 0.15
    minimum_cluster_members: int = 8
    minimum_grid_alignment: float = 0.70
    additional_reference_similarity: float = 0.70
    maximum_references: int = 9
    energy_window_seconds: float = 0.005
    energy_hop_seconds: float = 0.001
    energy_band_hz: tuple[float, float] = (20, 300)
    filter_order: int = 2
    onset_search_before_seconds: float = 0.030
    onset_search_after_seconds: float = 0.050
    background_seconds: float = 0.075
    background_mad_multiplier: float = 3
    rise_fraction: float = 0.10
    sustained_seconds: float = 0.003
    energy_floor: float = 1e-12


AUTOMATIC_PARAMETERS = AutomaticParameters()


class OnsetLocalizer:
    """Compute 1ms energy observations once, then inspect short event neighborhoods."""

    def __init__(self, analysis: AcousticAnalysis, parameters: AutomaticParameters = AUTOMATIC_PARAMETERS) -> None:
        self.parameters = parameters
        self.duration = analysis.duration_seconds
        rate = PARAMETERS.sample_rate
        self.step = round(parameters.energy_hop_seconds * rate)
        self.times = np.arange(0, len(analysis.mono), self.step, dtype=np.float64) / rate
        signal = analysis.mono.astype(np.float64)
        sos = butter(parameters.filter_order, parameters.energy_band_hz, btype="bandpass", fs=rate, output="sos")
        filtered = sosfiltfilt(sos, signal) if len(signal) > 30 else np.zeros_like(signal)
        window = round(parameters.energy_window_seconds * rate)
        ends = np.arange(0, len(signal), self.step) + 1
        starts = np.maximum(ends - window, 0)
        envelopes = []
        for values in (filtered, signal):
            cumulative = np.concatenate(([0.0], np.cumsum(values * values)))
            envelopes.append((cumulative[ends] - cumulative[starts]) / window)
        self.energy = np.column_stack(envelopes)

    def localize(self, timestamp: float) -> tuple[float, bool, str, float]:
        p = self.parameters
        first = int(np.searchsorted(self.times, max(0, timestamp - p.onset_search_before_seconds)))
        last = int(np.searchsorted(self.times, min(self.duration, timestamp + p.onset_search_after_seconds)))
        bg_first = int(
            np.searchsorted(self.times, max(0, timestamp - p.onset_search_before_seconds - p.background_seconds))
        )
        background = self.energy[bg_first:first]
        median = np.median(background, axis=0) if len(background) else np.zeros(2)
        mad = np.median(np.abs(background - median), axis=0) if len(background) else np.zeros(2)
        local = self.energy[first:last]
        if not len(local):
            return timestamp, False, "no context", 0.0
        increase = np.maximum(local.max(axis=0) - median, 0)
        low_increase = float(increase[0])
        if np.any(increase <= np.maximum(p.background_mad_multiplier * mad, p.energy_floor)):
            return timestamp, False, "no reliable joint energy rise", low_increase
        normalized = np.maximum(local - median, 0) / np.maximum(increase, p.energy_floor)
        dominant = int(np.argmax(normalized.mean(axis=1)))
        threshold = np.maximum(median + p.background_mad_multiplier * mad, median + p.rise_fraction * increase)
        crossing = np.all(local > np.maximum(threshold, p.energy_floor), axis=1)
        count = max(1, round(p.sustained_seconds / p.energy_hop_seconds))
        sustained = np.convolve(crossing.astype(int), np.ones(count, dtype=int), mode="valid") == count
        starts = np.flatnonzero(sustained[: dominant + 1])
        if not len(starts):
            return timestamp, False, "no sustained threshold crossing", low_increase
        # Follow the dominant rise backward, stopping at its most recent gap.
        onset = int(starts[-1])
        while onset > 0 and crossing[onset - 1]:
            onset -= 1
        localized = float(self.times[first + onset])
        reliable = onset > 0 and first + onset + count < last
        return (
            localized if reliable else timestamp,
            reliable,
            "energy crossing" if reliable else "search boundary",
            low_increase,
        )


def candidate_identity(proposal: Proposal, candidate: Candidate) -> str:
    """Identity uses the source observation, never a refined onset or prediction."""
    anchor = round(candidate["timestamp"] * PARAMETERS.sample_rate)
    source = (
        f"{proposal['audio_sha256'] or proposal['track_identifier']}:{candidate.get('provenance', 'acoustic')}:{anchor}"
    )
    return hashlib.sha256(source.encode()).hexdigest()[:24]


def grid_distances(times: NDArray[np.float64], beats: list[float]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if not beats:
        return np.full(len(times), np.inf), np.full(len(times), -1, dtype=np.int64)
    grid = np.asarray(beats)
    right = np.clip(np.searchsorted(grid, times), 0, len(grid) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(np.abs(times - grid[left]) <= np.abs(times - grid[right]), left, right)
    return times - grid[indices], indices.astype(np.int64)


def comprehensive_candidates(analysis: AcousticAnalysis, proposal: Proposal) -> list[AutomaticCandidate]:
    """Supplement both grid hypotheses; preserve every original acoustic candidate."""
    candidates: list[AutomaticCandidate] = []
    for original in proposal["kick_candidates"]:
        candidates.append(
            {
                **original,
                "candidate_id": "",
                "provenance": "acoustic",
                "unsupported_grids": [],
                "localized_timestamp": original["timestamp"],
                "final_timestamp": original["timestamp"],
                "localization_reliable": False,
                "timing_reason": "pending",
                "low_frequency_increase": 0.0,
                "automatic_kick": False,
                "low_confidence": True,
                "cluster_id": None,
                "uncertainty_reasons": [],
            }
        )
    # A weak local maximum still needs nonzero evidence, rather than a beat prior.
    peaks, _ = find_peaks(analysis.bass)
    original_times = np.asarray([c["timestamp"] for c in candidates])
    tolerance = TEMPLATE_PARAMETERS.duplicate_tolerance_seconds
    for grid_name, beats in (
        ("primary", proposal["beat_positions"]),
        ("alternative", proposal.get("alternative_beat_positions", [])),
    ):
        distances, indices = grid_distances(original_times, beats)
        supported = set(indices[np.abs(distances) <= PARAMETERS.association_radius_seconds].tolist())
        for index, beat in enumerate(beats):
            if index in supported:
                continue
            nearby = peaks[np.abs(analysis.timestamps[peaks] - beat) <= PARAMETERS.association_radius_seconds]
            peak = int(nearby[np.argmax(analysis.bass[nearby])]) if len(nearby) else None
            meaningful = peak is not None and analysis.bass[peak] > 1e-12
            time = float(analysis.timestamps[peak]) if meaningful and peak is not None else beat
            existing = next((c for c in candidates if abs(c["timestamp"] - time) <= tolerance), None)
            if existing is not None:
                existing["unsupported_grids"].append(grid_name)
                continue
            frame = peak if meaningful and peak is not None else min(round(time / 0.005), len(analysis.bass) - 1)
            candidates.append(
                {
                    "timestamp": time,
                    "bass_evidence": float(analysis.bass[frame]),
                    "prominence": 0.0,
                    "broadband_evidence": float(analysis.broadband[frame]),
                    "nearest_beat_distance_seconds": time - beat,
                    "grid_aligned": grid_name == "primary",
                    "unsupported_grids": [grid_name],
                    "provenance": "supplementary_acoustic" if meaningful else "beat_grid_only",
                    "candidate_id": "",
                    "localized_timestamp": time,
                    "final_timestamp": time,
                    "localization_reliable": False,
                    "timing_reason": "pending",
                    "low_frequency_increase": 0.0,
                    "automatic_kick": False,
                    "low_confidence": True,
                    "cluster_id": None,
                    "uncertainty_reasons": [],
                }
            )
    result: list[AutomaticCandidate] = sorted(candidates, key=lambda c: c["timestamp"])
    for candidate in result:
        candidate["candidate_id"] = candidate_identity(proposal, candidate)
    return result


def feature_matrix(
    analysis: AcousticAnalysis, timestamps: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Batched version of event_feature, with identical padding and normalization."""
    p = TEMPLATE_PARAMETERS
    step = PARAMETERS.hop_size / PARAMETERS.sample_rate
    centers = np.rint(timestamps / step).astype(int)
    offsets = np.arange(round(p.context_start_seconds / step), round(p.context_end_seconds / step) + 1)
    bg_offsets = np.arange(round(p.background_start_seconds / step), round(p.background_end_seconds / step) + 1)
    output = np.zeros((len(centers), len(offsets) * analysis.template_band_power.shape[1]), dtype=np.float64)
    power = analysis.template_band_power
    for start in range(0, len(centers), 256):
        anchors = centers[start : start + 256]
        indices = anchors[:, None] + offsets
        context = power[np.clip(indices, 0, len(power) - 1)].copy()
        context[(indices < 0) | (indices >= len(power))] = 0
        bg_indices = anchors[:, None] + bg_offsets
        background = power[np.clip(bg_indices, 0, len(power) - 1)].copy()
        background[(bg_indices < 0) | (bg_indices >= len(power))] = 0
        features = np.maximum(np.sqrt(context) - np.sqrt(np.median(background, axis=1))[:, None, :], 0)
        output[start : start + len(anchors)] = features.reshape(len(anchors), -1)
    norms = np.linalg.norm(output, axis=1)
    valid = (norms > 0) & (timestamps >= 0) & (timestamps < analysis.duration_seconds)
    output[valid] /= norms[valid, None]
    output[~valid] = 0
    return output, valid


def discover_references(
    analysis: AcousticAnalysis,
    proposal: Proposal[AutomaticCandidate],
    candidates: list[AutomaticCandidate],
    parameters: AutomaticParameters = AUTOMATIC_PARAMETERS,
) -> tuple[list[str], list[dict[str, object]], str | None]:
    times = np.asarray([c["localized_timestamp"] for c in candidates], dtype=np.float64)
    features, valid = feature_matrix(analysis, times)
    valid &= np.asarray([c["provenance"] != "beat_grid_only" for c in candidates], dtype=bool)
    available = np.flatnonzero(valid)
    if len(available) < 2:
        return [], [], None
    distances = np.maximum(pdist(features[available], metric="cosine"), 0)
    labels = fcluster(linkage(distances, method="average"), parameters.cluster_cosine_distance, criterion="distance")
    grid_info = {
        name: grid_distances(times, beats)
        for name, beats in (
            ("primary", proposal["beat_positions"]),
            ("alternative", proposal.get("alternative_beat_positions", [])),
        )
    }
    clusters: list[dict[str, object]] = []
    representatives: dict[int, list[int]] = {}
    medoids: dict[int, NDArray[np.float64]] = {}
    eligible: list[tuple[float, int, str]] = []
    for label in sorted(set(labels.tolist())):
        members = available[labels == label]
        matrix = features[members]
        # The member most similar to the cluster mean is its cosine medoid.
        similarities = matrix @ matrix.mean(axis=0)
        medoid = int(members[int(np.argmax(similarities))])
        chosen = [medoid]
        for endpoint in (int(members[0]), int(members[-1])):
            if endpoint not in chosen and all(
                abs(times[endpoint] - times[i]) >= max(0.180, analysis.duration_seconds / 6) for i in chosen
            ):
                chosen.append(endpoint)
        representatives[label] = chosen
        medoids[label] = features[medoid]
        increase = float(np.median([candidates[i]["low_frequency_increase"] for i in members]))
        grids: dict[str, object] = {}
        for name, (delta, beat_indices) in grid_info.items():
            aligned = members[np.abs(delta[members]) <= PARAMETERS.association_radius_seconds]
            fraction = len(aligned) / len(members)
            supported = len(set(beat_indices[aligned].tolist()))
            score = float(fraction * np.log1p(supported) * increase)
            grids[name] = {"alignment_fraction": fraction, "supported_beats": supported, "score": score}
            if (
                len(members) >= parameters.minimum_cluster_members
                and fraction >= parameters.minimum_grid_alignment
                and score > 0
            ):
                eligible.append((score, label, name))
        for i in members:
            candidates[i]["cluster_id"] = label
        distribution = np.histogram(times[members], bins=np.linspace(0, analysis.duration_seconds, 5))[0].tolist()
        within = (float(similarities.sum()) - 1) / (len(members) - 1) if len(members) > 1 else 1.0
        clusters.append(
            {
                "cluster_id": label,
                "member_count": len(members),
                "medoid_candidate_id": candidates[medoid]["candidate_id"],
                "representative_candidate_ids": [candidates[i]["candidate_id"] for i in chosen],
                "median_acoustic_evidence": float(np.median([candidates[i]["bass_evidence"] for i in members])),
                "median_low_frequency_power_increase": increase,
                "within_cluster_similarity": within,
                "recording_quarter_counts": distribution,
                "grids": grids,
            }
        )
    if not eligible:
        return [], clusters, None
    eligible.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, first_cluster, selected_grid = eligible[0]
    selected = [first_cluster]
    for _, label, grid in eligible[1:]:
        if (
            label not in selected
            and grid == selected_grid
            and any(
                float(medoids[label] @ medoids[prior]) >= parameters.additional_reference_similarity
                for prior in selected
            )
        ):
            selected.append(label)
    references = [candidates[i]["candidate_id"] for label in selected for i in representatives[label]][
        : parameters.maximum_references
    ]
    return references, clusters, selected_grid


def classify_candidates(
    analysis: AcousticAnalysis,
    candidates: list[AutomaticCandidate],
    positives: list[float],
    negatives: list[float] | None = None,
) -> list[AutomaticCandidate]:
    """Reuse the existing cosine/offset search, batched to avoid repeated features."""
    valid_positives = [(t, feature) for t in positives if (feature := event_feature(analysis, t)) is not None]
    positive_features = [feature for _, feature in valid_positives]
    negative_features = [feature for t in (negatives or []) if (feature := event_feature(analysis, t)) is not None]
    p = TEMPLATE_PARAMETERS
    step = PARAMETERS.hop_size / PARAMETERS.sample_rate
    radius = round(p.search_radius_seconds / step)
    offsets = np.asarray(sorted((i * step for i in range(-radius, radius + 1)), key=lambda x: (abs(x), x)))
    result = []
    for start in range(0, len(candidates), 128):
        batch = candidates[start : start + 128]
        anchors = np.asarray([c["localized_timestamp"] for c in batch])[:, None] + offsets
        features, valid = feature_matrix(analysis, anchors.ravel())
        pos = features @ np.asarray(positive_features).T if positive_features else np.zeros((len(features), 1))
        pos[~valid] = -1
        scores = pos.max(axis=1).reshape(len(batch), -1)
        shifts = scores.argmax(axis=1)
        neg = (
            (features @ np.asarray(negative_features).T).max(axis=1).reshape(len(batch), -1).max(axis=1)
            if negative_features
            else None
        )
        best_refs = pos.argmax(axis=1).reshape(len(batch), -1)
        for i, original in enumerate(batch):
            candidate = original.copy()
            positive = float(scores[i, shifts[i]]) if positive_features and scores[i, shifts[i]] >= 0 else None
            negative = float(neg[i]) if neg is not None else None
            state = classification(positive, negative, bool(negative_features)) if positive_features else "unclassified"
            offset = float(offsets[shifts[i]]) if positive is not None else None
            refined = (
                float(np.clip(candidate["localized_timestamp"] + offset, 0, np.nextafter(analysis.duration_seconds, 0)))
                if offset is not None
                else None
            )
            grid_only = candidate["provenance"] == "beat_grid_only"
            predicted = state == "likely" if positive_features else candidate["grid_aligned"] and not grid_only
            # A silent grid position cannot acquire a kick label just from expectation.
            if grid_only:
                predicted, refined = False, None
            candidate.update(
                positive_similarity=positive,
                negative_similarity=negative,
                classification=state,
                refined_timestamp=refined,
                final_timestamp=refined if refined is not None else candidate["localized_timestamp"],
                alignment_offset_seconds=offset,
                alignment_at_search_boundary=offset is not None and abs(offset) >= p.search_radius_seconds - 1e-9,
                automatic_kick=predicted,
                low_confidence=not bool(positive_features),
                best_positive_template_timestamp=valid_positives[int(best_refs[i, shifts[i]])][0]
                if positive_features
                else None,
            )
            reasons = []
            if (
                state in ("ambiguous", "unclassified")
                or positive is not None
                and min(abs(positive - p.weak_similarity), abs(positive - p.strong_similarity)) <= 0.03
            ):
                reasons.append("classification uncertainty")
            if negative is not None and positive is not None and positive - negative < p.negative_margin:
                reasons.append("conflicting references")
            if candidate.get("unsupported_grids"):
                reasons.append("unsupported beat")
            if predicted and not candidate["grid_aligned"]:
                reasons.append("off-grid prediction")
            if not candidate["localization_reliable"] or candidate.get("alignment_at_search_boundary", False):
                reasons.append("uncertain onset")
            candidate["uncertainty_reasons"] = reasons
            result.append(candidate)
    return result


def automatic_proposal(analysis: AcousticAnalysis, original: Proposal) -> Proposal[AutomaticCandidate]:
    proposal: Proposal[AutomaticCandidate] = {**original, "kick_candidates": []}
    proposal["annotation_algorithm_version"] = AUTOMATIC_ALGORITHM_VERSION
    proposal["acoustic_algorithm_version"] = 1
    proposal["automatic_parameters"] = asdict(AUTOMATIC_PARAMETERS)
    proposal["template_parameters"] = asdict(TEMPLATE_PARAMETERS)
    phase, bpm = proposal["beat_grid_phase_seconds"], proposal["bpm"]
    alt = alternative_phase(phase, bpm) if phase is not None and bpm is not None else None
    proposal["alternative_phase_seconds"] = alt
    proposal["alternative_beat_positions"] = (
        beat_positions(bpm, alt, 0, analysis.duration_seconds).tolist() if alt is not None and bpm is not None else []
    )
    candidates = comprehensive_candidates(analysis, {**proposal, "kick_candidates": original["kick_candidates"]})
    localizer = OnsetLocalizer(analysis)
    for candidate in candidates:
        time = candidate["timestamp"]
        localized, reliable, reason, increase = (
            localizer.localize(time)
            if candidate["provenance"] != "beat_grid_only"
            else (time, False, "estimated beat position", 0.0)
        )
        candidate.update(
            novelty_timestamp=time,
            localized_timestamp=localized,
            localization_reliable=reliable,
            timing_reason=reason,
            low_frequency_increase=increase,
            cluster_id=None,
        )
    refs, clusters, selected_grid = discover_references(analysis, proposal, candidates)
    by_id = {c["candidate_id"]: c for c in candidates}
    proposal["kick_candidates"] = classify_candidates(
        analysis, candidates, [by_id[key]["localized_timestamp"] for key in refs]
    )
    proposal["clusters"] = clusters
    proposal["automatic_reference_ids"] = refs
    proposal["positive_template_count"] = len(refs)
    proposal["selected_grid"] = selected_grid
    proposal["automatic_confidence"] = (
        "heuristic recurring signature"
        if refs
        else "low confidence rhythmic fallback"
        if bpm
        else "reliable automatic classification unavailable"
    )
    proposal["grid_comparisons"] = grid_comparisons({**proposal, "kick_candidates": list(proposal["kick_candidates"])})
    return proposal
