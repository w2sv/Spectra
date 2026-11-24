"""Independent, noncausal acoustic evidence and supporting beat grids.

This module does not import production spectral analysis or kick detection.
Array conventions: time along axis zero, frequency/bands along axis one.
"""

from dataclasses import asdict, dataclass
from math import ceil, gcd
from time import perf_counter
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample_poly
from scipy.signal.windows import hann

from spectra.audio import AudioSamples

ALGORITHM_VERSION = 1


@dataclass(frozen=True)
class AnnotationParameters:
    sample_rate: int = 48_000
    fft_size: int = 2048
    hop_size: int = 240
    bands: tuple[tuple[float, float], ...] = ((20, 60), (60, 150), (150, 300), (300, 3000))
    log_power_floor: float = 1e-12
    normalization_percentile: float = 95.0
    smoothing_frames: float = 1.0
    peak_separation_seconds: float = 0.080
    peak_prominence: float = 0.25
    grid_resolution_seconds: float = 0.005
    grid_evidence_radius_seconds: float = 0.020
    association_radius_seconds: float = 0.050
    section_seconds: float = 30.0
    drift_threshold_seconds: float = 0.030


PARAMETERS = AnnotationParameters()


@dataclass(frozen=True)
class AcousticAnalysis:
    mono: NDArray[np.float32]
    timestamps: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    low_spectral_power: NDArray[np.float32]
    band_power: NDArray[np.float64]
    band_novelty: NDArray[np.float64]
    unsmoothed_bass: NDArray[np.float64]
    unsmoothed_broadband: NDArray[np.float64]
    bass: NDArray[np.float64]
    broadband: NDArray[np.float64]
    duration_seconds: float


class Candidate(TypedDict):
    timestamp: float
    bass_evidence: float
    prominence: float
    broadband_evidence: float
    nearest_beat_distance_seconds: float | None
    grid_aligned: bool


class GridSection(TypedDict):
    start_seconds: float
    end_seconds: float
    local_phase_seconds: float
    phase_difference_seconds: float
    drift_flagged: bool


class Proposal(TypedDict):
    track_identifier: str
    audio_sha256: str | None
    annotation_algorithm_version: int
    analysis_parameters: dict[str, object]
    bpm: float | None
    beat_grid_phase_seconds: float | None
    beat_positions: list[float]
    unsupported_beat_positions: list[float]
    kick_candidates: list[Candidate]
    beat_grid_validation: list[GridSection]
    duration_seconds: float
    source_sample_rate: int
    informational: dict[str, float]


def analyze_audio(samples: AudioSamples, source_rate: int) -> AcousticAnalysis:
    p = PARAMETERS
    duration = len(samples) / source_rate
    mono = samples.mean(axis=1, dtype=np.float32)
    if source_rate != p.sample_rate:
        divisor = gcd(source_rate, p.sample_rate)
        mono = resample_poly(mono, p.sample_rate // divisor, source_rate // divisor).astype(np.float32)
    # Center k is exactly k*hop; boundary zeros are analysis padding only.
    count = ceil(len(mono) / p.hop_size)
    times = np.arange(count, dtype=np.float64) * p.hop_size / p.sample_rate
    frequencies = rfftfreq(p.fft_size, 1 / p.sample_rate)
    bin_edges = np.concatenate(([0.0], (frequencies[:-1] + frequencies[1:]) / 2, [p.sample_rate / 2]))
    limits = np.asarray(p.bands)
    overlap = np.minimum(limits[:, 1, None], bin_edges[None, 1:]) - np.maximum(limits[:, 0, None], bin_edges[None, :-1])
    weights = np.maximum(overlap, 0) / np.diff(bin_edges)
    window = hann(p.fft_size, sym=False).astype(np.float32)
    scale = np.full(len(frequencies), 2 / (p.fft_size * float(np.sum(window**2))))
    scale[[0, -1]] *= 0.5
    low = (frequencies >= 20) & (frequencies <= 500)
    band_power = np.empty((count, 4), dtype=np.float64)
    low_power = np.empty((count, int(low.sum())), dtype=np.float32)
    padded = np.pad(mono, (p.fft_size // 2, p.fft_size // 2))
    windows = np.lib.stride_tricks.sliding_window_view(padded, p.fft_size)[:: p.hop_size][:count]
    # Bound temporary FFT storage for complete multi-minute recordings.
    for start in range(0, count, 2048):
        spectrum = np.asarray(rfft(windows[start : start + 2048] * window, axis=1), dtype=np.complex64)
        power = (spectrum.real**2 + spectrum.imag**2) * scale
        band_power[start : start + len(power)] = power @ weights.T
        low_power[start : start + len(power)] = power[:, low]
    log_power = np.log(band_power + p.log_power_floor)
    differences = np.maximum(np.diff(log_power, axis=0, prepend=log_power[:1]), 0)
    novelty = np.zeros_like(differences)
    for band in range(4):
        positive = differences[:, band][differences[:, band] > 0]
        denominator = float(np.percentile(positive, p.normalization_percentile)) if len(positive) else 0.0
        if denominator > 0:
            novelty[:, band] = differences[:, band] / denominator
    bass = novelty[:, :3].mean(axis=1)
    broadband = novelty[:, 3].copy()
    return AcousticAnalysis(
        mono,
        times,
        frequencies[low],
        low_power,
        band_power,
        novelty,
        bass,
        broadband,
        gaussian_filter1d(bass, p.smoothing_frames),
        gaussian_filter1d(broadband, p.smoothing_frames),
        duration,
    )


def beat_positions(bpm: float, phase: float, start: float, end: float) -> NDArray[np.float64]:
    interval = 60 / bpm
    first = max(0, ceil((start - phase) / interval))
    indices = np.arange(first, max(first, ceil((end - phase) / interval)))
    return np.asarray(phase + indices * interval, dtype=np.float64)


def circular_phase_difference(local: float, global_phase: float, interval: float) -> float:
    return (local - global_phase + interval / 2) % interval - interval / 2


def estimate_phase(
    times: NDArray[np.float64], evidence: NDArray[np.float64], bpm: float, start: float, end: float
) -> float:
    """Mean local maxima at every beat, scoring only frames inside the section."""
    p = PARAMETERS
    inside = (times >= start) & (times < end)
    section_times, section = times[inside], evidence[inside]
    if not len(section):
        raise ValueError("No analysis frames in beat-grid section.")
    best_phase, best_score = 0.0, -1.0
    interval = 60 / bpm
    for phase in np.arange(0, interval, p.grid_resolution_seconds):
        beats = beat_positions(bpm, float(phase), start, end)
        if not len(beats):
            continue
        left = np.searchsorted(section_times, beats - p.grid_evidence_radius_seconds - 1e-10)
        right = np.searchsorted(section_times, beats + p.grid_evidence_radius_seconds + 1e-10, side="right")
        # Exact time-radius windows, including non-frame-aligned beat positions.
        maxima = [float(section[a:b].max()) if a < b else 0.0 for a, b in zip(left, right, strict=True)]
        score = float(np.mean(maxima))
        if score > best_score:
            best_phase, best_score = float(phase), score
    return best_phase


def validate_grid(analysis: AcousticAnalysis, bpm: float, phase: float) -> list[GridSection]:
    sections: list[GridSection] = []
    for start in np.arange(0, analysis.duration_seconds, PARAMETERS.section_seconds):
        end = min(float(start) + PARAMETERS.section_seconds, analysis.duration_seconds)
        local = estimate_phase(analysis.timestamps, analysis.bass, bpm, float(start), end)
        difference = circular_phase_difference(local, phase, 60 / bpm)
        sections.append(
            {
                "start_seconds": float(start),
                "end_seconds": end,
                "local_phase_seconds": local,
                "phase_difference_seconds": difference,
                "drift_flagged": abs(difference) > PARAMETERS.drift_threshold_seconds + 1e-10,
            }
        )
    return sections


def acoustic_candidates(analysis: AcousticAnalysis) -> list[Candidate]:
    p = PARAMETERS
    indices, properties = find_peaks(
        analysis.bass,
        distance=ceil(p.peak_separation_seconds * p.sample_rate / p.hop_size),
        prominence=p.peak_prominence,
    )
    return [
        {
            "timestamp": float(analysis.timestamps[index]),
            "bass_evidence": float(analysis.bass[index]),
            "prominence": float(prominence),
            "broadband_evidence": float(analysis.broadband[index]),
            "nearest_beat_distance_seconds": None,
            "grid_aligned": False,
        }
        for index, prominence in zip(indices, properties["prominences"], strict=True)
        if analysis.timestamps[index] < analysis.duration_seconds
    ]


def associate_candidates(
    candidates: list[Candidate], beats: NDArray[np.float64]
) -> tuple[list[Candidate], list[float]]:
    """Assign strongest evidence per chronological beat, retaining every off-grid peak."""
    assigned: set[int] = set()
    unsupported = []
    for beat in beats:
        available = [
            i
            for i, candidate in enumerate(candidates)
            if i not in assigned and abs(candidate["timestamp"] - beat) <= PARAMETERS.association_radius_seconds + 1e-10
        ]
        if available:
            assigned.add(max(available, key=lambda i: candidates[i]["bass_evidence"]))
        else:
            unsupported.append(float(beat))
    result: list[Candidate] = []
    for i, candidate in enumerate(candidates):
        nearest = int(np.argmin(np.abs(beats - candidate["timestamp"]))) if len(beats) else None
        associated: Candidate = candidate.copy()
        associated["grid_aligned"] = i in assigned
        associated["nearest_beat_distance_seconds"] = (
            float(candidate["timestamp"] - beats[nearest]) if nearest is not None else None
        )
        result.append(associated)
    return result, unsupported


def generate_proposal(
    analysis: AcousticAnalysis,
    identifier: str,
    checksum: str | None,
    source_rate: int,
    bpm: float | None,
    phase: float | None = None,
) -> Proposal:
    started = perf_counter()
    if bpm is not None and (not np.isfinite(bpm) or bpm <= 0):
        raise ValueError("BPM must be finite and positive.")
    if phase is not None and (bpm is None or not np.isfinite(phase) or not 0 <= phase < 60 / bpm):
        raise ValueError("Phase must be finite and within one beat interval.")
    candidates = acoustic_candidates(analysis)
    beats = np.empty(0, dtype=np.float64)
    unsupported: list[float] = []
    sections: list[GridSection] = []
    if bpm is not None:
        phase = (
            estimate_phase(analysis.timestamps, analysis.bass, bpm, 0, analysis.duration_seconds)
            if phase is None
            else phase
        )
        beats = beat_positions(bpm, phase, 0, analysis.duration_seconds)
        candidates, unsupported = associate_candidates(candidates, beats)
        sections = validate_grid(analysis, bpm, phase)
    return {
        "track_identifier": identifier,
        "audio_sha256": checksum,
        "annotation_algorithm_version": ALGORITHM_VERSION,
        "analysis_parameters": asdict(PARAMETERS),
        "bpm": bpm,
        "beat_grid_phase_seconds": phase,
        "beat_positions": beats.tolist(),
        "unsupported_beat_positions": unsupported,
        "kick_candidates": candidates,
        "beat_grid_validation": sections,
        "duration_seconds": analysis.duration_seconds,
        "source_sample_rate": source_rate,
        "informational": {"proposal_processing_seconds": perf_counter() - started},
    }
