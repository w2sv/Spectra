"""Locations of immutable, curated benchmark observations."""

from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
ACOUSTIC_BASELINE = RESULTS_ROOT / "annotation/acoustic_onset/v1_synthetic_baseline.json"
GUIDED_BASELINE = RESULTS_ROOT / "annotation/guided_template_matching/v2_synthetic_baseline.json"
