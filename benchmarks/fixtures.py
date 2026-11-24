"""Annotated in-memory or WAV fixtures and deterministic synthetic scenarios."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

from spectra.audio import DEFAULT_SAMPLE_RATE, AudioSamples

type Dataset = Literal["synthetic", "techno"]
type Partition = Literal["regression", "development", "evaluation"]


def file_checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_checksum(path: Path, expected: str | None) -> str:
    if expected is None:
        raise ValueError(f"{path.name}: no acquired checksum in the catalog; run prepare_tracks first.")
    actual = file_checksum(path)
    if actual != expected:
        raise ValueError(f"{path.name}: SHA-256 mismatch (expected {expected}, got {actual}).")
    return actual


def stereo_samples(samples: AudioSamples) -> AudioSamples:
    """Validate the audio boundary and duplicate mono into left/right channels."""
    if samples.ndim == 1:
        samples = np.column_stack((samples, samples))
    if samples.ndim != 2 or samples.shape[-1] != 2 or samples.dtype != np.float32:
        raise ValueError("Expected float32 mono or stereo audio.")
    if not len(samples) or not np.all(np.isfinite(samples)):
        raise ValueError("Expected nonempty, finite audio.")
    return samples


def load_wav(path: Path) -> tuple[int, AudioSamples]:
    """Load complete WAV audio without gain changes, trimming, or resampling."""
    rate, data = wavfile.read(path)
    if data.dtype == np.uint8:
        samples = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.signedinteger):
        samples = data.astype(np.float32) / float(2 ** (np.iinfo(data.dtype).bits - 1))
    elif np.issubdtype(data.dtype, np.floating):
        samples = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV encoding: {data.dtype}")
    if rate <= 0:
        raise ValueError("Fixture sample rate must be positive.")
    return rate, stereo_samples(samples)


@dataclass(frozen=True)
class AudioFixture:
    """Onsets are seconds from sample zero, before any resampling.

    In-memory audio is float32 mono (frames,) or stereo (frames, 2),
    nominally [-1, 1]. WAV headers must agree with sample_rate.
    """

    identifier: str
    category: str
    audio: AudioSamples | Path
    sample_rate: int
    kick_onsets: tuple[float, ...]
    generation_parameters: dict[str, object] = field(default_factory=dict)
    dataset: Dataset = "synthetic"
    partition: Partition = "regression"
    audio_sha256: str | None = None

    def load_audio(self) -> AudioSamples:
        if self.sample_rate <= 0:
            raise ValueError("Fixture sample rate must be positive.")
        if isinstance(self.audio, Path):
            if self.audio_sha256 is not None:
                verify_checksum(self.audio, self.audio_sha256)
            rate, samples = load_wav(self.audio)
            if rate != self.sample_rate:
                raise ValueError(f"{self.identifier}: sample rate disagrees with the WAV header.")
        else:
            samples = stereo_samples(self.audio)
        onsets = np.asarray(self.kick_onsets, dtype=np.float64)
        if (
            onsets.ndim != 1
            or not np.all(np.isfinite(onsets))
            or np.any(onsets < 0)
            or np.any(onsets >= len(samples) / self.sample_rate)
            or np.any(np.diff(onsets) <= 0)
        ):
            raise ValueError("Kick onsets must be finite, strictly increasing, and within the audio.")
        return samples


def load_manifest(path: Path) -> list[AudioFixture]:
    """Read external annotations; relative WAV paths resolve beside the manifest."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("A fixture manifest must contain a JSON list.")
    fixtures = []
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("identifier"), str)
            or not isinstance(record.get("category"), str)
            or not isinstance(record.get("audio"), str)
            or type(record.get("sample_rate")) is not int
            or not isinstance(record.get("kick_onsets"), list)
            or any(type(onset) not in (int, float) for onset in record["kick_onsets"])
        ):
            raise ValueError("Each fixture needs identifier, category, audio, sample_rate, and numeric kick_onsets.")
        fixtures.append(
            AudioFixture(
                identifier=record["identifier"],
                category=record["category"],
                audio=path.parent / record["audio"],
                sample_rate=record["sample_rate"],
                kick_onsets=tuple(float(onset) for onset in record["kick_onsets"]),
            )
        )
    return fixtures


def synthetic_fixture(
    identifier: str,
    category: str,
    onsets: tuple[float, ...],
    *,
    tempo: float | None = None,
    frequency: float = 55.0,
    amplitude: float = 0.6,
    decay: float = 0.12,
    background: str = "silence",
) -> AudioFixture:
    """Integrate an exponentially falling pitch with a soft attack and decaying body."""
    rate = DEFAULT_SAMPLE_RATE
    duration = 8.0
    seed = 20261124
    attack = 0.002
    pitch_decay = 0.025
    pitch_ratio = 3.5
    length = min(duration, 8 * decay)
    time = np.arange(round(length * rate)) / rate
    phase = 2 * np.pi * frequency * (time + (pitch_ratio - 1) * pitch_decay * (-np.expm1(-time / pitch_decay)))
    envelope = (-np.expm1(-time / attack)) * np.exp(-time / decay)
    kick = amplitude * envelope * np.sin(phase)
    # A short deterministic noise attack adds broadband beater energy.
    kick += amplitude * 0.08 * np.random.default_rng(seed).standard_normal(len(time)) * np.exp(-time / 0.003)
    mono = np.zeros(round(duration * rate), dtype=np.float64)
    positions = tuple(round(onset * rate) for onset in onsets)
    for position in positions:
        count = min(len(kick), len(mono) - position)
        mono[position : position + count] += kick[:count]

    timeline = np.arange(len(mono)) / rate
    if background in ("sub", "sustained"):
        mono += 0.3 * np.sin(2 * np.pi * 32 * timeline) * (-np.expm1(-timeline / 0.03))
    elif background == "rumble":
        noise = np.random.default_rng(seed + 1).standard_normal(len(mono))
        low_noise = np.asarray(sosfilt(butter(2, 110, fs=rate, output="sos"), noise), dtype=np.float64)
        low_noise *= 0.18 / np.sqrt(np.mean(low_noise**2))
        # A delayed, long bass tail on each beat, plus continuous low rumble.
        rumble = low_noise * 0.2
        for position in positions:
            start = position + round(0.09 * rate)
            count = min(round(1.2 * rate), len(mono) - start)
            tail_time = np.arange(count) / rate
            rumble[start : start + count] += low_noise[start : start + count] * np.exp(-tail_time / 0.45)
        mono += rumble
    elif background == "offbeat":
        note_time = np.arange(round(0.28 * rate)) / rate
        note = 0.35 * np.sin(2 * np.pi * 49 * note_time) * (-np.expm1(-note_time / 0.005)) * np.exp(-note_time / 0.1)
        for onset in np.arange(0.75, 7.5, 0.5):
            start = round(onset * rate)
            mono[start : start + len(note)] += note
    elif background != "silence":
        raise ValueError(f"Unknown synthetic background: {background}")

    samples = mono.astype(np.float32)
    samples.setflags(write=False)
    return AudioFixture(
        identifier,
        category,
        samples,
        rate,
        tuple(position / rate for position in positions),
        {
            "generator_version": 1,
            "seed": seed,
            "duration_seconds": duration,
            "tempo_bpm": tempo,
            "kick_sample_positions": positions,
            "frequency_hz": frequency,
            "amplitude": amplitude,
            "decay_seconds": decay,
            "attack_seconds": attack,
            "pitch_decay_seconds": pitch_decay,
            "pitch_ratio": pitch_ratio,
            "kick_duration_seconds": length,
            "beater_noise_ratio": 0.08,
            "background": background,
        },
    )


def synthetic_fixtures() -> list[AudioFixture]:
    regular = tuple(float(onset) for onset in np.arange(0.5, 7.5, 0.5))
    return [
        synthetic_fixture("four_on_floor", "clean", regular, tempo=120),
        synthetic_fixture("slow", "tempo", tuple(float(t) for t in np.arange(0.5, 7.5, 60 / 84)), tempo=84),
        synthetic_fixture("fast", "tempo", tuple(float(t) for t in np.arange(0.5, 7.5, 60 / 174)), tempo=174),
        synthetic_fixture("low_pitch", "frequency", regular, frequency=28),
        synthetic_fixture("high_pitch", "frequency", regular, frequency=210),
        synthetic_fixture("mid_pitch", "frequency", regular, frequency=95),
        synthetic_fixture("quiet", "amplitude", regular, amplitude=0.04),
        synthetic_fixture("loud", "amplitude", regular, amplitude=0.9),
        synthetic_fixture("long_decay", "decay", regular, decay=0.5),
        synthetic_fixture("with_sub", "sub_bass", regular, background="sub"),
        synthetic_fixture("with_rumble", "rumble", regular, background="rumble"),
        synthetic_fixture("sustained_bass", "negative", (), background="sustained"),
        synthetic_fixture("offbeat_bass", "negative", (), background="offbeat"),
        synthetic_fixture("silence_then_kick", "silence", (4.0,)),
        synthetic_fixture("irregular", "irregular", (0.13, 0.39, 1.11, 1.28, 2.7, 3.02, 3.19, 4.6, 6.15, 6.37, 7.1)),
    ]
