"""Local audio fixtures, isolated PipeWire playback, and offline analysis.

Run with uv run python scripts/audio_test.py --help from the repository root.
This development utility does not alter Spectra's production audio pipeline.
"""

import argparse
import hashlib
import json
import math
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
SINK_NAME = "spectra_test"
RATE = 48_000
SIGNAL_RMS_FLOOR = 1e-5  # -100 dB relative to unit PCM amplitude.


def require_tools(*names: str) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        packages = sorted({"ffmpeg" if name in ("ffmpeg", "ffprobe") else "pulseaudio-utils" for name in missing})
        raise RuntimeError(f"Missing {', '.join(missing)}. Install with: sudo apt install {' '.join(packages)}")


def pactl(*arguments: str) -> str:
    return subprocess.run(["pactl", *arguments], check=True, capture_output=True, text=True, timeout=5).stdout.strip()


def pulse_records(kind: str) -> list[dict[str, object]]:
    value: object = json.loads(pactl("--format=json", "list", kind))
    if not isinstance(value, list):
        raise RuntimeError(f"Unexpected pactl {kind} output.")
    records: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeError(f"Unexpected pactl {kind} record.")
        records.append(item)
    return records


def find_sink() -> dict[str, object] | None:
    return next((sink for sink in pulse_records("sinks") if sink.get("name") == SINK_NAME), None)


def is_null_sink(sink: dict[str, object]) -> bool:
    properties = sink.get("properties")
    return isinstance(properties, dict) and properties.get("factory.name") == "support.null-audio-sink"


@contextmanager
def test_sink() -> Iterator[str]:
    """Reuse an existing null sink, removing only a uniquely tagged owned module."""
    require_tools("pactl")
    existing = find_sink()
    if existing is not None:
        if not is_null_sink(existing):
            raise RuntimeError("spectra_test already exists but is not a null sink; refusing playback.")
        print("Reusing existing spectra_test; this run will not remove it.", file=sys.stderr)
        yield f"{SINK_NAME}.monitor"
        return

    owner = f"spectra.test.owner={uuid.uuid4().hex}"
    default_before = pactl("get-default-sink")
    try:
        pactl(
            "load-module",
            "module-null-sink",
            f"sink_name={SINK_NAME}",
            "format=s16le",
            f"rate={RATE}",
            "channels=2",
            "channel_map=front-left,front-right",
            f'sink_properties="device.description=Spectra_Test priority.session=-1000 {owner}"',
        )
        deadline = time.monotonic() + 5
        while (created := find_sink()) is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for spectra_test.")
            time.sleep(0.05)
        properties = created.get("properties")
        if not isinstance(properties, dict) or properties.get("spectra.test.owner") != owner.split("=", 1)[1]:
            raise RuntimeError("Another process owns spectra_test; refusing to use it.")
        if pactl("get-default-sink") != default_before:
            raise RuntimeError("The default output changed during sink creation; stopping without changing it back.")
        print("Created spectra_test (null sink); monitor: spectra_test.monitor", file=sys.stderr)
        yield f"{SINK_NAME}.monitor"
    finally:
        # Obtain the module ID only from a null sink with this run's unique
        # token. Module-list JSON does not include indices on all pactl versions.
        for sink in pulse_records("sinks"):
            properties = sink.get("properties")
            if (
                isinstance(properties, dict)
                and properties.get("spectra.test.owner") == owner.split("=", 1)[1]
                and is_null_sink(sink)
            ):
                module_id = sink.get("owner_module")
                if not isinstance(module_id, int):
                    raise RuntimeError("Cannot identify the owned test sink's module; leaving it intact.")
                pactl("unload-module", str(module_id))
                print("Removed this run's spectra_test module.", file=sys.stderr)


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def audio_file(path: Path) -> Iterator[wave.Wave_read]:
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (RATE, 2, 2):
            raise ValueError("Expected 48 kHz stereo 16-bit PCM WAV; use the convert command first.")
        yield audio


def excerpt_frames(audio: wave.Wave_read, start: float, duration: float | None) -> tuple[int, int]:
    if (
        not math.isfinite(start)
        or start < 0
        or (duration is not None and (not math.isfinite(duration) or duration <= 0))
    ):
        raise ValueError("Excerpt start must be nonnegative and duration must be positive.")
    first = round(start * RATE)
    count = audio.getnframes() - first
    if count <= 0:
        raise ValueError("Excerpt starts beyond the end of the file.")
    if duration is not None:
        count = min(count, round(duration * RATE))
    if count <= 0:
        raise ValueError("Excerpt contains no audio frames.")
    return first, count


@contextmanager
def playback_excerpt(path: Path, start: float, duration: float | None) -> Iterator[Path]:
    """Use the original WAV for full playback, or a temporary, frame-exact excerpt."""
    with audio_file(path) as audio:
        first, count = excerpt_frames(audio, start, duration)
        if start == 0 and duration is None:
            yield path
            return
        audio.setpos(first)
        with tempfile.TemporaryDirectory(prefix="spectra-playback-") as directory:
            excerpt = Path(directory) / "excerpt.wav"
            with wave.open(str(excerpt), "wb") as output:
                output.setparams(audio.getparams())
                remaining = count
                while remaining:
                    data = audio.readframes(min(65_536, remaining))
                    if not data:
                        raise ValueError("WAV data ended before the declared excerpt length.")
                    output.writeframesraw(data)
                    remaining -= len(data) // (audio.getnchannels() * audio.getsampwidth())
            yield excerpt


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def convert_audio(arguments: argparse.Namespace) -> None:
    require_tools("ffmpeg")
    if (
        not math.isfinite(arguments.start)
        or arguments.start < 0
        or arguments.duration is not None
        and (not math.isfinite(arguments.duration) or arguments.duration <= 0)
    ):
        raise ValueError("Excerpt start must be nonnegative and duration must be positive.")
    if arguments.output.suffix.lower() != ".wav":
        raise ValueError("Converted fixtures must use a .wav filename.")
    if arguments.output.exists() or arguments.output.with_suffix(".wav.json").exists():
        raise FileExistsError("The output WAV or provenance sidecar already exists; choose a new filename.")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(arguments.input)]
    if arguments.start:
        command += ["-ss", str(arguments.start)]
    if arguments.duration is not None:
        command += ["-t", str(arguments.duration)]
    command += ["-map", "0:a:0", "-vn", "-ar", str(RATE), "-ac", "2", "-c:a", "pcm_s16le", str(arguments.output)]
    subprocess.run(command, check=True)
    write_json(
        arguments.output.with_suffix(".wav.json"),
        {
            "source": arguments.source,
            "license": arguments.license,
            "original_file": str(arguments.input.resolve()),
            "start_seconds": arguments.start,
            "duration_seconds": arguments.duration,
            "conversion_command": command,
        },
    )


def synthesize(path: Path) -> None:
    """Generate an eight-second deterministic fixture; no third-party audio."""
    if path.suffix.lower() != ".wav":
        raise ValueError("Synthetic fixtures must use a .wav filename.")
    if path.exists() or path.with_suffix(".wav.json").exists():
        raise FileExistsError("The output WAV or provenance sidecar already exists; choose a new filename.")
    mono = np.zeros(8 * RATE, dtype=np.float64)
    for start, frequency, amplitude, decay, duration in (
        *((start, 65, 0.6, 0.04, 0.2) for start in (0.5, 1.0, 1.5, 2.0)),
        *((start, 8000, 0.18, 0.005, 0.04) for start in (4.0, 4.25, 4.5, 4.75)),
        *((start, 65, 0.06, 0.04, 0.2) for start in (5.5, 6.0)),
    ):
        timeline = np.arange(round(duration * RATE)) / RATE
        pulse = amplitude * np.sin(2 * np.pi * frequency * timeline) * np.exp(-timeline / decay)
        offset = round(start * RATE)
        mono[offset : offset + len(pulse)] += pulse
    timeline = np.arange(RATE) / RATE
    mono[round(2.5 * RATE) : round(3.5 * RATE)] += 0.15 * np.sin(2 * np.pi * 65 * timeline)
    frame = np.column_stack((mono, mono))
    pcm = np.rint(np.clip(frame, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output, wave.open(output, "wb") as audio:
        audio.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        audio.writeframes(pcm.tobytes())
    write_json(
        path.with_suffix(".wav.json"),
        {
            "source": "Generated by scripts/audio_test.py synth; no third-party recording",
            "license": "Project-generated test signal",
            "duration_seconds": 8,
            "sections": {
                "regular_bass_pulses": [0.5, 1, 1.5, 2],
                "sustained_bass": [2.5, 3.5],
                "high_frequency_pulses": [4, 4.25, 4.5, 4.75],
                "quiet_bass_pulses": [5.5, 6],
                "silence": [6.5, 8],
            },
        },
    )
    print(path)


def analyze_file(arguments: argparse.Namespace) -> dict[str, object]:
    from spectra.analysis import KickOnsetDetector, SpectrumAnalyzer

    analyzer = SpectrumAnalyzer()
    overrides = {
        name: value
        for name, value in (
            ("min_energy", arguments.min_energy),
            ("min_flux", arguments.min_flux),
            ("threshold_multiplier", arguments.threshold_multiplier),
            ("refractory_seconds", arguments.refractory),
        )
        if value is not None
    }
    detector = KickOnsetDetector(analyzer.band_edges, **overrides)
    detections = []
    with audio_file(arguments.file) as audio:
        first, count = excerpt_frames(audio, arguments.start, arguments.duration)
        audio.setpos(first)
        remaining = count
        while remaining:
            data = audio.readframes(min(1024, remaining))
            pcm = np.frombuffer(data, dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768.0
            if not len(pcm):
                raise ValueError("WAV data ended before the declared excerpt length.")
            remaining -= len(pcm)
            for spectrum in analyzer.process(pcm):
                event = detector.process(spectrum)
                if event is not None:
                    detections.append(
                        {
                            "timestamp": event.timestamp,
                            "source_timestamp": first / RATE + event.timestamp,
                            "strength": event.strength,
                        }
                    )
    with arguments.file.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    source_hashes = {}
    for name in ("spectrum.py", "kick.py"):
        with (ROOT / "src" / "spectra" / "analysis" / name).open("rb") as source:
            source_hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "input": str(arguments.file.resolve()),
        "sha256": checksum,
        "sample_rate": RATE,
        "start_frame": first,
        "frame_count": count,
        "timestamp_origin": "excerpt start; FFT window centers",
        "detector_overrides": overrides,
        "fft_size": analyzer.fft_size,
        "hop_size": analyzer.hop_size,
        "analysis_source_sha256": source_hashes,
        "detections": detections,
    }


def analyze(arguments: argparse.Namespace) -> None:
    if arguments.file.is_dir():
        paths = sorted(arguments.file.glob("*.wav"))
        if not paths:
            raise ValueError("The selected collection contains no WAV fixtures.")
        result: dict[str, object] = {
            "collection": str(arguments.file.resolve()),
            "results": [analyze_file(argparse.Namespace(**(vars(arguments) | {"file": path}))) for path in paths],
        }
    else:
        result = analyze_file(arguments)
    if arguments.output is not None:
        write_json(arguments.output, result)
        print(f"Analysis -> {arguments.output}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


def play(path: Path, capture: bool, start: float = 0.0, duration: float | None = None) -> None:
    require_tools("pactl", "paplay", *(("parec",) if capture else ()))
    with playback_excerpt(path, start, duration) as playback_path, test_sink() as monitor:
        from spectra.analysis import SpectrumAnalyzer
        from spectra.audio import AudioSourceInfo
        from spectra.audio.pipewire import PipeWireSource

        source = PipeWireSource(AudioSourceInfo(monitor, "Spectra test monitor")) if capture else None
        player = None
        signal_captured = False
        try:
            if source is not None:
                source.start()
                source.read()  # Wait for a connected monitor before starting playback.
            player = subprocess.Popen(
                [
                    "paplay",
                    f"--device={SINK_NAME}",
                    "--property=node.dont-fallback=true",
                    "--property=node.dont-move=true",
                    "--client-name=Spectra_Audio_Test",
                    str(playback_path.resolve()),
                ],
                start_new_session=True,
            )
            if source is None:
                player.wait()
            else:
                analyzer = SpectrumAnalyzer()
                report_at = time.monotonic()
                while player.poll() is None:
                    pcm = source.read()
                    rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
                    signal_captured |= rms > SIGNAL_RMS_FLOOR
                    spectra = analyzer.process(pcm)
                    if spectra and time.monotonic() >= report_at:
                        peak = float(np.max(np.abs(pcm)))
                        print(f"audio={spectra[-1].timestamp:.3f}s RMS={rms:.4f} Peak={peak:.4f}", flush=True)
                        report_at = time.monotonic() + 0.5
            if player.returncode != 0:
                raise RuntimeError(f"paplay failed with exit status {player.returncode}.")
            if source is not None:
                if not signal_captured:
                    raise RuntimeError(
                        "Playback completed, but spectra_test.monitor captured no signal above "
                        f"RMS {SIGNAL_RMS_FLOOR:g} (-100 dB). The excerpt may be silent, muted, or incorrectly routed."
                    )
                print("Verified non-silent audio on spectra_test.monitor.", flush=True)
        finally:
            if player is not None:
                stop_process(player)
            if source is not None:
                source.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Check required system utilities.")
    commands.add_parser("sink", help="Hold an isolated sink until Ctrl+C; remove only this run's sink.")
    synth = commands.add_parser("synth", help="Generate the reproducible eight-second smoke WAV.")
    synth.add_argument("output", type=Path, nargs="?", default=SAMPLES / "smoke.wav")
    convert = commands.add_parser("convert", help="Convert user-supplied/licensed audio to canonical PCM WAV.")
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--source", required=True, help="Source URL or description of the user-supplied file.")
    convert.add_argument("--license", required=True, help="Recording license or user-supplied usage information.")
    playback = commands.add_parser("play", help="Play a canonical WAV only into spectra_test.")
    playback.add_argument("file", type=Path)
    playback.add_argument(
        "--capture", action="store_true", help="Also exercise Spectra's capture and spectrum analysis."
    )
    analysis = commands.add_parser("analyze", help="Offline timestamped JSON for one WAV or every WAV in a directory.")
    analysis.add_argument("file", type=Path)
    analysis.add_argument("--output", type=Path)
    analysis.add_argument("--min-energy", type=float)
    analysis.add_argument("--min-flux", type=float)
    analysis.add_argument("--threshold-multiplier", type=float)
    analysis.add_argument("--refractory", type=float)
    for excerpt in (convert, analysis, playback):
        excerpt.add_argument("--start", type=float, default=0.0)
        excerpt.add_argument("--duration", type=float)
    arguments = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda _signal, _frame: sys.exit(130))
    try:
        if arguments.command == "check":
            for name in ("ffmpeg", "pactl", "paplay", "parec"):
                print(f"{name}: {shutil.which(name) or 'MISSING'}")
            require_tools("ffmpeg", "pactl", "paplay", "parec")
        elif arguments.command == "sink":
            with test_sink():
                print("Ready. Ctrl+C releases this session; existing sinks are left intact.", flush=True)
                signal.pause()
        elif arguments.command == "synth":
            synthesize(arguments.output)
        elif arguments.command == "convert":
            convert_audio(arguments)
        elif arguments.command == "analyze":
            analyze(arguments)
        elif arguments.command == "play":
            play(arguments.file, arguments.capture, arguments.start, arguments.duration)
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, wave.Error) as error:
        detail = (
            error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        )
        parser.exit(1, f"Audio test error: {detail}\n")


if __name__ == "__main__":
    main()
