"""Run isolated desktop workers, alternating renderer order between repetitions."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from benchmarks.annotation.cli import write_json
from benchmarks.review_performance.data import checksum, load_recording
from benchmarks.review_performance.metrics import distribution
from benchmarks.review_performance.runner import SCENARIOS, ExperimentWindow

RESULTS = Path("benchmarks/results/review_performance")


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() or result.stderr.strip()


def conditions(window: ExperimentWindow) -> dict:
    screen = window.screen()
    graphics = {}
    if window.backend == "vispy":
        from vispy.gloo import gl

        from benchmarks.review_performance.vispy_timeline import VispyTimeline

        assert isinstance(window.timeline, VispyTimeline)

        window.timeline.canvas.set_current()
        graphics = {
            key: vars(gl)["glGetParameter"](getattr(gl, key)) for key in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")
        }
        renderer = graphics["GL_RENDERER"].lower()
        graphics["hardware_accelerated"] = not any(
            value in renderer for value in ("llvmpipe", "softpipe", "swrast", "software")
        )
        graphics["physical_canvas_size"] = list(window.timeline.canvas.physical_size)
        graphics["samples"] = 4
        graphics["vsync_requested"] = False
    else:
        import pyqtgraph as pg

        graphics = {
            "useOpenGL": pg.getConfigOption("useOpenGL"),
            "antialias": pg.getConfigOption("antialias"),
            "segmentedLineMode": "on",
            "downsampling": "native peak",
            "device_coordinate_cache": True,
        }
    return {
        "recorded_utc": datetime.now(UTC).isoformat(),
        "revision": command_output(["git", "rev-parse", "HEAD"]),
        "worktree_diff_sha256": hashlib.sha256(
            subprocess.run(["git", "diff"], capture_output=True, check=True).stdout
        ).hexdigest(),
        "experiment_source_sha256": {
            str(path): checksum(path)
            for path in Path("benchmarks/review_performance").glob("*.py")
            if not path.name.startswith("test_") and path.name != "report.py"
        },
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("vispy", "PySide6", "pyqtgraph", "numpy", "scipy", "sounddevice")
        },
        "platform": platform.platform(),
        "cpu": command_output(["lscpu"]),
        "display": {
            "platform": QApplication.platformName(),
            "name": screen.name(),
            "reported_refresh_hz": screen.refreshRate(),
            "device_pixel_ratio": screen.devicePixelRatio(),
            "logical_dpi": screen.logicalDotsPerInch(),
            "physical_dpi": screen.physicalDotsPerInch(),
            "geometry": [screen.size().width(), screen.size().height()],
            "window": [window.width(), window.height()],
            "timeline": [window.timeline.width(), window.timeline.height()],
        },
        "graphics": graphics,
        "audio_devices": command_output(
            [sys.executable, "-c", "import sounddevice; print(sounddevice.query_devices())"]
        ),
        "gpu_utilization": None,
        "gpu_memory_bytes": None,
        "gpu_metrics_note": "No portable reliable per-process GPU counters used; buffer payload bytes are CPU-side enqueue counts, not driver residency or PCIe traffic.",
        "timing_note": "Qt paint/VisPy scene draw CPU completion; neither GPU execution nor swap/compositor presentation. No glFinish or synchronous GPU readbacks in measured runs.",
    }


def worker(args: argparse.Namespace) -> None:
    if args.prepared:
        with args.prepared.open("rb") as source:
            recording = pickle.load(source)
    else:
        recording = load_recording(args.track[0])
    app = QApplication.instance() or QApplication([])
    window = ExperimentWindow(
        recording, args.backend, args.target[0], args.waveform, args.mode == "integrated", args.axes
    )
    window.resize(*args.size)
    if not args.interactive:
        window.setFixedSize(*args.size)
    window.show()
    app.processEvents()
    environment: dict = {}
    profile = None
    underruns_at_start = 0
    buffering_at_start = False
    if args.profile:
        import cProfile

        profile = cProfile.Profile()

    def finished() -> None:
        if profile is not None:
            profile.disable()
            profile.dump_stats(str(args.output.with_suffix(".prof")))
        result = window.measurements.result(
            args.target[0], {"canvas"} if args.backend == "vispy" else {"wave", "evidence"}
        )
        if args.backend == "pyqtgraph":
            for key in (
                "buffer_upload_calls",
                "buffer_upload_bytes",
                "buffer_upload_bytes_per_second",
                "initial_and_warmup_upload_calls",
                "initial_and_warmup_upload_bytes",
            ):
                result[key] = None
        result["script_actions_dispatched"] = window.last_action + 1
        player = window.player
        result["audio"] = {
            "mode": args.mode,
            "underruns": player.underruns - underruns_at_start if player else None,
            "warmup_underruns": underruns_at_start if player else None,
            "warmup_reliable_buffering": buffering_at_start if player else None,
            "buffering_recovery": player.reliable_buffering if player else None,
            "final_speed": player.speed if player else None,
            "clock_generation": player.generation if player else None,
            "software_timing_only": True,
        }
        if args.backend == "vispy":
            from benchmarks.review_performance.vispy_timeline import VispyTimeline

            assert isinstance(window.timeline, VispyTimeline)
            gpu_timer = window.timeline.canvas.gpu_timer
            result["gpu_scene_execution_ms"] = distribution(gpu_timer.milliseconds) if gpu_timer else None
        result["stream"] = {
            "overwritten_before_consumption": window.stream_overwritten,
            "history_capacity_samples": window.stream.capacity if window.stream else None,
            "samples_ingested": window.stream.written if window.stream else None,
            "maximum_ingestion_lateness_ms": window.stream.maximum_ingestion_lateness_seconds * 1000
            if window.stream
            else None,
        }
        result["initialization_seconds"] = window.initialization_seconds
        result["waveform_preprocessing_seconds"] = getattr(window.timeline, "preprocessing_seconds", None)
        result["waveform_cpu_geometry_bytes"] = getattr(window.timeline, "geometry_cpu_bytes", None)
        result["recording_preparation_seconds"] = recording.preparation_seconds
        result["recording"] = {
            "track": recording.proposal["track_identifier"],
            "audio_sha256": recording.proposal["audio_sha256"],
            "duration_seconds": recording.analysis.duration_seconds,
            "candidate_count": len(window.editor.candidates),
            "saved_overrides": len(recording.draft.overrides),
            "saved_manual_events": len(recording.draft.manual_events),
            "protected_files_unchanged": all(
                checksum(Path(path)) == digest for path, digest in recording.protected_hashes.items()
            ),
        }
        if not result["recording"]["protected_files_unchanged"]:
            raise RuntimeError("Protected data changed during experiment.")
        write_json(
            args.output,
            {
                "conditions": environment,
                "parameters": {
                    "backend": args.backend,
                    "scenario": args.scenario[0],
                    "target_hz": args.target[0],
                    "duration_seconds": args.duration,
                    "warmup_seconds": args.warmup,
                    "waveform": args.waveform,
                    "axes": args.axes,
                    "mode": args.mode,
                    "profiled": args.profile,
                    "gpu_timing": args.gpu_timing,
                    "requested_window": args.size,
                    "stream_full_upload": args.stream_full,
                },
                "measurements": result,
            },
        )
        print(
            f"{args.output}: draw {result['logical_draw_hz']:.1f} Hz; frame P95 {result['frame_interval_ms']['p95']} ms",
            flush=True,
        )
        window.close()
        app.quit()

    def measured() -> None:
        nonlocal underruns_at_start, buffering_at_start
        if window.player is not None:
            underruns_at_start = window.player.underruns
            buffering_at_start = window.player.reliable_buffering
        if args.gpu_timing and args.backend == "vispy":
            from benchmarks.review_performance.gpu_timing import GPUTimer
            from benchmarks.review_performance.vispy_timeline import VispyTimeline

            assert isinstance(window.timeline, VispyTimeline)
            window.timeline.canvas.set_current()
            window.timeline.canvas.gpu_timer = GPUTimer()
        environment.update(conditions(window))
        window.begin(args.scenario[0], args.duration, finished, args.stream_full)
        window.measurements.start()
        if args.scenario[0] == "stream":
            from benchmarks.review_performance.vispy_timeline import VispyTimeline

            assert isinstance(window.timeline, VispyTimeline)
            window.timeline.strategy = "stream"
        if profile is not None:
            profile.enable()

    if args.interactive:
        window.begin("interactive", 1e9)
        window.measurements.active = False
    elif args.warmup:
        window.begin("follow_10", args.warmup, measured)
    else:
        QTimer.singleShot(0, measured)
    exit_code = app.exec()
    if exit_code:
        raise SystemExit(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", nargs="+", default=["universal", "no_trust"], choices=("universal", "no_trust"))
    parser.add_argument("--backend", choices=("all", "pyqtgraph", "vispy"), default="all")
    parser.add_argument("--scenario", nargs="+", choices=SCENARIOS, default=list(SCENARIOS[:-1]))
    parser.add_argument("--target", nargs="+", type=int, choices=(30, 60, 90, 120), default=[60, 120])
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--mode", choices=("rendering", "integrated"), default="rendering")
    parser.add_argument("--waveform", choices=("visible", "resident", "lod"), default="lod")
    parser.add_argument("--size", type=int, nargs=2, default=(1180, 620), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--gpu-timing", action="store_true", help="Separate diagnostic; asynchronous OpenGL scene timer queries"
    )
    parser.add_argument("--axes", choices=("stock", "cached", "none"), default="cached")
    parser.add_argument("--stream-full", action="store_true", help="Diagnostic full-history upload baseline")
    parser.add_argument("--output", type=Path, default=RESULTS / "vispy_comparison.json")
    parser.add_argument(
        "--resume", action="store_true", help="Skip existing worker results in the output's run directory"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepared", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.duration <= 0 or args.warmup < 0 or args.repetitions < 1 or min(args.size) < 200:
        parser.error("Positive duration/repetitions and window dimensions >= 200 required.")
    if not args.interactive and args.duration > 30:
        parser.error("Fixed measurement intervals support up to 30 seconds without reaching EOF.")
    if args.mode == "integrated" and "stream" in args.scenario:
        parser.error("Streaming simulates ingestion independently of playback; use --mode rendering.")
    if args.interactive:
        args.backend = "vispy" if args.backend == "all" else args.backend
        args.track, args.target = args.track[:1], args.target[:1]
        worker(args)
        return
    if args.worker:
        worker(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.output.parent / (args.output.stem + "_runs")
    run_dir.mkdir(exist_ok=True)
    results = []
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="spectra-rendering-") as temporary:
        for track in args.track:
            print(f"Preparing {track} (read-only)…", flush=True)
            prepared = Path(temporary) / f"{track}.pickle"
            with prepared.open("wb") as destination:
                pickle.dump(load_recording(track), destination, protocol=pickle.HIGHEST_PROTOCOL)
            for target in args.target:
                for scenario in args.scenario:
                    for repeat in range(args.repetitions):
                        backends = ["pyqtgraph", "vispy"] if args.backend == "all" else [args.backend]
                        if repeat % 2:
                            backends.reverse()
                        for backend in backends:
                            if scenario == "stream" and backend == "pyqtgraph":
                                continue
                            output = run_dir / f"{track}_{target}_{scenario}_{repeat}_{backend}.json"
                            command = [
                                sys.executable,
                                "-m",
                                "benchmarks.review_performance",
                                "--worker",
                                "--prepared",
                                str(prepared),
                                "--track",
                                track,
                                "--backend",
                                backend,
                                "--target",
                                str(target),
                                "--scenario",
                                scenario,
                                "--duration",
                                str(args.duration),
                                "--warmup",
                                str(args.warmup),
                                "--axes",
                                args.axes,
                                "--waveform",
                                args.waveform,
                                "--mode",
                                args.mode,
                                "--size",
                                *map(str, args.size),
                                "--output",
                                str(output),
                            ]
                            command += [
                                flag
                                for flag, enabled in (
                                    ("--profile", args.profile),
                                    ("--gpu-timing", args.gpu_timing),
                                    ("--no-axes", False),
                                    ("--stream-full", args.stream_full),
                                )
                                if enabled
                            ]
                            if not args.resume or not output.exists():
                                print(f"{track} {target} Hz {scenario} repeat {repeat + 1}: {backend}", flush=True)
                                subprocess.run(command, check=True, env=os.environ.copy())
                            result = json.loads(output.read_text())
                            result["repeat"] = repeat
                            result["source_file"] = str(output)
                            results.append(result)
                            write_json(args.output, {"schema_version": 1, "complete": False, "runs": results})
    write_json(
        args.output, {"schema_version": 1, "complete": True, "wall_seconds": perf_counter() - started, "runs": results}
    )
    print(f"Saved {len(results)} runs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
