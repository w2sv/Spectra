"""Short checks of the production VisPy timeline; historical comparisons are archived."""

import argparse
import copy
import json
import math
import platform
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from benchmarks.annotation.corrections import CorrectionEditor, effective_onsets
from benchmarks.annotation.review.playback import KickFlash, PCMPlayback, zoom_speed
from benchmarks.annotation.review.timeline import Timeline
from benchmarks.review_performance.data import checksum, load_recording
from benchmarks.review_performance.metrics import Measurements, resident_bytes


@contextmanager
def count_buffers(metrics: Measurements):
    from vispy.gloo.buffer import Buffer

    original_set, original_sub = Buffer.set_data, Buffer.set_subdata

    def record(data):
        if metrics.active:
            metrics.geometry_calls += 1
            metrics.geometry_bytes += data.nbytes
        else:
            metrics.initial_upload_calls += 1
            metrics.initial_upload_bytes += data.nbytes

    def set_data(self, data, copy=False):
        record(data)
        return original_set(self, data, copy=copy)

    def sub_data(self, data, offset=0, copy=False):
        record(data)
        return original_sub(self, data, offset=offset, copy=copy)

    Buffer.set_data, Buffer.set_subdata = set_data, sub_data
    try:
        yield
    finally:
        Buffer.set_data, Buffer.set_subdata = original_set, original_sub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("universal", "no_trust"), default="universal")
    parser.add_argument(
        "--scenario",
        choices=(
            "follow_2.5",
            "follow_10",
            "follow_40",
            "follow_120",
            "follow_full",
            "zoom",
            "seek",
            "markers",
            "idle",
        ),
        default="follow_10",
    )
    parser.add_argument("--target", type=int, choices=(30, 60, 90, 120), default=60)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--warmup", type=float, default=1)
    parser.add_argument("--mode", choices=("rendering", "integrated"), default="rendering")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/review_performance/local_check.json"))
    args = parser.parse_args()
    if not 0 < args.duration <= 30 or args.warmup < 0:
        parser.error("Use a positive duration up to 30 seconds and nonnegative warm-up.")
    recording = load_recording(args.track)
    app = QApplication.instance() or QApplication([])
    if args.interactive:
        from benchmarks.annotation.review.window import AnnotationReview

        with TemporaryDirectory(prefix="spectra-review-") as temporary:
            window = AnnotationReview(
                recording.analysis,
                recording.proposal,
                copy.deepcopy(recording.draft),
                Path(temporary) / "draft.json",
                60,
                10,
                audio_path=recording.audio_path,
            )
            window.show()
            app.exec()
        return
    metrics = Measurements()
    editor = CorrectionEditor(recording.proposal, copy.deepcopy(recording.draft))
    player = PCMPlayback.from_wav(recording.audio_path) if args.mode == "integrated" else None
    flash = KickFlash()
    flash.timestamps = effective_onsets(recording.proposal, editor.draft)
    with count_buffers(metrics):
        started = perf_counter()
        timeline = Timeline(recording.analysis, recording.proposal)
        initialization = perf_counter() - started
        timeline.update_candidates(
            editor, show_candidates=True, uncertainty=True, timing=True, scores=True, grid=True, alternative=False
        )
        window = QWidget()
        layout = QVBoxLayout(window)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(timeline)
        window.setFixedSize(1180, 620)
        window.setWindowTitle("Spectra — production VisPy rendering check")
        window.show()
        timer = QTimer(window)
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.setSingleShot(True)
        width = (
            recording.analysis.duration_seconds
            if args.scenario == "follow_full"
            else float(args.scenario.split("_")[1])
            if args.scenario.startswith("follow_")
            else 10
        )
        origin = monotonic()
        paint_start = [0.0]
        last_action = [-1]
        visibility = []
        last_exposed: list[bool | None] = [None]
        measured = [False]
        initial_underruns = [0]
        initial_uploads = [0, 0]
        generation = [0]
        seek_offset = [0.0]
        times = np.arange(math.ceil((args.duration + args.warmup) * 1000) + 2) / 1000
        speeds = np.full(len(times), zoom_speed(width))
        positions = 60 + np.cumsum(speeds) / 1000

        def before(event):
            paint_start[0] = perf_counter()

        def after(event):
            metrics.painted("canvas", paint_start[0])

        timeline.canvas.events["draw"].connect(before, position="first")
        timeline.canvas.events["draw"].connect(after, position="last")
        if player is not None:
            metrics.clock_position = lambda: player.position
            player.seek(60)
            player.set_speed(zoom_speed(width))
            if args.scenario != "idle":
                player.play()

        def finish():
            result = metrics.result(args.target, {"canvas"})
            result.update(
                initialization_seconds=initialization,
                waveform_preprocessing_seconds=timeline.waveform.pyramid.seconds,
                waveform_cpu_bytes=timeline.waveform.pyramid.bytes,
                waveform_uploads=timeline.waveform.uploads - initial_uploads[0],
                waveform_upload_bytes=timeline.waveform.uploaded_bytes - initial_uploads[1],
                visibility_changes=visibility,
                audio_underruns=None if player is None else player.underruns - initial_underruns[0],
                protected_files_unchanged=all(
                    checksum(Path(path)) == value for path, value in recording.protected_hashes.items()
                ),
            )
            from vispy.gloo import gl

            timeline.canvas.set_current()
            graphics = {
                name: vars(gl)["glGetParameter"](getattr(gl, name))
                for name in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")
            }
            document = {
                "parameters": vars(args) | {"output": str(args.output)},
                "python": sys.version,
                "platform": platform.platform(),
                "graphics": graphics,
                "display": {
                    "refresh_hz": window.screen().refreshRate(),
                    "scale": window.devicePixelRatioF(),
                    "window": [window.width(), window.height()],
                },
                "measurements": result,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
            print(
                f"{args.scenario}: {result['logical_draw_hz']:.1f} software draw Hz, P95 {result['frame_interval_ms']['p95']} ms",
                flush=True,
            )
            app.quit()

        def tick():
            try:
                now = monotonic()
                elapsed = now - origin
                if elapsed >= args.warmup + args.duration:
                    finish()
                    return
                if not measured[0] and elapsed >= args.warmup:
                    measured[0] = True
                    metrics.start()
                    initial_uploads[:] = [timeline.waveform.uploads, timeline.waveform.uploaded_bytes]
                    initial_underruns[0] = player.underruns if player else 0
                work_started = perf_counter()
                metrics.serial += 1
                if metrics.active:
                    metrics.updates.append(work_started)
                    if metrics.serial % args.target == 0:
                        metrics.memory.append(resident_bytes())
                exposed = window.windowHandle().isExposed()
                if exposed != last_exposed[0]:
                    visibility.append((elapsed, exposed))
                    last_exposed[0] = exposed
                action = int(elapsed * 4)
                if action != last_action[0]:
                    last_action[0] = action
                    if args.scenario == "zoom":
                        target = 10 * math.exp(3 * math.sin(elapsed * math.tau / 4))
                        metrics.input("zoom")
                        timeline.zoom(
                            target / timeline.target_width,
                            sum(timeline.interval) / 2,
                            playback_position=player.position
                            if player
                            else float(positions[min(len(positions) - 1, round(elapsed * 1000))]),
                        )
                    elif args.scenario == "seek" and action % 2 == 0:
                        seek_offset[0] += 5 if action % 4 == 0 else -1
                        generation[0] += 1
                        if player:
                            player.seek(60 + elapsed + seek_offset[0])
                    elif args.scenario == "markers":
                        key = recording.proposal["kick_candidates"][
                            action % len(recording.proposal["kick_candidates"])
                        ]["candidate_id"]
                        metrics.input("marker")
                        editor.toggle(key)
                        timeline.update_candidates(
                            editor,
                            show_candidates=True,
                            uncertainty=True,
                            timing=True,
                            scores=True,
                            grid=True,
                            alternative=False,
                        )
                position = (
                    player.poll()
                    if player
                    else 60
                    if args.scenario == "idle"
                    else float(positions[min(len(positions) - 1, round(elapsed * 1000))]) + seek_offset[0]
                )
                metrics.snapshot_time = perf_counter()
                metrics.cursor_position = position
                if not timeline.advance_zoom(monotonic(), playback_position=position):
                    timeline.follow(position, width=width if args.scenario != "zoom" else None)
                timeline.set_flash(
                    flash.update(
                        position,
                        playing=args.scenario != "idle",
                        generation=player.generation if player else generation[0],
                        playback_time=player.dac_time if player else None,
                        event_time=player.event_time if player else None,
                    )
                )
                if player:
                    player.set_speed(zoom_speed(timeline.interval[1] - timeline.interval[0]))
                if metrics.active:
                    metrics.update_costs.append((perf_counter() - work_started) * 1000)
                slot = math.floor((monotonic() - origin) * args.target) + 1
                timer.start(max(0, math.ceil((origin + slot / args.target - monotonic()) * 1000)))
            except Exception:
                import traceback

                traceback.print_exc()
                app.exit(1)

        timer.timeout.connect(tick)
        timer.start(0)
        try:
            status = app.exec()
        finally:
            timer.stop()
            if player:
                player.close()
            timeline.canvas.close()
            window.close()
        if status:
            raise SystemExit(status)


if __name__ == "__main__":
    main()
