"""Software frame accounting. Draw completion does not imply presentation."""

import os
import resource
from collections import defaultdict
from collections.abc import Callable, Sequence
from time import perf_counter, process_time

import numpy as np


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p95": None, "p99": None, "max": None}
    return dict(
        zip(("median", "p95", "p99", "max"), map(float, np.percentile(values, [50, 95, 99, 100])), strict=True),
        count=len(values),
    )


def resident_bytes() -> int:
    with open("/proc/self/statm") as source:
        return int(source.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class Measurements:
    def __init__(self) -> None:
        self.active = False
        self.serial = 0
        self.snapshot_time = 0.0
        self.cursor_position = 0.0
        self.pending_inputs: dict[str, list[float]] = defaultdict(list)
        self.updates: list[float] = []
        self.update_costs: list[float] = []
        self.paints: list[tuple[int, str, float, float, float]] = []
        self.latencies: dict[str, list[float]] = defaultdict(list)
        self.callback_costs: list[float] = []
        self.memory: list[int] = []
        self.clock_position: Callable[[], float] | None = None
        self.clock_gaps: list[float] = []
        self.initial_upload_calls = 0
        self.initial_upload_bytes = 0
        self.geometry_calls = 0
        self.geometry_bytes = 0
        self.rebuilds = 0
        self.cpu_geometry_bytes = 0
        self.started = 0.0
        self.cpu_started = 0.0
        self.peak_start = 0
        self.start()
        self.active = False

    def start(self) -> None:
        self.updates.clear()
        self.update_costs.clear()
        self.paints.clear()
        self.latencies.clear()
        self.callback_costs.clear()
        self.clock_gaps.clear()
        self.pending_inputs.clear()
        self.memory = [resident_bytes()]
        self.geometry_calls = self.geometry_bytes = self.rebuilds = self.cpu_geometry_bytes = 0
        self.started = perf_counter()
        self.cpu_started = process_time()
        self.peak_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.active = True

    def input(self, kind: str, scheduled: float | None = None) -> None:
        if self.active:
            self.pending_inputs[kind].append(perf_counter())
            if scheduled is not None:
                self.pending_inputs[kind + "_scheduled"].append(scheduled)

    def painted(self, component: str, started: float) -> None:
        if not self.active:
            return
        now = perf_counter()
        if self.clock_position is not None:
            self.clock_gaps.append((self.clock_position() - self.cursor_position) * 1000)
        self.paints.append((self.serial, component, now, (now - started) * 1000, (now - self.snapshot_time) * 1000))
        for kind, timestamps in self.pending_inputs.items():
            self.latencies[kind].extend((now - timestamp) * 1000 for timestamp in timestamps)
        self.pending_inputs.clear()

    def result(self, target: int, components: set[str]) -> dict:
        elapsed = perf_counter() - self.started
        self.active = False
        # Complete a logical frame once every component has painted since the last
        # completion. Timers can run between the two Qt paint events, so demanding
        # identical snapshot ids would incorrectly discard real composite frames.
        # Duplicate component paints of one snapshot are still collapsed.
        seen: set[str] = set()
        ends = []
        component_ends: dict[str, list[float]] = defaultdict(list)
        last_serial: dict[str, int] = {}
        for serial, component, end, _cost, _age in self.paints:
            if last_serial.get(component) == serial:
                continue
            last_serial[component] = serial
            component_ends[component].append(end)
            seen.add(component)
            if components <= seen:
                ends.append(end)
                seen.clear()
        intervals = (np.diff(ends) * 1000).tolist()
        update_intervals = (np.diff(self.updates) * 1000).tolist()
        deadline = 1000 / target
        return {
            "elapsed_seconds": elapsed,
            "update_hz": len(self.updates) / elapsed,
            "logical_draw_hz": len(ends) / elapsed,
            "paint_events": len(self.paints),
            "complete_logical_frames": len(ends),
            "incomplete_or_duplicate_events": len(self.paints) - len(ends) * len(components),
            "frame_interval_ms": distribution(intervals),
            "component_draw_hz": {key: len(values) / elapsed for key, values in component_ends.items()},
            "component_frame_interval_ms": {
                key: distribution((np.diff(values) * 1000).tolist()) for key, values in component_ends.items()
            },
            "frame_interval_histogram_ms": {
                "edges": [0, 8.3334, 16.6667, 25, 50, 100, 1000, 1e9],
                "counts": np.histogram(intervals, [0, 8.3334, 16.6667, 25, 50, 100, 1000, 1e9])[0].tolist(),
            },
            "deadline_misses": sum(value > deadline * 1.05 for value in intervals),
            "deadline_slots_missed": max(0, round(elapsed * target) - len(ends)),
            "update_interval_ms": distribution(update_intervals),
            "update_cpu_ms": distribution(self.update_costs),
            "paint_submission_ms": distribution([paint[3] for paint in self.paints]),
            "snapshot_to_paint_ms": distribution([paint[4] for paint in self.paints]),
            "input_to_first_paint_ms": {kind: distribution(values) for kind, values in self.latencies.items()},
            "cpu_percent_one_core": 100 * (process_time() - self.cpu_started) / elapsed,
            "rss_bytes": distribution(self.memory + [resident_bytes()]),
            "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "waveform_rebuilds": self.rebuilds,
            "waveform_prepared_bytes": self.cpu_geometry_bytes,
            "buffer_upload_calls": self.geometry_calls,
            "initial_and_warmup_upload_calls": self.initial_upload_calls,
            "initial_and_warmup_upload_bytes": self.initial_upload_bytes,
            "cursor_gap_recording_ms": distribution(self.clock_gaps),
            "buffer_upload_bytes": self.geometry_bytes,
            "buffer_upload_bytes_per_second": self.geometry_bytes / elapsed,
            "callback_ms": distribution(self.callback_costs),
            "raw": {"frame_intervals_ms": intervals, "update_intervals_ms": update_intervals},
        }
