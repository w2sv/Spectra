"""Instrument the current production Timeline without changing its rendering."""

from time import perf_counter

import pyqtgraph as pg
from PySide6.QtGui import QPaintEvent

from benchmarks.annotation.analysis import AcousticAnalysis, AutomaticCandidate, Proposal
from benchmarks.annotation.review import timeline
from benchmarks.review_performance.metrics import Measurements


def create_baseline(
    analysis: AcousticAnalysis, proposal: Proposal[AutomaticCandidate], measurements: Measurements
) -> timeline.Timeline:
    original = pg.PlotWidget
    count = 0

    class MeasuredPlotWidget(original):
        def __init__(self, *args, **kwargs) -> None:
            nonlocal count
            self.component = "wave" if count == 0 else "evidence"
            count += 1
            super().__init__(*args, **kwargs)

        def paintEvent(self, ev: QPaintEvent) -> None:
            started = perf_counter()
            super().paintEvent(ev)
            measurements.painted(self.component, started)

    # Constructor-local substitution; never changes the production source or GUI.
    pg.PlotWidget = MeasuredPlotWidget
    try:
        widget = timeline.Timeline(analysis, proposal)
        original_data = widget.waveform._getDisplayDataset
        previous: list[object] = [None]

        def display_data():
            dataset = original_data()
            if dataset is not None and dataset is not previous[0]:
                previous[0] = dataset
                if measurements.active:
                    measurements.rebuilds += 1
                    measurements.cpu_geometry_bytes += dataset.x.nbytes + dataset.y.nbytes
            return dataset

        widget.waveform._getDisplayDataset = display_data
        return widget
    finally:
        pg.PlotWidget = original
