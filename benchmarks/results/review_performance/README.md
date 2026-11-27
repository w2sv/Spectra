# Review rendering performance

[VisPy results and implementation decision](vispy_report.md) summarize the measured 41–55% P95 improvement and selection
of padded multiresolution waveform windows. [Underlying measurements](vispy_measurements.json.gz) retain the original
per-run data, input identity, short strategy checks and visibility diagnosis in a single lossless archive.

The following historical reports remain unchanged:

- [Matplotlib baseline](matplotlib_baseline.json): retired-renderer headless timings and a desktop smoke check.
- [Qt rendering comparison](qt_rendering_comparison.json): paired desktop paint/update costs, process CPU, underruns and
  software clock-to-cursor gaps before the VisPy migration.

Actual paint/draw completion is distinct from GPU execution, swapping and compositor presentation. The display reports
60 Hz; none of these measurements establishes physical 120 Hz presentation or physical audio-to-display synchronization.
