# VisPy timeline experiment

## Recommendation

**Prefer VisPy for a future annotation-timeline migration.** It clearly improves the measured following workloads: P95
software frame intervals fell **41–55%**, with moderately lower CPU use. The experiment leaves production rendering
unchanged. Migration should retain the current navigation, correction model and DAC-synchronized playback.

**Continue with VisPy for a dedicated runtime prototype**, reusing the bounded streaming buffer and batched marker
approach. Streaming is feasible; sustained 120 Hz, long-run memory stability and a scrolling spectrogram remain
unproven.

Collection was stopped at the user's request after the advantage was established. This is a focused positive result, not
completion of the originally proposed matrix.

## Repeated comparative results

Universal, rendering-only deterministic clock, 60 Hz requested, three independent 30-second runs per renderer and view,
three-second warm-up, alternating renderer order. Tables show medians of per-run statistics. Draw Hz means completion of
actual Qt composite paints or VisPy scene draws, not display presentation.

| View  | Qt draw Hz | VisPy draw Hz | Qt P95 ms | VisPy P95 ms | P95 reduction | Qt / VisPy P99 ms |
| ----- | ---------: | ------------: | --------: | -----------: | ------------: | ----------------: |
| 2.5 s |       25.1 |          44.4 |     47.30 |        28.05 |         40.7% |     57.26 / 35.23 |
| 10 s  |       19.4 |          42.5 |     60.56 |        31.51 |         48.0% |     73.19 / 36.40 |
| 40 s  |       15.2 |          37.8 |     81.56 |        36.77 |         54.9% |    103.05 / 43.91 |

The demanding-workload improvement target is met in these cases. Consistent 60 Hz is **not** achieved. Unsatisfied 60 Hz
draw slots fell approximately 51–57%, but neither implementation consistently meets the 16.7 ms deadline.

![Empirical frame-interval distributions](vispy_summary.svg)

Solid curves are VisPy; dashed curves are PyQtGraph. Vertical references mark 8.3 and 16.7 ms. The figure pools raw
intervals from the fully repeated cases; structured results retain individual runs, histograms and maxima.

| View  | Qt CPU % | VisPy CPU % | Qt RSS MB | VisPy RSS MB |
| ----- | -------: | ----------: | --------: | -----------: |
| 2.5 s |    100.7 |        92.5 |     556.8 |        629.4 |
| 10 s  |    100.5 |        94.8 |     556.6 |        629.3 |
| 40 s  |    100.4 |        93.8 |     555.8 |        629.3 |

CPU uses 100% for one occupied core; RSS uses decimal MB. VisPy uses about 73 MB more RSS here. Its constructor takes
roughly 1.1–1.2 seconds versus Qt's 0.7–0.75 seconds, including 0.65–0.69 seconds for an 81.2 MB waveform pyramid. GPU
physical memory and utilization were not reliably measured.

The four additional 120-second-view runs are retained but excluded from the headline comparison. They contain 4.9–12.9
second gaps between draws despite continued timer updates, and only two paired repetitions completed. Visibility,
compositor or scheduling effects were not isolated; these gaps cannot be assigned to rendering cost.

## Geometry and profiling

All fully repeated VisPy following cases recorded **zero waveform geometry rebuilds**. Ten-second following enqueued
about 1.9 kB/s of buffer payloads after warm-up, primarily axis updates. Changing from the warm-up view to the detailed
view incurs an approximately 15 MB resident-level upload. Following subsequently changes camera transforms.

The earlier, separately marked stock-axis profiles identified scene traversal, text preparation and repeated color
parsing as substantial costs. Marker refresh spent about 2.0 of 3.2 profiled seconds parsing colors. The retained
implementation caches marker colors and tick geometry, and moves a resident cursor mesh through a transform. These
changes were combined; their individual end-to-end gains were not isolated. Profiling timings are excluded from the
comparative tables.

The three waveform strategies are implemented: visible-range peak geometry, fully resident raw PCM, and resident
multiresolution extrema with original PCM at inspection zoom. Their comparative strategy matrix was not run before
collection stopped. The default pyramid preserves extrema, their sample times and incomplete final buckets. Numerical
checks protect these properties. Stock resident lines still submit offscreen vertices; padded envelope windows would be
a useful next experiment, not a measured optimization in this report.

Axes use the same coordinate scales with a different tick layout. Marker colors, sizes, diagnostic outlines, dashed
grids and cursor widths retain their meanings; pixel-identical rasterization is not claimed. Tests and desktop checks
cover hit-testing, individual corrections, selection, accumulated zoom, resizing and the interactive DAC-player launch.
The GUI requires more custom axis, cursor and interaction code than PyQtGraph. No permanent renderer interface was
added.

## Audio and streaming: exploratory evidence

These audio runs used Universal, actual PCM and the existing DAC clock, a 120 Hz requested update target, one
three-second run per case, and a 1280 × 650 window. Their size and duration differ from the primary rendering-only
matrix.

| Case             | Qt / VisPy draw Hz | Qt / VisPy cursor-gap P95 ms | Qt / VisPy callback P95 ms | Qt / VisPy measured underruns |
| ---------------- | -----------------: | ---------------------------: | -------------------------: | ----------------------------: |
| 10 s following   |        19.0 / 43.7 |                52.56 / 23.18 |              0.085 / 0.126 |                         0 / 0 |
| Adaptive seeking |        16.1 / 32.9 |                64.43 / 34.55 |              17.68 / 13.31 |                         2 / 0 |

Each VisPy audio run also had one warm-up underrun. Cursor gaps are recording milliseconds against a fresh DAC-clock
position at paint completion. These short checks favor VisPy's software cursor freshness but cannot establish absence of
audio regressions. No physical audio-to-display timing or human listening alignment was measured.

The three-second streaming check achieved 54.4 scene draws/s at a requested 120 Hz, approximately 92% of one CPU core,
0.76 MB/s of buffer payloads, and about 641 MB process RSS. Asynchronous GPU queries reported median/P95 scene elapsed
intervals of 2.52/5.80 ms; those intervals can include submission gaps. No synchronization wait was added.

Ingestion was independent at 100 Hz with a preallocated 480,000-sample CPU history and partial GPU writes. The check
consumed 142,560 samples without overwritten pending data, with maximum ingestion lateness around 10 ms. Three seconds
does not fill the ten-second history or establish long-run stability. Separate deterministic payload checks exercise
CPU/GPU ring wraparound, overwritten-history recovery and adjacent-sample continuity. Full-history upload comparisons
were not collected.

A scrolling spectrogram remains a plausible follow-up using offset texture updates supported by
[VisPy's texture API](https://vispy.org/api/vispy.gloo.texture.html). This is an inference from the API, not an
implemented or benchmarked result. Runtime DSP, capture and LED output were not changed.

## Conditions, coverage and artifacts

Python 3.15.2, VisPy 0.17.0, PySide6 6.12.0, PyQtGraph 0.14.0; Ubuntu/Wayland, Intel i7-8550U and hardware Mesa Intel
UHD 620 OpenGL 4.6, Mesa 25.2.8. The panel reports 60.011 Hz with 2× scaling. Primary windows were fixed at 1180 × 620
logical pixels. VisPy requested four-sample multisampling and disabled vsync; Qt retained native peak downsampling,
clipping, segmented lines and device-coordinate caching. No software OpenGL results support this recommendation.

The application-level update timer is independent of the editor's 16 ms timer. Actual paints/draws are counted
separately; Qt component paints and duplicate events are explicitly accounted for. Neither swapping nor compositor
presentation was measured. GPU timer runs are separate from the primary comparison. The desktop session was not an
isolated graphics lab.

No Trust, the complete rendering-only 120 Hz matrix, full-track following, three-repeat interaction/idle scenarios,
alternative waveform comparisons and long streaming runs remain unmeasured. The CLI supports them. Measurements record
their source hashes; subsequent interactive startup, input-handling and error-cleanup fixes were not rebenchmarked.

- [Repeated raw measurements](vispy_comparison.json), including the user-requested stop and flagged wide-view cases.
- [Audio checks](vispy_integrated.json), [streaming/GPU check](vispy_streaming.json), and
  [earlier profiling attribution](vispy_profiling.json).
- [Structured summary](vispy_summary.json), [complete tables with evidence labels](vispy_summary.md), and
  [input manifest](vispy_input_manifest.json).
- [Invocation and measurement methodology](../../review_performance/README.md).

All completed workers verified that protected proposal/draft bytes remained unchanged. Historical Matplotlib and Qt
reports are unchanged. Production annotation rendering and runtime visualization remain unchanged.
