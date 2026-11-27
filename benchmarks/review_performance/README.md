# Timeline rendering experiment

This isolated PySide6 application compares the current annotation `Timeline` with an embedded VisPy `SceneCanvas`. It
reads existing recordings, cached proposals and saved corrections. Interactive changes remain in a copied draft in
memory; there is no save, finalization, reclassification or catalog operation.

```bash
# Default: both recordings, both renderers, ten scenarios, 60/120 Hz, three 30-second runs each.
uv run --group visualization-experiment python -m benchmarks.review_performance

# Interactive prototype; Space plays actual PCM through the existing DAC-synchronized player.
uv run --group visualization-experiment python -m benchmarks.review_performance \
  --track universal --backend vispy --interactive --mode integrated

# Current PyQtGraph timeline with the same experimental transport controls.
uv run --group visualization-experiment python -m benchmarks.review_performance \
  --track no_trust --backend pyqtgraph --interactive --mode integrated

# Integrated measurements; rendering-only is the default mode.
uv run --group visualization-experiment python -m benchmarks.review_performance \
  --mode integrated --scenario follow_10 zoom seek \
  --output benchmarks/results/review_performance/vispy_integrated.json

# Alternative waveform strategies. Repeat separately with visible, resident and lod.
uv run --group visualization-experiment python -m benchmarks.review_performance \
  --backend vispy --waveform visible --target 120 \
  --scenario follow_2.5 follow_10 follow_40 follow_120 follow_full zoom \
  --output benchmarks/results/review_performance/vispy_visible.json

# Independent 10 ms ingestion blocks and partial writes to a bounded GPU ring.
uv run --group visualization-experiment python -m benchmarks.review_performance \
  --backend vispy --scenario stream \
  --output benchmarks/results/review_performance/vispy_streaming.json

# Separate diagnostics: do not mix these with unprofiled comparison runs.
uv run --group visualization-experiment python -m benchmarks.review_performance \
  --track universal --backend vispy --target 120 --scenario follow_120 --profile \
  --output benchmarks/results/review_performance/vispy_profile.json
uv run --group visualization-experiment python -m benchmarks.review_performance \
  --track universal --backend vispy --target 120 --scenario follow_120 --gpu-timing \
  --output benchmarks/results/review_performance/vispy_gpu.json

# Rebuild comparative tables, structured summaries and empirical CDF figures.
uv run --group visualization-experiment python -m benchmarks.review_performance.report \
  benchmarks/results/review_performance/vispy_comparison.json \
  benchmarks/results/review_performance/vispy_integrated.json
```

Use `--help` for scenario names, 30/60/90/120 Hz targets, dimensions, repetitions and warm-up duration. Shorten
`--duration` for smoke checks; comparative runs use 30 seconds with three repetitions. The default fixed window is 1180
× 620 logical pixels. Interactive windows remain resizable. A complete default matrix measures for two hours, plus
initialization and warm-up. Each run uses a fresh child process; execution order alternates between repetitions. The
parent prepares each recording once and passes a private temporary pickle to its own workers. Do not supply an untrusted
pickle to the internal worker option.

Interactive controls are Space (play/pause), F (follow), wheel (smooth zoom), horizontal drag (pan), Left/Right (seek
50% of the displayed duration), Shift+Left/Right (10%), Backspace (−10 seconds), Shift-click (change one marker),
right-click (inspect), G (primary/alternative grid), D (diagnostic outlines), and Q/Escape (close). Playback retains the
current 0.25–2.0× adaptive speed bounds and 40% following anchor. Rendering-only interactive mode uses a software clock;
choose `--mode integrated` to hear PCM. There is no persistence.

## Waveform and streaming strategies

- `visible` computes peak-preserving min/max geometry from the visible PCM range for each changed view. Close zoom uses
  the original PCM samples.
- `resident` retains the entire raw PCM geometry in a GPU buffer, using viewport transforms and clipping. It
  deliberately tests the cost of submitting offscreen vertices; it is not the default.
- `lod` precomputes extrema at powers-of-four resolutions. Their original sample times and chronological order are
  preserved, including incomplete final buckets. It selects the largest bucket no wider than one physical pixel and
  switches to visible original PCM at inspection zoom. Coarse levels remain resident; following changes camera
  transforms. At raw inspection zoom, local coordinates avoid quantizing absolute track times to float32.

Static evidence, batched markers, dashed beat segments and waveform levels remain in their visuals between changes. The
stock-axis diagnostic is selectable with `--axes stock`; the default cached axes retain tick glyph geometry between
changes of tick values. `--axes none` is a diagnostic, never a fidelity-equivalent baseline.

Streaming ingestion runs independently at 100 Hz, overwriting a preallocated ten-second CPU ring. Rendering copies only
new samples, or at most the bounded history after falling behind. A fixed GPU buffer receives corresponding dirty
regions; paired line segments prevent a false line across the circular boundary. Recorded kicks remain an immutable
indexed event population. `--stream-full` uses identical geometry and visuals but uploads the whole buffer as an
explicit transfer-cost diagnostic. Ingestion lateness, overwritten samples and resident memory are recorded.

## Measurement interpretation

The shared script starts at 60 seconds, resets transport between scenarios, and integrates the rendering-only clock on a
fixed 1 ms lattice. Inputs occur every 250 ms; seeking occurs every 500 ms. Both renderers sample the same requested
clock trajectory and scripts. Missed update slots are skipped, not queued. Integrated runs use actual PCM, the original
variable-rate resampler and PortAudio DAC clock, so audio and rendering scheduling may cause their trajectories to
differ.

A precise timer schedules absolute deadlines rather than using the editor's 16 ms timer. Update cadence and actual
paint/draw cadence are recorded separately. Qt's waveform and evidence paints are grouped when both have painted since
the last complete frame; duplicate paints within one update are excluded. Per-component cadence is also retained. This
conservative composite metric can be lower than waveform-only cadence. VisPy measures completion of its scene draw.
Neither metric establishes swap, compositor presentation or display FPS.

Intervals include median, P95, P99, maximum, raw samples and histograms. Deadline misses allow 5% timer jitter;
unsatisfied deadline slots are also counted. Idle intentionally avoids drawing, so its missing draw slots are not
failures. Input latency ends at the first paint; scheduled latency additionally includes Qt dispatch delay where
recorded. The `snapshot_to_paint_ms` measurement includes preparation before the audio clock snapshot. The independent
`cursor_gap_recording_ms` compares the rendered position with a fresh DAC-clock position at paint completion; its unit
is recording milliseconds, not physical audio-to-display delay or speed-normalized wall time. Callback timing measures
execution after acquiring the GIL; underruns provide separate evidence of scheduling contention.

Buffer counters measure enqueued payload bytes across all VisPy buffers. They do not measure driver allocation, physical
GPU residency, bus traffic or texture transfers. PyQtGraph buffer-transfer metrics are unavailable; its native waveform
geometry regeneration is counted separately. Initialization includes widget/player construction; waveform preprocessing
is reported separately, and initial/warm-up uploads are distinguished from measured uploads. Process CPU uses 100% for
one fully occupied core. RSS is sampled every 500 ms; the lifetime peak also includes initialization. GPU utilization
and physical memory remain unavailable when no reliable per-process counter is accessible.

`--gpu-timing` uses eight asynchronous OpenGL timer queries and skips sampling when all are pending. It never waits for
results. These separately marked runs measure the GL elapsed interval around scene submission, which can include gaps
between commands; they exclude swapping and compositor presentation. CPU profiling writes adjacent `.prof` files and
also perturbs timing, so neither diagnostic belongs in the primary comparison.

Each JSON records revision, source hashes, dependency/Python versions, CPU, display scaling/refresh, actual dimensions,
OpenGL vendor/renderer/driver, renderer settings, recording identity and protected-file integrity checks. Software
OpenGL results must be identified separately and cannot support hardware-acceleration claims. Historical Matplotlib and
Qt JSON reports are preserved unchanged. See the [retained results](../results/review_performance/README.md) for
findings and limitations.
