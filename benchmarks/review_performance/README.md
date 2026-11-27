# Production timeline rendering checks

Short, reproducible checks exercise the production VisPy timeline with existing recordings and copied corrections. They
do not modify annotation proposals, drafts or finalized annotations. Defaults run a single three-second case after a
one-second warm-up; this is a regression tool, not another extensive comparative campaign.

```bash
uv run python -m benchmarks.review_performance --track universal --scenario follow_10
uv run python -m benchmarks.review_performance --track no_trust --scenario follow_120 --target 120 --duration 3
uv run python -m benchmarks.review_performance --scenario seek --mode integrated
uv run python -m benchmarks.review_performance --interactive
```

`--interactive` opens the complete production annotation editor against a temporary draft. Saving there never replaces a
real draft; use the normal annotation CLI for persistent review. `--help` lists selectable scenarios, targets and short
durations. Generated local JSON remains ignored by Git.

Rendering-only uses a repeatable sample-time clock; integrated mode uses the existing PCM player and DAC clock. Actual
scene draws are measured separately from update callbacks. Visibility/exposure changes are recorded to identify
presentation pauses. Buffer counters record enqueued payloads, not physical GPU residency or bus traffic. CPU
submission, GPU execution, swapping and compositor presentation remain distinct; draw cadence is not confirmed display
FPS.

[Results and implementation decision](../results/review_performance/vispy_report.md) preserve the empirical
justification. The [compressed archive](../results/review_performance/vispy_measurements.json.gz) contains the original
per-run JSON records under their source filenames, plus the targeted waveform-strategy and visibility checks. Read it
with Python's standard `gzip` and `json` modules. The previous prototypes, PyQtGraph comparison path and redundant
derived reports were removed after migration. Historical Matplotlib and Qt reports retain their original data.
