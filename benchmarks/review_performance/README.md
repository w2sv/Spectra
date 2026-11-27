# Review rendering checks

Short, reproducible rendering checks use existing recordings and copied correction state. They never write annotation
proposals or drafts. The extensive comparative campaign is complete; defaults run one short VisPy case.

```bash
uv run --group visualization-experiment python -m benchmarks.review_performance \
  --track universal --backend vispy --scenario follow_10 --duration 3 --repetitions 1
```

`--help` lists scenarios and update targets. The experimental padded strategy is `--waveform windowed`. Rendering-only
uses a deterministic clock; `--mode integrated` uses the existing PCM player and DAC clock. Interactive corrections
remain in memory.

[Results and implementation decision](../results/review_performance/vispy_report.md) retain the empirical justification.
The compressed archive contains original JSON records under their source filenames plus `targeted_waveform_strategies`
and `targeted_visibility`. It is readable using Python's standard `gzip` and `json` modules.

Instrumentation counts actual scene draws separately from update callbacks. CPU submission, GPU timing and compositor
presentation are distinct; draw cadence is not confirmed display FPS. Buffer counters measure enqueued payloads, not
physical GPU memory. Short diagnostic checks settle implementation choices without establishing statistical guarantees.
