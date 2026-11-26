# Review performance observations

- [Matplotlib baseline](matplotlib_baseline.json) preserves retired-renderer headless operation timings and a desktop
  smoke check. It is historical evidence, not a measure of the current Qt review or its achieved frame rate.
- [Qt rendering comparison](qt_rendering_comparison.json) preserves paired Universal desktop measurements of actual
  paint intervals/costs, update costs, process CPU, underruns and software clock-to-cursor gaps. It also records the
  separate desktop interaction check and original measurement conditions.

The Qt comparison found ten-second waveform paint P95 31.7 → 26.1 ms and P99 53.1 → 27.9 ms. A unified DAC snapshot
reduced following delay. Wide viewports still incurred substantial native curve/aggregation work; consistent 60 Hz and
P95 below 20 ms were not achieved. Qt can paint more than once per timer update, so timer rates and paint counts are not
display FPS. CPU reduction was not shown.

Precomputed multiresolution min/max waveform prototypes did not reliably improve end-to-end desktop rendering and
sometimes regressed; they were not retained. The current path keeps native peak downsampling and clipping.

The paired retained run reported zero underruns, while the separate interaction run reported three initial underruns and
used recovery. Those observations must not be conflated. Software paint/clock measurements do not establish compositor
presentation times, physical audio-to-display synchronization or human listening alignment. The reports retain their
original flash settings; current controls and flash specifications are documented in the [tool README](../../README.md).
