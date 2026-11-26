# Kick detection evaluation

The detector consumes every raw `SpectrumResult`, independently of LED settings and GUI refreshes. Defaults remain
40–160 Hz, minimum power `1e-6`, minimum log-power flux `0.35`, adaptive threshold multiplier `2`, one-second
adaptation, and a 200 ms refractory period. Parameters live in the immutable `KickDetectionConfig`, passed to
`KickOnsetDetector`. The annotated offline benchmark and its recorded baseline are documented in
[`benchmarks/README.md`](../benchmarks/README.md).

Positive changes in individual bands can occur while total bass power falls: energy moves between bands during a sound's
release. The detector now also requires summed, frequency-weighted bass power to increase from the previous window. This
guard applies to event eligibility; all flux measurements still update the adaptive background statistics. No thresholds
were retuned.

## Reproducible comparison

The baseline is commit `cd0de17`, before the energy-rise guard. Both versions were evaluated on the complete curated WAV
excerpts, using default parameters and fresh state for each file. These are observed event counts, **not accuracy scores
or ground-truth annotations**. Fewer events do not establish better kick recognition. Changing eligibility can also
shift subsequent detections because of threshold crossings and the refractory period.

| Input                   | Baseline events | With energy-rise guard |
| ----------------------- | --------------: | ---------------------: |
| Synthetic smoke signal  |               8 |                      7 |
| Lyserb                  |              31 |                     32 |
| 160BPM Electronic Loop  |              39 |                     39 |
| Empacotatron            |              28 |                     27 |
| Algate                  |              43 |                     31 |
| Synth Pop full mix      |              28 |                     28 |
| Matching bass-only stem |               6 |                      6 |

The synthetic signal supplies independently known sound boundaries. Its seven remaining events correspond to four loud
bass pulses, the sustained tone's attack, and two quieter pulses. The spurious event at the sustained tone's release
(baseline timestamp approximately 3.477 s) is removed. High-frequency pulses and the final silence produce no events.
These findings motivated two focused regressions: falling power with inter-band redistribution, and a PCM tone's
attack/release through the real FFT analyzer.

Run the current detector using the existing development helper:

```bash
uv run python scripts/audio_test.py synth samples/kick-smoke.wav
uv run python scripts/audio_test.py analyze samples/kick-smoke.wav \
  --output samples/results/kick-smoke.json
uv run python scripts/audio_test.py analyze tests/fixtures/audio \
  --output samples/results/kick-collection.json
```

Choose unused output names; the helper does not overwrite experiments. JSON records input and analysis-source hashes,
timestamps, strengths, and overrides. Repeated collection analysis produced identical reports. Local comparison and
private-capture reports from this evaluation are in ignored `samples/results/kick-evaluation/`. Fixture manifests and
audio are unchanged.

The production `CaptureWorker` was also exercised through private PipeWire playback: smoke, Algate, and the bass-only
stem yielded 7, 32, and 7 events. Polling every 300 ms preserved all events even as continuous spectra were replaced.
These live counts need not match offline counts: monitor startup and capture alignment change FFT window placement.
Workers and playback processes were released, owned sinks removed, and the default output left unchanged.

## Limits

Bass-note attacks can still be mistaken for kicks, as the isolated bass stem demonstrates. Conversely, an overlapping
kick can be missed if other decaying bass masks its increase in total power. The aggregate guard does not classify
instruments. A sustained tone's attack is intentionally still a bass onset.

Window-center timestamps may precede an audible attack by part of the 4,096-frame window; they are audio-relative
measurements, not wall-clock playback times. No listening-based timing assessment or precision/recall claim is made
here. Further tuning needs independently reviewed kick timestamps, especially for quiet kicks, competing percussion, and
bass-heavy passages. Detector output must not be copied into the fixture manifest as reference annotations.
