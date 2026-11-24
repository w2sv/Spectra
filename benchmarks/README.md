# Kick detection benchmark

This offline harness runs the production `SpectrumAnalyzer` and
`KickOnsetDetector` against known kick onsets. It imports no capture backend,
rendering, Qt, preview, or output code. It processes successive audio reads as
fast as possible, with fresh analyzer and detector state for each fixture.
Production parameters and defaults live independently in the frozen
`spectra.analysis.KickDetectionConfig` dataclass.

The `dev` dependency group includes the separate `benchmark` group through
uv's `include-group` mechanism. Ordinary `uv run pytest` and `uv run pyright`
therefore work without an extra group flag. `--group benchmark` remains valid.

```bash
# Run every built-in synthetic fixture and print results.
uv run --group benchmark python -m benchmarks.kick_detection

# List fixture identifiers and categories.
uv run --group benchmark python -m benchmarks.kick_detection --list

# Select fixtures, or categories, and export an experiment.
uv run --group benchmark python -m benchmarks.kick_detection \
  --fixture four_on_floor --fixture irregular --json benchmarks/results/experiment.json
uv run --group benchmark python -m benchmarks.kick_detection \
  --category negative --category rumble

# Override existing detector parameters; application defaults stay unchanged.
uv run --group benchmark python -m benchmarks.kick_detection \
  --set refractory_seconds=0.15 --set threshold_multiplier=2.5

# Add externally annotated WAVs without downloading any dataset.
uv run --group benchmark python -m benchmarks.kick_detection \
  --manifest benchmarks/datasets/annotations.json --category external
```

`--fixture`, `--category`, `--dataset`, `--partition`, `--manifest`, and `--set` are repeatable. Values within
each selection option are combined; when both fixture and category filters are
given, a fixture must satisfy both. No filters retains the original synthetic-only run.
Unknown names, duplicate identifiers, and empty selections are errors. `--list`
lists all available fixtures and exits. `--json PATH` creates parent directories
and replaces an existing report at that path. `--help` shows the CLI options.

`--set NAME=VALUE` accepts these seven floating-point parameters:

| Field | Production default | Meaning |
| --- | ---: | --- |
| `min_frequency` | 40 | Lower bass-selection boundary, Hz |
| `max_frequency` | 160 | Upper bass-selection boundary, Hz |
| `min_energy` | 0.000001 | Minimum selected mean-square bass power |
| `min_flux` | 0.35 | Minimum summed positive log-power change |
| `threshold_multiplier` | 2 | Adaptive deviation multiplier |
| `adaptation_seconds` | 1 | Causal history time constant |
| `refractory_seconds` | 0.2 | Minimum interval between emitted events |

### Frequency boundaries and optimization

The analyzer publishes 24 integrated logarithmic bands from **40 to 16,000 Hz**,
not individual FFT bins. The first boundaries are approximately 40.000, 51.343,
65.902, 84.590, 108.577, 139.366, 178.885, and 229.612 Hz. For analyzer band
`[a, b]`, detector range `[low, high]` selects the fraction
`max(0, min(b, high) - max(a, low)) / (b - a)` of that band's existing power.
This is fractional weighting, not rounding to whole selected bands. The default
40–160 Hz range fully includes the first five bands and partly weights the
139.366–178.885 Hz band. No analyzer power below 40 Hz is available.

Ranges are functionally equivalent when their complete band-weight vectors
are identical. For a fixed upper bound, every valid lower bound at or below
40 Hz is equivalent: 0–160, 20–160, and 40–160 Hz produce the same weights.
Similarly, upper bounds at or above 16,000 Hz are equivalent for a fixed lower
bound. Within a single analyzer band, two entirely contained intervals of the
same width are equivalent; e.g. 52–62 and 54–64 Hz both select 10 Hz of the
51.343–65.902 Hz band. Ranges with no analyzer overlap are rejected.

Different upper bounds within a band generally **are not equivalent**: they
change its fractional power weight. That change can affect both the energy
gate and nonlinear log-power novelty. Fractional weighting cannot recover
where energy actually lay within the original band. Future optimization should
avoid redundant bounds below 40 Hz, account for identical weight vectors, and
retain these resolution limits; whole-band categorical selection would change
the existing detector and is not assumed here.

## Audio and timing

The harness shares the live source's default 48,000 Hz sample rate and 1,024
frame reads. It uses the production analyzer's 4,096 frame FFT, 512 frame hop,
arithmetic stereo-to-mono mix, periodic Hann window, power scaling, and
logarithmic bands. The report includes the actual band boundaries. Mono files
are duplicated into left/right channels; stereo retains its channel values and
is mixed by the analyzer. More than two channels are rejected. Integer WAVs
are converted to float32 full-scale PCM; floating-point WAVs retain their
amplitudes. There is no normalization, clipping, or fixture-specific gain.

Other sample rates use a causal polyphase FIR (`upfirdn`), with integer factors
reduced by their greatest common divisor, a Hamming window, cutoff
`1/max(up, down)` relative to the upsampled Nyquist frequency, and
`20*max(up, down)+1` taps. The filter's group delay is retained and reported;
annotations are never shifted to compensate. Output sample `m` depends only
on input samples through `floor(m*down/up)`. The resampler therefore exposes no
future audio. No resampler tail is flushed beyond the fixture duration.

The detector receives the same audio-relative **window-center** timestamps as
the live worker, preserving refractory and adaptive-history calculations.
Reported detection timestamps instead mark the **end of the capture read that
contains the completed window**. Thus they include the second half of the FFT
window and any read buffering. Two spectra from the same read can have the
same availability time. Only actual emitted `KickEvent`s are recorded.
Incomplete final FFT windows are discarded, with no zero padding or fabricated
tail events. A final short read becomes available at EOF.

All event times come from sample positions. No sleeps, playback clock, or wall
clock drives analysis. Timing estimates exclude OS/backend scheduling, DSP
execution time, presentation, and hardware transport delays; these cannot be
reproduced by an offline signal benchmark. Per-fixture `informational`
processing durations include resampling and DSP execution and are variable
throughput measurements, separate from accuracy and event latency.

## Fixtures

`AudioFixture` holds an identifier, dataset (`synthetic` or `techno`), musical
category, partition (`regression`, `development`, or `evaluation`), float32 samples or WAV path,
sample rate, annotated onset seconds, and generation parameters. Onsets must
be strictly increasing and within the audio. In-memory audio supports mono
`(frames,)` and left/right stereo `(frames, 2)`.

All 15 built-in fixtures belong to the synthetic dataset and regression
partition. They are eight seconds long. Kicks integrate an exponential
pitch sweep from 3.5 times the final frequency toward that frequency, with a
25 ms pitch-decay constant. Their amplitude has a 2 ms attack and exponential
decay, plus a short seeded noise beater. Each kick lasts eight amplitude-decay
constants, truncated at the fixture boundary. These are synthesized bass-drum
models, rather than impulses or recorded instruments. Ground truth is the
exact insertion sample index divided by the source sample rate; the seed is
`20261124` (rumble noise uses seed + 1).

| Fixtures | Category | Scenario |
| --- | --- | --- |
| `four_on_floor` | `clean` | 120 BPM, 55 Hz, amplitude 0.6, 120 ms decay |
| `slow`, `fast` | `tempo` | 84 and 174 BPM |
| `low_pitch`, `mid_pitch`, `high_pitch` | `frequency` | Final pitch 28, 95, and 210 Hz |
| `quiet`, `loud` | `amplitude` | Amplitudes 0.04 and 0.9 |
| `long_decay` | `decay` | 500 ms amplitude decay, overlapping tails |
| `with_sub` | `sub_bass` | Kicks over sustained 32 Hz bass, amplitude 0.3 |
| `with_rumble` | `rumble` | Seeded low-pass noise, 90 ms delayed decaying rumble tails |
| `sustained_bass` | `negative` | Sustained 32 Hz bass, no annotated kicks |
| `offbeat_bass` | `negative` | Repeated 49 Hz bass notes on offbeats, no kicks |
| `silence_then_kick` | `silence` | Four seconds of silence before one kick |
| `irregular` | `irregular` | Uneven spacing, including intervals below 200 ms |

Except for tempo variants, regular kicks start at 0.5 s and repeat every 0.5 s
until 7 s. Other parameters retain the clean fixture's values unless noted.
Pitch, level, and tempo variations were chosen as synthesis scenarios, without
restricting fixtures to the detector's frequency range or thresholds.
Reports store generation parameters, insertion positions, normalized source PCM
hashes, dependency versions, and implementation hashes, including fixture code.
Use the recorded generator version and source revision to reconstruct a recipe.

An external manifest is a JSON list, with relative WAV paths resolved beside
the manifest. Its declared rate must match the WAV header. For example:

```json
[
  {
    "identifier": "track_excerpt_01",
    "category": "external",
    "audio": "track_excerpt_01.wav",
    "sample_rate": 44100,
    "kick_onsets": [0.5, 1.0, 1.5]
  }
]
```

Generic `--manifest` fixtures retain the original regression-fixture behavior;
the fixed techno collection uses the catalog and reviewed manifest described
below. Local datasets stay ignored except the techno catalog and reviewed
annotations. The result directory retains the original production baseline,
the intentional synthetic annotation baseline, and the pilot summary; ad hoc
results stay ignored. Copyrighted audio, proposals, drafts, intermediate
analysis, and plots are never tracked. No recordings are downloaded by these tools.

## Metrics and interpretation

Each tolerance (±25, ±50, and ±100 ms) is evaluated independently using
`mir_eval.util.match_events`, a maximum-cardinality one-to-one matching with
inclusive tolerance boundaries. A reference or detection can appear in only
one matched pair; duplicate detections remain false positives. Matching is
not an optimization for the smallest total timing error. Timing summaries use
the pairs returned by mir_eval at that tolerance.

- TP = matched pairs; FP = unmatched detections; FN = unmatched annotations.
- Precision = TP/(TP+FP); recall = TP/(TP+FN).
- F1 = 2TP/(2TP+FP+FN).
- FP/min = FP divided by total audio duration in minutes, including silence.
- Signed error = detection availability minus annotated onset; positive is late.
- Median signed error, median absolute error, and P95 absolute error use only
  matched pairs. P95 uses NumPy's default linear percentile interpolation.

Zero denominators produce `null` in JSON and `n/a` in the terminal. Thus silence
with no detections has undefined precision, recall, and F1, rather than a perfect
score. A missed-only fixture has undefined precision, zero recall and F1;
a false-positive-only fixture has zero precision and F1, undefined recall.
Timing summaries are undefined when there are no matched pairs. FP/min remains
defined for every nonempty fixture, including negatives.

Aggregate precision, recall, and F1 use pooled TP/FP/FN counts, and FP/min uses
summed fixture duration. Aggregate timing statistics pool all matched errors,
rather than averaging fixture percentiles. The terminal prints aggregate
results for all three tolerances and per-fixture results at the primary ±50 ms
window; JSON contains per-fixture results at all tolerances, ground truth,
detections with strengths and original window-center times, DSP and detector
configuration, schema version, and Git commit/dirty status when available.
Apart from explicitly informational processing time and environment/Git
provenance, repeated runs with the same inputs and implementation are
deterministic. Source hashes identify code even when recording a dirty tree.
Schema version 2 adds dataset, category-within-dataset, and partition aggregates.
Every aggregate includes fixture count, total duration, and ground-truth kick
count. Category keys are nested under dataset, keeping unrelated categories
separate. The original schema-1 `baseline.json` is preserved byte-for-byte;
new synthetic runs have identical PCM, detections, and numerical metrics.

The initial default baseline at ±50 ms is **150 TP, 29 FP, 19 FN**, precision
**0.838**, recall **0.888**, F1 **0.862**, and **14.5 FP/min**. Matched median
signed/absolute error is **25.3 ms**, P95 absolute error **38.7 ms**. The ±25 ms
F1 is **0.420**; ±100 ms F1 is **0.868**. These figures describe this small
synthetic collection and do not establish real-world music accuracy.

Observed weaknesses are instrument ambiguity (all 14 offbeat bass notes and
the sustained bass attack trigger), rumble interference (12 FP and 3 FN),
complete failure on the 210 Hz final-pitch fixture (14 FN), and misses on close
irregular kicks (2 FN). Sub-bass adds 2 FP. The large drop at ±25 ms also shows
the importance of reporting availability rather than backdated window centers.
No algorithm or default parameter changes are part of this milestone.

To produce a current synthetic report for comparison:

```bash
uv run --group benchmark python -m benchmarks.kick_detection \
  --json benchmarks/results/current-synthetic.json
```

Repository verification (benchmark dependencies are included in dev):

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## Fixed techno collection and partitions

[`datasets/techno/tracks.json`](datasets/techno/tracks.json) is the single source
of truth for track identity, artist, title, supplied BPM/category, source URL,
partition, expected filename, acquired WAV SHA-256, rate, and duration.
Supplied BPMs and categories are initial metadata, not independently verified
acoustic measurements. Missing acquisition fields remain `null`.

Development tracks: `tennents` (120, minimal), `power_train` (135, hardgroove),
`losing_control` (140, melodic), `birds_in_the_basement` (145, hypnotic),
`universal` (147, acid), `a_grim_premonition` (148, industrial), `no_trust`
(150, distorted), and `the_question` (unknown BPM, schranz).

Held-out evaluation tracks: `rough` (150, reverse_bass), `hardcore_candy`
(151, hard_acid), and `untold` (170, acidcore). No replacements or partition
changes are made. Evaluation recordings are excluded from the annotation pilot.

```bash
uv run python -m benchmarks.prepare_tracks --source ~/Downloads/spectra-songs
uv run python -m benchmarks.prepare_tracks --verify

uv run python -m benchmarks.kick_detection --dataset synthetic
uv run python -m benchmarks.kick_detection --dataset techno --partition development
uv run python -m benchmarks.kick_detection --dataset techno --fixture universal
# Held-out benchmarking requires this explicit partition selection:
uv run python -m benchmarks.kick_detection --dataset techno --partition evaluation
```

Preparation compares normalized artist/title components, tolerating case,
punctuation, track numbering, download suffixes, and album prefixes. For
`HardtraX - A Grim Premonition - 02 No Trust.wav`, the final component is the
song title, preventing an album-title false match. Ambiguous matches fail.
Available development WAVs are validated through the shared WAV loader and
copied byte-for-byte to `datasets/techno/audio/<identifier>.wav`. Complete
originals remain untouched: no normalization, trimming, conversion, or
resampling happens during preparation. Existing matching checksums are skipped;
catalog or destination mismatches fail before copying. Missing recordings are
reported with source URLs and do not prevent available tracks being prepared.
There is no extraction, downloading, or acquisition automation.

Only fully reviewed records from `datasets/techno/annotations.json` enter the
techno benchmark. Missing audio/reviewed annotations are reported on explicit
techno selection; an empty selection exits with an error rather than scoring
proposals. Audio checksums are verified before processing. Without explicit
`--partition evaluation`, held-out recordings cannot run, even when selected
by identifier. The benchmark still uses the original causal production path;
annotation resampling and evidence never participate in detector evaluation.

## Independent annotation algorithm

`annotation_analysis.py` contains offline numerical analysis; `annotate.py`
handles proposals, validation, and finalization; `annotation_review.py` owns
Matplotlib interaction. None uses `SpectrumAnalyzer`, `KickOnsetDetector`,
`KickDetectionConfig`, production events, or production thresholds.

1. Load the complete WAV through the shared loader, arithmetic-mean stereo
   into mono, and resample to 48 kHz with `scipy.signal.resample_poly` if needed.
   This conversion may be noncausal and delay-compensated. No amplitude
   normalization is applied.
2. Compute centered 2,048-sample periodic Hann spectra every 240 samples (5 ms).
   Frame zero is centered on recording sample zero, with zero padding at the
   boundaries; timestamp `k` is `k*240/48000`, in the original time origin.
   Temporary FFT storage is bounded; the whole recording supplies evidence.
3. Compute one-sided mean-square bin powers, using fractional bin-cell
   overlaps for 20–60, 60–150, 150–300, and 300–3000 Hz band integration.
4. Calculate `log(power + 1e-12)` and positive successive differences. The
   difference is assigned to the later frame center; initial novelty is zero.
   Divide each band by its 95th percentile among **positive** differences,
   returning zeros if that denominator is zero. Values are not clipped to one.
5. Mean the three bass bands; retain 300–3000 Hz as separate broadband
   evidence. Gaussian-smooth both with sigma one frame (5 ms). The analysis
   object also preserves band powers, per-band novelty, and unsmoothed evidence.

With the catalog BPM (or `--bpm` override), search phase over `[0, 60/BPM)` at
5 ms resolution. Score each candidate by the mean maximum bass evidence within
±20 ms of every estimated beat across the **complete recording**. Ties select
the earliest phase. `--phase` overrides this search and must lie within one
beat interval. Unknown catalog BPM requires `--bpm`; no tempo is estimated.

Independently repeat phase search in consecutive 30 s sections (including the
short final section). Report `(local-global+interval/2) % interval - interval/2`;
flag magnitudes strictly greater than 30 ms. Evidence is restricted to each
section. The grid is never automatically warped or corrected. Ambiguous bass
rhythms, breakdowns, and near-silence can shift the preferred phase even with
accurate tempo; flags are review cues, not proof of tempo drift.

`scipy.signal.find_peaks` finds bass-evidence maxima with 80 ms minimum
separation and 0.25 minimum prominence. For each chronological beat, associate
the strongest available candidate within ±50 ms, using each candidate at most
once and retaining its acoustic timestamp. Unmatched beats are unsupported;
unmatched peaks remain off-grid candidates. Every candidate records timestamp,
bass evidence, prominence, broadband evidence, signed distance to the nearest
estimated beat, and alignment. Evidence is heuristic, not probability. Beat
alignment cannot distinguish a bass note from a kick.

## Proposal generation and interactive review

```bash
uv run python -m benchmarks.annotate --track universal
uv run python -m benchmarks.annotate --track universal --plot --start 60 --duration 10
uv run python -m benchmarks.annotate --track universal --bpm 147 --phase 0.125 --regenerate
# Save automatic diagnostic images without opening a window:
uv run python -m benchmarks.annotate --track universal --diagnostics
```

Proposals are written to `datasets/techno/proposals/<identifier>.json`, with
audio checksum, algorithm version and parameters, BPM and phase, expanded beat
positions, unsupported beats, candidates, and section-validation results.
Existing proposals are reused; replacing them (including applying overrides)
requires `--regenerate`. Regeneration never alters drafts or reviewed records.

The figure shows waveform, 20–500 Hz log-power spectrogram, and bass/broadband
evidence on a shared original-recording time axis. Gray dashed lines mark
estimated beats; green triangles are aligned proposals; orange triangles are
off-grid proposals; red crosses mark unsupported beats; blue circles mark
manually added kicks. The same markers appear on all three plots. Faint
original proposals remain for comparison after removal. The title shows track,
BPM, current range, and unsaved state. A default interval is 10 s.

- Use Matplotlib's toolbar to zoom and pan normally.
- Right/left arrows advance/back up by the current displayed interval.
- Shift + left click adds a kick at the exact selected time (blue).
- Ctrl + left click removes the nearest editable proposal or manually added
  kick within 50 ms. To correct a proposal's timing, remove it then add the
  corrected timestamp; adding does not silently delete a neighboring event.
- Ctrl + S saves the editable draft, rather than opening Matplotlib's image
  save dialog. Unsaved edits are not saved automatically on close.

On first opening, the editable draft contains all acoustic candidates as
**unreviewed suggestions**, including off-grid peaks. Existing drafts are
loaded unchanged and never reseeded on opening. Drafts live separately under
ignored `datasets/techno/drafts/`; proposals and reviewed annotations are
separate files. `confirmed_bpm` can be edited in the draft JSON before
finalization. Blue points distinguish manual additions; remaining suggestions
are accepted only through explicit full-recording review at finalization.

Listen using an external player while inspecting the same original timeline,
for example (if mpv is installed):

```bash
mpv --start=60 --length=10 benchmarks/datasets/techno/audio/universal.wav
```

There is no internal playback engine or synchronization with the player.

## Finalizing ground truth

```bash
uv run python -m benchmarks.annotate --track universal --finalize
```

Finalization loads the saved draft, verifies the actual audio checksum and
catalog rate/duration, checks finite, sorted, unique timestamps strictly inside
the recording duration, and requires a positive confirmed BPM. It then asks
you to type `REVIEWED universal ENTIRE RECORDING`, explicitly attesting that
the entire recording was reviewed, including sections with no proposals, and
every retained timestamp is a genuine kick. No displayed-section counter or
accepted-candidate count can satisfy this requirement. Declining leaves the
tracked manifest unchanged.

The reviewed record stores track identifier, checksum, annotation algorithm
version, explicit confirmed kick timestamps, confirmed BPM, and `reviewed`
status. There is no compact regular-sequence representation or partial-track
scoring. Initially `annotations.json` is an empty list: this milestone does
not fabricate ground truth or claim real-world precision/recall.

## Independent synthetic annotation validation

```bash
uv run python -m benchmarks.annotate --synthetic \
  --json benchmarks/results/annotation_synthetic_baseline.json
```

The recorded [annotation sanity check](results/annotation_synthetic_baseline.json)
evaluates all 15 unchanged synthetic fixtures independently of production
detection. Beat association uses tempo metadata only for `four_on_floor`,
`slow`, and `fast` (45 annotated kicks); all other fixtures evaluate acoustic
peaks without a grid. Reports separately retain grid-aligned and all-candidate
metrics at all three tolerance windows. Undefined grid metrics remain `null`
for fixtures without tempo metadata.

At ±50 ms, all candidates yield **169 TP, 78 FP, 0 FN**, precision **0.684**,
recall **1.000**, F1 **0.8125**. Grid-aligned proposals on the three tempo
fixtures yield **45 TP, 1 FP, 0 FN**, F1 **0.989**. Negative fixtures produce
one false candidate on sustained bass and fourteen on offbeat bass. Sub-bass
adds fourteen extra candidates and rumble adds forty-three.

Matched all-candidate median signed error is **−15 ms**, median absolute error
**15 ms**, P95 absolute error **15 ms**. Centered windows see the upcoming
transient up to half a window (21.3 ms) before their center reaches the physical
insertion. Positive log-power novelty is strongest on that early rise, so its
peak need not coincide with the maximum windowed power at the physical onset.
Impulse alignment and resampling tests confirm the frame timeline is correct.
No fixture-specific calibration or global timestamp correction is applied.
This evidence-timing bias and false positives must be considered during human
review; synthetic success does not establish accuracy on real music.

## Four-recording pilot

The original four downloads were prepared unchanged. All are **44,100 Hz**.
The [pilot report](results/annotation_pilot.json) records these unreviewed results:

| Recording | Duration (s) | Catalog BPM | Phase (s) | Beats | Aligned | Off-grid | Unsupported | Analysis time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `power_train` | 348.444 | 135 | 0.025 | 784 | 760 | 1212 | 24 | 3.43 |
| `universal` | 317.000 | 147 | 0.000 | 777 | 708 | 1063 | 69 | 3.77 |
| `a_grim_premonition` | 377.027 | 148 | 0.010 | 930 | 834 | 963 | 96 | 3.78 |
| `no_trust` | 378.800 | 150 | 0.010 | 947 | 835 | 1117 | 112 | 4.84 |

Analysis times include integrity verification, loading, resampling, evidence,
and proposal generation; they exclude plotting and are informational.
Local phase flags (original-recording seconds):

- `power_train`: 150–180 (+210 ms).
- `universal`: 60–90 (−203.2 ms).
- `a_grim_premonition`: 120–150 (+175 ms).
- `no_trust`: 0–30 (−35 ms), 60–90 (+180 ms), 90–120 (+185 ms),
  120–150 (+185 ms), 150–180 (+185 ms), 180–210 (+175 ms),
  210–240 (−195 ms), 270–300 (+190 ms), 330–360 (−35 ms),
  360–378.8 (+185 ms).

Diagnostics scan consecutive 10 s intervals independently for two explicit
criteria: a regular supported-grid interval has at least eight estimated beats
and at least 90% supported by associated candidates; a challenging interval has
at least five off-grid candidates or at least three unsupported beats. These
are proposal-based heuristics, not verified descriptions of genuine kick
content. The first interval satisfying each criterion is 0–10 s for every
pilot track, so each track's regular and challenging images show the same
interval. This also demonstrates that strong beat support can coexist with
substantial off-grid activity. If no interval qualifies, the tool reports it.
Eight local images are under `datasets/techno/proposals/diagnostics/`, named
`<identifier>-regular.png` and `<identifier>-challenging.png`.

All four proposal files are ready for interactive correction. No recording
was finalized, no real-world accuracy is claimed, and no held-out evaluation
recording was analyzed. Human review and the subsequent real-world production
baseline belong to the next milestone.
