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
Current reports use schema version 3 and additionally declare catalog
completeness, as described below. Older baseline files remain unchanged.

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
techno selection. Incomplete requested collections fail by default, including
explicit development collections and single-track requests. Use
`--allow-partial` to intentionally score the available reviewed subset:

```bash
uv run python -m benchmarks.kick_detection \
  --dataset techno --partition development --allow-partial \
  --json benchmarks/results/development-partial.json
```

Terminal and JSON results explicitly list requested/evaluated track identifiers,
missing audio, missing reviewed annotations, and `complete`. Partial reports
have `result_scope: "partial"`; they are not complete dataset baselines. If no
requested recording is ready, an allowed partial report has zero evaluated
tracks, `benchmark_executed: false`, and undefined accuracy/timing/FP-per-minute
values. It never scores unreviewed proposals. Single-track requests can be
complete independently of the remaining catalog. Corrupt audio still fails
integrity validation, even with `--allow-partial`.

Audio checksums are verified before processing. Without explicit
`--partition evaluation`, held-out recordings cannot run, even when selected
by identifier. The benchmark still uses the original causal production path;
annotation resampling and evidence never participate in detector evaluation.

## Acoustic annotation stage (version 1, retained)

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

## Track-specific template stage (version 2)

The original novelty envelopes, peak generation, and global/local grid searches
remain the high-recall first stage. `annotation_templates.py` classifies those
same candidates afterward, independently of production DSP and kick detection.
No candidates are discarded from proposal files.

Define representative sounds by listening and selecting acoustic **onsets** in
the waveform. Start with 3–5 positive kick references and optionally 0–5 recurring
negative sounds (bass notes, rumble, or synthesizer transients). These are
recommendations: classification works with one valid positive reference and
has no hard reference-count limit. No real-world references are automatically
selected. The checked-in `datasets/techno/templates/<identifier>.json` definitions
initially contain empty positive/negative onset lists, track identity, and audio
SHA-256. Definitions are validated against duration and checksum. Only these
definitions are tracked; extracted audio and feature arrays are not.

For every reference/candidate anchor, use the existing centered Hann STFT with
48 kHz rate, 2,048 FFT samples, 240-sample hops and 5 ms frames. Fractionally
integrate mean-square power into these bands (Hz):

`20–40, 40–60, 60–90, 90–130, 130–180, 180–250, 250–350, 350–500,
500–750, 750–1000, 1000–2000, 2000–3000`.

Extract the inclusive −20 to +160 ms context: **37 frames × 12 bands**. Estimate
each band's background from median power over the 16 frames from −100 to
−25 ms. Unavailable frames, including unavailable background context, have zero
power. Square-root-compress context power, subtract the square root of median
background power, and clamp negative differences to zero. Flatten into 444
values and L2-normalize **once across the entire vector**, preserving relative
band levels and temporal shape while reducing overall-amplitude dependence.
There is no independent per-band normalization. A zero norm is invalid and
never receives an arbitrary similarity or a likely-kick label.

Anchors use their nearest 5 ms frame (at most 2.5 ms quantization). At each
candidate, compare 13 anchors at −30 to +30 ms in 5 ms steps. Cosine similarity
finds the best positive reference and shift; negative similarity is independently
maximized over negative references/shifts. Exact ties prefer the smallest
absolute shift, then the negative shift. Initial fixed thresholds are:

| Classification | Rule |
| --- | --- |
| Likely | Positive ≥ 0.85 and no negative references, or positive ≥ 0.85 and positive ≥ negative + 0.10 |
| Ambiguous | Positive ≥ 0.70 but likely criteria fail |
| Unlikely | Positive < 0.70, or the candidate signature is invalid with valid positive references |
| Unclassified | No valid positive reference signatures exist |

An invalid supplied negative reference does not silently waive the margin
requirement. Reports identify invalid references. These thresholds and context
parameters live in the typed immutable `TemplateParameters`, separate from
`KickDetectionConfig`; the numerical API accepts alternative parameter objects.
The same thresholds apply on and off the grid. Similarities are not calibrated
probabilities, and a repeated non-kick sound can resemble a selected kick.

Each candidate retains `timestamp`/`novelty_timestamp`, refined onset (if valid),
positive/negative scores, the best positive reference's onset as identity,
alignment offset, classification, existing acoustic/grid evidence, and an
`alignment_at_search_boundary` flag. Refined onset = novelty timestamp + best
positive-match offset. Out-of-recording anchors are excluded. A match at ±30 ms
requires timing scrutiny. No unconditional +15 ms shift is applied. Human
references must be actual acoustic onsets: marking an early novelty peak as a
reference would transfer that error. Refinement is not sample-accurate onset
localization. `reference_alignment_checks` compare nearby refined candidates
with user references after classification; these are **reference-fit checks**,
not independent real-world accuracy measurements.

The original grid remains primary. A second grid has phase
`(global_phase + beat_interval/2) % beat_interval`. Each 30 s section retains its
original acoustic phase/discrepancy and reports, for both grids, associated
candidate count, mean positive-template similarity (undefined before references),
and likely-kick count. Association still uses ±50 ms and preserves candidate
identity. This helps distinguish competing bass/kick rhythms from actual drift;
neither phase nor tempo is switched automatically. G changes only the review
overlay and never accepted timestamps.

## Proposal generation and interactive review

```bash
uv run python -m benchmarks.annotate --track universal --regenerate
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
The current proposal algorithm is version 2: the unchanged version-1 acoustic
stage is followed by track-specific template comparison. New reports include
both versions, template parameters and the canonical template-definition hash.

The figure shows waveform, 20–500 Hz log-power spectrogram, and bass/broadband
evidence on a shared original-recording time axis, plus candidate cosine scores.
Likely kicks use circles, ambiguous candidates diamonds, unlikely candidates
crosses, and unclassified candidates triangles. Filled/high markers are
grid-aligned; hollow/low markers are off-grid. Accepted draft events use blue
plus signs, positive templates green stars, and negative templates red squares.
Gray dashed lines show estimated beats; red crosses mark unsupported beats.
State is encoded by shape and position as well as color. Hovering shows original
novelty and refined timestamps, classification, similarity, and the search-edge
warning. The title shows track, BPM, current range, active grid, stale-draft
warnings, and unsaved state. The default interval is 10 s.

| Interaction | Action |
| --- | --- |
| Toolbar zoom/pan | Inspect any section normally |
| Right / Left | Advance / go back by the displayed interval; seek and pause |
| Shift + left click | Add a manual kick at the selected timestamp |
| Ctrl + left click | Remove the nearest accepted draft event within 50 ms |
| K / N | Save a positive / negative reference at the cursor |
| Delete | Remove the nearest reference within 50 ms |
| R | Recompute suggestions; preserve the editable draft |
| A | Accept visible likely kicks |
| U | Remove visible automatic events associated with unlikely candidates |
| C | Clear all accepted draft events in the visible interval |
| G | Toggle primary / half-beat alternative grid; timestamps stay unchanged |
| H | Toggle unlikely-candidate visibility |
| Ctrl + Z | Undo the last edit, including bulk operations and template changes |
| Ctrl + S | Save the draft; does not invoke Matplotlib's image-save dialog |
| Space | Play / pause mpv |
| Home | Seek to the displayed interval's start |
| Alt + left click | Seek to the selected timestamp |

K/N require the cursor within 50 ms of a candidate or existing reference, but
save the **cursor's precise onset**, not the novelty timestamp. Zoom the
waveform and listen before choosing the onset. K/N near an existing reference
replace/adjust it; Delete removes it. Template changes are saved independently
and immediately, including when undone; R is required to apply them to
classifications. The concise shortcut reference is also displayed in the window.

Bulk operations use half-open `[start, end)` intervals. A rejects duplicate
candidate identities and skips suggestions within 30 ms of an existing event.
U acts only on recorded automatic provenance and preserves manual additions.
Legacy drafts without automatic provenance are treated conservatively. C also
removes manual events in the visible interval. Undo retains the last 100 editing
actions in memory; navigation, grid switches, and playback do not change events.
Unsaved draft edits are not automatically saved on close.

Before positive templates exist, new drafts contain **grid-aligned candidates
only**. Other candidates remain visible as unaccepted suggestions. Once positive
templates exist, new drafts contain only likely kicks at refined timestamps;
ambiguous and unlikely candidates remain available. An invalid positive template
never causes fallback acceptance of every grid event. Draft membership is never
human verification.

Drafts record an originating proposal checksum excluding informational runtime.
Existing drafts are never reseeded when reopening or reclassifying. A changed
proposal (or an older draft without provenance) produces an explicit **STALE
DRAFT** warning. Saving preserves that originating checksum rather than silently
claiming the draft was initialized from new suggestions. To intentionally discard
edits and recreate from current proposals:

```bash
uv run python -m benchmarks.annotate --track universal --reset-draft
```

Replacing an existing draft requires typing `RESET universal DRAFT`. Proposals,
drafts, and reviewed annotations are separate; resetting never changes reviewed
records. Drafts remain ignored under `datasets/techno/drafts/`.
`confirmed_bpm` can be edited in draft JSON before finalization.

Playback uses optional **mpv**, started paused with no video or separate window,
no user configuration, and a private temporary Unix IPC socket. It uses mpv's
[documented JSON IPC](https://mpv.io/manual/stable/#json-ipc), observing `time-pos`
and `pause`, and issuing absolute exact seeks. A 50 ms GUI timer uses reported
positions for the cursor on all three plots. Navigation seeks and pauses; polling
pauses/seeks at the interval end (subject to normal IPC/audio buffering latency).
Playback never adjusts annotation timestamps. Closing the figure releases the
process and temporary socket. Install it on Ubuntu with `sudo apt install mpv`.
If unavailable, the review window shows that instruction and all non-playback
annotation features remain available. Noninteractive generation and production
benchmarking do not depend on mpv.

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

## Original synthetic annotation validation (version 1)

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

## Original four-recording pilot (version 1)

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

## Reference-excluded refinement validation (version 2)

```bash
uv run python -m benchmarks.annotate --refined-synthetic \
  --json benchmarks/results/annotation_refinement_baseline.json
```

The recorded [refinement baseline](results/annotation_refinement_baseline.json)
uses the unchanged 15 synthetic signals and fixed thresholds. Synthetic ground
truth supplies first/middle/last unique kick references. All 37 reference kicks
and their one-to-one matched acoustic candidates are excluded from scoring,
leaving **132 non-reference kicks**. Matching references to candidates uses the
largest existing tolerance (100 ms), once before classification; every stage
then evaluates the same candidate identities and non-reference ground truth.
Extra/duplicate candidates remain errors. The one-kick fixture has no remaining
positive event, so it cannot establish classification recall. Negative fixtures
use positive signatures from the clean fixture, a cross-fixture stress test.

Acoustic timestamps are asserted identical to the original annotation baseline
before exclusions. Thus the acoustic row below is the original method on the
same non-reference population, not its earlier full-population score. Audio
duration remains the full 120 s in pooled FP/min; timing errors use matched
pairs. Grid metrics apply only to the three tempo-tagged fixtures (36 remaining
kicks across 24 s). All other rows use all fixtures. Combined likely/ambiguous
events use refined timestamps. Results at ±25, ±50, and ±100 ms are retained.

At the primary ±50 ms tolerance, with **positive references only**:

| Stage | TP / FP / FN | Precision | Recall | F1 | FP/min | Median signed / absolute (ms) | P95 absolute (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original acoustic | 132 / 78 / 0 | 0.629 | 1.000 | 0.772 | 39.0 | −13.21 / 13.21 | 15.00 |
| Grid-aligned (tempo fixtures only) | 36 / 1 / 0 | 0.973 | 1.000 | 0.986 | 2.5 | −12.10 / 12.10 | 15.00 |
| Likely, original times | 129 / 1 / 3 | 0.992 | 0.977 | 0.985 | 0.5 | −14.29 / 14.29 | 15.00 |
| Likely, refined times | 129 / 1 / 3 | 0.992 | 0.977 | 0.985 | 0.5 | 0.00 / 0.00 | 2.06 |
| Likely + ambiguous, refined | 132 / 1 / 0 | 0.992 | 1.000 | 0.996 | 0.5 | 0.00 / 0.00 | 2.10 |

The likely/combined precision, recall, and F1 are the same at all three supplied
tolerances in this run. Timing alignment substantially reduces the centered-
novelty bias without a global correction, but it does not guarantee the same
accuracy on human-selected references or real recordings.

The negative-reference case uses the independently known first offbeat bass
onset (0.75 s) from `offbeat_bass`, excluding that candidate in its source
fixture. Transferring this negative signature to other fixtures is intentionally
a stress test, not the intended track-specific workflow. Its acoustic population
has 77 FP instead of 78. Likely/refined results become **118 TP, 1 FP, 14 FN**,
precision **0.992**, recall **0.894**, F1 **0.940**, FP/min **0.5**, median
signed/absolute error **0 ms**, P95 absolute error **2.08 ms**. All eleven
remaining low-pitch kicks move to ambiguous because of the negative margin.
Likely + ambiguous retains **132 TP, 1 FP, 0 FN**.

Observed limitations:

- Offbeat-bass false positives fall from 14 to zero with positive-only
  classification (13 non-reference candidates in the negative-reference case).
- Sustained-bass and sustained-sub-bass extra candidates are rejected in these
  fixtures, but the test does not prove instrument specificity on real audio.
- Rumble leaves one likely false positive and makes three genuine kicks
  ambiguous; likely-only recall falls to 8/11 on its non-reference population.
- A negative bass signature can also resemble genuine low-pitched kicks,
  causing substantial recall loss. Thresholds were not tuned to remedy this.
- References are synthesized repeats from the same recording; real sounds,
  changing mix context, human onset errors, and different sections can be less
  consistent. Cross-fixture negatives are especially limited evidence.

## Template-selection pilot (version 2)

The four development recordings have refreshed **unclassified** version-2
proposals and empty checked-in template definitions. All **7,492** original
acoustic candidates, including **4,355 off-grid** candidates, retain identical
acoustic timestamps/evidence and global grids. No real reference identities
were invented. Original production and annotation baselines, the version-1
pilot report, existing editable drafts, and reviewed annotations are unchanged.
The [new pilot summary](results/annotation_refinement_pilot.json) includes
classification counts, both-grid section comparisons (scores undefined until
references exist), and ranked diagnostics.

Diagnostics rank consecutive 10 s intervals, breaking ties by earliest start.
With templates, regular ranks by the fraction of beats supported by likely
candidates and requires ≥8 beats. Challenging ranks by ambiguous candidates +
off-grid likely candidates + unsupported acoustic beats, and cannot overlap
regular. Before references, acoustic grid support and off-grid counts provide
fallback rankings. An additional image shows the largest absolute 30 s local
phase discrepancy when its earliest 10 s interval differs from the selected
regular/challenging sections. Classification counts and exact criteria are
included with the interval ranking. These are heuristic suggestions for review,
not verified descriptions of kick content.

| Track | Regular | Challenging | Largest phase discrepancy |
| --- | --- | --- | --- |
| `power_train` | 0–10 s | 160–170 s | 150–160 s |
| `universal` | 0–10 s | 60–70 s | Already the challenging interval |
| `a_grim_premonition` | 30–40 s | 220–230 s | 120–130 s |
| `no_trust` | 0–10 s | 290–300 s | 210–220 s |

Eleven local plots are under `datasets/techno/proposals/diagnostics_v2/`.
The older diagnostics remain in their original directory.

Start the intended interactive workflow:

```bash
uv run python -m benchmarks.annotate --track universal --regenerate
uv run python -m benchmarks.annotate --track universal --plot --start 60 --duration 10
```

Listen, choose 3–5 precise positive references with K (and recurring negatives
with N if useful), press R, then inspect other sections using navigation, G,
and H. A/U/C and Ctrl+Z reduce editing effort; Ctrl+S saves the draft. Existing
drafts remain intact; use the explicit confirmed reset only if a fresh likely-
kick initialization is desired. A full recording review is **not** required
before assessing template usefulness, but remains mandatory before finalizing
ground truth. No held-out recordings were processed or real-world classification
accuracy claimed. mpv was unavailable during implementation: IPC interactions
are tested without audio output, while actual desktop playback needs checking
after installation.
