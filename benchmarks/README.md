# Review kick annotations

```bash
uv run python -m benchmarks.annotation --track universal --plot
```

Open the existing proposals and saved review. Generation is automatic only when
proposals do not yet exist; `--regenerate` explicitly replaces predictions and is
unnecessary for navigation changes. Predictions are **not reviewed ground truth**.

On Linux, install PortAudio for sounddevice playback:

```bash
sudo apt install libportaudio2
uv sync --group review
```

PySide6 uses the desktop's Qt appearance and requires a working Wayland or X11
session. Minimal X11 installations may additionally need `libxcb-cursor0` and
`libxkbcommon-x11-0`. The `review` dependency group includes `benchmark`; `dev`
includes `review`. sounddevice is exclusive to review. Qt and PyQtGraph were
already Spectra GUI runtime dependencies; no production dependency was added.
Noninteractive annotation and benchmarking never import Qt, PyQtGraph or sounddevice.
Matplotlib is used only for saved static diagnostics.

## Listening and correcting

- **Click: seek playback**, including clicks on candidate markers. Seeking keeps
  the playing/paused state and never changes a classification.
- **Shift + Click: correct or add kick.** Clicking a displayed marker toggles its
  kick label without moving its onset. Elsewhere, review searches within 50 ms
  for a matching hidden acoustic candidate before automatically localizing a new
  manually identified kick. Ambiguous matches are highlighted for clarification:
  inspect and Shift-click the highlighted marker to correct it. Existing duplicate
  protection also runs after localization. All corrections are undoable.
- Right-click a marker to select and inspect it without changing labels.
- **Follow playback (F)** is the default navigation mode. The viewport follows
  playback with the cursor at 40% of the plot width wherever recording boundaries
  permit. Wheel zoom anchors at playback, both while playing and paused.
- Drag either plot horizontally to pan and enter free navigation. Playback keeps
  running; drags never seek or edit. In free navigation, wheel zoom anchors at the
  mouse pointer, and playback and seeks leave the viewport unchanged. The cursor
  is hidden outside the visible range. Press **F** or enable the checkbox to
  return to playback immediately, preserving zoom and play/pause state. Disabling
  the checkbox leaves playback and the viewport unchanged.

Space always toggles playback, and starts again at zero after the recording ends.
Audio continues across viewport boundaries. Following scrolls the graph beneath
the cursor while playing and remains stationary while paused. The viewport stays
within the recording: at either boundary, the cursor moves away from its anchor
while the visible duration is preserved. A full-recording viewport stays fixed.
`--start 60 --duration 10` sets the initial playback position and viewport width;
the default width is ten seconds. Seeks preserve zoom and immediately reposition
the viewport only in follow mode. The recording-wide slider previews a destination
while dragging and seeks on release, with the same mode-dependent behavior. Ordinary times are
integer seconds; subsecond zoom and Advanced inspection retain finer display.

The two plots show waveform and bass-onset/broadband evidence. A single marker
lane above the waveform normally shows green automatic kicks, blue human kicks,
and subdued red human rejections. Non-kicks remain available but hidden. Direct
review navigation temporarily reveals only the selected hidden event. The pink
selection and cyan playback indicators span both plots. The spectrogram remains
available in static diagnostics only.

| Shortcut | Action |
| --- | --- |
| Space | Play / pause; restart from zero at actual recording end |
| Backspace | Seek backward 10 seconds |
| Left / Right | Seek backward / forward 5 seconds |
| Shift + Left / Right | Seek backward / forward 1 second |
| F | Return to playback and enable following |
| , / . | Previous / next uncertain classification |
| [ / ] | Previous / next unsupported beat |
| < / > | Previous / next human correction |
| Ctrl+Z / Ctrl+S | Undo / save review |

Playback skips use the player's current clock, independent of zoom, clamp to the
recording boundaries, and retain play/pause state. Seeking to the recording end
leaves playback paused. Playback controls are disabled if audio is unavailable.
Shortcuts work with plot, button or slider focus; text entry keeps its usual keys.

Candidate navigation centers a two-second viewport, enters free navigation,
pauses and seeks to its beginning. Repeated navigation advances from the selected event even when events
share a viewport. Explicit human decisions leave both ordinary review queues.
Unsupported positions within 50 ms of an effective kick are excluded as before.
Classification uncertainty includes ambiguous/unclassified template decisions,
threshold proximity and conflicting references. Onset uncertainty, off-grid
predictions and unsupported provenance alone do not enter that queue.

**Advanced** contains non-kicks, classification uncertainty, grid/hypothesis,
similarity and timing overlays, correction navigation, override removal, and a
signed visual cursor offset in milliseconds. The estimated beat grid starts
visible as thin dashed gray lines behind the waveform, using the proposal's
selected primary or available alternative hypothesis. The unsupported-beat queue
uses the same hypothesis. Advanced can hide the grid or switch hypotheses;
neither changes annotations. All other optional overlays start hidden. A positive
offset moves only the drawn cursor forward; it never changes audio, seeking,
onsets, saved reviews or benchmark results. Leave it at zero unless residual
device-specific timing warrants calibration.

Similar-event actions display their affected count and require a second click.
They preserve explicit decisions. Reclassification is an explicit background
operation using a snapshot; results are discarded if corrections change while
it runs. It preserves earlier human decisions and remains undoable.

## Saving and finalizing

Ctrl+S saves a draft under `datasets/techno/drafts/`. Closing with unsaved edits
prompts to save, discard or cancel. Opening/closing a review does not write it.
Effective annotations are derived from predictions and explicit decisions; the UI
has no independent mutable kick list. Drafts retain identities, frozen corrected
onsets, manual events, group provenance and saved reclassification. Undo retains
100 actions in memory. Correction counts do **not** represent verified coverage.

Proposal identity excludes informational processing timings. Regeneration carries
only exactly matched identities and source anchors forward. Stale reviews require
inspection and saving before finalization. Unmatched decisions abort rather than
being guessed. Legacy drafts require explicit `--migrate-draft` or `--reset-draft`,
with backups and existing confirmation requirements. These commands never modify
reviewed annotation manifests.

```bash
uv run python -m benchmarks.annotation --track universal --finalize
```

Finalization consumes the saved review, checks the audio checksum, rate, duration,
BPM and precise sorted onsets, and still requires typing
`REVIEWED universal ENTIRE RECORDING`. Review the entire recording, including
sections with no proposals, before confirming. Saving alone does not finalize.

## Package and noninteractive tools

`annotation/analysis.py`, `templates.py`, and `automatic.py` retain the independent
numerical algorithms. `corrections.py` owns review data and undo. `diagnostics.py`
generates static plots. `validation/` retains the acoustic, refinement and automatic
synthetic experiments. `review/window.py`, `timeline.py`, and `playback.py` own Qt
interaction, native PyQtGraph rendering, and PCM/DAC-clock playback respectively.
`cli.py` implements the `python -m benchmarks.annotation` entry point. The separate
production benchmark remains `python -m benchmarks.kick_detection`.

```bash
uv run --group benchmark python -m benchmarks.annotation --synthetic
uv run --group benchmark python -m benchmarks.annotation --refined-synthetic
uv run --group benchmark python -m benchmarks.annotation --automatic-synthetic
uv run --group benchmark python -m benchmarks.annotation --track universal --diagnostics
```

Playback retains the original WAV's channel layout and amplitude in float32 PCM,
loading it once. Native-rate output is tried first; an unsupported rate is resampled
once to the device default while retaining original recording time. Output initially
requests backend-selected low-latency buffers. Repeated underruns trigger one retry
with 100 ms buffering; cursor synchronization still uses DAC timestamps. Device
errors leave editing and saving available. Pause/seek abort queued output rather
than waiting for a draining stop; obsolete timing generations cannot move a paused
cursor. Natural completion drains the final output buffer.

The cursor uses PortAudio's `outputBufferDacTime` and current stream clock, not
submitted sample count or GUI timer elapsed time. DAC times are **estimates**, not
a guarantee of sample-accurate physical speaker output. See
[methodology and verification](annotation-methodology.md) for numerical baselines,
experiments and the distinction between software-loopback and physical accuracy.


# Production kick detection benchmark

This offline harness runs the production `SpectrumAnalyzer` and
`KickOnsetDetector` against known kick onsets. It imports no capture backend,
rendering, Qt, preview, or output code. It processes successive audio reads as
fast as possible, with fresh analyzer and detector state for each fixture.
Production parameters and defaults live independently in the frozen
`spectra.analysis.KickDetectionConfig` dataclass.

The `dev` dependency group includes `review`, which includes `benchmark`, through
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
