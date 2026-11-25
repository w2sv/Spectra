# Automatic kick annotation and production benchmarking

Analyze one of the four local development recordings, then correct the predictions:

```bash
uv run python -m benchmarks.annotate --track universal --regenerate
uv run python -m benchmarks.annotate --track universal --plot
```

The complete recording supplies recurring sound references automatically. No template
selection or manual onset placement is needed before review. Listen with Space and
click incorrect candidate markers to toggle kick/non-kick. Clicking again reverses
the decision without moving the onset. Orange outlines flag uncertainty; they do
not prove an error. Predictions and saved reviews are **not reviewed ground truth**.

Use the right-side buttons for navigation, undo, saving, and optional corrections
of similar events. Group corrections display their affected count and require a
second click to apply. They preserve earlier explicit decisions. Reclassification
using corrections is an explicit action, not a calculation on every click. It
preserves corrected event identities, timestamps and labels and can be undone.
You need not individually confirm every correct algorithmic prediction, but must
review the entire recording before finalizing it.

Optional synchronized playback uses the installed `mpv` executable. On Ubuntu:
`sudo apt install mpv`. Generation, benchmarking, and review without playback do
not require it. Playback starts paused, uses private JSON IPC without a player
window, follows reported positions every 50 ms, and stops at the interval end.
Navigation seeks to the displayed start and pauses. Closing releases the player
and socket. Playback timing never changes annotations.

| Shortcut | Review action |
| --- | --- |
| Click a candidate / T | Toggle clicked / selected classification, retaining its onset |
| Space / Home | Play-pause / restart the displayed interval |
| Alt + Click | Seek playback |
| Left / Right | Previous / next displayed interval |
| , / . | Previous / next uncertain candidate |
| [ / ] | Previous / next unsupported-beat candidate |
| Ctrl + Z / Ctrl + S | Undo / save review |
| R | Reclassify uncorrected events using human corrections |
| < / > | Previous / next human correction (Advanced) |
| Escape | Remove the selected human override (Advanced) |
| G / H | Switch grid overlay / show-hide non-kicks (Advanced) |
| Shift + Click | Insert a completely missing event and localize its onset (Advanced) |

All buttons use the same handlers as the keyboard. Selected events are highlighted
on all three plots. Candidate navigation centers a two-second interval, shifts
it inward at recording boundaries, and never wraps. Consecutive unsupported
beats advance relative to the selected event, even within the same view.
Unsupported beats are ordinary candidates: toggle their labels rather than using
the former separate Y/X kick/gap mechanism. Corrected candidates and beats within
50 ms of an effective kick are excluded from the unsupported review queue.

Waveform, low-frequency spectrogram, and evidence plots share recording time.
Solid green circles are automatic kicks; hollow gray triangles are automatic
non-kicks. Solid blue squares and hollow red X markers distinguish human
corrections. Orange hollow diamonds independently indicate uncertainty. Small
vertical/horizontal marks independently indicate grid alignment/off-grid events.
The legend is below the data. The default view is ten seconds; `--start 60
--duration 10` selects another interval. Ordinary toolbar zoom/pan works.
The control panel scrolls at shorter heights; Advanced operations start collapsed.
Progress gives candidate/kick/non-kick counts, human corrections and uncorrected
uncertain candidates. A correction count is not a percentage of verified ground truth.

Reviews are ignored local files under `datasets/techno/drafts/`. They contain
candidate identities, explicit binary overrides with frozen onset timestamps,
group provenance, manual additions and any explicitly recomputed predictions.
There is no independently mutable accepted-timestamp list. Effective kicks are
derived from predictions plus overrides. Saved reclassification survives reopening.
Undo covers individual/group corrections, insertion, override removal, and
reclassification (100 actions in memory).

Proposal hashes exclude informational processing times. Regeneration preserves
exactly matched identities and source anchors; changed timing estimates do not
move corrected events. A stale-review warning requires inspecting new predictions
and saving before finalization. Unmatched human decisions abort replacement and
identify the unmatched IDs; no nearest-event guesses are made. Old draft formats
are preserved and require explicit `--migrate-draft` or `--reset-draft`. Migration
requires unambiguous matches and writes a legacy backup; reset requires typing
`RESET <identifier> DRAFT` and also keeps a backup. An old accepted list's omissions
are never inferred to be explicit rejections. Reviewed manifests are never changed
by generation, reclassification, migration, or draft initialization.

The original unreviewed Universal draft was backed up locally as
`universal.unreviewed-v2.json`, following confirmation that no manual annotation
had been done. Its new draft contains automatic predictions without fabricated
human corrections.

## Automatic annotation version 3

Version 3 reuses the unchanged version-1 acoustic stage and version-2 feature and
cosine-alignment methods. It is independent of production DSP and detection.
[Detailed earlier methods and baselines](annotation-methodology.md) remain intact.

1. Retain every original novelty peak. For unsupported beats in both primary and
   half-beat grids, find the strongest local novelty maximum within ±50 ms, even
   below the original prominence threshold. Otherwise retain the beat as a
   grid-only candidate. Supplementary events within 30 ms of an existing event
   are not duplicated; original acoustic candidates are never merged. Stable IDs
   hash audio identity, provenance and original 48 kHz sample position.
2. Independently localize acoustic events in −30 to +50 ms neighborhoods. A
   second-order zero-phase Butterworth filter supplies 20–300 Hz energy; original
   mono audio supplies broadband energy. Trailing 5 ms mean-square windows have
   1 ms hops and timestamps at their ends, avoiding centered-window backdating.
   Background is median/MAD over the 75 ms preceding the search interval. Follow
   the dominant joint energy rise backward to a crossing above both background
   +3 MAD and 10% of the local rise, sustained for at least 3 ms, in both channels.
   The raw-energy gate prevents filter pre-ringing alone from establishing onset.
   The numerical floor is 1e-12. Unreliable crossings retain their original time;
   grid-only positions remain estimated and cannot automatically become kicks.
3. At localized onsets, extract the existing 37×12 normalized features: centered
   2048-point Hann STFT, 48 kHz, 240-sample hops, −20 to +160 ms context, median
   −100 to −25 ms background, fractional band powers, square-root compression,
   subtraction/clamping and one L2 normalization. Invalid zero features are excluded.
4. Use SciPy average-linkage hierarchical clustering with cosine distance cut
   **0.15**, an experimental fixed parameter. Store cluster size, evidence, median
   nonnegative low-frequency increase, within-cluster similarity, recording-quarter
   distribution, medoid and primary/alternative grid support. Select the medoid
   and up to two members separated by at least one-sixth of the recording (and
   180 ms). Eligible clusters have ≥8 members and ≥70% alignment to a grid.
   Score each grid as `alignment_fraction * log1p(unique_supported_beats) *
   median_low_frequency_power_increase`. The best cluster supplies references.
   Further eligible clusters on that grid need medoid similarity ≥0.70 to an
   already selected cluster. At most nine references are used. No automatic
   negative references are discovered. Competing clusters and grids remain reported.
5. Compare every candidate with positive references at offsets −30…+30 ms, every
   5 ms. Similarity ≥0.85 predicts kick; ≥0.70 is ambiguous/non-kick; lower scores
   predict non-kick. The final onset is localized time plus the best match offset;
   search-boundary matches are flagged. Without references, only grid-aligned
   acoustic candidates predict kicks, with low confidence. Without a grid or
   references, reliable automatic classification is explicitly unavailable.
6. Human-confirmed kicks can supply additional positive references and explicit
   rejections can supply negatives on R. A positive-minus-negative margin of
   0.10 is retained. Rejected automatic references are excluded from positives.
   No algorithmic prediction is treated as human confirmation. Similarity is
   not a probability, and recurring bass sounds can win automatic cluster selection.

Parameters live in immutable `AutomaticParameters` and `TemplateParameters`,
independently of production configuration. No tuning was applied after recording
the automatic baseline. Proposals retain original, independently localized,
template-refined and final times; reliability and provenance; cluster identities;
similarity scores, predictions and uncertainty reasons. Uncertainty cues include
threshold proximity, conflicting references, unsupported positions, off-grid
predictions and unreliable localization. These are review cues, not error labels.
Both grid hypotheses and 30-second acoustic phase comparisons remain available;
neither BPM nor the primary grid is silently changed.

```bash
uv run python -m benchmarks.annotate --automatic-synthetic \
  --json benchmarks/results/annotation_automatic_baseline.json
uv run python -m benchmarks.annotate --track universal --diagnostics
```

Diagnostic ten-second sections rank likely beat support (≥8 beats) for a regular
section, and ambiguous + off-grid likely + unsupported activity for a disjoint
challenging section, with earliest ties. Without references they fall back to
acoustic/grid statistics. A distinct maximum-phase-discrepancy section is added
where possible. Plots of recurring cluster medoids expose competing signatures.
Selections and counts are heuristic; “regular” does not mean verified kicks.
Outputs stay ignored under `proposals/diagnostics_v3/`.

### Untuned automatic validation

[`annotation_automatic_baseline.json`](results/annotation_automatic_baseline.json)
uses all 15 unchanged fixtures and **no ground-truth reference templates**. Ground
truth enters evaluation only. The population retains 169/169 kicks at ±50 ms,
requiring zero manual insertions at that window. Initial classification gives
45 TP, 0 FP, 124 FN: precision 1.000, recall 0.266, F1 0.421, 0 FP/min, and
124 binary label corrections under the recorded one-to-one population assignment.
This assignment is not claimed to minimize editing effort in overlapping cases.

The three fixtures with supplied tempo metadata achieve 45/45 initial detections.
The other 12 have no tempo metadata; no eligible rhythmic cluster or reliable
gridless classifier is available, so their kicks remain unclassified/non-kick.
No known tempo was inferred from their ground-truth timestamps. This is a major
recall limitation, not evidence of strong general-purpose kick specificity.
Negative fixtures give zero initial predicted kicks, largely because references
are unavailable; that does not prove discrimination of offbeat bass.

On all matched population events at ±50 ms, novelty median signed error is
−15.0 ms; independent localization gives +3.27 ms median signed/absolute error
and 27.0 ms P95 absolute error. On the 45 automatically predicted kicks, localized
median is +3.0 ms and P95 3.83 ms; template refinement gives +3.0 ms and P95
3.93 ms, offering no clear extra timing improvement. Localization is threshold-
based and not sample accurate. Rumble, sub-bass and extended decay still demand
review, particularly when novelty or background energy masks an attack.

The JSON includes all three tolerance windows, timing stages, candidate-population
false negatives, cluster/reference counts, and per-fixture corrections versus
manual insertions. V1 candidate timestamps are asserted identical to the original
baseline. V2 comparisons additionally use its same non-reference event population;
those supplied-reference results must not be presented as fully automatic results.
Synthetic validation is a sanity check, not real-music accuracy evidence.

### Four-recording automatic pilot and review performance

[`annotation_automatic_pilot.json`](results/annotation_automatic_pilot.json) records
all four full development recordings, selected references, competing clusters/grid
evidence, uncertainty, supplemental candidates, reliability and processing times.
No held-out recording was processed and no annotation was finalized. Universal
is ready for the two-command workflow above. Human review is still required
before real-world precision or recall can be measured.

| Track | Clusters / references | Selected grid | Kicks / non-kicks | Uncertain | Supplementary | Unreliable onsets | Analysis time |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| power_train | 221 / 5 | primary | 666 / 1648 | 1709 | 342 | 1346 | 12.27 s |
| universal | 337 / 5 | primary | 656 / 1334 | 1148 | 219 | 529 | 11.72 s |
| a_grim_premonition | 613 / 7 | alternative | 672 / 1536 | 2138 | 411 | 1247 | 13.99 s |
| no_trust | 260 / 2 | primary | 778 / 1341 | 1322 | 167 | 1101 | 17.96 s |

Many candidates still have uncertain timing or classification. The alternative
hypothesis won for A Grim Premonition; this is an acoustic hypothesis, not a
verified change of kick-grid phase. Catalog BPM and recurring bass interference
remain possible sources of mistakes.

[`annotation_review_performance.json`](results/annotation_review_performance.json)
compares headless Agg on Universal 60–70 s at 1920×1080, interleaving old and new
operations in the same process to reduce differences in system load. Median
navigation fell from 1,821 to 420 ms; editing from 3,411 to 247 ms; edit/undo from
1,623 to 104 ms; cursor rendering from 488 to 3.2 ms; construction from 3,778
to 1,618 ms. The earlier standalone profile is retained as informational data. The older
edit operation is add/remove; the new one is binary toggling, so it is a workflow
comparison rather than identical operations. Repeated-hover fast-path timing
excludes first inspection. These measurements are machine/load dependent and
**not desktop responsiveness measurements**.

Profiling identified full axis reconstruction, repeated associations and layout,
and unconditional redraws as bottlenecks. The new view retains static audio artists,
caches logarithmic spectra, uses visible time slices and min/max waveform buckets,
and indexes sorted candidates for inspection. Edits blit dynamic overlays and
changed controls; cursor updates restore a cached background and draw only cursor
lines. Limits, layout, grid changes and resizing invalidate the relevant cache;
unsupported backends use `draw_idle`. Clustering and feature extraction never run
on hover or redraw. Actual Qt/mpv checks verified advancing reported time, interval-
end pause, navigation seek/pause, unchanged event times and process/socket cleanup.
Headless layouts were visually inspected at 1920×1080 and 1366×768 at 100 DPI.
The desktop uses high-DPI scaling. At 200% scaling, forcing a 1366×768 physical-
pixel window leaves insufficient space and the legend overlaps the controls;
use a larger window (at least 1366×768 logical pixels). The default window is
larger. Compact high-DPI layout remains a limitation; these rendering measurements
also do not measure audio-output latency.


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


# Production kick detection benchmark

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
