# Annotation methodology

## Objectives and relationship to production detection

Annotation builds reviewable kick hypotheses and recording-time onset estimates. It is independent of the production
`KickOnsetDetector`, its thresholds and its causal processing contract. Offline annotation may use centered windows,
future samples, zero-phase filters and complete-recording statistics. Those operations must not be transferred into the
real-time detector as though they were causal.

One interactive application reviews the current automatic proposals. Three numerical validation strategies preserve
different research assumptions:

| Stable strategy identifier    | Historical approach                   | Reference availability                     |
| ----------------------------- | ------------------------------------- | ------------------------------------------ |
| `acoustic-onset`              | V1 — Acoustic onset detection         | No template references                     |
| `guided-template-matching`    | V2 — Guided template matching         | Supplied reference onsets                  |
| `automatic-template-matching` | V3 — Automatic discovery and matching | References discovered without ground truth |

A strategy identifies an approach; an algorithm version identifies a numerical revision. V3 reuses substantial acoustic
analysis from V1 and feature extraction, similarity classification and grid diagnostics from V2. They are not three
maintained interactive workflows.

## V1 — Acoustic onset detection

### Acoustic analysis and candidate generation

`annotation/analysis.py` arithmetic-means stereo channels into mono and resamples to 48 kHz with
`scipy.signal.resample_poly` where needed. It applies no amplitude normalization. Centered 2,048-sample periodic Hann
spectra have 240-sample hops (5 ms), with zero padding at recording boundaries. Frame `k` is centered at
`k * 240 / 48000`, preserving the original recording time origin.

Fractional FFT-bin-cell overlaps integrate one-sided mean-square power over 20–60, 60–150, 150–300 and 300–3000 Hz.
Positive differences of `log(power + 1e-12)` are assigned to the later frame center. Each band is divided by the 95th
percentile of its positive differences; an empty denominator yields zero. Values are not clipped to one. The mean of the
first three bands supplies bass evidence; 300–3000 Hz supplies broadband evidence. Both are Gaussian-smoothed with sigma
one frame. Unsmoothed evidence and band powers remain available.

Bass-evidence peaks have at least 80 ms separation and 0.25 prominence. Evidence is heuristic, not a probability. High
acoustic candidate recall is valuable for review, but does not imply reliable kick classification.

### Beat-grid estimation

Given catalog BPM or an explicit override, phase is searched over one beat at 5 ms resolution. The score is mean maximum
bass evidence within ±20 ms of each beat across the complete recording; ties choose the earliest phase. An explicit
phase must lie within one beat interval. No automatic tempo estimation occurs.

The strongest available candidate within ±50 ms is associated with each chronological beat once, without changing its
acoustic timestamp. Unmatched peaks remain off-grid candidates and unmatched beats remain unsupported. A grid cannot
distinguish kick drums from bass notes.

Independent phase searches in consecutive 30-second sections report circular local/global differences; absolute
differences above 30 ms are flagged. The grid is not warped automatically. Bass rhythms, breakdowns and silence can make
phase ambiguous without establishing actual tempo drift.

### Experimental results and limitations

The [V1 synthetic baseline](results/annotation/acoustic_onset/v1_synthetic_baseline.json) contains 15 fixtures and 169
ground-truth kicks. At ±50 ms, all acoustic candidates produce **169 TP, 78 FP, 0 FN** (precision 0.684, recall 1.000,
F1 0.8125). Grid-aligned candidates on the three tempo-tagged fixtures produce 45 TP, 1 FP, 0 FN. Grid metrics exclude
fixtures without tempo metadata.

Matched all-candidate median signed error is −15 ms; median absolute and P95 absolute errors are 15 ms. Centered windows
observe an upcoming transient before its physical onset, so novelty maxima can lead the actual attack. This is an
evidence-timing bias, not a broken frame timeline; no global correction is applied. Offbeat bass contributes 14 false
candidates, sub-bass 14 extra candidates and rumble 43. Candidate discovery alone is insufficient for instrument
specificity.

## V2 — Guided template matching

### Feature representation

`annotation/templates.py` retains supplied-reference matching as an independent numerical experiment. Validation
extracts reference features directly from fixture ground truth; no persisted manual-template definitions are required.

The existing centered spectra are integrated over these bands in Hz:

`20–40, 40–60, 60–90, 90–130, 130–180, 180–250, 250–350, 350–500, 500–750, 750–1000, 1000–2000, 2000–3000`.

An inclusive −20 to +160 ms context contains 37 frames × 12 bands. Background is median band power over −100 to −25 ms.
Missing context is zero padded. Features subtract square-root background power from square-root context power, clamp
negative values to zero, flatten to 444 values and L2-normalize the whole vector. There is no separate per-band
normalization: relative spectral and temporal shape is retained. Zero-norm signatures are invalid rather than assigned
arbitrary similarity. Nearest-frame anchors introduce at most 2.5 ms quantization.

### Similarity classification and temporal refinement

Each candidate searches −30 to +30 ms in 5 ms steps. Cosine similarity independently maximizes positive and negative
matches. Exact positive-score ties favor smallest absolute shift, then negative shift. `TemplateParameters` defines the
defaults:

| Label        | Rule                                                                |
| ------------ | ------------------------------------------------------------------- |
| Likely       | Positive ≥0.85, and when negatives exist, positive ≥ negative +0.10 |
| Ambiguous    | Positive ≥0.70 but likely criteria fail                             |
| Unlikely     | Positive \<0.70 or invalid candidate with valid positive references |
| Unclassified | No valid positive signatures                                        |

Invalid supplied negatives do not waive the margin requirement. Scores are not probabilities. Refined time is the
original candidate anchor plus its best positive shift; matches at the search boundary require scrutiny. This is
template alignment, not sample-accurate onset localization. Early or inaccurate references transfer their own timing
errors to matching events.

The alternative grid is half a beat from the acoustic phase. Per-section grid comparisons retain original acoustic phase
evidence, supported-candidate counts, similarities and likely counts. They do not silently change tempo or phase.

### Reference-excluded validation

Unique first/middle/last ground-truth kicks supply positive references. Fixtures without kicks use the clean fixture's
signatures as a cross-fixture stress test. Reference events and their one-to-one acoustic matches at the largest
evaluation tolerance (100 ms) are excluded from every stage. Duplicates remain errors. All stages use the same remaining
candidate identities and ground-truth population, independently of subsequent classification or refinement.

The [V2 synthetic baseline](results/annotation/guided_template_matching/v2_synthetic_baseline.json) excludes 37
references, leaving **132 non-reference kicks**. The one-kick fixture has no non-reference positives and cannot
establish classification recall. Audio duration remains the complete 120 seconds for FP/min; timing uses matched pairs.
At ±50 ms, with positive references only:

| Stage on the same non-reference population | TP / FP / FN | Precision / recall / F1 | Median absolute / P95 absolute error |
| ------------------------------------------ | ------------ | ----------------------- | ------------------------------------ |
| Acoustic candidates                        | 132 / 78 / 0 | 0.629 / 1.000 / 0.772   | 13.21 / 15.00 ms                     |
| Likely, original times                     | 129 / 1 / 3  | 0.992 / 0.977 / 0.985   | 14.29 / 15.00 ms                     |
| Likely, refined times                      | 129 / 1 / 3  | 0.992 / 0.977 / 0.985   | 0.00 / 2.06 ms                       |
| Likely + ambiguous, refined                | 132 / 1 / 0  | 0.992 / 1.000 / 0.996   | 0.00 / 2.10 ms                       |

The negative-reference experiment uses the independently known first offbeat bass note at 0.75 seconds and excludes its
candidate in its source fixture. Transferred negative signatures reduce likely results to 118 TP, 1 FP, 14 FN; eleven
low-pitch kicks become ambiguous. Combined likely/ambiguous results remain 132 TP, 1 FP, 0 FN. This is an intentional
cross-fixture stress test, not proof that arbitrary negative templates transfer safely. Rumble still leaves a likely
false positive.

## V3 — Automatic template discovery and matching

### Supplementary candidate generation and onset localization

`annotation/automatic.py` keeps every V1 acoustic candidate and visits unsupported beats on both grid hypotheses. It
adds the strongest nonzero local novelty maximum within ±50 ms, even below V1's prominence threshold, or a grid-only
expectation when no acoustic maximum exists. Supplementary positions within 30 ms of an existing event are not
duplicated. Stable IDs hash audio identity, provenance and original 48 kHz sample position; original candidates are not
merged.

A separate localizer searches −30 to +50 ms around each acoustic event. A second-order zero-phase Butterworth filter
provides 20–300 Hz energy; original mono PCM supplies raw broadband energy. Trailing 5 ms mean-square windows at 1 ms
hops use recording timestamps, with background median/MAD over the preceding 75 ms. The dominant joint energy rise is
followed backward to a threshold above both background +3 MAD and 10% of the local increase, sustained for at least 3 ms
in both channels. The numerical floor is 1e-12. Requiring raw-energy rise prevents filter pre-ringing alone from
creating an onset. Unreliable crossings retain the original time. Grid-only expectations never automatically become
kicks.

### Clustering, classification and human-assisted reclassification

Localized signatures reuse V2's 37×12 representation. SciPy average-linkage hierarchical clustering uses cosine-distance
cut 0.15. An eligible cluster has at least eight members and 70% alignment to a grid. Grid score is
`alignment_fraction * log1p(unique_supported_beats) * median_low_frequency_increase`. The best cluster supplies its
medoid and up to two members separated by at least one-sixth of the recording and 180 ms. Further clusters on that grid
need medoid similarity ≥0.70 to selected clusters. At most nine references are used, without automatically discovered
negatives. Competing clusters/grids remain diagnostics.

Every candidate searches V2's ±30 ms offsets at its localized anchor. Similarity ≥0.85 predicts kick; ambiguous or
weaker matches remain candidates. Final time includes the best match offset. Without references, only grid-aligned
acoustic candidates predict kicks, with low confidence. Without a grid or references, reliable gridless classification
is unavailable.

Explicit reclassification can add human-confirmed kick signatures and negative signatures from rejected candidates,
retaining the 0.10 margin. Rejected automatic references are removed from positives. Existing explicit corrections keep
frozen timestamps and labels. Predictions never become human confirmation by themselves.

### Experimental results and limitations

The [V3 synthetic baseline](results/annotation/automatic_template_matching/v3_synthetic_baseline.json) uses all 15
fixtures with **no ground-truth templates during generation or classification**. At ±50 ms, the candidate population
covers 169/169 kicks with zero required insertions. Initial automatic classification yields **45 TP, 0 FP, 124 FN**
(precision 1.000, recall 0.266, F1 0.421), requiring 124 binary corrections under the recorded one-to-one assignment.
That assignment is not a claim of minimum editing effort for overlapping events.

All 45 kicks in the three tempo-tagged fixtures are detected. The other twelve fixtures lack tempo metadata and eligible
rhythmic references, explaining the major recall limitation. Zero predicted kicks on negative fixtures largely reflect
unavailable references, not demonstrated non-kick discrimination.

For matched population events, independent localization changes median signed error from −15 ms to +3.27 ms, with 27 ms
P95 absolute error. On the 45 predicted kicks, localized median error is +3.0 ms and P95 3.83 ms; template refinement
produces +3.0 ms and P95 3.93 ms, without clear extra timing benefit. Rumble, sub-bass, long decays and masked attacks
still need review.

## Validation methodology and algorithm revisions

All strategies use deterministic synthetic audio and one-to-one event matching at ±25, ±50 and ±100 ms. Positive signed
error means late. Precision, recall and F1 are undefined when their denominators are zero; reports use `null`. Timing
statistics use matched pairs, and FP/min uses complete evaluated audio duration.

Acoustic discovery recall, guided classification recall and fully automatic classification recall describe different
populations and reference assumptions. Do not rank these strategies by directly comparing their headline metrics.
`--strategy all` returns independent reports, with no cross-strategy aggregate.

Historical comparisons report acoustic population counts, added/removed timestamps, current/historical metrics,
parameter changes and algorithm versions rather than asserting exact equality. Missing fixtures or changed ground-truth
populations are marked incompatible. Automatic primary metrics remain full-population metrics; a separate compatible
guided comparison excludes its recorded references only for evaluation, after automatic generation/classification.
Historical reference selection never supplies automatic templates. Correctness invariants such as valid timestamps,
deterministic generation and consistent one-to-one reference exclusion remain testable independently of historical
equality.

`AUTOMATIC_ALGORITHM_VERSION = 3` identifies current cached automatic proposals. Increment it for meaningful numerical
changes to acoustic analysis, candidate generation, beat grids, onset localization, feature extraction, reference
discovery, classification/refinement or relevant numerical defaults, including changes to shared V1/V2 components
consumed by V3. Do not increment it for UI, playback, documentation, package/CLI restructuring or numerically equivalent
refactoring. Strategy identifiers remain stable across revisions. This is algorithm provenance, not a serialization
schema-versioning framework; the proposal checksum identifies the precise proposal independently of that version.

## Real-recording pilot results

The [V1 pilot](results/annotation/acoustic_onset/v1_techno_pilot.json) analyzed four unchanged 44.1 kHz development
recordings. Durations were 348.444, 317.000, 377.027 and 378.800 seconds for Power Train, Universal, A Grim Premonition
and No Trust. Acoustic peak counts were 1,972, 1,771, 1,797 and 1,952 respectively. Local phase discrepancies reached
about half a beat; they are review cues, not verified tempo drift.

The [V2 pilot](results/annotation/guided_template_matching/v2_techno_pilot.json) retained all 7,492 acoustic candidates,
including 4,355 off-grid events. No real references had been established, so predictions remained unclassified. It
records candidate/grid diagnostics rather than real-track classification accuracy.

The [V3 pilot](results/annotation/automatic_template_matching/v3_techno_pilot.json) records automatic hypotheses, not
ground truth:

| Track              | Clusters / references | Grid        | Predicted kicks / remaining candidates | Uncertain / unreliable onsets |
| ------------------ | --------------------- | ----------- | -------------------------------------- | ----------------------------- |
| Power Train        | 221 / 5               | Primary     | 666 / 1,648                            | 1,709 / 1,346                 |
| Universal          | 337 / 5               | Primary     | 656 / 1,334                            | 1,148 / 529                   |
| A Grim Premonition | 613 / 7               | Alternative | 672 / 1,536                            | 2,138 / 1,247                 |
| No Trust           | 260 / 2               | Primary     | 778 / 1,341                            | 1,322 / 1,101                 |

Diagnostic sections rank beat support for a regular interval and ambiguous, off-grid and unsupported activity for a
disjoint challenging interval. A distinct maximum phase-discrepancy interval may be added. These are acoustic
heuristics, not independently verified musical descriptions. No held-out evaluation track was analyzed and no track was
finalized in these pilots.

## Remaining limitations

Synthesized repeated sounds and supplied-reference experiments are sanity checks, not estimates of real-music accuracy.
Recurring bass signatures can win automatic reference discovery. Similarities are not calibrated probabilities;
uncertain onsets, competing grids and invalid signatures must remain visible for review. There is no automatic tempo
estimation or reliable general gridless classifier. Only explicit full-recording human confirmation can establish
finalized annotations.

Operational instructions are in the [benchmark README](README.md). GUI timing observations and their limits are separate
in the [review performance notes](results/review_performance/README.md).
