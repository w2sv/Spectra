# Annotation methodology and earlier experiments

The current ordinary workflow is automatic generation followed by binary corrections; see [README](README.md). The version-1 and version-2 methods and measurements below are retained for comparison. Version 2 used manually established reference onsets and is not equivalent to fully automatic discovery. Its manual-template UI has been replaced; its numerical APIs and synthetic validation remain available.

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
has no hard reference-count limit. In version 2, real-world references were not automatically selected. The checked-in `datasets/techno/templates/<identifier>.json` definitions
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
