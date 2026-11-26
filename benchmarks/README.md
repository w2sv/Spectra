# Benchmarks

Tools for annotating kick onsets and evaluating Spectra's production kick detector against synthetic fixtures and
annotated music.

## Setup

```bash
uv sync
```

Interactive annotation requires a graphical desktop session and PortAudio. On Ubuntu:

```bash
sudo apt install libportaudio2
```

## Annotation

### Prepare recordings

The [track catalog](datasets/techno/tracks.json) defines the recordings, metadata and dataset partitions. Obtain the
corresponding WAV files and import them:

```bash
uv run python -m benchmarks.prepare_tracks --source ~/Downloads/spectra-songs
uv run python -m benchmarks.prepare_tracks --verify
```

Original recordings are preserved unchanged.

### Review

```bash
uv run python -m benchmarks.annotation --track universal
```

Proposals are generated automatically when needed. Existing human corrections are preserved where possible; incompatible
corrections prevent regeneration.

Additional operations:

| Arguments following `--track universal` | Purpose                                        |
| --------------------------------------- | ---------------------------------------------- |
| `--regenerate`                          | Regenerate automatic proposals                 |
| `--prepare-only`                        | Prepare proposals without opening the editor   |
| `--diagnostics`                         | Generate static diagnostic plots               |
| `--bpm 147 --phase 0.01`                | Override tempo and beat-grid phase             |
| `--start 60 --duration 5`               | Set the initial playback position and viewport |
| `--reset-draft`                         | Discard the existing review with confirmation  |

Changing BPM or phase for an existing proposal requires `--regenerate`.

### Controls

| Input                 | Action                                   |
| --------------------- | ---------------------------------------- |
| Click                 | Seek                                     |
| Shift+click           | Toggle or insert a kick                  |
| Right-click           | Inspect candidate                        |
| Drag                  | Pan timeline                             |
| Scroll                | Smooth zoom with adaptive speed          |
| Space                 | Play/pause                               |
| Left / Right          | Seek ±50% of visible duration            |
| Shift+Left / Right    | Seek ±10% of visible duration            |
| Backspace             | Seek backward 10 seconds                 |
| F                     | Follow playback                          |
| 1                     | Reset to the 10-second view and 1x speed |
| `,` / `.`             | Previous/next uncertain candidate        |
| `[` / `]`             | Previous/next unsupported beat           |
| `<` / `>`             | Previous/next human correction           |
| Ctrl+Z                | Undo                                     |
| Ctrl+Shift+Z / Ctrl+Y | Redo                                     |
| Ctrl+S                | Save                                     |
| Escape / Q            | Close                                    |

The upper marker lane displays automatic and manually confirmed kicks; the lower lane displays remaining candidates and
explicit rejections. The graphical legend identifies their respective symbols. Similar-event corrections and
reclassification can propagate human decisions across acoustically similar candidates without overriding existing
explicit corrections.

Adaptive speed is enabled by default: zooming in slows playback and zooming out speeds it up, within 0.25x–2x, with
corresponding pitch changes. Disable **Adaptive speed** for 1x playback without changing the view; arrow-key seeks still
follow the displayed viewport. Press **1** to restore the 10-second view and 1x speed while preserving playback position
and navigation mode.

### Finalization

Automatic proposals are not ground truth. Review the entire recording, including sections without candidates, before
finalizing. Save the review with Ctrl+S, then run:

```bash
uv run python -m benchmarks.annotation --track universal --finalize
```

Finalization requires explicit confirmation and writes the reviewed onsets to
[annotations.json](datasets/techno/annotations.json). Saving clears undo/redo history but does not finalize the review.

### Numerical validation

```bash
uv run python -m benchmarks.annotation --validate --strategy all
```

Available strategies:

| Strategy                      | Approach                                           |
| ----------------------------- | -------------------------------------------------- |
| `acoustic-onset`              | Acoustic candidate discovery                       |
| `guided-template-matching`    | Classification using supplied reference signatures |
| `automatic-template-matching` | Automatic reference discovery and classification   |

The default is `automatic-template-matching`. Use `--json PATH` to export results.

See [Annotation methodology](annotation-methodology.md) for algorithm descriptions, historical versions, experimental
results and limitations.

## Production kick detection

The benchmark evaluates the causal production detector against known kick onsets.

```bash
# Synthetic fixtures
uv run python -m benchmarks.kick_detection

# List available fixtures
uv run python -m benchmarks.kick_detection --list

# Select fixtures and export results
uv run python -m benchmarks.kick_detection \
  --fixture four_on_floor --fixture irregular \
  --json benchmarks/results/experiment.json

# Experiment with detector parameters
uv run python -m benchmarks.kick_detection \
  --set refractory_seconds=0.15 --set threshold_multiplier=2.5

# Benchmark annotated music
uv run python -m benchmarks.kick_detection --dataset techno --partition development
```

Parameter overrides do not modify application defaults. Only finalized annotations enter real-recording benchmarks.
Incomplete dataset selections fail unless `--allow-partial` is supplied. The evaluation partition is reserved for
held-out evaluation and must not be used for parameter tuning.

### Metrics

Events are matched one-to-one at ±25, ±50 and ±100 ms tolerances. Reports include precision, recall, F1, false positives
per minute and timing errors. Positive signed timing errors indicate late detections. Offline timing measurements
exclude system scheduling, rendering and hardware transport latency.

## Results and methodology

- [Annotation methodology](annotation-methodology.md)
- [Production kick detection baseline](results/kick_detection/synthetic_baseline.json)
- [Annotation experiments](results/annotation/)
- [Review performance](results/review_performance/)

Historical results are retained for comparison. Audio recordings, drafts, generated proposals, diagnostics and ad hoc
results remain local. Use `--help` with the respective commands for all available options.
