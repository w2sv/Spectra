# Reproducible audio testing

Run these commands from the repository root. The helper is development tooling;
production capture, analysis, LED processing, and the committed kick detector are unchanged.

## Setup

```bash
uv sync
uv run python scripts/audio_test.py check
```

Ubuntu packages, if missing:

```bash
sudo apt install ffmpeg              # conversion (also supplies ffprobe)
sudo apt install pulseaudio-utils    # pactl, paplay, parec
```

PipeWire's PulseAudio compatibility server must be running for playback/capture.
Offline analysis and synthetic generation do not require a running audio server.

## Local sample library

`samples/` is entirely gitignored. Keep additional recordings or excerpts there:
`four-on-floor.wav`, `bass-heavy.wav`, `percussion.wav`, `quiet-kicks.wav`, and
`breakdown.wav` are useful categories, not files bundled with this project.
Use freely licensed recordings or files you supply and are entitled to use.

The version-controlled [curated collection](../tests/fixtures/audio/README.md) lives in
`tests/fixtures/audio/`. It contains six CC0 excerpts, including a matching full-mix/bass-only
pair. Its manifest records licensing, original downloads, excerpt selections, checksums, and
format. Full downloaded sources remain in ignored `samples/originals/`.

```bash
uv run python scripts/audio_test.py synth
uv run python scripts/audio_test.py convert /path/to/recording.flac samples/four-on-floor.wav \
  --start 30 --duration 45 --source 'user-supplied recording' --license 'user-supplied; local testing only'
```

For downloaded music, use the original recording URL as `--source` and its actual
license as `--license`. Conversion writes a `.wav.json` provenance sidecar with
the source, license, original path, excerpt, and FFmpeg command. Preserve the
original recording and its attribution requirements. Nothing downloads automatically.
Outputs are 48,000 Hz, stereo, signed 16-bit PCM WAV, without gain normalization.
Existing outputs are not overwritten.

The deterministic eight-second `smoke.wav` contains silence, four 65 Hz damped
bass pulses, sustained bass, high-frequency pulses, quieter bass pulses, then
silence. Its sidecar describes the exact sections. It is a routing/analysis smoke
signal, not a musical recording or a ground-truth kick-detection benchmark.

## Isolated playback and capture

```bash
uv run python scripts/audio_test.py play samples/smoke.wav --capture
uv run python scripts/audio_test.py play tests/fixtures/audio/lyserb.wav \
  --start 1 --duration 5 --capture
```

This creates a low-priority `spectra_test` null sink, starts its monitor capture
and playback without waiting for an initial buffer, and reports captured audio times,
RMS, and peaks through the existing `PipeWireSource` and `SpectrumAnalyzer`.
Playback has a deadline of the selected audio duration plus five seconds. A
watcher cancels blocked capture reads when playback ends or reaches that deadline.
Every captured chunk is checked for signal, independently of diagnostic printing.
After playback, `--capture` fails if no chunk exceeded RMS `1e-5` (-100 dB
relative to unit PCM amplitude). Initial silence is allowed; a wholly silent or
muted excerpt therefore fails verification, with normal resource cleanup.
Playback alone is available by omitting `--capture`. EOF, Ctrl+C, errors, and
SIGTERM stop the child processes and remove only the module created by that run.

`--start` and `--duration` use seconds and the same frame rounding, validation,
and end-of-file clamping as offline analysis. Omit `--duration` to play from
`--start` to the end; omit both to play the entire WAV. Selected excerpts are
copied to a temporary PCM WAV in bounded chunks and deleted after playback.

The helper never selects a default device or creates a loopback to speakers.
Playback explicitly targets `spectra_test`, with PipeWire properties disabling
fallback and target moves. An existing null sink is reused and never removed;
a same-named non-null sink is rejected. Owned modules are identified by a unique
creation token before unloading, not by sink name alone.

To hold the sink for another process or GUI session:

```bash
uv run python scripts/audio_test.py sink
# In another terminal, while that command remains running:
uv run spectra --list-sources
uv run spectra --source spectra_test.monitor --gui
paplay --device=spectra_test samples/smoke.wav
```

The monitor appears through Spectra's existing output-monitor discovery. Restart
an already-open GUI to refresh its source list. Stop the holder with Ctrl+C to
remove its sink. Do not unload modules by name; another session may own them.

## Offline excerpts and JSON detections

```bash
uv run python scripts/audio_test.py analyze samples/smoke.wav
uv run python scripts/audio_test.py analyze tests/fixtures/audio/lyserb.wav \
  --start 1 --duration 10 --output samples/results/lyserb-default.json
uv run python scripts/audio_test.py analyze tests/fixtures/audio \
  --output samples/results/collection-default.json
uv run python scripts/audio_test.py analyze tests/fixtures/audio \
  --min-flux 0.5 --refractory 0.2 --output samples/results/collection-adjusted.json
```

Offline analysis reuses the committed `SpectrumAnalyzer` and `KickOnsetDetector`
directly, processing every FFT result. No new detector is implemented here.
Each run starts fresh state at the excerpt boundary. JSON includes input and
analysis-source SHA-256 hashes, the selected frame range, FFT/hop sizes, detector
overrides, and detections with strength and timestamps. `timestamp` is relative
to the excerpt; `source_timestamp` includes its offset within the input WAV.
Both use spectrum-window centers, not playback wall-clock time. Incomplete final
FFT windows are not padded. Compare identical excerpt ranges and source hashes
when evaluating parameter changes. Live monitor timing includes server startup
and buffering and should not be treated as sample-exact file timing.

For a directory input, every `.wav` is analyzed in filename order, with fresh state per file;
the JSON's `results` array contains the individual reports. A file input keeps the single-report
format. Future reference kick timestamps belong in the fixture manifest only after independent
annotation. Detector outputs are experiment results, never ground truth. In particular, the
bass-only stem is an intended negative example, not an assertion that the present detector
must already produce zero events.

This step establishes fixtures, playback, and export only. It does not change
thresholds, add detection effects, or provide accuracy scores without annotations.
