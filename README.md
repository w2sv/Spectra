# Spectra

Spectra is a real-time, music-reactive LED visualization application for Ubuntu 24.04.
The current CLI captures desktop playback without changing audio routing and prints RMS/peak
levels, with optional streaming frequency analysis. A virtual LED strip and physical LED output
over Wi-Fi will follow later.

The selected stack is Python 3.15, NumPy/SciPy for signal processing, and PySide6/pyqtgraph
for the future desktop interface and live plots. Audio capture uses `parec` through PipeWire's
PulseAudio compatibility server; `pactl` discovers output monitors.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python and
dependencies. The project pins regular CPython 3.15 (with the GIL); uv downloads it when needed.

```bash
uv sync
uv run spectra
```

Audio capture requires a running desktop PipeWire session with `pipewire-pulse`, plus
`parec` and `pactl` from Ubuntu's `pulseaudio-utils` package (`sudo apt install pulseaudio-utils`).
Play music and watch RMS/peak levels change; press Ctrl+C to stop and release the capture process.
Capture is 48 kHz, stereo float32, with left/right columns in `(frames, channels)` NumPy arrays.

```bash
uv run spectra --list-sources
uv run spectra --source '<monitor identifier from --list-sources>'
uv run spectra --spectrum
```

`--spectrum` adds the three strongest frequency bands about twice per second and can be combined
with `--source`. Analysis downmixes stereo to mono, applies a 4,096-sample Hann window, and runs
an overlapping real FFT every 512 samples. Power is summed into 24 logarithmic bands from
40–16,000 Hz, sharing bin power across band boundaries by fractional frequency overlap.
Values are mean-square power in squared PCM amplitude units; an in-range sine with peak
amplitude A contributes approximately A²/2. Timestamps mark the window center relative to
the stream's first frame. `SpectrumAnalyzer` can also process synthetic PCM and reset for a new stream.

The default is `@DEFAULT_MONITOR@`, resolved when capture starts. Restart capture after
selecting a different default output.

`uv sync` creates `.venv` and installs runtime and development dependencies from the committed
`uv.lock`. In PyCharm, select the existing interpreter at `.venv/bin/python`.

Development commands:

```bash
uv run pyright
uv run ruff check .
uv run ruff format .
uv run pytest
```

Tests cover PCM conversion, frame alignment, spectral power, and streaming continuity without an audio server.
Ruff is the sole linter and formatter.
