# Spectra

Spectra is a real-time, music-reactive LED visualization application for Ubuntu 24.04.
The current CLI captures desktop playback without changing audio routing and prints RMS and
peak levels. Frequency analysis, a virtual LED strip, and physical LED output over Wi-Fi will follow later.

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
```

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

Tests cover PCM conversion and frame alignment without requiring an audio server.
Ruff is the sole linter and formatter.
