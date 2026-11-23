# Spectra

Spectra is a real-time, music-reactive LED visualization application for Ubuntu 24.04.
Spectra captures desktop playback without changing audio routing. Its optional desktop interface
shows a live frequency spectrum and a virtual strip of 300 RGB LEDs; CLI diagnostics remain available.
Physical LED output over Wi-Fi will follow later.

The selected stack is Python 3.15, NumPy/SciPy for signal processing, and PySide6/pyqtgraph
for the desktop interface and live plots. Audio capture uses `parec` through PipeWire's
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
uv run spectra --gui
```

In graphical mode, choose an output monitor and click **Start**. **Stop** releases capture;
closing the window also stops its worker and subprocess. Capture/analysis run off the GUI thread,
and a 30 FPS timer displays the latest result. Source selection also works with `--gui --source`.
If results stop arriving for 500 ms, the display clears while capture stays active and recovers
automatically when results resume.

The spectrum displays the 24 log-spaced bands on a fixed −80 to 0 dB scale, with a power floor
of 10⁻¹² (−120 dB). LED colors progress from red at low frequencies to blue at high frequencies.
LED intensity uses a fixed −60 to −6 dB range, with adjustable sensitivity and a fixed frequency
balance tilt. The initial 6 dB tilt applies −3 dB to the lowest band and +3 dB to the highest;
there is no automatic per-band gain or per-frame normalization. Each band controls 12 or 13 LEDs.

The GUI's **LED sensitivity**, **Frequency balance**, **Attack**, and **Release** controls update
during playback. Defaults are 0 dB sensitivity, 20 ms attack, and 180 ms release. Positive balance
favors higher frequencies; negative balance favors lower ones. Attack/release use elapsed audio
time for every FFT result, independently of GUI rendering. Changes preserve the current envelope;
zero milliseconds disables smoothing in that direction. Silence releases to an unlit strip.
These controls affect only LED brightness: the spectrum graph and CLI retain raw power readings.
The Qt-independent `frequency_frame` renderer accepts normalized intensities, LED count, and hues.

The **Kick** indicator briefly highlights detected bass transients; its session count resets on
capture restart. Detection uses raw 40–160 Hz power changes, independently of LED controls, with
an adaptive flux threshold, minimum and rising bass-power gates, and a 200 ms refractory period. It detects
kick-like low-frequency onsets, including some non-kick sounds. Defaults are adjustable in
`analysis/kick.py`; no tempo estimation or onset-driven LED effects are applied. The worker checks
every FFT result and drains pending events on GUI updates (newest 32 retained during long stalls,
with the full detection count preserved).
See [kick detection evaluation](docs/kick-detection.md) for the fixture comparison and remaining limitations.

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
