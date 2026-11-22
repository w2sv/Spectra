# Spectra

Spectra is a real-time, music-reactive LED visualization application for Ubuntu 24.04.
The current application prints a startup confirmation. Desktop audio capture, frequency
analysis, a virtual LED strip, and physical LED output over Wi-Fi will follow later.

The selected stack is Python 3.15, NumPy/SciPy for signal processing, and PySide6/pyqtgraph
for the desktop interface and live plots. PipeWire audio capture is planned.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python and
dependencies. The project pins regular CPython 3.15 (with the GIL); uv downloads it when needed.

```bash
uv sync
uv run spectra
```

`uv sync` creates `.venv` and installs runtime and development dependencies from the committed
`uv.lock`. In PyCharm, select the existing interpreter at `.venv/bin/python`.

Development commands:

```bash
uv run pyright
uv run ruff check .
uv run ruff format .
uv run pytest
```

Tests will live in `tests/` when deterministic behavior needs coverage. No tests exist yet;
pytest will report no tests collected (exit code 5). Ruff is the sole linter and formatter.
