"""Qt application entry point, imported only for graphical mode."""

import signal

from spectra.audio import AudioSourceInfo


def run_gui(sources: list[AudioSourceInfo], selected: AudioSourceInfo, discovery_error: str | None = None) -> None:
    """Run Qt on the main thread and release capture on every exit path."""
    from PySide6.QtWidgets import QApplication

    from spectra.ui.window import SpectraWindow

    application = QApplication([])
    window = SpectraWindow(sources, selected, discovery_error)
    previous_handler = signal.signal(signal.SIGINT, lambda _signal, _frame: window.close())
    window.show()
    try:
        application.exec()
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        window.shutdown()
