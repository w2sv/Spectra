"""Spectra's graphical and diagnostic entry points."""

import argparse
import math
from time import monotonic

import numpy as np

from spectra.analysis import SpectrumAnalyzer
from spectra.audio import AudioSource, AudioSourceError
from spectra.audio.pipewire import DEFAULT_OUTPUT_MONITOR, PipeWireSource, discover_output_monitors


def capture_diagnostics(source: AudioSource, *, spectrum: bool = False) -> None:
    """Report signal levels and optional spectral power every half second."""
    analyzer = SpectrumAnalyzer(sample_rate=source.sample_rate) if spectrum else None
    try:
        source.start()
        print(
            f"Capturing: {source.info.display_name}\n"
            f"Sample rate: {source.sample_rate} Hz\nChannels: {source.channels}\n"
            "Press Ctrl+C to stop.\n",
            flush=True,
        )
        squared_sum = 0.0
        sample_count = 0
        peak = 0.0
        latest_spectrum = None
        next_report = monotonic() + 0.5

        while True:
            samples = source.read()
            if analyzer is not None:
                results = analyzer.process(samples)
                if results:
                    latest_spectrum = results[-1]
            squared_sum += float(np.sum(np.square(samples, dtype=np.float64)))
            sample_count += samples.size
            peak = max(peak, float(np.max(np.abs(samples))))
            if monotonic() >= next_report:
                rms = math.sqrt(squared_sum / sample_count)
                print(f"RMS: {rms:.3f} | Peak: {peak:.3f}", flush=True)
                if analyzer is not None and latest_spectrum is not None:
                    if np.any(latest_spectrum.band_energies):
                        strongest = np.argsort(latest_spectrum.band_energies)[-3:][::-1]
                        bands = " | ".join(
                            f"{analyzer.band_edges[index]:.0f}-{analyzer.band_edges[index + 1]:.0f} Hz: "
                            f"{latest_spectrum.band_energies[index]:.3e}"
                            for index in strongest
                        )
                    else:
                        bands = "silence"
                    print(f"Spectrum @ {latest_spectrum.timestamp:.3f}s (mean-square): {bands}", flush=True)
                squared_sum = 0.0
                sample_count = 0
                peak = 0.0
                next_report = monotonic() + 0.5
    finally:
        source.stop()


def main() -> None:
    """Select an output monitor and launch the requested presentation."""
    parser = argparse.ArgumentParser(description="Capture and visualize desktop audio.")
    parser.add_argument("--list-sources", action="store_true", help="List available output monitors and exit.")
    parser.add_argument("--source", metavar="ID", help="Capture a monitor identifier from --list-sources.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--spectrum", action="store_true", help="Also display the strongest frequency bands.")
    modes.add_argument("--gui", action="store_true", help="Open the graphical spectrum and virtual LED strip.")
    arguments = parser.parse_args()

    try:
        sources = [DEFAULT_OUTPUT_MONITOR]
        discovery_error = None
        if arguments.gui or arguments.list_sources or arguments.source not in (None, DEFAULT_OUTPUT_MONITOR.identifier):
            try:
                sources.extend(discover_output_monitors())
            except AudioSourceError as error:
                if not arguments.gui or arguments.list_sources or arguments.source:
                    raise
                discovery_error = str(error)
        if arguments.list_sources:
            for info in sources:
                print(f"{info.identifier}\t{info.display_name}")
            return

        selected = DEFAULT_OUTPUT_MONITOR
        if arguments.source and arguments.source != selected.identifier:
            selected = next(
                (info for info in sources if info.identifier == arguments.source),
                None,
            )
            if selected is None:
                parser.error("Unknown output monitor. Use --list-sources to see available identifiers.")
        if arguments.gui:
            from spectra.ui import run_gui

            run_gui(sources, selected, discovery_error)
        else:
            capture_diagnostics(PipeWireSource(selected), spectrum=arguments.spectrum)
    except AudioSourceError as error:
        parser.exit(1, f"Audio capture error: {error}\n")
    except KeyboardInterrupt:
        print("\nCapture stopped.")
