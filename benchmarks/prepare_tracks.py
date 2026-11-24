"""Copy already downloaded WAVs byte-for-byte into the fixed recording catalog."""

import argparse
import json
import re
import shutil
import unicodedata
from dataclasses import asdict, replace
from pathlib import Path

from benchmarks.fixtures import file_checksum, load_wav
from benchmarks.tracks import TECHNO, Track, read_catalog, verified_audio


def normalized_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).casefold()
    return "".join(character for character in text if character.isalnum())


def matches_track(path: Path, track: Track) -> bool:
    """Match artist then title; a numbered suffix overrides an album prefix.

    In 'Artist - Album - 02 Title', only the last component is the song title.
    This prevents an album named after song 01 from also matching song 02.
    """
    parts = re.split(r"\s+[-–—]\s+", path.stem)
    artists = [normalized_name(re.sub(r"^\s*\d+[.\s_-]+", "", part)) for part in parts[:-1]]
    if len(parts) < 2 or normalized_name(track.artist) not in artists:
        return False
    title = re.sub(r"^\s*\d+[.\s_-]+", "", parts[-1])
    title = re.sub(r"\s*[([]\s*free\s+download\s*[)\]]\s*$", "", title, flags=re.IGNORECASE)
    return normalized_name(title) == normalized_name(track.title)


def prepare(source: Path | None, verify: bool = False, root: Path = TECHNO) -> None:
    tracks = read_catalog(root)
    paths = sorted(source.rglob("*")) if source is not None else []
    wavs = [path for path in paths if path.is_file() and path.suffix.casefold() == ".wav"]
    assignments: dict[str, Path] = {}
    for path in wavs:
        matches = [track for track in tracks if matches_track(path, track)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous catalog match: {path.name}")
        if matches:
            identifier = matches[0].identifier
            if identifier in assignments:
                raise ValueError(f"Multiple recordings match {identifier}; refusing to guess.")
            assignments[identifier] = path
        else:
            print(f"Unmatched WAV: {path.name}")
    updated = []
    # Validate identities before copying or changing the catalog.
    for track in tracks:
        destination = root / "audio" / track.filename
        candidate = assignments.get(track.identifier)
        if candidate is not None and track.partition == "evaluation":
            print(f"Held out, not prepared: {track.identifier}")
            candidate = None
        if verify or candidate is None:
            if destination.is_file():
                verified_audio(track, root)
                print(f"Verified: {track.identifier}")
            else:
                print(f"Missing: {track.identifier} — {track.source}")
            updated.append(track)
            continue
        checksum = file_checksum(candidate)
        if track.sha256 is not None and checksum != track.sha256:
            raise ValueError(f"{track.identifier}: downloaded recording disagrees with catalog checksum.")
        if destination.exists() and file_checksum(destination) != checksum:
            raise ValueError(f"{destination}: existing audio has a different checksum; refusing overwrite.")
        rate, samples = load_wav(candidate)
        updated.append(replace(track, sha256=checksum, sample_rate=rate, duration_seconds=len(samples) / rate))
        del samples
    if verify:
        return
    for track in updated:
        candidate = assignments.get(track.identifier)
        if candidate is None or track.partition == "evaluation":
            continue
        destination = root / "audio" / track.filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print(f"Unchanged: {track.identifier} (matching SHA-256)")
        else:
            with candidate.open("rb") as original, destination.open("xb") as output:
                shutil.copyfileobj(original, output)
            if file_checksum(destination) != track.sha256:
                raise ValueError(f"{track.identifier}: copy checksum mismatch.")
            print(f"Prepared: {track.identifier}")
    catalog = root / "tracks.json"
    temporary = catalog.with_suffix(".json.tmp")
    temporary.write_text(json.dumps([asdict(track) for track in updated], indent=2) + "\n", encoding="utf-8")
    temporary.replace(catalog)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--source", type=Path, help="Directory of extracted, already downloaded WAVs.")
    selection.add_argument("--verify", action="store_true", help="Verify all locally available catalog audio.")
    arguments = parser.parse_args()
    try:
        if arguments.source is not None and not arguments.source.is_dir():
            raise ValueError("Source must be an existing directory.")
        prepare(arguments.source, arguments.verify)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Preparation error: {error}\n")


if __name__ == "__main__":
    main()
