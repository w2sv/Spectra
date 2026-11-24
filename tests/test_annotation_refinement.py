"""Numerical signatures, provenance, editing, completeness, and audio-free IPC."""

import json
import socket
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from scipy.io import wavfile

from benchmarks.annotation_analysis import Candidate, Proposal, analyze_audio
from benchmarks.annotation_editing import DraftEditor, is_stale, load_draft, new_draft, reset_draft, save_draft
from benchmarks.annotation_templates import (
    TemplateDefinition,
    alternative_phase,
    classification,
    event_feature,
    load_templates,
    refine_candidates,
    save_templates,
)
from benchmarks.fixtures import file_checksum, synthetic_fixture
from benchmarks.kick_detection import main as benchmark_main
from benchmarks.review_playback import MpvPlayback
from benchmarks.tracks import Track, collection_status, reviewed_fixtures


def candidate(
    timestamp: float,
    label: Literal["unclassified", "likely", "ambiguous", "unlikely"] = "unclassified",
    aligned: bool = True,
    refined: float | None = None,
) -> Candidate:
    return {
        "timestamp": timestamp,
        "bass_evidence": 1.0,
        "prominence": 0.5,
        "broadband_evidence": 0.3,
        "nearest_beat_distance_seconds": 0.0,
        "grid_aligned": aligned,
        "classification": label,
        "refined_timestamp": refined,
    }


def proposal(candidates: list[Candidate], positive_count: int = 0) -> Proposal:
    return {
        "track_identifier": "test",
        "audio_sha256": "0" * 64,
        "annotation_algorithm_version": 2,
        "analysis_parameters": {},
        "bpm": 120.0,
        "beat_grid_phase_seconds": 0.0,
        "beat_positions": [0.0, 0.5, 1.0, 1.5],
        "unsupported_beat_positions": [],
        "kick_candidates": candidates,
        "beat_grid_validation": [],
        "duration_seconds": 2.0,
        "source_sample_rate": 48_000,
        "informational": {},
        "positive_template_count": positive_count,
    }


def test_features_preserve_spectral_shape_and_normalize_global_amplitude() -> None:
    fixture = synthetic_fixture("features", "clean", (0.5, 1.0))
    samples = fixture.load_audio()
    analysis = analyze_audio(samples, fixture.sample_rate)
    feature = event_feature(analysis, 0.5)
    quieter = event_feature(analyze_audio(samples * np.float32(0.2), fixture.sample_rate), 0.5)
    assert feature is not None and quieter is not None
    assert feature.shape == (37 * 12,)
    assert np.linalg.norm(feature) == pytest.approx(1.0)
    np.testing.assert_allclose(feature, quieter, atol=1e-7)
    powers = analysis.template_band_power.copy()
    powers[:, 3] *= 9
    changed = event_feature(replace(analysis, template_band_power=powers), 0.5)
    assert changed is not None and not np.allclose(feature, changed)
    empty = analyze_audio(np.zeros((48_000, 2), dtype=np.float32), 48_000)
    assert event_feature(empty, 0.5) is None
    assert event_feature(replace(empty, template_band_power=np.ones_like(empty.template_band_power)), 0.5) is None
    powers = np.ones_like(empty.template_band_power)
    boundary = event_feature(replace(empty, template_band_power=powers), 0.0)
    assert boundary is not None
    np.testing.assert_array_equal(boundary.reshape(37, 12)[:4], np.zeros((4, 12)))


@pytest.mark.parametrize(
    ("positive", "negative", "has_negative", "expected"),
    [
        (0.85, None, False, "likely"),
        (0.85, 0.8, True, "ambiguous"),
        (0.7, None, False, "ambiguous"),
        (0.69, None, False, "unlikely"),
        (1.0, 0.9, True, "likely"),
        (None, None, False, "unlikely"),
        (0.95, None, True, "ambiguous"),
    ],
)
def test_similarity_classification(
    positive: float | None, negative: float | None, has_negative: bool, expected: str
) -> None:
    assert classification(positive, negative, has_negative) == expected


def test_cosine_alignment_uses_template_onset_and_flags_search_boundary() -> None:
    fixture = synthetic_fixture("alignment", "clean", (0.5, 2.0, 3.5))
    analysis = analyze_audio(fixture.load_audio(), fixture.sample_rate)
    templates = TemplateDefinition("alignment", "0" * 64, [0.5])
    refined, _, _ = refine_candidates(analysis, [candidate(1.985, aligned=False), candidate(3.455)], templates)
    assert refined[0]["timestamp"] == 1.985
    assert refined[0].get("refined_timestamp") == pytest.approx(2.0, abs=0.0025)
    assert refined[0].get("positive_similarity") == pytest.approx(1.0, abs=1e-6)
    assert refined[0].get("classification") == "likely"
    assert not refined[0]["grid_aligned"]
    assert not refined[0].get("alignment_at_search_boundary")
    assert refined[1].get("alignment_at_search_boundary")
    zeros = replace(analysis, template_band_power=np.zeros_like(analysis.template_band_power))
    invalid, invalid_refs, _ = refine_candidates(zeros, [candidate(0.985)], templates)
    assert invalid_refs == [0.5]
    assert invalid[0].get("positive_similarity") is None
    assert invalid[0].get("classification") != "likely"


def test_template_persistence_checks_audio_identity_and_timestamps(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"original recording bytes")
    checksum = file_checksum(audio)
    path = tmp_path / "templates.json"
    templates = TemplateDefinition("track", checksum, [0.125, 1.0], [0.75])
    save_templates(path, templates, 2.0)
    assert load_templates(path, "track", checksum, 2.0, audio) == templates
    with pytest.raises(ValueError, match="checksum"):
        load_templates(path, "other", checksum, 2.0)
    with pytest.raises(ValueError, match="unique and sorted"):
        save_templates(path, TemplateDefinition("track", checksum, [1.0, 0.5]), 2.0)
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_templates(path, "track", checksum, 2.0, audio)


def test_draft_initialization_staleness_and_confirmed_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initial = proposal([candidate(0.5), candidate(0.75, aligned=False), candidate(1.0)])
    draft = new_draft(initial)
    assert draft.kick_onsets == [0.5, 1.0]
    assert not is_stale(draft, initial)
    timed = initial.copy()
    timed["informational"] = {"seconds": 3.0}
    assert not is_stale(draft, timed)
    classified = proposal(
        [candidate(0.5, "likely", refined=0.515), candidate(0.75, "ambiguous"), candidate(1.0, "unlikely")],
        positive_count=1,
    )
    assert new_draft(classified).kick_onsets == [0.515]
    assert new_draft(proposal([candidate(0.5)], positive_count=1)).kick_onsets == []
    assert is_stale(draft, classified)
    path = tmp_path / "draft.json"
    save_draft(path, draft, 2.0)
    content = path.read_bytes()
    assert load_draft(path, classified).kick_onsets == [0.5, 1.0]
    assert path.read_bytes() == content
    monkeypatch.setattr("builtins.input", lambda _: "no")
    with pytest.raises(ValueError, match="not confirmed"):
        reset_draft(path, classified)
    assert path.read_bytes() == content
    monkeypatch.setattr("builtins.input", lambda _: "RESET test DRAFT")
    assert reset_draft(path, classified).kick_onsets == [0.515]


def test_bulk_edits_use_half_open_intervals_deduplicate_and_undo() -> None:
    suggested = proposal(
        [candidate(0.5, "likely", refined=0.515), candidate(0.7, "unlikely"), candidate(1.0, "likely")], 1
    )
    editor = DraftEditor(new_draft(suggested), TemplateDefinition("test", "0" * 64), suggested)
    editor.add(0.52)
    # A manual annotation close to a suggestion blocks a second automatic event.
    editor.remove(0.515)
    editor.bulk("accept", 0, 1)
    assert editor.draft.kick_onsets == [0.52, 1.0]
    editor.bulk("clear", 0, 1)
    assert editor.draft.kick_onsets == [1.0]
    assert editor.undo()
    assert editor.draft.kick_onsets == [0.52, 1.0]
    editor.add(0.7)
    # Explicit manual confirmation must survive U even if it was automatic before.
    editor.draft.automatic_events["0.7"] = 0.7
    editor.bulk("unlikely", 0, 1)
    assert 0.7 in editor.draft.kick_onsets
    editor.set_template(0.6, True)
    editor.set_template(0.605, False, replace_timestamp=0.6)
    assert editor.templates.negative_onsets == [0.605]
    assert editor.templates.positive_onsets == []
    assert editor.undo()
    assert editor.templates.positive_onsets == [0.6]
    assert editor.templates.negative_onsets == []


def test_alternative_phase_is_half_a_beat_without_changing_events() -> None:
    assert alternative_phase(0.4, 120) == pytest.approx(0.15)
    assert alternative_phase(0.0, 150) == pytest.approx(0.2)


def test_diagnostic_ranking_selects_distinct_intervals_and_uses_likely_support() -> None:
    from benchmarks.annotation_review import diagnostic_sections

    candidates = [candidate(i * 0.5, "likely") for i in range(4)]
    candidates += [candidate(10 + i * 0.5, "likely") for i in range(18)]
    candidates += [candidate(20.1 + i * 0.2, "ambiguous", aligned=False) for i in range(10)]
    current = proposal(candidates, 1)
    current["duration_seconds"] = 30.0
    current["beat_positions"] = [i * 0.5 for i in range(60)]
    current["beat_grid_validation"] = [
        {
            "start_seconds": 0.0,
            "end_seconds": 30.0,
            "local_phase_seconds": 0.04,
            "phase_difference_seconds": 0.04,
            "drift_flagged": True,
        }
    ]
    selected = diagnostic_sections(current)
    assert selected["regular"] is not None and selected["regular"]["start"] == 10
    assert selected["challenging"] is not None and selected["challenging"]["start"] == 20
    assert selected["phase_discrepancy"] is not None and selected["phase_discrepancy"]["start"] == 0


def test_complete_and_partial_catalog_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path
    (root / "audio").mkdir()
    audio = root / "audio" / "available.wav"
    wavfile.write(audio, 48_000, np.zeros((48_000, 2), dtype=np.int16))
    available = Track(
        "available",
        "Artist",
        "Track",
        120,
        "test",
        "local",
        "development",
        "available.wav",
        file_checksum(audio),
        48_000,
        1.0,
    )
    missing = replace(available, identifier="missing", filename="missing.wav")
    (root / "annotations.json").write_text(
        json.dumps(
            [
                {
                    "track_identifier": "available",
                    "audio_sha256": available.sha256,
                    "confirmed_bpm": 120,
                    "confirmed_kick_onsets": [0.5],
                    "annotation_algorithm_version": 2,
                    "review_status": "reviewed",
                }
            ]
        )
    )
    monkeypatch.setattr("benchmarks.kick_detection.read_catalog", lambda: [available, missing])
    monkeypatch.setattr("benchmarks.kick_detection.reviewed_fixtures", lambda tracks: reviewed_fixtures(tracks, root))
    monkeypatch.setattr(
        "benchmarks.kick_detection.collection_status",
        lambda tracks, fixtures: collection_status(tracks, fixtures, root),
    )
    output = tmp_path / "result.json"
    command = ["benchmark", "--dataset", "techno", "--partition", "development", "--json", str(output)]
    monkeypatch.setattr("sys.argv", command)
    with pytest.raises(SystemExit) as error:
        benchmark_main()
    assert error.value.code == 2
    assert not output.exists()
    monkeypatch.setattr("sys.argv", [*command, "--allow-partial"])
    benchmark_main()
    report = json.loads(output.read_text())
    status = report["collection_completeness"]
    assert report["result_scope"] == "partial"
    assert not status["complete"]
    assert status["requested_track_identifiers"] == ["available", "missing"]
    assert status["evaluated_track_identifiers"] == ["available"]
    assert status["missing_audio"] == status["missing_reviewed_annotations"] == ["missing"]
    monkeypatch.setattr("sys.argv", [*command, "--fixture", "available"])
    benchmark_main()
    assert json.loads(output.read_text())["result_scope"] == "complete"


def test_playback_ipc_without_audio_output() -> None:
    client, server = socket.socketpair()
    server.setblocking(False)
    temporary = tempfile.TemporaryDirectory(prefix="spectra-ipc-test-")
    directory = Path(temporary.name)
    player = MpvPlayback(client, directory=temporary)

    def commands() -> list[dict[str, object]]:
        data = bytearray()
        while True:
            try:
                data.extend(server.recv(65_536))
            except BlockingIOError:
                break
        return [json.loads(line) for line in data.splitlines()]

    def send(value: object) -> None:
        server.sendall((json.dumps(value) + "\n").encode())

    try:
        assert [message["command"] for message in commands()] == [
            ["observe_property", 1, "time-pos"],
            ["observe_property", 2, "pause"],
        ]
        player.interval(1.0, 2.0)
        initial = commands()
        seek_id = initial[-1]["request_id"]
        send({"request_id": seek_id, "error": "success"})
        send({"event": "property-change", "name": "time-pos", "data": 1.25})
        assert player.poll() == 1.25
        player.toggle()
        assert commands()[-1]["command"] == ["set_property", "pause", False]
        send({"event": "property-change", "name": "time-pos", "data": 2.02})
        player.poll()
        stopped = commands()
        assert stopped[0]["command"] == ["set_property", "pause", True]
        assert stopped[-1]["command"] == ["seek", 2.0, "absolute+exact"]
        player.interval(3.0, 4.0)
        commands()
        send({"event": "property-change", "name": "time-pos", "data": 1.8})
        assert player.poll() is None  # stale pre-seek observation is ignored
        send({"event": "file-loaded"})
        player.poll()
        assert commands()[-1]["command"] == ["seek", 3.0, "absolute+exact"]
    finally:
        player.close()
        player.close()
        server.close()
    assert not directory.exists()
    assert client.fileno() == -1


def test_owned_player_startup_and_cleanup_without_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "fake-mpv"
    arguments = tmp_path / "arguments.json"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, socket, sys\n"
        f"open({str(arguments)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
        "path = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--input-ipc-server='))\n"
        "server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "server.bind(path)\nserver.listen(1)\n"
        "client, _ = server.accept()\n"
        "with client.makefile('r') as reader:\n"
        "    for line in reader:\n"
        "        message = json.loads(line)\n"
        "        if message['command'][0] == 'quit':\n"
        "            break\n"
        "client.close()\nserver.close()\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    monkeypatch.setattr("benchmarks.review_playback.shutil.which", lambda _: str(executable))
    player = MpvPlayback.start(tmp_path / "unused.wav")
    assert player is not None and player.directory is not None and player.process is not None
    directory = Path(player.directory.name)
    process = player.process
    try:
        flags = json.loads(arguments.read_text())
        assert "--pause=yes" in flags and "--no-video" in flags and "--force-window=no" in flags
        assert (directory / "ipc.sock").exists()
    finally:
        player.close()
    assert process.poll() is not None
    assert not directory.exists()
