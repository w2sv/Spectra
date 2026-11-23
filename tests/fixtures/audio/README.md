# Curated audio references

Six short references, each stored once as **48 kHz stereo signed 16-bit PCM WAV**.
Full sources and archive contents stay in ignored `samples/originals/`; experiment
reports belong under ignored `samples/results/`.

All source recordings are published under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
Artist credits are retained for provenance. The source pages and original
downloads, source/fixture SHA-256 hashes, conversion settings, excerpt boundaries,
and decoded signal checks are recorded in [manifest.json](manifest.json).

| Fixture | Source excerpt | Characteristics and purpose |
| --- | --- | --- |
| `lyserb.wav` | 16–28 s | [Lyserb — cinameng / James Gargette](https://opengameart.org/content/lyserb): lo-fi techno drum loop; basic rhythmic bass transients. |
| `160bpm-electronic.wav` | 12–24 s | [160BPM Electronic Loop — RUOK](https://opengameart.org/content/160bpm-electronic-loop): faster beat and closely spaced percussion; 32 beats at the source's stated tempo. |
| `empacotatron.wav` | 20–32 s | [Empacotatron — Fupi](https://opengameart.org/content/empacotatron): dense electronic arrangement, bleeps, and competing transients; taken from lossless FLAC. |
| `algate.wav` | Complete 15.118125 s loop | [Algate — cinameng / James Gargette](https://opengameart.org/content/algate): bass-heavy dub-techno full mix, taken from `Algate/wav/algate.wav` in the ZIP, not its MP3 copy. |
| `synth-pop-four-on-floor.wav` | 45–57 s | [Synth Pop with 4 on the Floor — Mesostic](https://commons.wikimedia.org/wiki/File:Synth_pop_with_4_on_the_floor.ogg): regular drum pattern with synth pads, bass, and melody. |
| `synth-pop-bass-only.wav` | 45–57 s | [Bass stem — Mesostic](https://commons.wikimedia.org/wiki/File:Stem_3_from_%22Synth_pop_with_10_on_the_floor%22.ogg): the matching bass-only track; intended negative case for kick discrimination, with bass-note attacks but no drum track. |

The Commons bass filename says “10 on the floor”; its description identifies it
as stem 3 of the “4 on the Floor” full mix. Both excerpts use the same source-time
range. A bass-only reference is not a promise that the present detector will
produce zero events: it detects bass transients and has not been retuned here.

Audio payload: **14,422,944 bytes (13.75 MiB), 75.118125 seconds**.
Sources were decoded with FFmpeg's error checks before inclusion; each excerpt
was checked for the required format, length, nonzero content, peak/RMS, and
frequency-energy distribution. Selections did not use detector event output.

## Analysis

```bash
uv run python scripts/audio_test.py analyze tests/fixtures/audio/lyserb.wav
uv run python scripts/audio_test.py analyze tests/fixtures/audio \
  --output samples/results/collection-default.json
```

See [the workflow guide](../../../docs/audio-testing.md) for private-sink playback,
excerpt selection, configuration comparisons, and importing other recordings.

## Future annotations

`reference_kicks_seconds` is currently `null` in every manifest entry. Add
independently reviewed, excerpt-relative reference timestamps there later.
Do not copy detector output into those fields. Detection reports are experiment
artifacts, not ground truth, and no expected detection counts are frozen here.
