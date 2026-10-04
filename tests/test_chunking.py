import subprocess
from pathlib import Path

import pytest

from dubshift.converters.chunking import (
    DEFAULT_MAX_CHUNK_SECONDS,
    ChunkSpan,
    _choose_chunk_boundaries,
    _parse_silencedetect_output,
    convert_long_audio,
)
from dubshift.converters.exceptions import (
    ChunkDurationMismatchError,
    ConversionFailedError,
    FinalDurationMismatchError,
)
from dubshift.converters.fake import FakeVoiceConverter


def _fail_if_called(*args: object, **kwargs: object) -> None:
    raise AssertionError("this should not be called")

# ---------------------------------------------------------------------------
# Pure unit tests: silencedetect output parsing
# ---------------------------------------------------------------------------


def test_parse_silencedetect_output_pairs_start_and_end_lines() -> None:
    stderr_text = (
        "[silencedetect @ 0x0] silence_start: 12.5\n"
        "[silencedetect @ 0x0] silence_end: 13.1 | silence_duration: 0.6\n"
        "[silencedetect @ 0x0] silence_start: 40.0\n"
        "[silencedetect @ 0x0] silence_end: 40.8 | silence_duration: 0.8\n"
    )

    intervals = _parse_silencedetect_output(stderr_text, total_duration=100.0)

    assert intervals == [(12.5, 13.1), (40.0, 40.8)]


def test_parse_silencedetect_output_handles_trailing_unclosed_silence() -> None:
    stderr_text = "[silencedetect @ 0x0] silence_start: 90.0\n"

    intervals = _parse_silencedetect_output(stderr_text, total_duration=100.0)

    assert intervals == [(90.0, 100.0)]


def test_parse_silencedetect_output_with_no_silence_returns_empty_list() -> None:
    assert _parse_silencedetect_output("no relevant lines here\n", total_duration=100.0) == []


# ---------------------------------------------------------------------------
# Pure unit tests: chunk boundary selection
# ---------------------------------------------------------------------------


def test_default_max_chunk_seconds_has_safety_margin_below_the_provider_limit() -> None:
    # Must stay below ElevenLabs' documented 300s limit by design.
    assert DEFAULT_MAX_CHUNK_SECONDS == 295.0


def test_choose_chunk_boundaries_covers_entire_source_exactly_once() -> None:
    spans = _choose_chunk_boundaries(
        total_duration=700.0,
        silence_intervals=[],
        target_chunk_seconds=270.0,
        max_chunk_seconds=295.0,
        search_window_seconds=30.0,
    )

    assert spans[0].start == 0.0
    assert spans[-1].end == 700.0
    for earlier, later in zip(spans, spans[1:]):
        assert earlier.end == later.start  # contiguous: no gap, no overlap
    assert sum(span.duration for span in spans) == pytest.approx(700.0)


def test_choose_chunk_boundaries_never_exceeds_max_chunk_seconds() -> None:
    spans = _choose_chunk_boundaries(
        total_duration=1000.0,
        silence_intervals=[],
        target_chunk_seconds=270.0,
        max_chunk_seconds=295.0,
        search_window_seconds=30.0,
    )

    assert len(spans) > 1
    for span in spans:
        assert span.duration <= 295.0 + 1e-9


def test_no_chunk_exceeds_295_seconds() -> None:
    """Explicit, literal check that no chunk is ever allowed to reach the
    documented 300s ElevenLabs limit -- or even the old 300s default."""
    spans = _choose_chunk_boundaries(
        total_duration=1000.0,
        silence_intervals=[],
        target_chunk_seconds=270.0,
        max_chunk_seconds=295.0,
        search_window_seconds=30.0,
    )

    assert len(spans) > 1
    for span in spans:
        assert span.duration <= 295.0
        assert span.duration < 300.0


def test_choose_chunk_boundaries_falls_back_to_hard_limit_without_silence() -> None:
    spans = _choose_chunk_boundaries(
        total_duration=700.0,
        silence_intervals=[],
        target_chunk_seconds=270.0,
        max_chunk_seconds=295.0,
        search_window_seconds=30.0,
    )

    # No silence anywhere: every non-final cut lands exactly at the hard max.
    assert spans[0].end == 295.0
    assert spans[1].end == 590.0
    assert spans[2].end == 700.0


def test_no_silence_fallback_cuts_at_295_not_300() -> None:
    spans = _choose_chunk_boundaries(
        total_duration=700.0,
        silence_intervals=[],
        target_chunk_seconds=270.0,
        max_chunk_seconds=295.0,
        search_window_seconds=30.0,
    )

    assert spans[0].end == 295.0
    assert spans[0].end != 300.0


def test_choose_chunk_boundaries_prefers_silence_near_target_over_hard_limit() -> None:
    # A clean pause sits almost exactly at the ideal (target-length) cut.
    spans = _choose_chunk_boundaries(
        total_duration=400.0,
        silence_intervals=[(265.0, 275.0)],
        target_chunk_seconds=270.0,
        max_chunk_seconds=295.0,
        search_window_seconds=30.0,
    )

    assert spans[0].end == pytest.approx(270.0)
    assert spans[0].duration < 295.0


def test_choose_chunk_boundaries_ignores_silence_outside_the_search_window() -> None:
    # This pause is real, but far outside the window around the ideal cut
    # (ideal=270, window=[240, 295]), so it must not be used.
    spans = _choose_chunk_boundaries(
        total_duration=700.0,
        silence_intervals=[(10.0, 11.0)],
        target_chunk_seconds=270.0,
        max_chunk_seconds=295.0,
        search_window_seconds=30.0,
    )

    assert spans[0].end == 295.0


def test_choose_chunk_boundaries_no_chunking_needed_for_short_audio() -> None:
    spans = _choose_chunk_boundaries(
        total_duration=60.0,
        silence_intervals=[],
        target_chunk_seconds=270.0,
        max_chunk_seconds=295.0,
        search_window_seconds=30.0,
    )

    assert spans == [ChunkSpan(index=0, start=0.0, end=60.0)]


# ---------------------------------------------------------------------------
# convert_long_audio: orchestration (all ffmpeg/ffprobe calls mocked)
# ---------------------------------------------------------------------------


def _make_input_audio(tmp_path: Path) -> Path:
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fake pcm audio bytes")
    return input_audio


def test_short_input_invokes_converter_once_and_skips_chunking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.converters.chunking._probe_duration_seconds", lambda path: 60.0)
    monkeypatch.setattr("dubshift.converters.chunking._detect_silence_intervals", _fail_if_called)
    monkeypatch.setattr("dubshift.converters.chunking._extract_chunk", _fail_if_called)

    def fake_decode_to_wav(input_path: Path, output_path: Path) -> None:
        output_path.write_bytes(input_path.read_bytes() + b"-wav")

    monkeypatch.setattr("dubshift.converters.chunking._decode_to_wav", fake_decode_to_wav)

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "output.wav"
    converter = FakeVoiceConverter()

    result = convert_long_audio(input_audio, output_path, "voice-1", converter)

    assert result == output_path
    assert output_path.exists()

    # Invoked exactly once -- no chunk extraction/splitting for a short input.
    assert len(converter.calls) == 1
    called_input, called_voice, called_output_path = converter.calls[0]
    assert called_input == input_audio
    assert called_voice == "voice-1"
    # The provider's raw output must land in a temp file, never directly
    # at the caller's output_path (which must end up holding normalized
    # WAV bytes, not provider-native bytes).
    assert called_output_path != output_path


def test_short_input_normalizes_provider_output_to_canonical_wav(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A short, ElevenLabs-like mp3 response must be decoded, not passed through."""
    monkeypatch.setattr("dubshift.converters.chunking._probe_duration_seconds", lambda path: 10.0)
    monkeypatch.setattr("dubshift.converters.chunking._detect_silence_intervals", _fail_if_called)
    monkeypatch.setattr("dubshift.converters.chunking._extract_chunk", _fail_if_called)

    decode_calls: list[bytes] = []

    def fake_decode_to_wav(input_path: Path, output_path: Path) -> None:
        # Capture what decoding actually received, before the temp
        # directory holding it is cleaned up.
        decode_calls.append(input_path.read_bytes())
        # Simulate ffmpeg actually transcoding the mp3 bytes into WAV bytes.
        output_path.write_bytes(b"RIFF....WAVEfmt " + input_path.read_bytes())

    monkeypatch.setattr("dubshift.converters.chunking._decode_to_wav", fake_decode_to_wav)

    class FakeElevenLabsLikeConverter:
        """Writes mp3-shaped bytes, the way ElevenLabsVoiceConverter does."""

        def convert(self, input_audio: Path, *, voice: str, output_path: Path) -> Path:
            output_path.write_bytes(b"ID3\x03\x00\x00\x00fake mp3 bytes")
            return output_path

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.wav"

    result = convert_long_audio(input_audio, output_path, "voice-1", FakeElevenLabsLikeConverter())

    assert result == output_path
    assert len(decode_calls) == 1
    # The provider's mp3 bytes were the *input* to decoding, not what ended
    # up written verbatim at output_path.
    assert decode_calls[0].startswith(b"ID3")
    assert output_path.read_bytes().startswith(b"RIFF")


def test_short_input_duration_validation_still_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_probe(path: Path) -> float:
        if path.name == "normalized.wav":
            return 20.0  # drastically different from the 10.0s source
        return 10.0

    monkeypatch.setattr("dubshift.converters.chunking._probe_duration_seconds", fake_probe)
    monkeypatch.setattr("dubshift.converters.chunking._detect_silence_intervals", _fail_if_called)
    monkeypatch.setattr("dubshift.converters.chunking._extract_chunk", _fail_if_called)

    def fake_decode_to_wav(input_path: Path, output_path: Path) -> None:
        output_path.write_bytes(b"fake normalized wav bytes")

    monkeypatch.setattr("dubshift.converters.chunking._decode_to_wav", fake_decode_to_wav)

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "output.wav"
    converter = FakeVoiceConverter()

    with pytest.raises(FinalDurationMismatchError, match="Reconstructed output duration"):
        convert_long_audio(input_audio, output_path, "voice-1", converter)

    assert not output_path.exists()


def test_short_input_failure_leaves_no_partial_output_or_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.converters.chunking._probe_duration_seconds", lambda path: 10.0)
    monkeypatch.setattr("dubshift.converters.chunking._detect_silence_intervals", _fail_if_called)
    monkeypatch.setattr("dubshift.converters.chunking._extract_chunk", _fail_if_called)

    captured_tmp_dirs: list[Path] = []

    class FailingConverter:
        def convert(self, input_audio: Path, *, voice: str, output_path: Path) -> Path:
            captured_tmp_dirs.append(output_path.parent)
            raise ConversionFailedError("provider exploded")

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "output.wav"

    with pytest.raises(ConversionFailedError):
        convert_long_audio(input_audio, output_path, "voice-1", FailingConverter())

    assert not output_path.exists()
    assert captured_tmp_dirs, "expected the converter to have been invoked before failing"
    for tmp_dir in captured_tmp_dirs:
        assert not tmp_dir.exists()


def test_decode_to_wav_uses_canonical_48khz_mono_pcm16_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The normalization step's own ffmpeg command must target the canonical format."""
    from dubshift.converters.chunking import _decode_to_wav

    captured_command: list = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        captured_command.extend(command)
        Path(command[-1]).write_bytes(b"fake wav bytes")
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("dubshift.converters.chunking.subprocess.run", fake_run)

    input_path = tmp_path / "provider_output.mp3"
    input_path.write_bytes(b"fake mp3 bytes")
    output_path = tmp_path / "normalized.wav"

    _decode_to_wav(input_path, output_path)

    assert captured_command[captured_command.index("-ar") + 1] == "48000"
    assert captured_command[captured_command.index("-ac") + 1] == "1"
    assert captured_command[captured_command.index("-c:a") + 1] == "pcm_s16le"


def _patch_chunking_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    total_duration: float,
    chunk_durations: dict[int, float] | None = None,
    converted_duration_overrides: dict[int, float] | None = None,
    final_output_duration_override: float | None = None,
) -> dict[str, list]:
    """Patch every ffmpeg-backed helper with simple, file-producing fakes.

    Returns a dict of call logs the test can assert against.
    """
    calls: dict[str, list] = {
        "extract": [],
        "decode": [],
        "concat": [],
    }
    chunk_durations = chunk_durations or {}
    converted_duration_overrides = converted_duration_overrides or {}

    def fake_probe_duration(path: Path) -> float:
        if path.name.startswith("source_"):
            index = int(path.stem.split("_")[1])
            return chunk_durations.get(index, 90.0)
        if path.name.startswith("converted_"):
            index = int(path.stem.split("_")[1])
            if index in converted_duration_overrides:
                return converted_duration_overrides[index]
            return chunk_durations.get(index, 90.0)
        # "reconstructed.wav" (long/chunked path) and "normalized.wav"
        # (short path) are the not-yet-published temp files whose
        # duration is checked against the source before being moved into
        # the real output_path.
        if path.name in ("reconstructed.wav", "normalized.wav") and final_output_duration_override is not None:
            return final_output_duration_override
        return total_duration

    def fake_extract_chunk(source_path: Path, span, output_path: Path) -> None:
        calls["extract"].append(span.index)
        output_path.write_bytes(f"source-chunk-{span.index}".encode())

    def fake_decode_to_wav(input_path: Path, output_path: Path) -> None:
        calls["decode"].append(input_path.name)
        output_path.write_bytes(input_path.read_bytes() + b"-decoded")

    def fake_concatenate_wavs(wav_paths: list[Path], output_path: Path, *, work_dir: Path) -> None:
        calls["concat"].append(list(wav_paths))
        output_path.write_bytes(b"".join(p.read_bytes() for p in wav_paths))

    monkeypatch.setattr("dubshift.converters.chunking._probe_duration_seconds", fake_probe_duration)
    monkeypatch.setattr(
        "dubshift.converters.chunking._detect_silence_intervals", lambda path, duration: []
    )
    monkeypatch.setattr("dubshift.converters.chunking._extract_chunk", fake_extract_chunk)
    monkeypatch.setattr("dubshift.converters.chunking._decode_to_wav", fake_decode_to_wav)
    monkeypatch.setattr("dubshift.converters.chunking._concatenate_wavs", fake_concatenate_wavs)

    return calls


def test_long_input_produces_multiple_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _patch_chunking_pipeline(monkeypatch, total_duration=700.0)

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "output.wav"
    converter = FakeVoiceConverter()

    result = convert_long_audio(input_audio, output_path, "voice-1", converter)

    assert result == output_path
    assert output_path.exists()
    # 700s with target=270/max=295 -> three chunks: [0,295],[295,590],[590,700].
    assert calls["extract"] == [0, 1, 2]
    assert len(converter.calls) == 3
    assert len(calls["decode"]) == 3
    assert len(calls["concat"]) == 1


def test_reconstruction_preserves_chunk_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _patch_chunking_pipeline(monkeypatch, total_duration=700.0)

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "output.wav"
    converter = FakeVoiceConverter()

    convert_long_audio(input_audio, output_path, "voice-1", converter)

    concatenated_paths = calls["concat"][0]
    decoded_names = [p.name for p in concatenated_paths]
    assert decoded_names == ["decoded_000.wav", "decoded_001.wav", "decoded_002.wav"]

    # The final file's bytes reflect the chunks concatenated in that exact order.
    expected = b"".join((f"source-chunk-{i}".encode() + b"-decoded") for i in range(3))
    assert output_path.read_bytes() == expected


def test_duration_mismatch_is_detected_and_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _patch_chunking_pipeline(
        monkeypatch,
        total_duration=700.0,
        chunk_durations={0: 300.0, 1: 300.0, 2: 100.0},
        converted_duration_overrides={1: 305.0},  # 5s drift, exceeds default 0.5s tolerance
    )

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "output.wav"
    converter = FakeVoiceConverter()

    with pytest.raises(ChunkDurationMismatchError, match="Chunk 1"):
        convert_long_audio(input_audio, output_path, "voice-1", converter)

    # Concatenation must never run once a mismatch is found.
    assert calls["concat"] == []
    assert not output_path.exists()


def test_duration_within_tolerance_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_chunking_pipeline(
        monkeypatch,
        total_duration=700.0,
        chunk_durations={0: 300.0, 1: 300.0, 2: 100.0},
        converted_duration_overrides={1: 300.2},  # within default 0.5s tolerance
    )

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "output.wav"
    converter = FakeVoiceConverter()

    result = convert_long_audio(input_audio, output_path, "voice-1", converter)

    assert result == output_path
    assert output_path.exists()


def test_final_duration_check_passes_for_correctly_reconstructed_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A correctly reconstructed file (no per-chunk or cumulative drift) must pass cleanly."""
    _patch_chunking_pipeline(
        monkeypatch,
        total_duration=700.0,
        chunk_durations={0: 300.0, 1: 300.0, 2: 100.0},
        # No overrides: every probe reports exactly the true duration, so
        # both the per-chunk and final whole-file invariants hold exactly.
    )

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "output.wav"
    converter = FakeVoiceConverter()

    result = convert_long_audio(input_audio, output_path, "voice-1", converter)

    assert result == output_path
    assert output_path.exists()


def test_cumulative_final_drift_rejected_even_if_every_chunk_is_individually_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _patch_chunking_pipeline(
        monkeypatch,
        total_duration=700.0,
        chunk_durations={0: 300.0, 1: 300.0, 2: 100.0},
        # Each chunk drifts by only 0.3s -- within the 0.5s per-chunk tolerance.
        converted_duration_overrides={0: 300.3, 1: 300.3, 2: 100.3},
        # But the fully reconstructed output measures 0.9s off overall,
        # exceeding the 0.5s whole-file tolerance.
        final_output_duration_override=700.9,
    )

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "output.wav"
    converter = FakeVoiceConverter()

    with pytest.raises(FinalDurationMismatchError, match="Reconstructed output duration"):
        convert_long_audio(input_audio, output_path, "voice-1", converter)

    # Every chunk passed its own check, so concatenation did run; only the
    # final whole-file invariant caught the cumulative drift.
    assert len(calls["concat"]) == 1
    assert not output_path.exists()


def test_temp_chunk_files_are_cleaned_up_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_tmp_dirs: list[Path] = []
    original_extract = None

    calls = _patch_chunking_pipeline(monkeypatch, total_duration=700.0)

    def capturing_extract_chunk(source_path: Path, span, output_path: Path) -> None:
        captured_tmp_dirs.append(output_path.parent)
        output_path.write_bytes(f"source-chunk-{span.index}".encode())

    monkeypatch.setattr("dubshift.converters.chunking._extract_chunk", capturing_extract_chunk)

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "output.wav"
    converter = FakeVoiceConverter()

    convert_long_audio(input_audio, output_path, "voice-1", converter)

    assert captured_tmp_dirs, "expected at least one chunk to be extracted"
    for tmp_dir in captured_tmp_dirs:
        assert not tmp_dir.exists()


def test_temp_chunk_files_are_cleaned_up_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_tmp_dirs: list[Path] = []

    _patch_chunking_pipeline(
        monkeypatch,
        total_duration=700.0,
        chunk_durations={0: 300.0, 1: 300.0, 2: 100.0},
        converted_duration_overrides={0: 310.0},  # fails on the very first chunk
    )

    def capturing_extract_chunk(source_path: Path, span, output_path: Path) -> None:
        captured_tmp_dirs.append(output_path.parent)
        output_path.write_bytes(f"source-chunk-{span.index}".encode())

    monkeypatch.setattr("dubshift.converters.chunking._extract_chunk", capturing_extract_chunk)

    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "output.wav"
    converter = FakeVoiceConverter()

    with pytest.raises(ChunkDurationMismatchError):
        convert_long_audio(input_audio, output_path, "voice-1", converter)

    assert captured_tmp_dirs, "expected at least one chunk to be extracted before failing"
    for tmp_dir in captured_tmp_dirs:
        assert not tmp_dir.exists()
    assert not output_path.exists()
