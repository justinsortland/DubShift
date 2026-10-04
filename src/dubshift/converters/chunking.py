"""Provider-agnostic long-audio chunking and reconstruction.

Allows voice conversion of audio longer than a provider's single-request
duration limit by splitting it into speech-safe chunks (cutting at a
silence / low-energy pause near the target boundary where one exists),
converting each chunk independently through the existing ``VoiceConverter``
interface (via ``convert_audio``), and reconstructing one continuous WAV
file from the converted chunks.

Splitting and reconstruction are done with ffmpeg/ffprobe rather than a
dedicated audio-processing library, consistent with the rest of DubShift.
Nothing here is specific to ElevenLabs or any other provider -- the chunk
size limit is a parameter, not a hardcoded assumption.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dubshift.cache import ConvertedChunkCache, compute_cache_key, provider_identity_for
from dubshift.converters.base import VoiceConverter
from dubshift.converters.exceptions import (
    ChunkDurationMismatchError,
    ChunkingError,
    ConversionOutputExistsError,
    FinalDurationMismatchError,
    InvalidAudioPathError,
)
from dubshift.converters.orchestration import convert_audio
from dubshift.media import DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE
from dubshift.progress import ProgressCallback, noop_progress

# Target length for each chunk: comfortably under the 5-minute provider
# limit so that a handful of seconds of boundary slack (searching for a
# nearby pause) never pushes a chunk over the hard cap.
DEFAULT_TARGET_CHUNK_SECONDS = 4.5 * 60  # 270s

# Hard ceiling no chunk may ever exceed. Deliberately kept below
# ElevenLabs' documented 300s (5 minute) Voice Changer request limit (see
# dubshift.converters.elevenlabs.MAX_INPUT_DURATION_SECONDS) -- the 5s
# margin absorbs codec/container/duration-probe rounding so a chunk that
# measures at (or just under) this cap never gets rejected for actually
# landing at or past the provider's real limit. Chunking itself is
# provider-agnostic, so this is a parameter here rather than an import
# from a specific provider module.
DEFAULT_MAX_CHUNK_SECONDS = 295.0  # 5s of safety margin below the 300s provider limit

# How far around the ideal (target-length) cut point to look for a silent
# pause, in seconds. The search window is centered on the ideal cut and
# clamped so it never extends past the hard max.
DEFAULT_BOUNDARY_SEARCH_SECONDS = 30.0

# ffmpeg silencedetect parameters. -35dB is quiet enough to catch natural
# pauses between sentences without tripping on soft speech; 0.3s is a
# typical minimum pause length in natural speech.
_SILENCE_NOISE_THRESHOLD_DB = -35
_SILENCE_MIN_DURATION_SECONDS = 0.3

# Allowed drift between a source chunk's duration and its converted
# duration before DubShift refuses to assemble the final output. A real
# ElevenLabs smoke test showed near-perfect preservation (15.000000s in,
# 15.000091s out), so this has ample headroom for normal codec/container
# rounding while still catching a chunk that was clearly mishandled
# (truncated, duplicated, or decoded at the wrong rate).
DEFAULT_DURATION_TOLERANCE_SECONDS = 0.5

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


@dataclass(frozen=True)
class ChunkSpan:
    """A [start, end) slice of the source timeline, in seconds."""

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _probe_duration_seconds(audio_path: Path) -> float:
    """Return the duration of ``audio_path`` in seconds, via ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise ChunkingError(
            "ffprobe is required for audio chunking but was not found on "
            "PATH. Install ffmpeg (which includes ffprobe), e.g. "
            "`brew install ffmpeg` on macOS, and try again."
        ) from exc

    duration_text = result.stdout.strip()
    if result.returncode != 0 or not duration_text:
        raise ChunkingError(
            f"Could not determine the duration of {audio_path} via ffprobe "
            f"(exit code {result.returncode}): {result.stderr.strip()}"
        )

    try:
        return float(duration_text)
    except ValueError as exc:
        raise ChunkingError(
            f"ffprobe returned an unparseable duration for {audio_path}: {duration_text!r}"
        ) from exc


def _parse_silencedetect_output(stderr_text: str, total_duration: float) -> list[tuple[float, float]]:
    """Pure parsing of ffmpeg's ``silencedetect`` filter log lines into (start, end) pairs.

    Separated from the subprocess call so the parsing logic itself can be
    unit-tested with canned ffmpeg output, with no subprocess involved.
    """
    intervals: list[tuple[float, float]] = []
    pending_start: float | None = None

    for line in stderr_text.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            intervals.append((pending_start, float(end_match.group(1))))
            pending_start = None

    if pending_start is not None:
        # Silence was still ongoing when the file ended (no matching
        # silence_end line was logged for it).
        intervals.append((pending_start, total_duration))

    return intervals


def _detect_silence_intervals(audio_path: Path, total_duration: float) -> list[tuple[float, float]]:
    """Return (start, end) pairs, in seconds, of near-silent spans in ``audio_path``."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={_SILENCE_NOISE_THRESHOLD_DB}dB:duration={_SILENCE_MIN_DURATION_SECONDS}",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise ChunkingError(
            "ffmpeg is required for audio chunking but was not found on "
            "PATH. Install it (e.g. `brew install ffmpeg` on macOS) and "
            "try again."
        ) from exc

    if result.returncode != 0:
        raise ChunkingError(
            f"ffmpeg silence detection failed for {audio_path} "
            f"(exit code {result.returncode}): {result.stderr.strip()}"
        )

    return _parse_silencedetect_output(result.stderr, total_duration)


def _choose_chunk_boundaries(
    total_duration: float,
    silence_intervals: list[tuple[float, float]],
    *,
    target_chunk_seconds: float,
    max_chunk_seconds: float,
    search_window_seconds: float,
) -> list[ChunkSpan]:
    """Pick contiguous [start, end) spans covering [0, total_duration] exactly once.

    Walks forward from 0, and for each chunk looks for a silence interval
    whose midpoint falls within ``search_window_seconds`` of the ideal
    (target-length) cut point, clamped so the chunk never exceeds
    ``max_chunk_seconds``. If no such pause exists, falls back to cutting
    exactly at the hard max -- never further.
    """
    boundaries: list[float] = []
    cursor = 0.0
    epsilon = 1e-6

    while total_duration - cursor > max_chunk_seconds + epsilon:
        ideal_cut = cursor + target_chunk_seconds
        hard_limit_cut = cursor + max_chunk_seconds

        window_start = max(cursor, ideal_cut - search_window_seconds)
        window_end = min(hard_limit_cut, ideal_cut + search_window_seconds)

        best_cut: float | None = None
        best_distance: float | None = None
        for silence_start, silence_end in silence_intervals:
            clipped_start = max(silence_start, window_start)
            clipped_end = min(silence_end, window_end)
            if clipped_start >= clipped_end:
                continue
            midpoint = (clipped_start + clipped_end) / 2
            distance = abs(midpoint - ideal_cut)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_cut = midpoint

        cut = best_cut if best_cut is not None else hard_limit_cut
        cut = min(cut, hard_limit_cut)  # never exceed the hard limit, even from float error
        boundaries.append(cut)
        cursor = cut

    spans: list[ChunkSpan] = []
    start = 0.0
    for boundary in boundaries:
        spans.append(ChunkSpan(index=len(spans), start=start, end=boundary))
        start = boundary
    spans.append(ChunkSpan(index=len(spans), start=start, end=total_duration))
    return spans


def _run_ffmpeg(command: list[str], *, context: str) -> None:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise ChunkingError(
            "ffmpeg is required for audio chunking but was not found on "
            "PATH. Install it (e.g. `brew install ffmpeg` on macOS) and "
            "try again."
        ) from exc

    if result.returncode != 0:
        raise ChunkingError(f"ffmpeg failed while {context} (exit code {result.returncode}): {result.stderr.strip()}")


def _extract_chunk(source_path: Path, span: ChunkSpan, output_path: Path) -> None:
    """Losslessly trim [span.start, span.end) from ``source_path`` into a PCM WAV."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-ss",
        f"{span.start:.3f}",
        "-t",
        f"{span.duration:.3f}",
        "-i",
        str(source_path),
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-ac",
        str(DEFAULT_CHANNELS),
        "-y",
        str(output_path),
    ]
    _run_ffmpeg(command, context=f"extracting chunk {span.index}")


def _decode_to_wav(input_path: Path, output_path: Path) -> None:
    """Decode (or re-encode) ``input_path`` to DubShift's canonical PCM WAV format.

    Converted chunks may come back from a provider in any container/codec
    (e.g. mp3 from ElevenLabs); every chunk is normalized to the same
    format here before concatenation, so the concat step can be a lossless
    stream copy.
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(input_path),
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-ac",
        str(DEFAULT_CHANNELS),
        "-y",
        str(output_path),
    ]
    _run_ffmpeg(command, context=f"decoding {input_path.name} to WAV")


def _escape_concat_path(path: Path) -> str:
    # ffmpeg concat demuxer file format: single-quoted, with embedded
    # single quotes escaped as '\''.
    return str(path).replace("'", "'\\''")


def _concatenate_wavs(wav_paths: list[Path], output_path: Path, *, work_dir: Path) -> None:
    """Losslessly concatenate same-format WAV files, in order, into ``output_path``."""
    list_file = work_dir / "concat_list.txt"
    list_file.write_text("".join(f"file '{_escape_concat_path(p)}'\n" for p in wav_paths))

    command = [
        "ffmpeg",
        "-hide_banner",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        "-y",
        str(output_path),
    ]
    _run_ffmpeg(command, context="concatenating converted chunks")


def _verify_final_duration(result_path: Path, expected_duration: float, tolerance: float) -> None:
    """Verify ``result_path``'s duration matches ``expected_duration`` within ``tolerance``.

    Removes ``result_path`` and raises ``FinalDurationMismatchError`` if it
    doesn't -- DubShift must never publish a result with unexplained drift
    from the source timeline, whether that result came from a single
    provider call or from reassembling many chunks. ``result_path`` is
    typically a not-yet-published temporary file, verified before it is
    moved into the real output path.
    """
    final_duration = _probe_duration_seconds(result_path)
    final_drift = abs(final_duration - expected_duration)
    if final_drift > tolerance:
        result_path.unlink(missing_ok=True)
        raise FinalDurationMismatchError(
            f"Reconstructed output duration ({final_duration:.3f}s) differs "
            f"from the source duration ({expected_duration:.3f}s) by "
            f"{final_drift:.3f}s, exceeding the {tolerance:.3f}s tolerance. "
            "Refusing to publish a result with this much drift from the "
            "source timeline."
        )


@dataclass(frozen=True)
class _CacheableConversion:
    """The outcome of converting one unit (a chunk, or a whole short input)."""

    was_cache_hit: bool
    cache_key: str | None


def _convert_with_cache(
    source_path: Path,
    dest_path: Path,
    voice: str,
    converter: VoiceConverter,
    cache: ConvertedChunkCache | None,
) -> _CacheableConversion:
    """Run ``convert_audio``, reusing a cached result for identical content when possible.

    If ``cache`` is ``None``, caching is disabled entirely: always calls
    the provider, never reads or writes the cache (this is what
    ``use_cache=False`` / ``--no-cache`` resolves to).

    Otherwise, computes a content-addressed key from ``source_path``'s
    exact bytes plus ``voice`` and the converter's identity (see
    ``dubshift.cache``). On a hit, copies the cached bytes to ``dest_path``
    and skips the provider call entirely. On a miss, calls the provider as
    usual; the caller is responsible for persisting the result to the
    cache -- but only *after* validating it, since a failed, partial, or
    duration-invalid conversion must never be cached.
    """
    if cache is None:
        convert_audio(source_path, dest_path, voice, converter)
        return _CacheableConversion(was_cache_hit=False, cache_key=None)

    cache_key = compute_cache_key(
        source_path, voice=voice, provider_identity=provider_identity_for(converter)
    )
    cached_path = cache.get(cache_key)
    if cached_path is not None:
        shutil.copyfile(cached_path, dest_path)
        return _CacheableConversion(was_cache_hit=True, cache_key=cache_key)

    convert_audio(source_path, dest_path, voice, converter)
    return _CacheableConversion(was_cache_hit=False, cache_key=cache_key)


def convert_long_audio(
    input_path: Path,
    output_path: Path,
    voice: str,
    converter: VoiceConverter,
    *,
    target_chunk_seconds: float = DEFAULT_TARGET_CHUNK_SECONDS,
    max_chunk_seconds: float = DEFAULT_MAX_CHUNK_SECONDS,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    on_progress: ProgressCallback = noop_progress,
    use_cache: bool = True,
    cache: ConvertedChunkCache | None = None,
) -> Path:
    """Convert ``input_path`` through ``converter``, always producing DubShift's
    canonical WAV (PCM signed 16-bit little-endian, 48kHz, mono) at ``output_path``.

    If ``input_path`` is already at or under ``max_chunk_seconds``, the
    existing fast/single-request path is used -- ``converter`` is invoked
    exactly once, with no chunk extraction -- but whatever format it
    returns (e.g. an mp3 from ElevenLabs) is then decoded into the
    canonical WAV via the same normalization step the chunked path uses,
    so a caller can always rely on ``output_path`` containing real
    WAV/PCM data, never provider-native bytes under a ``.wav`` name.

    Otherwise, the audio is split into chunks (cut at a silence/low-energy
    pause near each target boundary when one exists, else at the hard
    ``max_chunk_seconds`` limit), each chunk is converted independently
    through ``converter`` via ``convert_audio``, every converted chunk's
    duration is checked against its source chunk's duration within
    ``duration_tolerance_seconds``, each converted chunk is decoded to the
    same canonical PCM WAV format, and all chunks are losslessly
    concatenated, in order, into ``output_path`` as one continuous WAV
    file.

    In both cases, the complete output's total duration is finally
    checked against the complete source duration (within
    ``duration_tolerance_seconds``) -- catching drift a provider
    introduced (for a short input) or that accumulated across chunks that
    each individually passed their own per-chunk check (for a long one) --
    and the output is removed and a ``FinalDurationMismatchError`` raised
    if that invariant fails. All intermediate files live in a temporary
    directory that is removed whether this succeeds or raises.

    Unless ``use_cache`` is ``False``, each unit converted through the
    provider (the whole input for a short file, or each chunk for a long
    one) is looked up in a persistent, content-addressed cache first --
    see ``dubshift.cache`` -- and skipped entirely on a hit. A fresh
    conversion is only written to the cache *after* it passes its
    duration check, so a failed, partial, or duration-invalid result is
    never cached. ``cache`` lets a caller supply a specific
    ``ConvertedChunkCache`` instance (mainly for testing); by default a
    cache rooted at the on-disk default location is used.

    ``on_progress``, if given, is called with short human-readable status
    strings (e.g. ``"Converting chunk 2/4..."``) as each stage starts.
    """
    input_path = Path(input_path)
    if not input_path.exists() or not input_path.is_file():
        raise InvalidAudioPathError(f"Input audio file not found: {input_path}")

    output_path = Path(output_path)
    if output_path.exists():
        raise ConversionOutputExistsError(f"Refusing to overwrite existing file: {output_path}")

    active_cache = (cache if cache is not None else ConvertedChunkCache()) if use_cache else None

    total_duration = _probe_duration_seconds(input_path)

    if total_duration <= max_chunk_seconds:
        on_progress("Converting audio...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="dubshift-convert-") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            provider_output_path = tmp_dir / "provider_output.chunk"
            result = _convert_with_cache(
                input_path, provider_output_path, voice, converter, active_cache
            )

            normalized_path = tmp_dir / "normalized.wav"
            _decode_to_wav(provider_output_path, normalized_path)
            _verify_final_duration(normalized_path, total_duration, duration_tolerance_seconds)

            if not result.was_cache_hit and result.cache_key is not None:
                active_cache.put(result.cache_key, provider_output_path)

            shutil.move(str(normalized_path), str(output_path))

        return output_path

    on_progress("Detecting chunk boundaries...")
    silence_intervals = _detect_silence_intervals(input_path, total_duration)
    spans = _choose_chunk_boundaries(
        total_duration,
        silence_intervals,
        target_chunk_seconds=target_chunk_seconds,
        max_chunk_seconds=max_chunk_seconds,
        search_window_seconds=DEFAULT_BOUNDARY_SEARCH_SECONDS,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dubshift-chunks-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        decoded_chunk_paths: list[Path] = []

        for span in spans:
            on_progress(f"Converting chunk {span.index + 1}/{len(spans)}...")
            source_chunk_path = tmp_dir / f"source_{span.index:03d}.wav"
            _extract_chunk(input_path, span, source_chunk_path)
            source_chunk_duration = _probe_duration_seconds(source_chunk_path)

            converted_chunk_path = tmp_dir / f"converted_{span.index:03d}.chunk"
            result = _convert_with_cache(
                source_chunk_path, converted_chunk_path, voice, converter, active_cache
            )
            converted_chunk_duration = _probe_duration_seconds(converted_chunk_path)

            drift = abs(converted_chunk_duration - source_chunk_duration)
            if drift > duration_tolerance_seconds:
                raise ChunkDurationMismatchError(
                    f"Chunk {span.index} duration mismatch: source chunk is "
                    f"{source_chunk_duration:.3f}s but the converted chunk is "
                    f"{converted_chunk_duration:.3f}s (drift {drift:.3f}s "
                    f"exceeds tolerance {duration_tolerance_seconds:.3f}s). "
                    "Refusing to assemble the final output to avoid "
                    "cumulative audio/video drift."
                )

            if not result.was_cache_hit and result.cache_key is not None:
                active_cache.put(result.cache_key, converted_chunk_path)

            decoded_chunk_path = tmp_dir / f"decoded_{span.index:03d}.wav"
            _decode_to_wav(converted_chunk_path, decoded_chunk_path)
            decoded_chunk_paths.append(decoded_chunk_path)

        on_progress("Reconstructing audio...")
        reconstructed_path = tmp_dir / "reconstructed.wav"
        _concatenate_wavs(decoded_chunk_paths, reconstructed_path, work_dir=tmp_dir)
        _verify_final_duration(reconstructed_path, total_duration, duration_tolerance_seconds)

        shutil.move(str(reconstructed_path), str(output_path))

    return output_path
