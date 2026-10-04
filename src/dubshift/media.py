"""Audio extraction from video files via FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# This WAV is a generic, canonical intermediate for speech-to-speech voice
# conversion, not a representation tailored to any one model. DubShift
# shouldn't throw away source audio bandwidth here: 48 kHz preserves the
# full fidelity of typical source recordings. Individual voice-conversion
# providers may require a specific (often lower, e.g. 16 kHz for a
# HuBERT/ContentVec-style content encoder) internal rate -- that
# provider-specific resampling belongs in the provider implementation, not
# in this generic extraction layer.
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 1

# PCM signed 16-bit little-endian: the uncompressed, universally supported
# sample format expected as input by virtually every speech-to-speech /
# voice-conversion toolkit.
_AUDIO_CODEC = "pcm_s16le"

# AAC is natively supported by the MP4 container and playable essentially
# everywhere (browsers, phones, editors) without a transcode step, unlike
# the PCM WAV a VoiceConverter produces after decoding -- which MP4 does
# not support as an audio codec at all.
_REPLACEMENT_AUDIO_CODEC = "aac"

# Maximum allowed difference between a video's duration and its
# replacement audio's duration before replace_audio() refuses to mux them.
# Replacing a voice must never silently truncate (or extend) the original
# video to match an audio track of a different length -- small drift
# (sub-second, from codec/container rounding) is tolerated, anything more
# is treated as a real mismatch DubShift should surface, not paper over.
DEFAULT_REMUX_DURATION_TOLERANCE_SECONDS = 0.5


class InvalidVideoPathError(ValueError):
    """Raised when the given input video path doesn't exist or isn't a file."""


class InvalidReplacementAudioPathError(ValueError):
    """Raised when the given replacement audio path doesn't exist or isn't a file."""


class MediaError(RuntimeError):
    """Base class for media-processing failures."""


class MissingFFmpegError(MediaError):
    """Raised when ffmpeg is not available on PATH."""


class OutputExistsError(MediaError):
    """Raised when the resolved output path already exists."""


class AudioExtractionError(MediaError):
    """Raised when ffmpeg fails to extract audio, or produces no output."""


class AudioReplacementError(MediaError):
    """Raised when ffmpeg fails to replace a video's audio track, or produces no output."""


class MediaDurationMismatchError(MediaError):
    """Raised when a video's and its replacement audio's durations differ beyond tolerance.

    Replacing a voice must not silently truncate (or extend) the original
    video to match a differently-sized audio track.
    """


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _probe_duration_seconds(path: Path) -> float:
    """Return the duration of ``path`` in seconds, via ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise AudioReplacementError(
            "ffprobe is required to check video/audio durations before "
            "replacing audio, but was not found on PATH. Install ffmpeg "
            "(which includes ffprobe), e.g. `brew install ffmpeg` on "
            "macOS, and try again."
        ) from exc

    duration_text = result.stdout.strip()
    if result.returncode != 0 or not duration_text:
        raise AudioReplacementError(
            f"Could not determine the duration of {path} via ffprobe "
            f"(exit code {result.returncode}): {result.stderr.strip()}"
        )

    try:
        return float(duration_text)
    except ValueError as exc:
        raise AudioReplacementError(
            f"ffprobe returned an unparseable duration for {path}: {duration_text!r}"
        ) from exc


def extract_audio(video_path: Path, output_path: Path | None = None) -> Path:
    """Extract the audio track from ``video_path`` into a speech-ready WAV file.

    The video stream is never touched or re-encoded; only the audio stream
    is demuxed and transcoded to PCM signed 16-bit little-endian at
    ``DEFAULT_SAMPLE_RATE`` Hz, ``DEFAULT_CHANNELS`` channel(s).

    If ``output_path`` is omitted, the WAV file is written next to
    ``video_path`` using the same stem (e.g. ``clip.mp4`` -> ``clip.wav``).
    An existing output file is never overwritten.
    """
    video_path = Path(video_path)
    if not video_path.exists() or not video_path.is_file():
        raise InvalidVideoPathError(f"Video file not found: {video_path}")

    if not _ffmpeg_available():
        raise MissingFFmpegError(
            "FFmpeg is required to extract audio but was not found on PATH. "
            "Install it (e.g. `brew install ffmpeg` on macOS, or see "
            "https://ffmpeg.org/download.html) and try again."
        )

    resolved_output = Path(output_path) if output_path is not None else video_path.with_suffix(".wav")

    if resolved_output.exists():
        raise OutputExistsError(f"Refusing to overwrite existing file: {resolved_output}")

    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        _AUDIO_CODEC,
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-ac",
        str(DEFAULT_CHANNELS),
        "-n",
        str(resolved_output),
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MissingFFmpegError(
            "FFmpeg is required to extract audio but was not found on PATH. "
            "Install it (e.g. `brew install ffmpeg` on macOS, or see "
            "https://ffmpeg.org/download.html) and try again."
        ) from exc

    if result.returncode != 0:
        raise AudioExtractionError(
            f"FFmpeg failed to extract audio from {video_path} "
            f"(exit code {result.returncode}):\n{result.stderr.strip()}"
        )

    if not resolved_output.exists():
        raise AudioExtractionError(
            f"ffmpeg reported success but the output file is missing: {resolved_output}"
        )

    return resolved_output


def replace_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    """Replace ``video_path``'s audio track with ``audio_path``, keeping its video untouched.

    The video stream is stream-copied (``-c:v copy``) and never re-encoded;
    every original video frame is preserved. Only the video's primary
    video stream and the replacement audio's primary audio stream are
    mapped into the output -- the original audio track is dropped
    entirely, not merged or preserved alongside the new one.

    For DubShift, the original video's timeline is authoritative. The
    video's and replacement audio's durations are probed up front and must
    agree within ``DEFAULT_REMUX_DURATION_TOLERANCE_SECONDS``; if they
    don't, this raises ``MediaDurationMismatchError`` *before* running
    ffmpeg at all, rather than muxing something with a large mismatch.
    Within that tolerance, the replacement audio is normalized -- padded
    with silence if it's slightly shorter than the video, or trimmed if
    it's slightly longer -- as part of the mux's own audio filter graph
    (no extra intermediate file), so the final container's duration
    matches the video's duration as closely as ffmpeg/container precision
    permits, rather than drifting by whatever small mismatch was present.

    An existing output file is never overwritten.
    """
    video_path = Path(video_path)
    if not video_path.exists() or not video_path.is_file():
        raise InvalidVideoPathError(f"Video file not found: {video_path}")

    audio_path = Path(audio_path)
    if not audio_path.exists() or not audio_path.is_file():
        raise InvalidReplacementAudioPathError(f"Replacement audio file not found: {audio_path}")

    if not _ffmpeg_available():
        raise MissingFFmpegError(
            "FFmpeg is required to replace audio but was not found on PATH. "
            "Install it (e.g. `brew install ffmpeg` on macOS, or see "
            "https://ffmpeg.org/download.html) and try again."
        )

    output_path = Path(output_path)
    if output_path.exists():
        raise OutputExistsError(f"Refusing to overwrite existing file: {output_path}")

    video_duration = _probe_duration_seconds(video_path)
    audio_duration = _probe_duration_seconds(audio_path)
    drift = abs(video_duration - audio_duration)
    if drift > DEFAULT_REMUX_DURATION_TOLERANCE_SECONDS:
        raise MediaDurationMismatchError(
            f"Video duration ({video_duration:.3f}s) and replacement audio "
            f"duration ({audio_duration:.3f}s) differ by {drift:.3f}s, "
            f"exceeding the {DEFAULT_REMUX_DURATION_TOLERANCE_SECONDS:.3f}s "
            "remux tolerance. Refusing to mux: this would either truncate "
            "the original video or leave it with a long silent/frozen "
            "tail."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize the replacement audio to exactly the video's duration:
    # atrim drops anything past that point (a no-op if audio is already
    # shorter), and apad's whole_dur then pads with silence up to that
    # point (a no-op if it's already that long). Harmless when the two
    # durations already match. The video stream is untouched by this --
    # only the mapped audio stream passes through this filter.
    audio_filter = f"atrim=end={video_duration:.6f},apad=whole_dur={video_duration:.6f}"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-af",
        audio_filter,
        "-c:a",
        _REPLACEMENT_AUDIO_CODEC,
        "-n",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MissingFFmpegError(
            "FFmpeg is required to replace audio but was not found on PATH. "
            "Install it (e.g. `brew install ffmpeg` on macOS, or see "
            "https://ffmpeg.org/download.html) and try again."
        ) from exc

    if result.returncode != 0:
        raise AudioReplacementError(
            f"FFmpeg failed to replace audio in {video_path} "
            f"(exit code {result.returncode}):\n{result.stderr.strip()}"
        )

    if not output_path.exists():
        raise AudioReplacementError(
            f"ffmpeg reported success but the output file is missing: {output_path}"
        )

    return output_path
