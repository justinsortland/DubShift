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


class InvalidVideoPathError(ValueError):
    """Raised when the given input path doesn't exist or isn't a file."""


class MediaError(RuntimeError):
    """Base class for audio-extraction failures."""


class MissingFFmpegError(MediaError):
    """Raised when ffmpeg is not available on PATH."""


class OutputExistsError(MediaError):
    """Raised when the resolved output path already exists."""


class AudioExtractionError(MediaError):
    """Raised when ffmpeg fails to extract audio, or produces no output."""


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


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
