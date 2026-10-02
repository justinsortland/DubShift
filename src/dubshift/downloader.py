"""YouTube video downloading via yt-dlp."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadError as YtDlpDownloadError

DEFAULT_DOWNLOAD_DIR = Path("downloads")

# Prefer separate best-quality video/audio streams (merged into mp4 via
# ffmpeg); fall back to a pre-merged mp4, then to whatever is best overall.
FORMAT_SELECTOR = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
MERGE_OUTPUT_FORMAT = "mp4"

_YOUTUBE_URL_RE = re.compile(
    r"^https?://"
    r"(www\.|m\.)?"
    r"(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)"
    r"[\w-]{6,}",
    re.IGNORECASE,
)


class InvalidURLError(ValueError):
    """Raised when the given input is not a usable YouTube video URL."""


class DownloadError(RuntimeError):
    """Raised when yt-dlp fails to download a video."""


class MissingFFmpegError(DownloadError):
    """Raised when ffmpeg is required to merge streams but isn't installed."""


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def validate_youtube_url(url: str) -> str:
    """Return a cleaned URL, or raise InvalidURLError for obviously bad input."""
    cleaned = (url or "").strip()
    if not cleaned or not _YOUTUBE_URL_RE.match(cleaned):
        raise InvalidURLError(f"Not a valid YouTube video URL: {url!r}")
    return cleaned


def download_video(url: str, output_dir: Path | str = DEFAULT_DOWNLOAD_DIR) -> Path:
    """Download a single YouTube video and return the path to the downloaded file.

    Playlist URLs are accepted but only the single referenced video is
    downloaded (``noplaylist: True``). The preferred format selects separate
    best-quality video and audio streams, which yt-dlp merges into a single
    mp4 via ffmpeg -- so ffmpeg must be installed.
    """
    cleaned_url = validate_youtube_url(url)

    if not _ffmpeg_available():
        raise MissingFFmpegError(
            "FFmpeg is required to merge the separately downloaded video and "
            "audio streams used by DubShift's preferred download format, but "
            "it was not found on PATH. Install it (e.g. `brew install ffmpeg` "
            "on macOS, or see https://ffmpeg.org/download.html) and try again."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "noplaylist": True,
        "outtmpl": str(output_path / "%(id)s.%(ext)s"),
        "format": FORMAT_SELECTOR,
        "merge_output_format": MERGE_OUTPUT_FORMAT,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(cleaned_url, download=True)
            filename = ydl.prepare_filename(info)
    except YtDlpDownloadError as exc:
        raise DownloadError(f"Failed to download {cleaned_url}: {exc}") from exc

    downloaded_path = Path(filename)
    if not downloaded_path.exists():
        # When yt-dlp merges separate streams, the final file is written
        # with the merge_output_format extension rather than the
        # pre-merge extension prepare_filename() computed.
        merged_path = downloaded_path.with_suffix(f".{MERGE_OUTPUT_FORMAT}")
        if merged_path.exists():
            downloaded_path = merged_path

    if not downloaded_path.exists():
        raise DownloadError(
            f"yt-dlp reported success but the output file is missing: {downloaded_path}"
        )

    return downloaded_path
