"""YouTube video downloading via yt-dlp."""

from __future__ import annotations

import re
from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadError as YtDlpDownloadError

DEFAULT_DOWNLOAD_DIR = Path("downloads")

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


def validate_youtube_url(url: str) -> str:
    """Return a cleaned URL, or raise InvalidURLError for obviously bad input."""
    cleaned = (url or "").strip()
    if not cleaned or not _YOUTUBE_URL_RE.match(cleaned):
        raise InvalidURLError(f"Not a valid YouTube video URL: {url!r}")
    return cleaned


def download_video(url: str, output_dir: Path | str = DEFAULT_DOWNLOAD_DIR) -> Path:
    """Download a single YouTube video and return the path to the downloaded file.

    Playlist URLs are accepted but only the single referenced video is
    downloaded (``noplaylist: True``).
    """
    cleaned_url = validate_youtube_url(url)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "noplaylist": True,
        "outtmpl": str(output_path / "%(id)s.%(ext)s"),
        # A single progressive (pre-merged audio+video) stream avoids
        # requiring ffmpeg for this milestone.
        "format": "best[ext=mp4]/best",
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
        raise DownloadError(
            f"yt-dlp reported success but the output file is missing: {downloaded_path}"
        )

    return downloaded_path
