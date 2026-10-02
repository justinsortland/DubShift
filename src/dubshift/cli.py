"""DubShift command-line interface."""

from pathlib import Path
from typing import Optional

import typer

from dubshift import __version__
from dubshift.downloader import (
    DEFAULT_DOWNLOAD_DIR,
    DownloadError,
    InvalidURLError,
    download_video,
)
from dubshift.media import (
    InvalidVideoPathError,
    MediaError,
    extract_audio,
)

app = typer.Typer(
    name="dubshift",
    help="Replace the spoken voice in educational videos while preserving "
    "the original video's timing and visual content.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """DubShift: voice replacement for educational videos."""


@app.command()
def version() -> None:
    """Print the installed DubShift version."""
    typer.echo(__version__)


@app.command()
def download(
    url: str = typer.Argument(..., help="YouTube video URL to download."),
    output_dir: Path = typer.Option(
        DEFAULT_DOWNLOAD_DIR,
        "--output-dir",
        "-o",
        help="Local directory to save the downloaded video in.",
    ),
) -> None:
    """Download a single YouTube video (playlists are not supported)."""
    try:
        downloaded_path = download_video(url, output_dir)
    except (InvalidURLError, DownloadError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(str(downloaded_path))


@app.command()
def extract(
    video_path: Path = typer.Argument(..., help="Path to a downloaded video file."),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write the extracted WAV file to "
        "(default: alongside the video, same name with a .wav extension).",
    ),
) -> None:
    """Extract a speech-ready WAV audio track from a video file."""
    try:
        wav_path = extract_audio(video_path, output_path)
    except (InvalidVideoPathError, MediaError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(str(wav_path))


if __name__ == "__main__":
    app()
