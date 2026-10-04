"""DubShift command-line interface."""

from pathlib import Path
from typing import Optional

import typer

from dubshift import __version__
from dubshift.converters import InvalidAudioPathError, VoiceConversionError, convert_audio
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

SUPPORTED_CONVERSION_PROVIDERS = {"elevenlabs"}

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


@app.command()
def convert(
    input_audio: Path = typer.Argument(..., help="Path to the audio file to convert."),
    provider: str = typer.Option(
        ...,
        "--provider",
        help="Voice-conversion backend to use. Currently only 'elevenlabs' is supported.",
    ),
    voice: str = typer.Option(
        ...,
        "--voice",
        help="Backend-specific voice identifier (e.g. an ElevenLabs voice_id).",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to write the converted audio to.",
    ),
) -> None:
    """Convert the voice in an audio file using a pluggable voice-conversion provider."""
    if provider == "elevenlabs":
        from dubshift.converters.elevenlabs import ElevenLabsVoiceConverter

        try:
            converter = ElevenLabsVoiceConverter()
        except VoiceConversionError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    else:
        supported = ", ".join(sorted(SUPPORTED_CONVERSION_PROVIDERS))
        typer.echo(f"Error: Unknown provider {provider!r}. Supported providers: {supported}.", err=True)
        raise typer.Exit(code=1)

    try:
        result_path = convert_audio(input_audio, output_path, voice, converter)
    except (InvalidAudioPathError, VoiceConversionError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(str(result_path))


if __name__ == "__main__":
    app()
