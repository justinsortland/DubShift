"""DubShift command-line interface."""

from pathlib import Path
from typing import Optional

import typer

from dubshift import __version__
from dubshift.cache import ConvertedChunkCache
from dubshift.converters import InvalidAudioPathError, VoiceConversionError, convert_audio
from dubshift.downloader import (
    DEFAULT_DOWNLOAD_DIR,
    DownloadError,
    InvalidURLError,
    download_video,
)
from dubshift.media import (
    InvalidReplacementAudioPathError,
    InvalidVideoPathError,
    MediaError,
    extract_audio,
    replace_audio,
)
from dubshift.pipeline import PipelineError, dub_video

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


@app.command()
def remux(
    video_path: Path = typer.Argument(..., help="Path to the source video file."),
    audio_path: Path = typer.Argument(..., help="Path to the replacement audio file."),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to write the remuxed MP4 to.",
    ),
) -> None:
    """Replace a video's audio track with another audio file.

    This exists to directly test the remux primitive; it is not yet part
    of a full download/extract/convert/remux pipeline command.
    """
    try:
        result_path = replace_audio(video_path, audio_path, output_path)
    except (InvalidVideoPathError, InvalidReplacementAudioPathError, MediaError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(str(result_path))


@app.command()
def dub(
    url: str = typer.Argument(..., help="YouTube video URL to dub."),
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
        help="Path to write the final dubbed MP4 to.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Bypass the converted-chunk cache: always call the provider, "
        "never read or write cached results.",
    ),
) -> None:
    """Download a YouTube video, convert its spoken voice, and remux the result into one MP4.

    Runs the full download -> extract -> convert -> remux pipeline,
    printing progress as each stage starts.
    """
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
        result_path = dub_video(
            url,
            output_path,
            voice,
            converter,
            on_progress=typer.echo,
            use_cache=not no_cache,
        )
    except (
        PipelineError,
        InvalidURLError,
        DownloadError,
        InvalidVideoPathError,
        InvalidReplacementAudioPathError,
        MediaError,
        InvalidAudioPathError,
        VoiceConversionError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Done: {result_path}")


cache_app = typer.Typer(help="Inspect or manage DubShift's converted-chunk cache.")
app.add_typer(cache_app, name="cache")


@cache_app.command("clear")
def cache_clear() -> None:
    """Delete all cached converted-chunk results."""
    cache = ConvertedChunkCache()
    removed = cache.clear()
    noun = "entry" if removed == 1 else "entries"
    typer.echo(f"Removed {removed} cached {noun} from {cache.cache_dir}")


if __name__ == "__main__":
    app()
