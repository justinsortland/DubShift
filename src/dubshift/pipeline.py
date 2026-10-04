"""End-to-end download -> extract -> convert -> remux orchestration.

Wires together the already-existing, independently testable building
blocks -- ``dubshift.downloader.download_video``,
``dubshift.media.extract_audio``, ``dubshift.converters.convert_long_audio``,
and ``dubshift.media.replace_audio`` -- into the single ``dubshift dub``
pipeline. No stage's implementation is duplicated here; this module only
decides what gets called, in what order, and where intermediate files
live.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from dubshift.converters import VoiceConverter, convert_long_audio
from dubshift.downloader import download_video
from dubshift.media import extract_audio, replace_audio
from dubshift.progress import ProgressCallback, noop_progress


class PipelineError(RuntimeError):
    """Base class for dub-pipeline orchestration failures."""


class PipelineOutputExistsError(PipelineError):
    """Raised when the pipeline's final output path already exists.

    Checked before any downloading or provider/API calls, since voice
    conversion can consume paid API credits -- DubShift should never spend
    those only to discover at the very end that the output can't be
    written.
    """


def dub_video(
    url: str,
    output_path: Path,
    voice: str,
    converter: VoiceConverter,
    *,
    on_progress: ProgressCallback = noop_progress,
    use_cache: bool = True,
) -> Path:
    """Download ``url``, convert its spoken voice to ``voice``, and remux the result.

    Runs, in order: ``download_video`` -> ``extract_audio`` ->
    ``convert_long_audio`` (which transparently chunks long audio when
    needed -- the caller never has to care) -> ``replace_audio``, writing
    the final dubbed MP4 to ``output_path``.

    Every intermediate artifact (the downloaded video, the extracted
    source WAV, the converted WAV) lives in one temporary directory that
    is removed whether this succeeds or raises; ``output_path`` is never
    inside that directory, and is the only file this leaves behind.

    ``output_path`` is checked up front, before any network or provider
    call: an existing file there is left completely untouched and this
    raises ``PipelineOutputExistsError`` immediately.

    ``on_progress``, if given, is called with short human-readable status
    strings (e.g. ``"Downloading video..."``) as each stage starts;
    ``convert_long_audio``'s own finer-grained progress (chunk-by-chunk)
    is reported through the same callback. ``use_cache`` is forwarded to
    ``convert_long_audio`` -- set it to ``False`` to bypass the
    converted-chunk cache entirely (``--no-cache``).
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise PipelineOutputExistsError(
            f"Refusing to overwrite existing file: {output_path}. Checked "
            "before downloading or converting anything, since voice "
            "conversion can consume API credits."
        )

    with tempfile.TemporaryDirectory(prefix="dubshift-dub-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)

        on_progress("Downloading video...")
        video_path = download_video(url, tmp_dir / "download")

        on_progress("Extracting audio...")
        source_audio_path = extract_audio(video_path, tmp_dir / "source.wav")

        converted_audio_path = convert_long_audio(
            source_audio_path,
            tmp_dir / "converted.wav",
            voice,
            converter,
            on_progress=on_progress,
            use_cache=use_cache,
        )

        on_progress("Remuxing video...")
        replace_audio(video_path, converted_audio_path, output_path)

    return output_path
