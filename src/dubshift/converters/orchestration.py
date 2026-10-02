"""Backend-agnostic orchestration for running voice conversion."""

from __future__ import annotations

from pathlib import Path

from dubshift.converters.base import VoiceConverter
from dubshift.converters.exceptions import (
    ConversionFailedError,
    ConversionOutputExistsError,
    InvalidAudioPathError,
)


def convert_audio(
    input_path: Path,
    output_path: Path,
    voice: str,
    converter: VoiceConverter,
) -> Path:
    """Run voice conversion on ``input_path`` using ``converter``.

    Validates the generic preconditions that apply no matter which
    converter is used -- the input file exists, the output file doesn't
    already exist -- then delegates the actual conversion work to
    ``converter``. Callers only need a ``VoiceConverter``; they never need
    to know whether it calls a cloud API, runs a local model, or anything
    else.

    ``converter`` must write to, and return, the same ``output_path`` it
    was given (after path resolution); a converter that silently writes
    somewhere else is treated as a failure.
    """
    input_path = Path(input_path)
    if not input_path.exists() or not input_path.is_file():
        raise InvalidAudioPathError(f"Input audio file not found: {input_path}")

    output_path = Path(output_path)
    if output_path.exists():
        raise ConversionOutputExistsError(f"Refusing to overwrite existing file: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_path = Path(converter.convert(input_path, voice=voice, output_path=output_path))

    if result_path.resolve() != output_path.resolve():
        raise ConversionFailedError(
            f"{type(converter).__name__} wrote to {result_path} instead of "
            f"the requested output path {output_path}"
        )

    if not output_path.exists():
        raise ConversionFailedError(
            f"{type(converter).__name__} reported success but the output "
            f"file is missing: {output_path}"
        )

    return output_path
