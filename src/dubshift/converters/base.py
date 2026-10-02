"""The interface every voice-conversion backend must implement."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class VoiceConverter(Protocol):
    """Converts the voice in an audio file to a different voice.

    Implementations are free to call a cloud API, run a local model, or do
    anything else needed to produce the converted audio -- the rest of
    DubShift never needs to know which. All backend-specific concerns
    (resampling to whatever rate the model expects, chunking long audio,
    authentication, retries, picking a container format, etc.) belong
    inside the converter, not in the caller.
    """

    def convert(
        self,
        input_audio: Path,
        *,
        voice: str,
        output_path: Path,
    ) -> Path:
        """Convert ``input_audio`` to ``voice`` and write the result to ``output_path``.

        Args:
            input_audio: Path to the source audio (e.g. produced by
                ``dubshift.media.extract_audio``).
            voice: A backend-specific voice identifier -- an ElevenLabs
                voice ID, a path to a reference clip for Seed-VC, an RVC
                model name, etc. Its meaning is entirely up to the
                converter.
            output_path: Where the converted audio should be written.

        Returns:
            The path to the converted audio file (ordinarily
            ``output_path``).

        Raises:
            dubshift.converters.exceptions.VoiceConversionError: on any
                backend-specific failure.
        """
        ...
