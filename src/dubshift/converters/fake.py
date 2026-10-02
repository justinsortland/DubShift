"""A dependency-free VoiceConverter for tests.

Not a real voice-conversion backend -- it just copies the input audio to
the output path and records what it was asked to do, so that code which
depends only on the ``VoiceConverter`` protocol can be exercised without a
cloud API or an ML model.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FakeVoiceConverter:
    """A VoiceConverter stand-in that copies the input audio to the output path."""

    calls: list[tuple[Path, str, Path]] = field(default_factory=list)

    def convert(
        self,
        input_audio: Path,
        *,
        voice: str,
        output_path: Path,
    ) -> Path:
        self.calls.append((input_audio, voice, output_path))
        shutil.copyfile(input_audio, output_path)
        return output_path
