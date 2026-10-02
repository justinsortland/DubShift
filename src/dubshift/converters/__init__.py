"""Pluggable voice-conversion backends.

This package defines the abstraction the rest of DubShift uses to convert
a speaker's voice to another voice, without knowing whether the actual
conversion happens via a cloud API, a local model, or anything else.

Concrete providers (e.g. ElevenLabs- or Seed-VC-backed converters) are
implemented elsewhere as separate modules that satisfy the
``VoiceConverter`` protocol -- none exist yet.
"""

from dubshift.converters.base import VoiceConverter
from dubshift.converters.exceptions import (
    ConversionFailedError,
    ConversionOutputExistsError,
    InvalidAudioPathError,
    UnsupportedVoiceError,
    VoiceConversionError,
)
from dubshift.converters.fake import FakeVoiceConverter
from dubshift.converters.orchestration import convert_audio

__all__ = [
    "VoiceConverter",
    "VoiceConversionError",
    "InvalidAudioPathError",
    "ConversionOutputExistsError",
    "UnsupportedVoiceError",
    "ConversionFailedError",
    "FakeVoiceConverter",
    "convert_audio",
]
