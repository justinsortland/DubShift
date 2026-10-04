"""Exceptions for the voice-conversion abstraction.

These are generic to the abstraction itself, not tied to any specific
backend, so that orchestration code, the CLI, and tests can all handle
failures the same way regardless of which converter is plugged in.
"""


class VoiceConversionError(RuntimeError):
    """Base class for voice-conversion failures."""


class InvalidAudioPathError(ValueError):
    """Raised when an input audio path doesn't exist or isn't a file."""


class ConversionOutputExistsError(VoiceConversionError):
    """Raised when the resolved output path already exists."""


class UnsupportedVoiceError(VoiceConversionError):
    """Raised when a converter doesn't recognize or support a requested voice."""


class ConversionFailedError(VoiceConversionError):
    """Raised when a converter fails to produce valid output audio."""


class MissingCredentialsError(VoiceConversionError):
    """Raised when a converter requires credentials that were not provided."""


class InputDurationLimitExceededError(VoiceConversionError):
    """Raised when input audio exceeds a provider's maximum supported duration."""


class ChunkingError(VoiceConversionError):
    """Raised when splitting input audio into chunks, or reconstructing the final output, fails."""


class ChunkDurationMismatchError(VoiceConversionError):
    """Raised when a converted chunk's duration differs materially from its source chunk's."""


class FinalDurationMismatchError(ChunkDurationMismatchError):
    """Raised when the fully reconstructed output's total duration differs materially from the source's."""
