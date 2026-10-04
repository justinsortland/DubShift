"""ElevenLabs Voice Changer (speech-to-speech) VoiceConverter provider.

Implements ``dubshift.converters.base.VoiceConverter`` on top of the
official ElevenLabs Python SDK's speech-to-speech ("Voice Changer")
endpoint. ``voice`` is treated as an ElevenLabs voice_id.

This module only performs voice conversion. It does not transcribe audio,
does not perform text-to-speech, and does not chunk audio longer than
ElevenLabs' current Voice Changer request limit -- such input is rejected
up front with a clear error.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx
from elevenlabs import ElevenLabs
from elevenlabs.core.api_error import ApiError

from dubshift.converters.exceptions import (
    ConversionFailedError,
    InputDurationLimitExceededError,
    MissingCredentialsError,
    UnsupportedVoiceError,
)

# HTTP status ElevenLabs uses for a voice_id that doesn't exist (or isn't
# available to the caller) on this voice-scoped endpoint.
_VOICE_NOT_FOUND_STATUS_CODE = 404

# ElevenLabs' documented machine-readable error code for the same
# condition, checked as a structured field (never a substring match
# against a human-readable message) in case a future response uses a
# different status code for it.
_VOICE_NOT_FOUND_DETAIL_STATUS = "voice_not_found"

# Fields ElevenLabs may use for a structured error's machine-readable
# identifier (e.g. "quota_exceeded", "invalid_api_key",
# "missing_permissions"), in preference order. "status" matches the field
# already used above for voice_not_found detection; "code" and "type" are
# included defensively in case a given response uses one of those instead.
_ERROR_DETAIL_IDENTIFIER_FIELDS = ("status", "code", "type")

# The speech-to-speech model requested for every conversion.
MODEL_ID = "eleven_multilingual_sts_v2"

# mp3 at 44.1kHz/128kbps is available on every ElevenLabs plan; PCM/WAV
# output at 44.1kHz requires a paid Pro-or-above subscription tier.
OUTPUT_FORMAT = "mp3_44100_128"

# ElevenLabs' current Voice Changer request limit. DubShift does not yet
# split long audio across multiple requests, so anything longer than this
# is rejected up front rather than failing deep inside (or being silently
# truncated by) the API call.
MAX_INPUT_DURATION_SECONDS = 5 * 60

_API_KEY_ENV_VAR = "ELEVENLABS_API_KEY"


def _probe_duration_seconds(audio_path: Path) -> float:
    """Return the duration of ``audio_path`` in seconds, via ffprobe.

    Reuses the ffprobe binary that ships with the ffmpeg install DubShift
    already requires (see ``dubshift.media``), rather than adding a
    dedicated audio-parsing library just to read a duration.
    """
    if shutil.which("ffprobe") is None:
        raise ConversionFailedError(
            "ffprobe is required to check audio duration before calling "
            "ElevenLabs, but was not found on PATH. Install ffmpeg (which "
            "includes ffprobe), e.g. `brew install ffmpeg` on macOS."
        )

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError as exc:
        raise ConversionFailedError(
            "ffprobe is required to check audio duration before calling "
            "ElevenLabs, but was not found on PATH. Install ffmpeg (which "
            "includes ffprobe), e.g. `brew install ffmpeg` on macOS."
        ) from exc

    duration_text = result.stdout.strip()
    if result.returncode != 0 or not duration_text:
        raise ConversionFailedError(
            f"Could not determine the duration of {audio_path} via ffprobe "
            f"(exit code {result.returncode}): {result.stderr.strip()}"
        )

    try:
        return float(duration_text)
    except ValueError as exc:
        raise ConversionFailedError(
            f"ffprobe returned an unparseable duration for {audio_path}: {duration_text!r}"
        ) from exc


def _is_unsupported_voice_error(exc: ApiError) -> bool:
    """Detect a "voice_id doesn't exist / isn't available" response.

    Uses only structured fields the SDK already parses for us -- the HTTP
    status code and, defensively, the API's machine-readable error code --
    never substring matching against a human-readable message.
    """
    if exc.status_code == _VOICE_NOT_FOUND_STATUS_CODE:
        return True

    body = exc.body
    detail = body.get("detail") if isinstance(body, dict) else None
    return isinstance(detail, dict) and detail.get("status") == _VOICE_NOT_FOUND_DETAIL_STATUS


def _extract_error_detail(body: object) -> dict | None:
    """Safely pull ElevenLabs' structured ``detail`` object out of an error body.

    Only ever looks at the already-parsed response *body* ElevenLabs sent
    back -- never request headers or anything else that could carry
    credentials. Returns ``None`` (never raises) if the body isn't shaped
    as expected, so callers can always fall back to a generic message.
    """
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    if not isinstance(detail, dict):
        return None
    return detail


def _describe_api_error(exc: ApiError) -> str:
    """Build a one-line(-plus-detail) description of an ElevenLabs ApiError.

    Tries to surface the structured cause (e.g. ``quota_exceeded``,
    ``invalid_api_key``, ``missing_permissions``) and human-readable
    message ElevenLabs included in its response body. Falls back cleanly
    to the plain status-only message if the body doesn't contain anything
    usable.
    """
    base_message = f"ElevenLabs speech-to-speech request failed (status {exc.status_code})."

    detail = _extract_error_detail(exc.body)
    if detail is None:
        return base_message

    identifier = None
    for field in _ERROR_DETAIL_IDENTIFIER_FIELDS:
        value = detail.get(field)
        if isinstance(value, str) and value:
            identifier = value
            break

    message = detail.get("message")
    message = message if isinstance(message, str) and message else None

    if identifier is None and message is None:
        return base_message

    summary = " — ".join(part for part in (identifier, message) if part)
    return (
        f"ElevenLabs speech-to-speech request failed (status {exc.status_code}):\n"
        f"{summary}"
    )


def _write_output_atomically(data: bytes, output_path: Path) -> None:
    """Write ``data`` to ``output_path`` without ever leaving a partial file there.

    Writes to a temporary file in the same directory as ``output_path``
    (so the final swap is a same-filesystem, atomic rename) and only
    replaces ``output_path`` after the write has fully succeeded. The
    temporary file is removed if anything goes wrong.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(data)
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


class ElevenLabsVoiceConverter:
    """A VoiceConverter backed by ElevenLabs' Voice Changer (speech-to-speech) API."""

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get(_API_KEY_ENV_VAR)
        if not resolved_key:
            raise MissingCredentialsError(
                "No ElevenLabs API key provided. Pass api_key explicitly or "
                f"set the {_API_KEY_ENV_VAR} environment variable."
            )
        self._client = ElevenLabs(api_key=resolved_key)

    @property
    def cache_identity(self) -> str:
        """A stable identity for the converted-chunk cache: provider + model + settings.

        Never includes the API key -- only the (non-secret) model and
        output-format identifiers that actually affect the converted
        result.
        """
        return f"elevenlabs:{MODEL_ID}:{OUTPUT_FORMAT}"

    def convert(
        self,
        input_audio: Path,
        *,
        voice: str,
        output_path: Path,
    ) -> Path:
        input_audio = Path(input_audio)
        output_path = Path(output_path)

        duration_seconds = _probe_duration_seconds(input_audio)
        if duration_seconds > MAX_INPUT_DURATION_SECONDS:
            raise InputDurationLimitExceededError(
                f"Input audio is {duration_seconds:.1f}s long, which exceeds "
                f"ElevenLabs' current {MAX_INPUT_DURATION_SECONDS:.0f}s "
                "(5 minute) Voice Changer request limit. Splitting long "
                "audio across multiple requests (chunking) is not "
                "implemented yet."
            )

        try:
            with open(input_audio, "rb") as audio_file:
                audio_bytes = b"".join(
                    self._client.speech_to_speech.convert(
                        voice_id=voice,
                        audio=audio_file,
                        model_id=MODEL_ID,
                        output_format=OUTPUT_FORMAT,
                    )
                )
        except ApiError as exc:
            if _is_unsupported_voice_error(exc):
                raise UnsupportedVoiceError(
                    f"ElevenLabs voice {voice!r} was not found or is not "
                    f"available for this account (status {exc.status_code})."
                ) from exc
            raise ConversionFailedError(_describe_api_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise ConversionFailedError(
                "ElevenLabs speech-to-speech request failed due to a "
                f"network error: {type(exc).__name__}"
            ) from exc

        if not audio_bytes:
            raise ConversionFailedError(
                "ElevenLabs returned no audio data for the speech-to-speech request."
            )

        _write_output_atomically(audio_bytes, output_path)
        return output_path
