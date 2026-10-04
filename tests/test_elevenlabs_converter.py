from pathlib import Path

import pytest

from dubshift.converters.elevenlabs import (
    MAX_INPUT_DURATION_SECONDS,
    MODEL_ID,
    OUTPUT_FORMAT,
    ElevenLabsVoiceConverter,
)
from dubshift.converters.exceptions import (
    ConversionFailedError,
    InputDurationLimitExceededError,
    MissingCredentialsError,
    UnsupportedVoiceError,
)


class _FakeSpeechToSpeechResource:
    """Stand-in for client.speech_to_speech on the real ElevenLabs SDK."""

    def __init__(self, chunks: list[bytes] | None = None, error: Exception | None = None) -> None:
        self._chunks = chunks if chunks is not None else [b"converted audio bytes"]
        self._error = error
        self.calls: list[dict] = []

    def convert(self, *, voice_id: str, audio, model_id: str, output_format: str):
        self.calls.append(
            {
                "voice_id": voice_id,
                "audio": audio,
                "model_id": model_id,
                "output_format": output_format,
            }
        )
        if self._error is not None:
            raise self._error
        yield from self._chunks


class _FakeElevenLabsClient:
    """Stand-in for elevenlabs.ElevenLabs."""

    instances: list["_FakeElevenLabsClient"] = []

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key
        self.speech_to_speech = _FakeSpeechToSpeechResource()
        _FakeElevenLabsClient.instances.append(self)


@pytest.fixture(autouse=True)
def fake_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests assume a short, well-under-the-limit input; one test overrides this."""
    monkeypatch.setattr(
        "dubshift.converters.elevenlabs._probe_duration_seconds", lambda path: 10.0
    )


@pytest.fixture(autouse=True)
def clear_fake_client_instances() -> None:
    _FakeElevenLabsClient.instances.clear()
    yield
    _FakeElevenLabsClient.instances.clear()


def _make_input_audio(tmp_path: Path) -> Path:
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fake pcm audio bytes")
    return input_audio


def test_converts_successfully_and_writes_exact_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FakeElevenLabsClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    result = converter.convert(input_audio, voice="voice-123", output_path=output_path)

    assert result == output_path
    assert output_path.read_bytes() == b"converted audio bytes"


def test_missing_api_key_raises_missing_credentials_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FakeElevenLabsClient)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    with pytest.raises(MissingCredentialsError):
        ElevenLabsVoiceConverter()


def test_reads_api_key_from_environment_when_not_passed_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FakeElevenLabsClient)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "from-env")

    ElevenLabsVoiceConverter()

    assert _FakeElevenLabsClient.instances[-1].api_key == "from-env"


def test_explicit_api_key_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FakeElevenLabsClient)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "from-env")

    ElevenLabsVoiceConverter(api_key="explicit-key")

    assert _FakeElevenLabsClient.instances[-1].api_key == "explicit-key"


def test_passes_correct_voice_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FakeElevenLabsClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    converter.convert(input_audio, voice="specific-voice-id", output_path=output_path)

    call = _FakeElevenLabsClient.instances[-1].speech_to_speech.calls[0]
    assert call["voice_id"] == "specific-voice-id"


def test_passes_required_model_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FakeElevenLabsClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    converter.convert(input_audio, voice="voice-123", output_path=output_path)

    call = _FakeElevenLabsClient.instances[-1].speech_to_speech.calls[0]
    assert call["model_id"] == "eleven_multilingual_sts_v2"
    assert call["model_id"] == MODEL_ID
    assert call["output_format"] == OUTPUT_FORMAT


def test_maps_provider_api_error_to_conversion_failed_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from elevenlabs.core.api_error import ApiError

    class _FailingClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(
                error=ApiError(status_code=500, body={"detail": "internal error"})
            )

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FailingClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(ConversionFailedError) as exc_info:
        converter.convert(input_audio, voice="voice-123", output_path=output_path)

    assert "test-key" not in str(exc_info.value)
    assert not output_path.exists()


def test_maps_network_error_to_conversion_failed_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import httpx

    class _FailingClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(
                error=httpx.ConnectError("connection refused")
            )

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FailingClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(ConversionFailedError):
        converter.convert(input_audio, voice="voice-123", output_path=output_path)


def test_empty_provider_output_raises_conversion_failed_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _EmptyClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(chunks=[])

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _EmptyClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(ConversionFailedError, match="no audio"):
        converter.convert(input_audio, voice="voice-123", output_path=output_path)

    assert not output_path.exists()


def test_voice_not_found_status_code_raises_unsupported_voice_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from elevenlabs.core.api_error import ApiError

    class _FailingClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(
                error=ApiError(status_code=404, body={"detail": "Voice not found"})
            )

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FailingClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(UnsupportedVoiceError, match="voice-123"):
        converter.convert(input_audio, voice="voice-123", output_path=output_path)

    assert not output_path.exists()


def test_voice_not_found_detail_status_raises_unsupported_voice_error_regardless_of_status_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from elevenlabs.core.api_error import ApiError

    class _FailingClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(
                error=ApiError(
                    status_code=400,
                    body={"detail": {"status": "voice_not_found", "message": "no such voice"}},
                )
            )

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FailingClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(UnsupportedVoiceError):
        converter.convert(input_audio, voice="voice-123", output_path=output_path)


def test_unrelated_api_error_with_structured_body_remains_conversion_failed_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from elevenlabs.core.api_error import ApiError

    class _FailingClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(
                error=ApiError(
                    status_code=400,
                    body={"detail": {"status": "quota_exceeded", "message": "..."}},
                )
            )

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FailingClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(ConversionFailedError):
        converter.convert(input_audio, voice="voice-123", output_path=output_path)


def test_quota_exceeded_error_is_surfaced_with_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from elevenlabs.core.api_error import ApiError

    class _FailingClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(
                error=ApiError(
                    status_code=401,
                    body={
                        "detail": {
                            "status": "quota_exceeded",
                            "message": "You have insufficient quota to complete this request.",
                        }
                    },
                )
            )

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FailingClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(ConversionFailedError) as exc_info:
        converter.convert(input_audio, voice="voice-123", output_path=output_path)

    message = str(exc_info.value)
    assert "status 401" in message
    assert "quota_exceeded" in message
    assert "You have insufficient quota to complete this request." in message


def test_missing_permissions_error_is_surfaced_with_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from elevenlabs.core.api_error import ApiError

    class _FailingClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(
                error=ApiError(
                    status_code=403,
                    body={
                        "detail": {
                            "status": "missing_permissions",
                            "message": "This API key is missing the required permission.",
                        }
                    },
                )
            )

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FailingClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(ConversionFailedError) as exc_info:
        converter.convert(input_audio, voice="voice-123", output_path=output_path)

    message = str(exc_info.value)
    assert "status 403" in message
    assert "missing_permissions" in message
    assert "This API key is missing the required permission." in message


def test_invalid_api_key_error_is_surfaced_with_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from elevenlabs.core.api_error import ApiError

    class _FailingClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(
                error=ApiError(
                    status_code=401,
                    body={"detail": {"status": "invalid_api_key", "message": "Invalid API key."}},
                )
            )

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FailingClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(ConversionFailedError) as exc_info:
        converter.convert(input_audio, voice="voice-123", output_path=output_path)

    message = str(exc_info.value)
    assert "status 401" in message
    assert "invalid_api_key" in message
    assert "Invalid API key." in message
    assert "test-key" not in message


def test_non_dict_body_falls_back_to_status_only_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from elevenlabs.core.api_error import ApiError

    class _FailingClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(
                error=ApiError(status_code=500, body="not a dict at all")
            )

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FailingClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(ConversionFailedError) as exc_info:
        converter.convert(input_audio, voice="voice-123", output_path=output_path)

    assert str(exc_info.value) == "ElevenLabs speech-to-speech request failed (status 500)."


def test_detail_without_identifier_or_message_falls_back_to_status_only_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from elevenlabs.core.api_error import ApiError

    class _FailingClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(
                error=ApiError(status_code=500, body={"detail": {"unrelated_field": 123}})
            )

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FailingClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(ConversionFailedError) as exc_info:
        converter.convert(input_audio, voice="voice-123", output_path=output_path)

    assert str(exc_info.value) == "ElevenLabs speech-to-speech request failed (status 500)."


def test_api_key_never_appears_in_structured_error_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even if the SDK's exception object carries request headers somewhere,
    DubShift's message construction never touches them -- only exc.body."""
    from elevenlabs.core.api_error import ApiError

    secret_key = "sk_test_totally_secret_value_12345"

    class _FailingClient(_FakeElevenLabsClient):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key)
            self.speech_to_speech = _FakeSpeechToSpeechResource(
                error=ApiError(
                    status_code=401,
                    headers={"Authorization": f"Bearer {secret_key}"},
                    body={"detail": {"status": "invalid_api_key", "message": "Invalid API key."}},
                )
            )

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FailingClient)

    converter = ElevenLabsVoiceConverter(api_key=secret_key)
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(ConversionFailedError) as exc_info:
        converter.convert(input_audio, voice="voice-123", output_path=output_path)

    assert secret_key not in str(exc_info.value)


def test_output_is_written_atomically_with_no_leftover_temp_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FakeElevenLabsClient)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    converter.convert(input_audio, voice="voice-123", output_path=output_path)

    assert output_path.read_bytes() == b"converted audio bytes"
    leftover_temp_files = [p for p in tmp_path.iterdir() if p.name.startswith(".converted.mp3.")]
    assert leftover_temp_files == []


def test_output_write_failure_leaves_no_partial_or_temp_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FakeElevenLabsClient)

    def failing_replace(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("dubshift.converters.elevenlabs.os.replace", failing_replace)

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(OSError, match="disk full"):
        converter.convert(input_audio, voice="voice-123", output_path=output_path)

    assert not output_path.exists()
    leftover_temp_files = [p for p in tmp_path.iterdir() if p.name.startswith(".converted.mp3.")]
    assert leftover_temp_files == []


def test_rejects_input_longer_than_duration_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FakeElevenLabsClient)
    monkeypatch.setattr(
        "dubshift.converters.elevenlabs._probe_duration_seconds",
        lambda path: MAX_INPUT_DURATION_SECONDS + 1,
    )

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    with pytest.raises(InputDurationLimitExceededError):
        converter.convert(input_audio, voice="voice-123", output_path=output_path)

    assert not output_path.exists()


def test_accepts_input_exactly_at_duration_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabs", _FakeElevenLabsClient)
    monkeypatch.setattr(
        "dubshift.converters.elevenlabs._probe_duration_seconds",
        lambda path: MAX_INPUT_DURATION_SECONDS,
    )

    converter = ElevenLabsVoiceConverter(api_key="test-key")
    input_audio = _make_input_audio(tmp_path)
    output_path = tmp_path / "converted.mp3"

    result = converter.convert(input_audio, voice="voice-123", output_path=output_path)

    assert result == output_path
