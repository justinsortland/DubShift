from pathlib import Path

import pytest

from dubshift.converters import (
    ConversionFailedError,
    ConversionOutputExistsError,
    FakeVoiceConverter,
    InvalidAudioPathError,
    VoiceConverter,
    convert_audio,
)


class _ReversingConverter:
    """A second, independent VoiceConverter implementation.

    Deliberately does not inherit from anything in dubshift.converters --
    it only defines the method the protocol expects, to prove that
    orchestration code cares about structure, not a shared base class.
    """

    def convert(self, input_audio: Path, *, voice: str, output_path: Path) -> Path:
        output_path.write_bytes(input_audio.read_bytes()[::-1])
        return output_path


def _run_as_generic_client(
    converter: VoiceConverter, tmp_path: Path, voice: str = "any-voice"
) -> Path:
    """Stand-in for client code that only knows about the VoiceConverter protocol."""
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"fake pcm audio bytes")
    output_path = tmp_path / "output.wav"
    return convert_audio(input_path, output_path, voice, converter)


def test_fake_converter_satisfies_the_protocol() -> None:
    assert isinstance(FakeVoiceConverter(), VoiceConverter)


def test_structurally_compatible_class_satisfies_the_protocol() -> None:
    assert isinstance(_ReversingConverter(), VoiceConverter)


@pytest.mark.parametrize("converter", [FakeVoiceConverter(), _ReversingConverter()])
def test_client_code_works_against_any_voice_converter(
    converter: VoiceConverter, tmp_path: Path
) -> None:
    result = _run_as_generic_client(converter, tmp_path)

    assert result == tmp_path / "output.wav"
    assert result.exists()


def test_convert_audio_passes_arguments_through_to_the_converter(tmp_path: Path) -> None:
    converter = FakeVoiceConverter()

    result = _run_as_generic_client(converter, tmp_path, voice="narrator-1")

    assert len(converter.calls) == 1
    called_input, called_voice, called_output = converter.calls[0]
    assert called_input == tmp_path / "input.wav"
    assert called_voice == "narrator-1"
    assert called_output == result


def test_convert_audio_returns_converter_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"fake pcm audio bytes")
    output_path = tmp_path / "output.wav"

    convert_audio(input_path, output_path, "any-voice", _ReversingConverter())

    assert output_path.read_bytes() == b"fake pcm audio bytes"[::-1]


def test_convert_audio_rejects_missing_input(tmp_path: Path) -> None:
    missing_input = tmp_path / "does-not-exist.wav"
    output_path = tmp_path / "output.wav"

    with pytest.raises(InvalidAudioPathError):
        convert_audio(missing_input, output_path, "any-voice", FakeVoiceConverter())


def test_convert_audio_rejects_directory_input(tmp_path: Path) -> None:
    a_directory = tmp_path / "a-directory"
    a_directory.mkdir()
    output_path = tmp_path / "output.wav"

    with pytest.raises(InvalidAudioPathError):
        convert_audio(a_directory, output_path, "any-voice", FakeVoiceConverter())


def test_convert_audio_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"fake pcm audio bytes")
    existing_output = tmp_path / "output.wav"
    existing_output.write_bytes(b"already here")

    with pytest.raises(ConversionOutputExistsError):
        convert_audio(input_path, existing_output, "any-voice", FakeVoiceConverter())


def test_convert_audio_creates_output_parent_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"fake pcm audio bytes")
    output_path = tmp_path / "nested" / "output.wav"

    result = convert_audio(input_path, output_path, "any-voice", FakeVoiceConverter())

    assert result == output_path
    assert result.exists()


def test_convert_audio_raises_if_converter_does_not_produce_output(tmp_path: Path) -> None:
    class _BrokenConverter:
        def convert(self, input_audio: Path, *, voice: str, output_path: Path) -> Path:
            # Claims success but never actually writes the file.
            return output_path

    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"fake pcm audio bytes")
    output_path = tmp_path / "output.wav"

    with pytest.raises(ConversionFailedError, match="missing"):
        convert_audio(input_path, output_path, "any-voice", _BrokenConverter())


def test_convert_audio_raises_if_converter_writes_to_a_different_path(tmp_path: Path) -> None:
    class _SneakyConverter:
        def convert(self, input_audio: Path, *, voice: str, output_path: Path) -> Path:
            # Ignores the requested output_path and writes elsewhere instead.
            actual_path = output_path.parent / "elsewhere.wav"
            actual_path.write_bytes(b"not where it was asked to write")
            return actual_path

    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"fake pcm audio bytes")
    output_path = tmp_path / "output.wav"

    with pytest.raises(ConversionFailedError, match="instead of"):
        convert_audio(input_path, output_path, "any-voice", _SneakyConverter())


def test_convert_audio_accepts_an_equivalent_but_unnormalized_output_path(tmp_path: Path) -> None:
    class _UnnormalizedPathConverter:
        def convert(self, input_audio: Path, *, voice: str, output_path: Path) -> Path:
            output_path.write_bytes(b"converted")
            # Returns a path that resolves to the same file but isn't
            # written identically (extra "./" segment).
            return output_path.parent / "." / output_path.name

    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"fake pcm audio bytes")
    output_path = tmp_path / "output.wav"

    result = convert_audio(input_path, output_path, "any-voice", _UnnormalizedPathConverter())

    assert result == output_path
    assert result.read_bytes() == b"converted"
