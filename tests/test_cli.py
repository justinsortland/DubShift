import builtins
import importlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dubshift import __version__
from dubshift.cli import app
from dubshift.converters.exceptions import (
    ConversionFailedError,
    MissingCredentialsError,
)
from dubshift.downloader import DownloadError, InvalidURLError
from dubshift.media import AudioExtractionError, InvalidVideoPathError, MissingFFmpegError

runner = CliRunner()


def test_help_exits_successfully() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_help_mentions_dubshift() -> None:
    result = runner.invoke(app, ["--help"])
    assert "dubshift" in result.output.lower()


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_version_command_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_download_command_prints_path_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_path = tmp_path / "video123.mp4"
    fake_path.write_bytes(b"fake video bytes")

    def fake_download_video(url: str, output_dir: Path) -> Path:
        return fake_path

    monkeypatch.setattr("dubshift.cli.download_video", fake_download_video)

    result = runner.invoke(app, ["download", "https://youtu.be/dQw4w9WgXcQ"])

    assert result.exit_code == 0
    assert result.output.strip() == str(fake_path)


def test_download_command_reports_invalid_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download_video(url: str, output_dir: Path) -> Path:
        raise InvalidURLError(f"Not a valid YouTube video URL: {url!r}")

    monkeypatch.setattr("dubshift.cli.download_video", fake_download_video)

    result = runner.invoke(app, ["download", "not a url"])

    assert result.exit_code == 1
    assert "Not a valid YouTube video URL" in result.output


def test_download_command_reports_download_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download_video(url: str, output_dir: Path) -> Path:
        raise DownloadError("Failed to download: video unavailable")

    monkeypatch.setattr("dubshift.cli.download_video", fake_download_video)

    result = runner.invoke(app, ["download", "https://youtu.be/dQw4w9WgXcQ"])

    assert result.exit_code == 1
    assert "video unavailable" in result.output


def test_extract_command_prints_path_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_wav = tmp_path / "clip.wav"
    fake_wav.write_bytes(b"fake wav bytes")

    def fake_extract_audio(video_path: Path, output_path: Path | None) -> Path:
        return fake_wav

    monkeypatch.setattr("dubshift.cli.extract_audio", fake_extract_audio)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    result = runner.invoke(app, ["extract", str(video)])

    assert result.exit_code == 0
    assert result.output.strip() == str(fake_wav)


def test_extract_command_reports_invalid_video_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract_audio(video_path: Path, output_path: Path | None) -> Path:
        raise InvalidVideoPathError(f"Video file not found: {video_path}")

    monkeypatch.setattr("dubshift.cli.extract_audio", fake_extract_audio)

    result = runner.invoke(app, ["extract", "does-not-exist.mp4"])

    assert result.exit_code == 1
    assert "Video file not found" in result.output


def test_extract_command_reports_missing_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_extract_audio(video_path: Path, output_path: Path | None) -> Path:
        raise MissingFFmpegError("FFmpeg is required to extract audio but was not found on PATH.")

    monkeypatch.setattr("dubshift.cli.extract_audio", fake_extract_audio)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    result = runner.invoke(app, ["extract", str(video)])

    assert result.exit_code == 1
    assert "FFmpeg is required" in result.output


def test_extract_command_reports_extraction_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_extract_audio(video_path: Path, output_path: Path | None) -> Path:
        raise AudioExtractionError("FFmpeg failed to extract audio: corrupt input")

    monkeypatch.setattr("dubshift.cli.extract_audio", fake_extract_audio)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    result = runner.invoke(app, ["extract", str(video)])

    assert result.exit_code == 1
    assert "corrupt input" in result.output


def test_cli_module_and_unrelated_commands_do_not_require_elevenlabs_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dubshift.cli must not import the ElevenLabs provider (or its SDK) at module load time.

    Simulates the ElevenLabs SDK being completely uninstalled by making any
    attempt to import it raise ImportError, then reloads dubshift.cli's
    *existing* module object in place (so this doesn't leave a second,
    orphaned module object behind for other tests to trip over) and
    exercises commands unrelated to voice conversion.
    """
    import dubshift.cli as cli_module

    real_import = builtins.__import__

    def blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "elevenlabs" or name.startswith("elevenlabs."):
            raise ImportError("simulated: the elevenlabs SDK is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    importlib.reload(cli_module)

    fresh_runner = CliRunner()

    assert fresh_runner.invoke(cli_module.app, ["--help"]).exit_code == 0
    assert fresh_runner.invoke(cli_module.app, ["download", "--help"]).exit_code == 0
    assert fresh_runner.invoke(cli_module.app, ["extract", "--help"]).exit_code == 0


class _FakeElevenLabsConverter:
    """Stand-in for dubshift.converters.elevenlabs.ElevenLabsVoiceConverter."""

    def __init__(self) -> None:
        pass


def test_convert_command_prints_path_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_output = tmp_path / "converted.mp3"

    def fake_convert_audio(input_path: Path, output_path: Path, voice: str, converter) -> Path:
        return fake_output

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabsVoiceConverter", _FakeElevenLabsConverter)
    monkeypatch.setattr("dubshift.cli.convert_audio", fake_convert_audio)

    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fake pcm audio bytes")

    result = runner.invoke(
        app,
        [
            "convert",
            str(input_audio),
            "--provider",
            "elevenlabs",
            "--voice",
            "voice-123",
            "--output",
            str(fake_output),
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == str(fake_output)


def test_convert_command_passes_voice_and_paths_through_to_orchestration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}

    def fake_convert_audio(input_path: Path, output_path: Path, voice: str, converter) -> Path:
        captured["input_path"] = input_path
        captured["output_path"] = output_path
        captured["voice"] = voice
        captured["converter"] = converter
        return output_path

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabsVoiceConverter", _FakeElevenLabsConverter)
    monkeypatch.setattr("dubshift.cli.convert_audio", fake_convert_audio)

    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fake pcm audio bytes")
    output_path = tmp_path / "converted.mp3"

    runner.invoke(
        app,
        [
            "convert",
            str(input_audio),
            "--provider",
            "elevenlabs",
            "--voice",
            "narrator-1",
            "--output",
            str(output_path),
        ],
    )

    assert captured["input_path"] == input_audio
    assert captured["output_path"] == output_path
    assert captured["voice"] == "narrator-1"
    assert isinstance(captured["converter"], _FakeElevenLabsConverter)


@pytest.mark.parametrize("missing_flag", ["--voice", "--output", "--provider"])
def test_convert_command_requires_core_options(missing_flag: str, tmp_path: Path) -> None:
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fake pcm audio bytes")

    all_args = [
        str(input_audio),
        "--provider",
        "elevenlabs",
        "--voice",
        "voice-123",
        "--output",
        str(tmp_path / "converted.mp3"),
    ]
    # Drop the flag under test and its value.
    flag_index = all_args.index(missing_flag)
    args = all_args[:flag_index] + all_args[flag_index + 2 :]

    result = runner.invoke(app, ["convert", *args])

    assert result.exit_code != 0
    assert missing_flag in result.output


def test_convert_command_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_if_called() -> None:
        raise AssertionError("ElevenLabsVoiceConverter should not be constructed for an unknown provider")

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabsVoiceConverter", fail_if_called)

    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fake pcm audio bytes")

    result = runner.invoke(
        app,
        [
            "convert",
            str(input_audio),
            "--provider",
            "seedvc",
            "--voice",
            "voice-123",
            "--output",
            str(tmp_path / "converted.mp3"),
        ],
    )

    assert result.exit_code == 1
    assert "Unknown provider" in result.output
    assert "seedvc" in result.output


def test_convert_command_reports_provider_construction_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_elevenlabs_converter() -> None:
        raise MissingCredentialsError("No ElevenLabs API key provided.")

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabsVoiceConverter", fake_elevenlabs_converter)

    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fake pcm audio bytes")

    result = runner.invoke(
        app,
        [
            "convert",
            str(input_audio),
            "--provider",
            "elevenlabs",
            "--voice",
            "voice-123",
            "--output",
            str(tmp_path / "converted.mp3"),
        ],
    )

    assert result.exit_code == 1
    assert "No ElevenLabs API key provided" in result.output


def test_convert_command_reports_conversion_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_convert_audio(input_path: Path, output_path: Path, voice: str, converter) -> Path:
        raise ConversionFailedError("ElevenLabs speech-to-speech request failed (status 500).")

    monkeypatch.setattr("dubshift.converters.elevenlabs.ElevenLabsVoiceConverter", _FakeElevenLabsConverter)
    monkeypatch.setattr("dubshift.cli.convert_audio", fake_convert_audio)

    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fake pcm audio bytes")

    result = runner.invoke(
        app,
        [
            "convert",
            str(input_audio),
            "--provider",
            "elevenlabs",
            "--voice",
            "voice-123",
            "--output",
            str(tmp_path / "converted.mp3"),
        ],
    )

    assert result.exit_code == 1
    assert "request failed" in result.output
