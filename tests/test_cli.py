from pathlib import Path

import pytest
from typer.testing import CliRunner

from dubshift import __version__
from dubshift.cli import app
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
