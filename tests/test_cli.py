from pathlib import Path

import pytest
from typer.testing import CliRunner

from dubshift import __version__
from dubshift.cli import app
from dubshift.downloader import DownloadError, InvalidURLError

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
