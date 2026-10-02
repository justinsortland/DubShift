import subprocess
from pathlib import Path

import pytest

from dubshift.media import (
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    AudioExtractionError,
    InvalidVideoPathError,
    MissingFFmpegError,
    OutputExistsError,
    extract_audio,
)


@pytest.fixture(autouse=True)
def ffmpeg_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assume ffmpeg is installed by default; the dedicated test below overrides this."""
    monkeypatch.setattr("dubshift.media._ffmpeg_available", lambda: True)


def _fake_run_that_writes_output(output_index: int = -1):
    """Build a fake subprocess.run that creates the requested output file."""

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        output_path = Path(command[output_index])
        output_path.write_bytes(b"fake wav bytes")
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    return fake_run


def test_extract_audio_rejects_missing_input(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.mp4"

    with pytest.raises(InvalidVideoPathError):
        extract_audio(missing)


def test_extract_audio_rejects_directory_input(tmp_path: Path) -> None:
    a_directory = tmp_path / "a-directory"
    a_directory.mkdir()

    with pytest.raises(InvalidVideoPathError):
        extract_audio(a_directory)


def test_extract_audio_raises_when_ffmpeg_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.media._ffmpeg_available", lambda: False)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run should not be called when ffmpeg is missing")

    monkeypatch.setattr("dubshift.media.subprocess.run", fail_if_called)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    with pytest.raises(MissingFFmpegError, match="(?i)ffmpeg"):
        extract_audio(video)


def test_extract_audio_refuses_to_overwrite_existing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run should not be called when output already exists")

    monkeypatch.setattr("dubshift.media.subprocess.run", fail_if_called)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    existing_output = tmp_path / "clip.wav"
    existing_output.write_bytes(b"already here")

    with pytest.raises(OutputExistsError):
        extract_audio(video)


def test_extract_audio_defaults_output_path_next_to_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.media.subprocess.run", _fake_run_that_writes_output())

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    result = extract_audio(video)

    assert result == tmp_path / "clip.wav"
    assert result.exists()


def test_extract_audio_uses_custom_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.media.subprocess.run", _fake_run_that_writes_output())

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    custom_output = tmp_path / "nested" / "audio.wav"

    result = extract_audio(video, output_path=custom_output)

    assert result == custom_output
    assert result.exists()


def test_extract_audio_builds_expected_ffmpeg_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_command: list = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        captured_command.extend(command)
        Path(command[-1]).write_bytes(b"fake wav bytes")
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    result = extract_audio(video)

    assert captured_command == [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-ac",
        str(DEFAULT_CHANNELS),
        "-n",
        str(result),
    ]


def test_extract_audio_does_not_reencode_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured_command: list = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        captured_command.extend(command)
        Path(command[-1]).write_bytes(b"fake wav bytes")
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    extract_audio(video)

    assert "-vn" in captured_command
    assert not any(flag.startswith("-vcodec") or flag == "-c:v" for flag in captured_command)


def test_extract_audio_raises_on_ffmpeg_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            command, returncode=1, stdout="", stderr="Invalid data found when processing input"
        )

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not actually a video")

    with pytest.raises(AudioExtractionError, match="Invalid data found"):
        extract_audio(video)


def test_extract_audio_raises_if_ffmpeg_vanishes_mid_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    with pytest.raises(MissingFFmpegError):
        extract_audio(video)


def test_extract_audio_raises_if_output_file_missing_after_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        # ffmpeg reports success but, unexpectedly, writes nothing.
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    with pytest.raises(AudioExtractionError, match="missing"):
        extract_audio(video)
