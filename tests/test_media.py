import subprocess
from pathlib import Path

import pytest

from dubshift.media import (
    DEFAULT_CHANNELS,
    DEFAULT_REMUX_DURATION_TOLERANCE_SECONDS,
    DEFAULT_SAMPLE_RATE,
    AudioExtractionError,
    AudioReplacementError,
    InvalidReplacementAudioPathError,
    InvalidVideoPathError,
    MediaDurationMismatchError,
    MissingFFmpegError,
    OutputExistsError,
    extract_audio,
    replace_audio,
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


# ---------------------------------------------------------------------------
# replace_audio
# ---------------------------------------------------------------------------


def _make_video_and_audio(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    audio = tmp_path / "dub.wav"
    audio.write_bytes(b"fake replacement audio bytes")
    return video, audio


def _fake_run_for_remux(
    *,
    video_path: Path,
    audio_path: Path,
    video_duration: float = 5.0,
    audio_duration: float = 5.0,
    mux_returncode: int = 0,
    mux_stderr: str = "",
):
    """Build a fake subprocess.run that answers ffprobe duration probes and
    the final ffmpeg mux call distinctly, as replace_audio now issues both.
    """

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        if command[0] == "ffprobe":
            probed_path = command[-1]
            if probed_path == str(video_path):
                duration = video_duration
            elif probed_path == str(audio_path):
                duration = audio_duration
            else:
                raise AssertionError(f"Unexpected ffprobe target: {probed_path}")
            return subprocess.CompletedProcess(command, returncode=0, stdout=f"{duration}\n", stderr="")

        assert command[0] == "ffmpeg"
        if mux_returncode == 0:
            Path(command[-1]).write_bytes(b"fake mp4 bytes")
        return subprocess.CompletedProcess(command, returncode=mux_returncode, stdout="", stderr=mux_stderr)

    return fake_run


def test_replace_audio_rejects_missing_video(tmp_path: Path) -> None:
    _, audio = _make_video_and_audio(tmp_path)
    missing_video = tmp_path / "does-not-exist.mp4"

    with pytest.raises(InvalidVideoPathError):
        replace_audio(missing_video, audio, tmp_path / "out.mp4")


def test_replace_audio_rejects_directory_as_video(tmp_path: Path) -> None:
    _, audio = _make_video_and_audio(tmp_path)
    a_directory = tmp_path / "a-directory"
    a_directory.mkdir()

    with pytest.raises(InvalidVideoPathError):
        replace_audio(a_directory, audio, tmp_path / "out.mp4")


def test_replace_audio_rejects_missing_replacement_audio(tmp_path: Path) -> None:
    video, _ = _make_video_and_audio(tmp_path)
    missing_audio = tmp_path / "does-not-exist.wav"

    with pytest.raises(InvalidReplacementAudioPathError):
        replace_audio(video, missing_audio, tmp_path / "out.mp4")


def test_replace_audio_rejects_directory_as_replacement_audio(tmp_path: Path) -> None:
    video, _ = _make_video_and_audio(tmp_path)
    a_directory = tmp_path / "a-directory"
    a_directory.mkdir()

    with pytest.raises(InvalidReplacementAudioPathError):
        replace_audio(video, a_directory, tmp_path / "out.mp4")


def test_replace_audio_raises_when_ffmpeg_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.media._ffmpeg_available", lambda: False)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run should not be called when ffmpeg is missing")

    monkeypatch.setattr("dubshift.media.subprocess.run", fail_if_called)

    video, audio = _make_video_and_audio(tmp_path)

    with pytest.raises(MissingFFmpegError, match="(?i)ffmpeg"):
        replace_audio(video, audio, tmp_path / "out.mp4")


def test_replace_audio_refuses_to_overwrite_existing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run should not be called when output already exists")

    monkeypatch.setattr("dubshift.media.subprocess.run", fail_if_called)

    video, audio = _make_video_and_audio(tmp_path)
    existing_output = tmp_path / "out.mp4"
    existing_output.write_bytes(b"already here")

    with pytest.raises(OutputExistsError):
        replace_audio(video, audio, existing_output)


def test_replace_audio_builds_expected_ffmpeg_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)
    output = tmp_path / "out.mp4"
    commands: list[list] = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return _fake_run_for_remux(video_path=video, audio_path=audio)(command)

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    result = replace_audio(video, audio, output)

    mux_command = next(c for c in commands if c[0] == "ffmpeg")
    assert mux_command == [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-af",
        "atrim=end=5.000000,apad=whole_dur=5.000000",
        "-c:a",
        "aac",
        "-n",
        str(result),
    ]


def test_replace_audio_omits_shortest_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video, audio = _make_video_and_audio(tmp_path)
    commands: list[list] = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return _fake_run_for_remux(video_path=video, audio_path=audio)(command)

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    replace_audio(video, audio, tmp_path / "out.mp4")

    mux_command = next(c for c in commands if c[0] == "ffmpeg")
    assert "-shortest" not in mux_command


def test_replace_audio_copies_video_stream_without_reencoding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)
    commands: list[list] = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return _fake_run_for_remux(video_path=video, audio_path=audio)(command)

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    replace_audio(video, audio, tmp_path / "out.mp4")

    mux_command = next(c for c in commands if c[0] == "ffmpeg")
    assert "-c:v" in mux_command
    c_v_index = mux_command.index("-c:v")
    assert mux_command[c_v_index + 1] == "copy"


def test_replace_audio_uses_replacement_audio_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)
    commands: list[list] = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return _fake_run_for_remux(video_path=video, audio_path=audio)(command)

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    replace_audio(video, audio, tmp_path / "out.mp4")

    # input index 1 is the replacement audio file; its first audio stream
    # must be mapped into the output.
    mux_command = next(c for c in commands if c[0] == "ffmpeg")
    assert "1:a:0" in mux_command


def test_replace_audio_does_not_map_original_audio_track(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)
    commands: list[list] = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return _fake_run_for_remux(video_path=video, audio_path=audio)(command)

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    replace_audio(video, audio, tmp_path / "out.mp4")

    # input index 0 is the original video; none of its audio streams
    # (0:a, 0:a:0, ...) may be mapped.
    mux_command = next(c for c in commands if c[0] == "ffmpeg")
    mapped_targets = [mux_command[i + 1] for i, token in enumerate(mux_command) if token == "-map"]
    assert not any(target.startswith("0:a") for target in mapped_targets)
    assert mapped_targets == ["0:v:0", "1:a:0"]


def test_replace_audio_raises_on_ffmpeg_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)

    monkeypatch.setattr(
        "dubshift.media.subprocess.run",
        _fake_run_for_remux(
            video_path=video,
            audio_path=audio,
            mux_returncode=1,
            mux_stderr="Invalid data found when processing input",
        ),
    )

    with pytest.raises(AudioReplacementError, match="Invalid data found"):
        replace_audio(video, audio, tmp_path / "out.mp4")


def test_replace_audio_raises_if_ffmpeg_vanishes_during_mux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        if command[0] == "ffprobe":
            return _fake_run_for_remux(video_path=video, audio_path=audio)(command)
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    with pytest.raises(MissingFFmpegError):
        replace_audio(video, audio, tmp_path / "out.mp4")


def test_replace_audio_raises_if_ffprobe_vanishes_during_duration_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    video, audio = _make_video_and_audio(tmp_path)

    with pytest.raises(AudioReplacementError, match="(?i)ffprobe"):
        replace_audio(video, audio, tmp_path / "out.mp4")


def test_replace_audio_raises_if_output_file_missing_after_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        if command[0] == "ffprobe":
            return _fake_run_for_remux(video_path=video, audio_path=audio)(command)
        # ffmpeg reports success but, unexpectedly, writes nothing.
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    with pytest.raises(AudioReplacementError, match="missing"):
        replace_audio(video, audio, tmp_path / "out.mp4")


# ---------------------------------------------------------------------------
# replace_audio: duration-mismatch tolerance
# ---------------------------------------------------------------------------


def test_replace_audio_rejects_materially_shorter_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)
    commands: list[list] = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return _fake_run_for_remux(
            video_path=video, audio_path=audio, video_duration=5.0, audio_duration=4.0
        )(command)

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    output = tmp_path / "out.mp4"
    with pytest.raises(MediaDurationMismatchError):
        replace_audio(video, audio, output)

    # The mux must never run -- no truncation, no output file.
    assert not any(c[0] == "ffmpeg" for c in commands)
    assert not output.exists()


def test_replace_audio_rejects_materially_longer_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)
    commands: list[list] = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return _fake_run_for_remux(
            video_path=video, audio_path=audio, video_duration=5.0, audio_duration=6.5
        )(command)

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    output = tmp_path / "out.mp4"
    with pytest.raises(MediaDurationMismatchError):
        replace_audio(video, audio, output)

    assert not any(c[0] == "ffmpeg" for c in commands)
    assert not output.exists()


def test_replace_audio_succeeds_with_tiny_difference_within_tolerance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)

    # Comfortably inside the default 0.5s tolerance.
    assert 0.2 < DEFAULT_REMUX_DURATION_TOLERANCE_SECONDS
    monkeypatch.setattr(
        "dubshift.media.subprocess.run",
        _fake_run_for_remux(video_path=video, audio_path=audio, video_duration=5.0, audio_duration=5.2),
    )

    output = tmp_path / "out.mp4"
    result = replace_audio(video, audio, output)

    assert result == output
    assert output.exists()


def test_replace_audio_rejects_difference_exactly_above_tolerance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)

    monkeypatch.setattr(
        "dubshift.media.subprocess.run",
        _fake_run_for_remux(
            video_path=video,
            audio_path=audio,
            video_duration=5.0,
            audio_duration=5.0 + DEFAULT_REMUX_DURATION_TOLERANCE_SECONDS + 0.01,
        ),
    )

    with pytest.raises(MediaDurationMismatchError):
        replace_audio(video, audio, tmp_path / "out.mp4")


# ---------------------------------------------------------------------------
# replace_audio: small accepted mismatches are normalized, not left as drift
# ---------------------------------------------------------------------------


def test_replace_audio_pads_slightly_short_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)
    commands: list[list] = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return _fake_run_for_remux(
            video_path=video, audio_path=audio, video_duration=5.0, audio_duration=4.8
        )(command)

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    replace_audio(video, audio, tmp_path / "out.mp4")

    mux_command = next(c for c in commands if c[0] == "ffmpeg")
    af_index = mux_command.index("-af")
    audio_filter = mux_command[af_index + 1]
    assert "apad=whole_dur=5.000000" in audio_filter
    assert "atrim=end=5.000000" in audio_filter


def test_replace_audio_trims_slightly_long_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)
    commands: list[list] = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return _fake_run_for_remux(
            video_path=video, audio_path=audio, video_duration=5.0, audio_duration=5.3
        )(command)

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    replace_audio(video, audio, tmp_path / "out.mp4")

    mux_command = next(c for c in commands if c[0] == "ffmpeg")
    af_index = mux_command.index("-af")
    audio_filter = mux_command[af_index + 1]
    assert "atrim=end=5.000000" in audio_filter
    assert "apad=whole_dur=5.000000" in audio_filter


def test_replace_audio_normalization_target_is_video_duration_not_audio_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The output's governing duration must be the video's, in both directions."""
    video, audio = _make_video_and_audio(tmp_path)

    for video_duration, audio_duration in [(5.0, 4.8), (5.0, 5.3)]:
        output = tmp_path / f"out-{audio_duration}.mp4"
        commands: list[list] = []

        def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
            commands.append(command)
            return _fake_run_for_remux(
                video_path=video,
                audio_path=audio,
                video_duration=video_duration,
                audio_duration=audio_duration,
            )(command)

        monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

        replace_audio(video, audio, output)

        mux_command = next(c for c in commands if c[0] == "ffmpeg")
        audio_filter = mux_command[mux_command.index("-af") + 1]
        assert f"{video_duration:.6f}" in audio_filter
        assert f"{audio_duration:.6f}" not in audio_filter
        assert "-shortest" not in mux_command


def test_replace_audio_omits_shortest_flag_with_accepted_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, audio = _make_video_and_audio(tmp_path)
    commands: list[list] = []

    def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return _fake_run_for_remux(
            video_path=video, audio_path=audio, video_duration=5.0, audio_duration=4.8
        )(command)

    monkeypatch.setattr("dubshift.media.subprocess.run", fake_run)

    replace_audio(video, audio, tmp_path / "out.mp4")

    mux_command = next(c for c in commands if c[0] == "ffmpeg")
    assert "-shortest" not in mux_command
