"""Tests for dubshift.pipeline -- the download/extract/convert/remux orchestration.

Every stage (download_video, extract_audio, convert_long_audio,
replace_audio) is mocked at the dubshift.pipeline import site. No real
network, subprocess, or ElevenLabs calls happen here.
"""

from pathlib import Path

import pytest

from dubshift.converters.exceptions import ConversionFailedError
from dubshift.converters.fake import FakeVoiceConverter
from dubshift.downloader import DownloadError
from dubshift.media import AudioExtractionError, AudioReplacementError
from dubshift.pipeline import PipelineOutputExistsError, dub_video


def _install_stage_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    call_log: list[str],
    captured: dict,
    download_error: Exception | None = None,
    extract_error: Exception | None = None,
    convert_error: Exception | None = None,
    remux_error: Exception | None = None,
) -> None:
    """Patch every pipeline stage with a simple, file-producing fake.

    Each fake records what it was called with (into ``captured``) and its
    name (into ``call_log``), then either raises the configured error or
    writes a small marker file to the path it was asked to produce and
    returns that path -- mirroring each real function's contract.
    """

    def fake_download_video(url: str, output_dir: Path) -> Path:
        call_log.append("download")
        captured["download_url"] = url
        captured["download_output_dir"] = output_dir
        if download_error is not None:
            raise download_error
        result = output_dir / "video.mp4"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_bytes(b"fake video bytes")
        return result

    def fake_extract_audio(video_path: Path, output_path: Path) -> Path:
        call_log.append("extract")
        captured["extract_video_path"] = video_path
        captured["extract_output_path"] = output_path
        if extract_error is not None:
            raise extract_error
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake source wav bytes")
        return output_path

    def fake_convert_long_audio(input_path: Path, output_path: Path, voice: str, converter) -> Path:
        call_log.append("convert")
        captured["convert_input_path"] = input_path
        captured["convert_output_path"] = output_path
        captured["convert_voice"] = voice
        captured["convert_converter"] = converter
        if convert_error is not None:
            raise convert_error
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake converted wav bytes")
        return output_path

    def fake_replace_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
        call_log.append("remux")
        captured["remux_video_path"] = video_path
        captured["remux_audio_path"] = audio_path
        captured["remux_output_path"] = output_path
        if remux_error is not None:
            raise remux_error
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake dubbed mp4 bytes")
        return output_path

    monkeypatch.setattr("dubshift.pipeline.download_video", fake_download_video)
    monkeypatch.setattr("dubshift.pipeline.extract_audio", fake_extract_audio)
    monkeypatch.setattr("dubshift.pipeline.convert_long_audio", fake_convert_long_audio)
    monkeypatch.setattr("dubshift.pipeline.replace_audio", fake_replace_audio)


def test_dub_video_runs_stages_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(monkeypatch, call_log=call_log, captured=captured)

    output_path = tmp_path / "dubbed.mp4"
    result = dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "voice-1", FakeVoiceConverter())

    assert call_log == ["download", "extract", "convert", "remux"]
    assert result == output_path
    assert output_path.exists()


def test_dub_video_passes_url_through_to_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(monkeypatch, call_log=call_log, captured=captured)

    output_path = tmp_path / "dubbed.mp4"
    dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "voice-1", FakeVoiceConverter())

    assert captured["download_url"] == "https://youtu.be/dQw4w9WgXcQ"


def test_dub_video_passes_paths_between_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(monkeypatch, call_log=call_log, captured=captured)

    output_path = tmp_path / "dubbed.mp4"
    dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "voice-1", FakeVoiceConverter())

    downloaded_video = captured["download_output_dir"] / "video.mp4"

    # The downloaded video feeds both extraction's input AND remux's
    # original-video input (remux must use the real downloaded video, not
    # anything derived from extraction/conversion).
    assert captured["extract_video_path"] == downloaded_video
    assert captured["remux_video_path"] == downloaded_video

    # Extraction's output feeds conversion's input.
    assert captured["convert_input_path"] == captured["extract_output_path"]

    # Conversion's output feeds remux's replacement-audio input.
    assert captured["remux_audio_path"] == captured["convert_output_path"]

    # The pipeline's own output_path is exactly what reaches remux's output.
    assert captured["remux_output_path"] == output_path


def test_dub_video_passes_converter_and_voice_into_convert_long_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(monkeypatch, call_log=call_log, captured=captured)

    output_path = tmp_path / "dubbed.mp4"
    converter = FakeVoiceConverter()

    dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "narrator-1", converter)

    assert captured["convert_voice"] == "narrator-1"
    assert captured["convert_converter"] is converter


def test_dub_video_rejects_existing_output_before_any_stage_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(monkeypatch, call_log=call_log, captured=captured)

    output_path = tmp_path / "dubbed.mp4"
    output_path.write_bytes(b"already here")

    with pytest.raises(PipelineOutputExistsError):
        dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "voice-1", FakeVoiceConverter())

    assert call_log == []
    assert output_path.read_bytes() == b"already here"


def test_download_failure_stops_later_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(
        monkeypatch, call_log=call_log, captured=captured, download_error=DownloadError("network broke")
    )

    output_path = tmp_path / "dubbed.mp4"

    with pytest.raises(DownloadError):
        dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "voice-1", FakeVoiceConverter())

    assert call_log == ["download"]
    assert not output_path.exists()


def test_extraction_failure_stops_conversion_and_remux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(
        monkeypatch,
        call_log=call_log,
        captured=captured,
        extract_error=AudioExtractionError("bad codec"),
    )

    output_path = tmp_path / "dubbed.mp4"

    with pytest.raises(AudioExtractionError):
        dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "voice-1", FakeVoiceConverter())

    assert call_log == ["download", "extract"]
    assert not output_path.exists()


def test_conversion_failure_stops_remux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(
        monkeypatch,
        call_log=call_log,
        captured=captured,
        convert_error=ConversionFailedError("provider exploded"),
    )

    output_path = tmp_path / "dubbed.mp4"

    with pytest.raises(ConversionFailedError):
        dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "voice-1", FakeVoiceConverter())

    assert call_log == ["download", "extract", "convert"]
    assert not output_path.exists()


def test_remux_failure_propagates_cleanly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(
        monkeypatch,
        call_log=call_log,
        captured=captured,
        remux_error=AudioReplacementError("mux failed: corrupt input"),
    )

    output_path = tmp_path / "dubbed.mp4"

    with pytest.raises(AudioReplacementError, match="mux failed"):
        dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "voice-1", FakeVoiceConverter())

    assert call_log == ["download", "extract", "convert", "remux"]
    assert not output_path.exists()


def test_intermediates_are_cleaned_up_on_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(monkeypatch, call_log=call_log, captured=captured)

    output_path = tmp_path / "dubbed.mp4"
    dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "voice-1", FakeVoiceConverter())

    tmp_dir = captured["download_output_dir"].parent
    assert not tmp_dir.exists()
    assert output_path.exists()


def test_intermediates_are_cleaned_up_on_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(
        monkeypatch,
        call_log=call_log,
        captured=captured,
        convert_error=ConversionFailedError("boom"),
    )

    output_path = tmp_path / "dubbed.mp4"

    with pytest.raises(ConversionFailedError):
        dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "voice-1", FakeVoiceConverter())

    tmp_dir = captured["download_output_dir"].parent
    assert not tmp_dir.exists()
    assert not output_path.exists()


def test_final_output_survives_temporary_directory_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call_log: list[str] = []
    captured: dict = {}
    _install_stage_fakes(monkeypatch, call_log=call_log, captured=captured)

    output_path = tmp_path / "dubbed.mp4"
    result = dub_video("https://youtu.be/dQw4w9WgXcQ", output_path, "voice-1", FakeVoiceConverter())

    tmp_dir = captured["download_output_dir"].parent
    assert not tmp_dir.exists()
    assert result.exists()
    assert result.read_bytes() == b"fake dubbed mp4 bytes"
