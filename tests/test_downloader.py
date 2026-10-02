from pathlib import Path

import pytest

from dubshift.downloader import (
    FORMAT_SELECTOR,
    MERGE_OUTPUT_FORMAT,
    DownloadError,
    InvalidURLError,
    MissingFFmpegError,
    download_video,
    validate_youtube_url,
)


@pytest.fixture(autouse=True)
def ffmpeg_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assume ffmpeg is installed by default; the dedicated test below overrides this."""
    monkeypatch.setattr("dubshift.downloader._ffmpeg_available", lambda: True)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "http://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    ],
)
def test_validate_youtube_url_accepts_valid_urls(url: str) -> None:
    assert validate_youtube_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not a url",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://vimeo.com/123456",
        "https://www.youtube.com/playlist?list=PL12345",
        "https://www.youtube.com/",
    ],
)
def test_validate_youtube_url_rejects_invalid_urls(url: str) -> None:
    with pytest.raises(InvalidURLError):
        validate_youtube_url(url)


def test_download_video_rejects_invalid_url_before_touching_yt_dlp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("yt_dlp.YoutubeDL should not be constructed for invalid input")

    monkeypatch.setattr("dubshift.downloader.yt_dlp.YoutubeDL", fail_if_called)

    with pytest.raises(InvalidURLError):
        download_video("not a url", output_dir=tmp_path)


def test_download_video_raises_when_ffmpeg_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.downloader._ffmpeg_available", lambda: False)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("yt_dlp.YoutubeDL should not be constructed when ffmpeg is missing")

    monkeypatch.setattr("dubshift.downloader.yt_dlp.YoutubeDL", fail_if_called)

    with pytest.raises(MissingFFmpegError, match="(?i)ffmpeg"):
        download_video("https://youtu.be/dQw4w9WgXcQ", output_dir=tmp_path)


class _FakeYoutubeDL:
    """Stand-in for yt_dlp.YoutubeDL that simulates separate video/audio streams
    being downloaded and merged into a single file by ffmpeg.

    This mirrors real yt-dlp behavior: prepare_filename() reflects the
    pre-merge (video-only) extension, while the file that actually ends up
    on disk uses the configured merge_output_format extension.
    """

    def __init__(self, opts: dict) -> None:
        self.opts = opts

    def __enter__(self) -> "_FakeYoutubeDL":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def extract_info(self, url: str, download: bool = True) -> dict:
        video_id = "fakeid123"
        out_dir = Path(self.opts["outtmpl"]).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        merge_ext = self.opts.get("merge_output_format", "mp4")
        (out_dir / f"{video_id}.{merge_ext}").write_bytes(b"fake merged video bytes")
        return {"id": video_id, "ext": "webm"}

    def prepare_filename(self, info: dict) -> str:
        out_dir = Path(self.opts["outtmpl"]).parent
        return str(out_dir / f"{info['id']}.{info['ext']}")


def test_download_video_creates_output_dir_and_returns_merged_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.downloader.yt_dlp.YoutubeDL", _FakeYoutubeDL)

    target_dir = tmp_path / "nested" / "downloads"
    assert not target_dir.exists()

    result = download_video("https://youtu.be/dQw4w9WgXcQ", output_dir=target_dir)

    assert target_dir.exists()
    # prepare_filename() would say "fakeid123.webm" (pre-merge); the actual
    # merged file on disk is "fakeid123.mp4".
    assert result == target_dir / "fakeid123.mp4"
    assert result.exists()


def test_download_video_requests_separate_streams_merged_to_mp4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_opts: dict = {}

    class _CapturingYoutubeDL(_FakeYoutubeDL):
        def __init__(self, opts: dict) -> None:
            captured_opts.update(opts)
            super().__init__(opts)

    monkeypatch.setattr("dubshift.downloader.yt_dlp.YoutubeDL", _CapturingYoutubeDL)

    download_video("https://youtu.be/dQw4w9WgXcQ", output_dir=tmp_path)

    assert captured_opts["noplaylist"] is True
    assert captured_opts["format"] == FORMAT_SELECTOR
    assert FORMAT_SELECTOR == "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    assert captured_opts["merge_output_format"] == MERGE_OUTPUT_FORMAT == "mp4"


def test_download_video_wraps_yt_dlp_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from yt_dlp.utils import DownloadError as YtDlpDownloadError

    class _FailingYoutubeDL(_FakeYoutubeDL):
        def extract_info(self, url: str, download: bool = True) -> dict:
            raise YtDlpDownloadError("video unavailable")

    monkeypatch.setattr("dubshift.downloader.yt_dlp.YoutubeDL", _FailingYoutubeDL)

    with pytest.raises(DownloadError, match="video unavailable"):
        download_video("https://youtu.be/dQw4w9WgXcQ", output_dir=tmp_path)


def test_download_video_errors_if_output_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _SilentlyFailingYoutubeDL(_FakeYoutubeDL):
        def extract_info(self, url: str, download: bool = True) -> dict:
            return {"id": "missingfile", "ext": "mp4"}

    monkeypatch.setattr("dubshift.downloader.yt_dlp.YoutubeDL", _SilentlyFailingYoutubeDL)

    with pytest.raises(DownloadError, match="missing"):
        download_video("https://youtu.be/dQw4w9WgXcQ", output_dir=tmp_path)
