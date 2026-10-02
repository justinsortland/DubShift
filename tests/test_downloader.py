from pathlib import Path

import pytest

from dubshift.downloader import (
    DownloadError,
    InvalidURLError,
    download_video,
    validate_youtube_url,
)


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


class _FakeYoutubeDL:
    """Stand-in for yt_dlp.YoutubeDL that writes a fake file instead of downloading."""

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
        (out_dir / f"{video_id}.mp4").write_bytes(b"fake video bytes")
        return {"id": video_id, "ext": "mp4"}

    def prepare_filename(self, info: dict) -> str:
        out_dir = Path(self.opts["outtmpl"]).parent
        return str(out_dir / f"{info['id']}.{info['ext']}")


def test_download_video_creates_output_dir_and_returns_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dubshift.downloader.yt_dlp.YoutubeDL", _FakeYoutubeDL)

    target_dir = tmp_path / "nested" / "downloads"
    assert not target_dir.exists()

    result = download_video("https://youtu.be/dQw4w9WgXcQ", output_dir=target_dir)

    assert target_dir.exists()
    assert result == target_dir / "fakeid123.mp4"
    assert result.exists()


def test_download_video_disables_playlist_downloads(
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
