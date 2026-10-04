from pathlib import Path

import pytest

from dubshift.cache import (
    ConvertedChunkCache,
    compute_cache_key,
    default_cache_dir,
    provider_identity_for,
)


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_default_cache_dir_honors_env_var_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "custom-cache"
    monkeypatch.setenv("DUBSHIFT_CACHE_DIR", str(override))

    assert default_cache_dir() == override


def test_default_cache_dir_falls_back_to_home_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DUBSHIFT_CACHE_DIR", raising=False)

    result = default_cache_dir()

    assert result == Path.home() / ".cache" / "dubshift" / "converted-chunks"


def test_provider_identity_for_uses_cache_identity_attribute_when_present() -> None:
    class _ConverterWithIdentity:
        cache_identity = "elevenlabs:model-x:format-y"

        def convert(self, *args: object, **kwargs: object) -> None:
            raise NotImplementedError

    assert provider_identity_for(_ConverterWithIdentity()) == "elevenlabs:model-x:format-y"


def test_provider_identity_for_falls_back_to_class_name_when_absent() -> None:
    class _PlainConverter:
        def convert(self, *args: object, **kwargs: object) -> None:
            raise NotImplementedError

    identity = provider_identity_for(_PlainConverter())

    assert identity.endswith("_PlainConverter")
    assert identity == f"{_PlainConverter.__module__}.{_PlainConverter.__qualname__}"


def test_compute_cache_key_is_deterministic(tmp_path: Path) -> None:
    chunk = _write(tmp_path / "chunk.wav", b"identical audio bytes")

    key_one = compute_cache_key(chunk, voice="voice-1", provider_identity="elevenlabs:model:format")
    key_two = compute_cache_key(chunk, voice="voice-1", provider_identity="elevenlabs:model:format")

    assert key_one == key_two
    # A SHA-256 hex digest.
    assert len(key_one) == 64
    assert all(c in "0123456789abcdef" for c in key_one)


def test_compute_cache_key_changes_with_source_audio_content(tmp_path: Path) -> None:
    chunk_a = _write(tmp_path / "a.wav", b"audio content A")
    chunk_b = _write(tmp_path / "b.wav", b"audio content B")

    key_a = compute_cache_key(chunk_a, voice="voice-1", provider_identity="elevenlabs:model:format")
    key_b = compute_cache_key(chunk_b, voice="voice-1", provider_identity="elevenlabs:model:format")

    assert key_a != key_b


def test_compute_cache_key_changes_with_voice(tmp_path: Path) -> None:
    chunk = _write(tmp_path / "chunk.wav", b"identical audio bytes")

    key_voice_1 = compute_cache_key(chunk, voice="voice-1", provider_identity="elevenlabs:model:format")
    key_voice_2 = compute_cache_key(chunk, voice="voice-2", provider_identity="elevenlabs:model:format")

    assert key_voice_1 != key_voice_2


def test_compute_cache_key_changes_with_provider_identity(tmp_path: Path) -> None:
    chunk = _write(tmp_path / "chunk.wav", b"identical audio bytes")

    key_provider_a = compute_cache_key(chunk, voice="voice-1", provider_identity="elevenlabs:model-a:format")
    key_provider_b = compute_cache_key(chunk, voice="voice-1", provider_identity="elevenlabs:model-b:format")

    assert key_provider_a != key_provider_b


def test_compute_cache_key_is_not_based_on_filename(tmp_path: Path) -> None:
    chunk_one = _write(tmp_path / "first_name.wav", b"identical audio bytes")
    chunk_two = _write(tmp_path / "completely_different_name.wav", b"identical audio bytes")

    key_one = compute_cache_key(chunk_one, voice="voice-1", provider_identity="elevenlabs:model:format")
    key_two = compute_cache_key(chunk_two, voice="voice-1", provider_identity="elevenlabs:model:format")

    assert key_one == key_two


def test_cache_get_is_none_on_miss(tmp_path: Path) -> None:
    cache = ConvertedChunkCache(tmp_path / "cache")

    assert cache.get("some-key-that-was-never-written") is None


def test_cache_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = ConvertedChunkCache(tmp_path / "cache")
    source = _write(tmp_path / "converted.mp3", b"converted audio bytes")

    cache.put("some-key", source)
    cached_path = cache.get("some-key")

    assert cached_path is not None
    assert cached_path.read_bytes() == b"converted audio bytes"


def test_cache_put_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ConvertedChunkCache(cache_dir)
    source = _write(tmp_path / "converted.mp3", b"converted audio bytes")

    cache.put("some-key", source)

    entries = list(cache_dir.iterdir())
    assert len(entries) == 1
    assert not entries[0].name.startswith(".")


def test_cache_put_failure_leaves_no_partial_or_temp_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    cache = ConvertedChunkCache(cache_dir)
    source = _write(tmp_path / "converted.mp3", b"converted audio bytes")

    def failing_replace(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("dubshift.cache.os.replace", failing_replace)

    with pytest.raises(OSError, match="disk full"):
        cache.put("some-key", source)

    assert cache.get("some-key") is None
    assert list(cache_dir.iterdir()) == []


def test_cache_clear_removes_all_entries_and_reports_count(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ConvertedChunkCache(cache_dir)
    cache.put("key-one", _write(tmp_path / "one.mp3", b"one"))
    cache.put("key-two", _write(tmp_path / "two.mp3", b"two"))

    removed = cache.clear()

    assert removed == 2
    assert cache.get("key-one") is None
    assert cache.get("key-two") is None


def test_cache_clear_on_nonexistent_directory_returns_zero(tmp_path: Path) -> None:
    cache = ConvertedChunkCache(tmp_path / "never-created")

    assert cache.clear() == 0
