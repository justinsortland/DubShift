"""Persistent, content-addressed cache for converted audio chunks.

Caches successful provider conversions keyed by a SHA-256 hash of
everything that materially affects the result: the exact source chunk
audio bytes, the provider's identity (including its model and any other
conversion settings it exposes), and the target voice. This lets a
``dubshift dub`` run that fails partway through (e.g. a provider
quota/network error on a later chunk) be retried without re-paying for
chunks that already converted successfully.

Never keyed by filename, path, or timestamp -- only by content. Lives
outside any ``TemporaryDirectory`` so it survives both pipeline cleanup
and process restarts.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

_CACHE_DIR_ENV_VAR = "DUBSHIFT_CACHE_DIR"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "dubshift" / "converted-chunks"

_READ_BLOCK_SIZE = 1024 * 1024  # 1 MiB


def default_cache_dir() -> Path:
    """Where cached converted chunks live, overridable via DUBSHIFT_CACHE_DIR."""
    override = os.environ.get(_CACHE_DIR_ENV_VAR)
    return Path(override) if override else _DEFAULT_CACHE_DIR


def provider_identity_for(converter: object) -> str:
    """A stable string identifying a converter's provider/model/settings for caching.

    Converters may expose a ``cache_identity`` attribute or property with a
    precise identity (e.g. including the model and output format actually
    used); if absent, falls back to the converter's fully-qualified class
    name, which is still a reasonable identity for a converter that
    doesn't vary its settings.
    """
    identity = getattr(converter, "cache_identity", None)
    if isinstance(identity, str) and identity:
        return identity
    return f"{type(converter).__module__}.{type(converter).__qualname__}"


def compute_cache_key(chunk_path: Path, *, voice: str, provider_identity: str) -> str:
    """A SHA-256 hex digest over the chunk's exact bytes plus voice and provider identity.

    Depends only on content -- never on a filename, path, or timestamp --
    so the same source audio, voice, and provider/model/settings always
    produce the same key, and anything that could materially change the
    converted result changes the key too.
    """
    hasher = hashlib.sha256()
    hasher.update(provider_identity.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(voice.encode("utf-8"))
    hasher.update(b"\x00")
    with open(chunk_path, "rb") as source_file:
        for block in iter(lambda: source_file.read(_READ_BLOCK_SIZE), b""):
            hasher.update(block)
    return hasher.hexdigest()


class ConvertedChunkCache:
    """An on-disk cache of provider conversion results, keyed by content hash."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _entry_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{cache_key}.cache"

    def get(self, cache_key: str) -> Path | None:
        """Return the path to a cached result for ``cache_key``, or ``None`` on a miss."""
        entry_path = self._entry_path(cache_key)
        return entry_path if entry_path.is_file() else None

    def put(self, cache_key: str, source_path: Path) -> Path:
        """Persist ``source_path``'s bytes under ``cache_key``, atomically.

        Writes to a temporary file inside the cache directory first, then
        renames it into place, so a crash or a concurrent reader never
        sees a partially-written cache entry.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        entry_path = self._entry_path(cache_key)

        fd, tmp_name = tempfile.mkstemp(dir=self._cache_dir, prefix=f".{cache_key}.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            os.close(fd)
            shutil.copyfile(source_path, tmp_path)
            os.replace(tmp_path, entry_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return entry_path

    def clear(self) -> int:
        """Delete every cached entry. Returns the number of entries removed."""
        if not self._cache_dir.exists():
            return 0
        removed = 0
        for entry in self._cache_dir.iterdir():
            if entry.is_file():
                entry.unlink()
                removed += 1
        return removed
