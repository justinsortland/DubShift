"""A minimal progress-reporting callback shared across pipeline stages.

Keeps user-facing progress reporting out of low-level media/provider code:
orchestration functions (``dub_video``, ``convert_long_audio``) accept an
optional callback and call it with short, human-readable status strings as
each stage starts. The CLI supplies a callback that just prints; anything
else (tests, a future UI) can supply its own, or rely on the default no-op.
"""

from __future__ import annotations

from typing import Callable

ProgressCallback = Callable[[str], None]


def noop_progress(message: str) -> None:
    """The default progress callback: does nothing."""
