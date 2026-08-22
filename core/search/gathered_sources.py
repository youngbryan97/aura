"""Keep what a search already fetched, so a deadline cannot erase it.

LIVE, 2026-08-22. Asked to look up a company, the runtime searched, fetched
five pages, and then had its synthesis step run past the caller's 35-second
budget. `asyncio.wait_for` cancels the task it is waiting on, and the five
pages went with it. The turn ended in "I couldn't get to an answer I'd stand
behind", with the header reading REPLY PATH BLOCKED, while five usable sources
had been in memory a moment earlier.

Gathering and summarising are different pieces of work with different failure
modes. What was gathered is recorded here the moment it exists, so a caller
whose deadline expires during the summary can still answer from the sources
instead of discarding them.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GatheredSource", "record_gathered", "take_gathered", "clear_gathered"]

#: Older than this and it belongs to a previous question.
_STALE_SECONDS = 180.0


@dataclass(frozen=True, slots=True)
class GatheredSource:
    title: str
    url: str
    text: str = ""
    snippet: str = ""


@dataclass(slots=True)
class _Gathered:
    query: str = ""
    at: float = field(default_factory=time.time)
    sources: tuple[GatheredSource, ...] = ()


#: A mutable holder, replaced per turn: a child task must be able to write
#: where the parent can read, and a ContextVar set inside a child does not
#: propagate back.
_HOLDER: contextvars.ContextVar[dict[str, _Gathered] | None] = contextvars.ContextVar(
    "aura_gathered_sources", default=None
)


def _holder() -> dict[str, _Gathered]:
    current = _HOLDER.get()
    if current is None:
        current = {}
        _HOLDER.set(current)
    return current


def record_gathered(query: object, sources: Any) -> int:
    """Note the pages a search has in hand. Returns how many were kept."""
    kept: list[GatheredSource] = []
    for item in list(sources or []):
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or "").strip()
            url = str(item.get("url") or item.get("link") or "").strip()
            text = str(item.get("text") or item.get("content") or "").strip()
            snippet = str(item.get("snippet") or item.get("summary") or "").strip()
        else:
            title = str(getattr(item, "title", "") or "").strip()
            url = str(getattr(item, "url", "") or "").strip()
            text = str(getattr(item, "text", "") or getattr(item, "content", "") or "").strip()
            snippet = str(getattr(item, "snippet", "") or "").strip()
        if not (url or text or snippet):
            continue
        kept.append(GatheredSource(title=title, url=url, text=text[:4000], snippet=snippet[:600]))
    if not kept:
        return 0
    _holder()["last"] = _Gathered(query=str(query or ""), at=time.time(), sources=tuple(kept))
    return len(kept)


def take_gathered(*, max_age_s: float = _STALE_SECONDS) -> _Gathered | None:
    """What the last search fetched, if it is recent enough to be this one's."""
    held = _holder().get("last")
    if held is None:
        return None
    if time.time() - held.at > max_age_s:
        return None
    return held


def clear_gathered() -> None:
    _holder().pop("last", None)
