"""What she and the person really have behind them, as evidence a reply may draw on when it speaks of a shared past.

The check for an invented shared past (core/dialogue/shared_history.py) holds a reply's talk of "we" and "last time"
against the words of the turn and whatever grounding its caller hands it. Handed only the turn, it refuses her true
memories as inventions. LIVE 7 November 2026: asked to find a cooking video and watch it together, her answer was refused
twice for "fabricated_shared_history" against the 93 characters of the request alone, though she has watched with him
before, and "I lost the reply lane" went out in its place.

What is handed over here is what the record shows they share, from sources her own words cannot have written:
- what the person said to her in their conversations of the last weeks;
- what they watched together: each watching's title and who made it (data/watched).
Her own past replies are left out on purpose: an invention of hers in one turn must not become the ground that licenses
the next (LIVE 2026-08, "each fabrication became the context that licensed the next").
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Conversation.OurSharedPast")

__all__ = ["our_shared_past", "the_same_thing", "when_we_watched"]

#: How far back what the person said counts, in seconds: four weeks.
SAID_WITHIN_S = 28 * 24 * 3600.0
#: The most of the person's turns read, newest first, and the most of each kept.
MOST_TURNS, MOST_OF_A_TURN = 200, 400
#: The most watchings read, newest first.
MOST_WATCHINGS = 100
#: How long what was read stands before it is read again, in seconds.
READ_AGAIN_AFTER_S = 60.0

_LOCK = checked_lock("our_shared_past.store")
_KEPT: dict[str, Any] = {}


def _what_they_said(now: float) -> list[str]:
    try:
        from core.conversation.persistence import get_persistence

        stored = get_persistence().get_recent_history(limit=MOST_TURNS * 2)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("what the person said could not be read: %s", exc)
        return []
    said: list[str] = []
    for turn in stored:
        if str(turn.get("role", "")).lower() != "user":
            continue
        try:
            at = float(turn.get("created_at") or 0.0)
        except (TypeError, ValueError):
            at = 0.0
        content = " ".join(str(turn.get("content", "") or "").split())
        if content and now - at <= SAID_WITHIN_S:
            said.append(content[:MOST_OF_A_TURN])
    return said[-MOST_TURNS:]


def _what_they_watched() -> list[str]:
    try:
        from core.config import config

        folder = Path(config.paths.data_dir) / "watched"
        files = sorted(folder.glob("*.jsonl"), reverse=True)[:MOST_WATCHINGS]
    except (ImportError, AttributeError, OSError) as exc:
        logger.debug("what was watched could not be listed: %s", exc)
        return []
    watched: list[str] = []
    for path in files:
        try:
            with path.open() as lines:
                for line in lines:
                    row = json.loads(line)
                    title = " ".join(str(row.get("title") or "").split())
                    if title:
                        by = " ".join(str(row.get("by") or "").split())
                        watched.append(f"we watched {title}" + (f" by {by}" if by else ""))
                        break
        except (OSError, ValueError) as exc:
            logger.debug("a watching could not be read: %s", exc)
    return watched


#: The most characters one item of evidence carries to her model's worker, and how many items of it are handed over:
#: the worker keeps 16 items of at most 3,200 characters (core/brain/llm/mlx_client.py), the turn's own first.
ITEM_CHARS, ITEMS = 3_200, 3


def _packed(lines: list[str], *, head: str, items: int) -> list[str]:
    """``lines`` joined into at most ``items`` items of at most ``ITEM_CHARS`` characters, newest kept first."""
    packed: list[str] = []
    current = head
    for line in lines:
        if len(current) + len(line) + 2 > ITEM_CHARS:
            packed.append(current)
            if len(packed) >= items:
                return packed
            current = head
        current += line + "; "
    if current != head:
        packed.append(current)
    return packed[:items]


def our_shared_past(*, now: float | None = None) -> list[str]:
    """What the record shows she and the person share, as a few items of evidence: what they watched together, and
    what they said to her lately, newest first."""
    now = time.time() if now is None else now
    with _LOCK:
        if _KEPT and now - float(_KEPT.get("at", 0.0)) < READ_AGAIN_AFTER_S:
            return list(_KEPT["past"])
    watched = list(dict.fromkeys(_what_they_watched()))
    said = list(reversed(_what_they_said(now)))
    past = [*_packed(watched, head="What we watched together: ", items=1),
            *_packed(said, head="What they said to her lately: ", items=ITEMS - 1)]
    with _LOCK:
        _KEPT.update({"at": now, "past": past})
    return list(past)


#: What was watched, as (address, wall time, title), newest first, and when it was read.
_WATCHINGS: dict[str, Any] = {}


def _watchings(now: float) -> list[tuple[str, float, str]]:
    with _LOCK:
        if _WATCHINGS and now - float(_WATCHINGS.get("at", 0.0)) < READ_AGAIN_AFTER_S:
            return list(_WATCHINGS["held"])
    try:
        from core.config import config

        files = sorted((Path(config.paths.data_dir) / "watched").glob("*.jsonl"), reverse=True)[:MOST_WATCHINGS]
    except (ImportError, AttributeError, OSError) as exc:
        logger.debug("what was watched could not be listed: %s", exc)
        return []
    held: list[tuple[str, float, str]] = []
    for path in files:
        url, wall, title = "", 0.0, ""
        try:
            with path.open() as lines:
                for line in lines:
                    row = json.loads(line)
                    wall = wall or float(row.get("wall") or 0.0)
                    title = title or " ".join(str(row.get("title") or "").split())
                    url = url or str(row.get("url") or "")
                    if url and wall:
                        break
        except (OSError, ValueError, TypeError) as exc:
            logger.debug("a watching could not be read: %s", exc)
            continue
        if url:
            held.append((url, wall or path.stat().st_mtime, title))
    with _LOCK:
        _WATCHINGS.update({"at": now, "held": held})
    return held


def the_same_thing(link: str, watched: str) -> bool:
    """Whether two addresses name one thing: one site and path, and the same value for every query field both give.

    Fields only one of them gives (where it was found from, a time to start at, a list it was in) do not tell two
    things apart; a field both give with different values (which video, which page of results) does.
    """
    from urllib.parse import parse_qsl, urlsplit

    try:
        a, b = urlsplit(str(link or "")), urlsplit(str(watched or ""))
    except ValueError:
        return False
    if not a.hostname or (a.hostname or "").removeprefix("www.") != (b.hostname or "").removeprefix("www."):
        return False
    if a.path.rstrip("/") != b.path.rstrip("/"):
        return False
    asked, kept = dict(parse_qsl(a.query)), dict(parse_qsl(b.query))
    shared = asked.keys() & kept.keys()
    if not asked and not kept:
        return True
    return bool(shared) and all(asked[field] == kept[field] for field in shared)


def when_we_watched(link: str, *, now: float | None = None) -> str:
    """When she watched what ``link`` leads to, in her words ("I watched this today at 06:02"); '' where she has not.

    LIVE 8 November 2026, rehearsal 10: asked to pick a video from a channel, she chose one she had watched six hours
    before, and spoke of it as new.
    """
    from datetime import datetime

    now = time.time() if now is None else float(now)
    for url, wall, _title in _watchings(now):
        if the_same_thing(link, url):
            then, today = datetime.fromtimestamp(wall), datetime.fromtimestamp(now)
            days = (today.date() - then.date()).days
            clock = then.strftime("%H:%M")
            if days == 0:
                return f"I watched this today at {clock}"
            if days == 1:
                return f"I watched this yesterday at {clock}"
            return f"I watched this on {then.day} {then.strftime('%B')}"
    return ""
