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
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.Conversation.OurSharedPast")

__all__ = ["our_shared_past"]

#: How far back what the person said counts, in seconds: four weeks.
SAID_WITHIN_S = 28 * 24 * 3600.0
#: The most of the person's turns read, newest first, and the most of each kept.
MOST_TURNS, MOST_OF_A_TURN = 200, 400
#: The most watchings read, newest first.
MOST_WATCHINGS = 100
#: How long what was read stands before it is read again, in seconds.
READ_AGAIN_AFTER_S = 60.0

_LOCK = threading.Lock()
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
