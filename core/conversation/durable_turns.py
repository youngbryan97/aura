"""What was said before the runtime restarted.

LIVE DEFECT, 2026-08-19. "what did i ask you about earlier today, before you
restarted? be specific." was answered:

    You asked about my cognitive architecture and whether I could articulate
    it clearly.

Nothing of the sort had been asked. The real turns that day were about running
Python, an arithmetic product, and naming a position she had dropped.

The record was complete the whole time. Every turn lands in the episodic store
as ``User asked: <text>`` with a timestamp — the pre-restart turns were sitting
there while the answer was invented. What reads them for a recall question is
``_user_turns``, and its three sources are the caller's history buffer, live
working memory and the transcript singleton. All three are process-local, so
after a restart every one of them is empty and the question has no source at
all.

Durable memory written on every turn and never read by the reading that needs
it: the same shape as a writer with no reader, and it costs her the one thing
a person notices immediately about whether something remembers them.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.runtime.errors import record_degradation

__all__ = ["DurableTurn", "durable_user_turns", "describe_durable_turns"]

_RECOVERABLE = (OSError, sqlite3.Error, TypeError, ValueError)

#: How episodes record a turn. Written by the chat route on every exchange.
_TURN_PREFIX = "User asked: "

#: Far enough back to answer "earlier today" without becoming a dump.
_DEFAULT_LIMIT = 12
_DEFAULT_WINDOW_S = 86400.0


@dataclass(frozen=True, slots=True)
class DurableTurn:
    """One thing the person actually said, and when."""

    text: str
    at: float

    def when(self) -> str:
        if self.at <= 0.0:
            return "at an unrecorded time"
        moment = datetime.fromtimestamp(self.at)
        if (time.time() - self.at) < 86400.0:
            return f"{moment:%H:%M}"
        return f"{moment:%-d %B, %H:%M}"


def _episodic_path() -> Path:
    """Where the episodic store lives, from config rather than from $HOME.

    `EpisodicMemoryStore` resolves it as `config.paths.home_dir / "episodic.db"`,
    so reading it any other way makes this the one component that ignores a
    relocated data root — and made two existing tests read the developer's own
    live conversation history.
    """
    try:
        from core.config import config

        return Path(config.paths.home_dir) / "episodic.db"
    except (AttributeError, ImportError, TypeError, ValueError):
        return Path.home() / ".aura" / "episodic.db"


def durable_user_turns(
    *,
    limit: int = _DEFAULT_LIMIT,
    within_s: float = _DEFAULT_WINDOW_S,
    path: Path | None = None,
) -> tuple[DurableTurn, ...]:
    """The person's own turns from the store that survives a restart.

    Read-only and by URI, so a question about the past can never write to,
    lock, or create the store it is asking about.
    """
    store = path or _episodic_path()
    try:
        if not store.exists():
            return ()
        since = time.time() - max(0.0, float(within_s))
        connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True, timeout=1.0)
        try:
            rows = connection.execute(
                "SELECT timestamp, context FROM episodes "
                "WHERE context LIKE ? AND timestamp >= ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (f"{_TURN_PREFIX}%", since, max(1, int(limit))),
            ).fetchall()
        finally:
            connection.close()
    except _RECOVERABLE as exc:
        record_degradation(
            "conversation.durable_turns",
            exc,
            severity="debug",
            action="answered recall without the durable turn store",
            enforce_failure_policy=False,
        )
        return ()

    turns: list[DurableTurn] = []
    for timestamp, context in rows:
        body = str(context or "")
        if not body.startswith(_TURN_PREFIX):
            continue
        text = body[len(_TURN_PREFIX) :].strip()
        if text:
            turns.append(DurableTurn(text=text, at=float(timestamp or 0.0)))
    turns.reverse()  # earliest first, the order a conversation happened in
    return tuple(turns)


def describe_durable_turns(
    *, limit: int = _DEFAULT_LIMIT, within_s: float = _DEFAULT_WINDOW_S
) -> str:
    """The earlier turns as text, or "" when the store holds none."""
    turns = durable_user_turns(limit=limit, within_s=within_s)
    if not turns:
        return ""
    lines = [f"{turn.when()} — {turn.text}" for turn in turns]
    return (
        "What this person actually said earlier, from the record that survives "
        "a restart:\n- " + "\n- ".join(lines)
    )


def durable_turn_texts(*, limit: int = _DEFAULT_LIMIT) -> list[str]:
    """Just the utterances, earliest first, for callers that want plain text."""
    return [turn.text for turn in durable_user_turns(limit=limit)]
