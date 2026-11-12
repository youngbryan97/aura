"""The assistant message a turn produced, found in working memory."""
from __future__ import annotations

from typing import Any

__all__ = ["this_turns_reply"]


def this_turns_reply(working_memory: Any, *, mark: int, since: float | None = None) -> dict[str, Any] | None:
    """The assistant message THIS turn produced, or None.

    Extraction used to take the last message if its role was "assistant". A duplicate or suppressed user append
    leaves an older assistant message at the end of working memory, and that previous answer went out again as this
    turn's: correct-looking, addressed to the wrong question.

    ``mark`` is how long working memory was when the turn began. It is pruned while the turn runs (stale thoughts and
    reflections, and to its cap), so a reply can end up at or before the mark: 84 of about 2,600 cycles in her state
    log to 12 November 2026 ended on a new reply with working memory no longer than before, each read as unanswered
    and charged as an unresolved objective. A reply stamped at or after ``since``, when the turn began, is this
    turn's wherever it sits.
    """
    memory = list(working_memory or [])
    if not memory:
        return None
    last = memory[-1]
    if not isinstance(last, dict) or str(last.get("role", "") or "").strip().lower() != "assistant":
        return None
    if len(memory) > max(0, int(mark)):
        return last
    stamp = last.get("timestamp")
    if since is not None and isinstance(stamp, (int, float)) and stamp >= since:
        return last
    return None
