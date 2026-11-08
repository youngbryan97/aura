"""What she says in the chat without being asked, kept in the conversation where she said it.

LIVE 7 November 2026, dress rehearsal at 22:44. Over three videos she made twenty remarks and three last words in the
chat window, and the conversation she keeps held none of them: her store had the person's four requests and her four
answers, eight turns. Asked "So what did you think of all three?", her model read a conversation whose last turn was
her choosing Salad Fingers' *Spoons* and guessing what it would be like, and it answered by choosing it again: "I'm
picking Salad Fingers 1: Spoons. I've actually watched it now, so this is my real read rather than a prediction."
Whoever began it, what was said in the chat is the conversation.

A remark is kept with the exchange it follows, the last one complete when it was said, because the chat window shows
it under that exchange. An exchange is a person's message and everything she said until the next one
(``messages_the_request_reaches`` in core/conversation/delivered_history.py), so her model reads a remark as part of
what she said there.

The conversation itself lives with the chat surface, which registers a keeper here (``keep_with``); this module only
carries what was said to whoever keeps it, so anything that speaks for her without being asked can use it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Conversation.SaidUnasked")

__all__ = ["keep_with", "she_said_unasked"]

#: Keeps one thing she said unasked: (words, origin, when) -> whether it was kept.
Keeper = Callable[[str, str, float], Awaitable[bool]]

_KEEPERS: list[Keeper] = []


def keep_with(keeper: Keeper) -> None:
    """Register ``keeper`` as where what she says unasked is kept. Registering it twice keeps it once."""
    if keeper not in _KEEPERS:
        _KEEPERS.append(keeper)


async def she_said_unasked(words: str, *, origin: str) -> bool:
    """Keep ``words``, said in the chat without being asked, in her conversation. True when a keeper kept them."""
    said = str(words or "").strip()
    if not said:
        return False
    at = time.time()
    kept = False
    for keeper in list(_KEEPERS):
        try:
            kept = bool(await keeper(said, str(origin or ""), at)) or kept
        except (OSError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
            record_degradation("conversation", exc, severity="warning",
                               action="something she said unasked was shown but not kept in her conversation")
    if not kept:
        logger.info("💬 said unasked, and no conversation kept it (%d keeper(s)): %r", len(_KEEPERS), said[:80])
    return kept
