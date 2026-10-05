"""What a drafted reply says, checked where the cognitive engine's reply is first assessed.

Two checks of content rather than form:

* a question that can only be answered with a number gets a reply with one;
* a dated claim agrees with what the turn's own sources say about the same
  thing (core/conversation/claims_against_sources.py). When one does not, the
  source's sentence joins the turn's sources and she is asked again, through
  the same repair retry every rejected draft goes through.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Server.Chat")

__all__ = ["checked_reply"]

#: The reason a draft is asked again for, as the repair retry names it.
DISAGREES_WITH_ITS_SOURCES = "dates_a_claim_otherwise_than_its_sources"

Retry = Callable[[str, Sequence[str]], Awaitable[str | None]]


async def checked_reply(
    visible: str,
    text: str,
    assessment_text: str,
    assessment: Any,
    *,
    retry: Retry,
    recent_user_messages: list[str],
    grounding: list[str],
    antecedent: str,
) -> tuple[str, str, Any]:
    """The reply, the text assessed, and its assessment, after both checks."""
    from core.conversation.response_reliability import (
        assess_user_facing_reply,
        numeric_answer_missing,
    )

    def assess(reply: str) -> Any:
        return assess_user_facing_reply(
            visible, reply, recent_user_messages=recent_user_messages, grounding=grounding, antecedent=antecedent,
        )

    # The engine path does not leave through _finalize_fastpath, so the
    # numeric floor installed there never saw these replies. Live
    # 2026-07-26, "What is 17 minus 8, and then times 3?" was answered with
    # "A quick refresh on classic habits: green tea, journaling, and
    # standing by the window to watch the light change." — no number, and
    # every gate passed it because they check form, not whether the
    # question was answered.
    if numeric_answer_missing(visible, text):
        logger.warning(
            "🔢 CognitiveEngine reply carried no number for a question that "
            "can only be answered with one (%d chars); refusing it rather "
            "than serving an answer to a different question.",
            len(text),
        )
        text = (
            "I didn't actually work that out — what I had wasn't an answer, "
            "and I won't dress it up as one. Ask me again and I'll do the "
            "arithmetic properly."
        )
        return text, text, assess(text)

    revised = await _against_its_sources(visible, text, retry)
    if revised:
        return revised, revised, assess(revised)
    return text, assessment_text, assessment


async def _against_its_sources(visible: str, text: str, retry: Retry) -> str | None:
    """Ask again, with the sources' own sentences, when a dated claim disagrees with them.

    LIVE 2026-10-04: "FTX Arena after they bought the rights for $135 million
    in 2022", from a turn whose Wikipedia page says "In March 2021, FTX
    acquired the naming rights". The page was read; the passage she was shown
    was about the opening date. Returns None when nothing disagrees, or when
    asking again produced nothing; the draft then stands.
    """
    from core.conversation.claims_against_sources import claims_with_years, disagreements
    from core.conversation.turn_evidence_custody import (
        record_turn_grounding,
        record_turn_world_evidence,
        turn_world_sources,
    )
    from core.conversation.what_the_world_says import WorldEvidence, WorldSource

    sources = turn_world_sources()
    if not sources:
        return None
    try:
        found = await asyncio.to_thread(disagreements, text, sources, visible)
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        record_degradation("chat.claims_against_sources", exc, severity="info", action="served the draft unchecked")
        return None
    if not found:
        claims = claims_with_years(text, visible)
        if claims:
            logger.info(
                "📚 Read %d dated claim(s) of her draft against %d source(s); none dated otherwise.",
                len(claims), len(sources),
            )
        return None
    for one in found:
        logger.warning(
            "📚 Her draft dates a claim otherwise than her sources: %r — %s says %r (%.3f).",
            one.claim[:160], one.title[:80], one.says[:200], one.score,
        )
    rendered = WorldEvidence(sources=[
        WorldSource(f"{one.origin}, the sentence on this", one.title, one.location, one.says) for one in found
    ]).render()
    record_turn_world_evidence(rendered)
    record_turn_grounding(rendered)
    return await retry(text, (DISAGREES_WITH_ITS_SOURCES,))
