"""Her reply in a conversation on a page: her model, given the conversation as it went and the time it keeps.

The other side's lines are the person's turns and hers are her own; what she was asked to do there and what the page
says are shown to her as the grounding evidence her chat gives (``core.utils.injected_blocks``), data and not a
sentence addressed to her. The turn carries the time a conversation keeps (``answer_within_s``,
core/brain/llm/an_answer_in_the_time_given.py), so her answer is fitted to it rather than the wait stretched to her
answer. How she plays a role a game asks of her is hers: nothing here tells her.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from core.perception.a_conversation_on_a_page import Line

logger = logging.getLogger(__name__)

#: How long a line in a live conversation may take, from the other side's last line: about what a person waits.
A_CONVERSATION_S = 15.0


def messages_for(lines: list[Line], *, request: str, page_text: list[str]) -> list[dict[str, str]]:
    """The turns her model is given: the evidence, then the conversation as the two sides wrote it."""
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

    evidence = ("[GROUNDING EVIDENCE FOR THIS TURN]\n"
                + json.dumps({"asked_of_her": request, "the_page_says": page_text[:20]}, ensure_ascii=False)
                + "\n[END GROUNDING EVIDENCE FOR THIS TURN]")
    turns = [{"role": "user" if line.who == "them" else "assistant", "content": line.text} for line in lines]
    return [{"role": RUNTIME_EVIDENCE_ROLE, "content": evidence}, *turns]


async def her_reply(lines: list[Line], *, request: str, page_text: list[str], within_s: float = A_CONVERSATION_S,
                    router: Any = None) -> str:
    """Her next line in the conversation, or "" when her model gave none in the time."""
    if router is None:
        from core.container import ServiceContainer

        router = ServiceContainer.get("llm_router", default=None)
    if router is None:
        return ""
    result = await router.generate_with_metadata(
        "", None, within_s * 2, prefer_tier="primary", messages=messages_for(lines, request=request, page_text=page_text),
        origin="conversation_on_a_page", purpose="conversation_on_a_page", foreground_request=True,
        # As the pursuit's own calls are: the turn that sent her to the page is waiting on this, which the runtime
        # honours only while that turn is open (llm_health_router._the_turn_is_waiting_on_this).
        serves_current_turn=True, answer_within_s=within_s)
    text = str((result or {}).get("text") or "").strip()
    logger.info("her line in a conversation on a page (%d chars, ok=%s)", len(text), (result or {}).get("ok"))
    return text
