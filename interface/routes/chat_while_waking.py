"""A typed turn while her runtime is still starting: her cortex answers, in her words, if it is running.

Bryan, 11 November 2026: "She should say she's still waking up (in her words) if she still is. I can wait a few
seconds." Her voice mode does this on her live context (core/voice/duplex/live_context.py). Her chat route did not:
LIVE 12 November, the first typed turn a minute after her app opened was not admitted to her chat lane while it warmed,
and the route served a line it wrote in her first person, "the live answer lane could not finish preparing before a
reasoning turn began. I recorded the readiness failure separately from Aura's answer quality."

Here the turn goes to her cortex's running client, waited for within the turn's own time, on her recent conversation
with the turn's evidence, which says she is still starting. Nothing tells her what to say. None when she is not
waking, when no cortex comes up in time, or when nothing formed; the route then says what it said before.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.Chat.WhileWaking")


async def her_reply_while_waking(words: str, *, session_id: str, timeout_s: float) -> str | None:
    """Her answer to ``words`` on her running cortex while she is still starting, or None."""
    from core.conversation.delivered_history import VISIBLE_CONVERSATION_EXCHANGES
    from core.voice.duplex.live_context import (
        _ANSWER_OPENS,
        _generate_on_her_cortex,
        rendered,
        still_waking,
        the_turns_evidence,
        what_she_reads_first,
    )

    said = " ".join(str(words or "").split())
    if not said or not still_waking():
        return None
    try:
        from interface.routes import chat_memory_state

        exchanges: Any = await chat_memory_state._recent_completed_conversation_exchanges(
            current_user_message=said, session_id=session_id, limit=VISIBLE_CONVERSATION_EXCHANGES,
            allow_cross_session=True)
    except (ImportError, RuntimeError, ValueError, OSError, TypeError) as exc:
        logger.debug("her recent conversation could not be read while she wakes: %s", exc)
        exchanges = []
    prompt = (rendered(what_she_reads_first(exchanges or [])) + "<|im_start|>user\n" + the_turns_evidence() + said
              + _ANSWER_OPENS)
    reply = await _generate_on_her_cortex(prompt, scope=f"chat_waking:{str(session_id or 'chat')[:40]}", words=said,
                                          reply_stream=None, timeout_s=timeout_s)
    reply = str(reply or "").strip()
    if not reply:
        return None
    logger.info("🌅 a typed turn answered on her cortex while she is still starting (%d characters)", len(reply))
    try:
        from interface.routes import chat_preflight

        await chat_preflight._log_exchange(said, reply, session_id=session_id)
    except (ImportError, RuntimeError, ValueError, OSError, TypeError) as exc:
        logger.debug("the exchange answered while waking was not recorded: %s", exc)
    return reply
