"""What a person did in a typed exchange, handed to her model of them.

Her model of a person learns what they did facing her last move and what their question was about
(core/consciousness/theory_of_mind.watch_the_turn). The conversational dynamics phase was its only caller, and a turn
typed into her chat answers without running a phase, so on 13 November 2026 her model of Bryan had learned nothing
from anything he typed. Each typed exchange now reaches it once, from the chat's durable post-reply step
(interface/routes/chat_preflight.py), classified by the same dynamics engine the phase uses.
"""
from __future__ import annotations

import logging

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.WatchingPeople")

#: Her last reply to each person in this process: the move their next turn faces.
_HER_LAST: dict[str, str] = {}


def watch_a_typed_exchange(who: str, said: str, she_replied: str) -> bool:
    """Hand her model of `who` what they did and asked in this exchange; False when nothing reached it."""
    who = " ".join(str(who or "").split())
    said = str(said or "").strip()
    if not who or not said:
        return False
    try:
        from core.consciousness.theory_of_mind import watch_the_turn
        from core.conversational.dynamics import get_dynamics_engine

        engine = get_dynamics_engine()
        her_last = _HER_LAST.get(who, "")
        memory = [{"role": "assistant", "content": her_last}] if her_last else []
        dynamics = engine.update(message=said, role="user", working_memory=memory)
        her_act = engine._classify_speech_act(her_last)[0] if her_last else ""
        their_act, topic = str(dynamics.last_speech_act or ""), str(dynamics.current_topic or "")
        watch_the_turn(who, their_act=their_act, her_act=her_act, topic=topic)
        logger.info("👁️ Her model of %s saw a %s about %s, facing her %s.", who, their_act or "turn",
                    topic or "nothing named", her_act or "opening")
        if she_replied:
            engine.update(message=she_replied, role="assistant",
                          working_memory=memory + [{"role": "user", "content": said}])
            _HER_LAST[who] = str(she_replied)
        return True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("watching_people", exc, severity="warning",
                           action="her model of the person did not see this typed exchange")
        return False
