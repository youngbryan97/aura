"""Her turn when nobody asked: her recent conversation, a reading of what she noticed, and her cortex.

Two parts of her speak unprompted. Her remark on a fault she sees on the screen (core/perception/ambient_utterance.py)
gave her cortex the screen as a reading beside her conversation, and the sentence was hers. Her host monitor
(core/fictional/jarvis.py) wrote its sentences in code and sent them as hers, with a label in front: LIVE 15 November
2026, 16:11, "[Proactive/JARVIS] CPU usage is at 100% — something is running hot. Want me to check what's consuming
resources?" in her conversation (F06, canned answers out). Both now come here. What was noticed is a reading; what she
says of it, and whether it is worth saying, is her model's, and her Will decides after.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Brain.HerUnpromptedTurn")

#: How much of her conversation her model reads before the reading: the last few turns.
CONVERSATION_TURNS = 6

#: Her unprompted turn is worked out in its own generation: nothing upstream has thought about what she noticed. Named,
#: her model's private channel is opened and bounded by the decoder, and a generation that ends mid-thought shows
#: nothing. Unnamed, the channel was the template's default, unbounded, and all of it was handed over as her remark.
#: LIVE 14 November 2026, 14:56, her first remark from her cortex in the bubble: "The user is asking what's on their
#: screen. I have the OCR/screen capture data from the tool result. Let me parse what's visible"
#: (core/brain/llm/chat_format.py, thinking_enabled_for_generation).
_HER_MODE = "reflective"


#: Her look at the screen, as the call of hers that reads it (core/skills/computer_use.py, read_screen_text).
A_LOOK_AT_THE_SCREEN: tuple[str, dict[str, Any]] = ("computer_use", {"action": "read_screen_text"})


def her_conversation_now_with(heading: str, reading: str, *,
                              looked_with: tuple[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The conversation of the sitting going on now, its last few turns, and ``reading`` after it.

    An earlier sitting is not the conversation in front of her (core/conversation/sitting.py): it reaches her as
    memory, with when it was. LIVE 16 November 2026, 08:32 local, keeping Bryan company at the desk, her remark was
    "I'm sorry - I wasn't able to highlight it just now ... Could you try asking again in a moment?": the last turns
    she was given were "Can you highlight the back button?" from 00:40, eight hours and a sitting before, and she
    answered them, and pointed at an arrow for them.

    While somebody's words wait for her, the reading stands beside them as runtime evidence. Otherwise, given
    ``looked_with``, it is the call of hers that read it and what came back, in her template's form for a call of
    hers, as her calls during a person's turn reach her (core/conversation/surface_disposition.her_calls_this_turn).
    Without it the reading went in as the person's turn, under a heading: LIVE 17 November 2026, 09:50 and 12:08
    local, her remarks were "Here's what I can see on your screen:" and "What can I help you with? Are you looking
    to: ...", the answers to a person who had shown her their screen. Her template takes no turn of hers before
    somebody's ("No user query found in messages"), so with no words of anyone's in the sitting the person's turn is
    there and empty: they are with her and have said nothing.
    """
    import time

    from core.conversation.sitting import sitting_begins
    from core.together.remarks import said_since
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE, stamp_grounding

    said = said_since(0.0)
    now = said[sitting_begins([at for at, _role, _content in said], time.time()):][-CONVERSATION_TURNS:]
    messages = [{"role": role, "content": content} for _at, role, content in now]
    waiting = bool(messages) and messages[-1]["role"] == "user"
    if waiting or looked_with is None:
        messages.append(stamp_grounding({"role": RUNTIME_EVIDENCE_ROLE if waiting else "user",
                                         "content": f"## {heading}\n{reading}"}))
        return messages
    if not any(message["role"] == "user" for message in messages):
        messages.insert(0, {"role": "user", "content": ""})
    tool, arguments = looked_with
    messages.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": "her_look", "type": "function", "function": {"name": tool, "arguments": dict(arguments)}}]})
    messages.append({"role": "tool", "tool_call_id": "her_look", "name": tool, "content": reading})
    return messages


async def her_turn_on(heading: str, reading: str, *, origin: str, least_s: float) -> str:
    """Her next turn in her conversation, with ``reading`` under ``heading`` beside it; "" when nothing formed.

    ``origin`` names the generation in her log and to the lane. ``least_s`` is the least time it may take; a longer
    prompt gets what it needs on this machine.
    """
    messages = her_conversation_now_with(heading, reading)
    try:
        text = await on_her_cortex(messages, origin=origin, least_s=least_s)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
        record_degradation(
            origin, exc, severity="debug",
            action="stayed silent because her own voice was unavailable",
        )
        return ""
    return what_she_finished(str(text or ""))


def what_she_finished(text: str) -> str:
    """Her turn to its last finished sentence, its lines kept; "" when she finished none.

    It was cut at 400 characters. LIVE 14 November 2026, 15:03, her remark in the bubble ended "Both ran the same
    command:" and an opened code fence: her allowance ran out mid-sentence, and then the slice cut it again.
    """
    from core.utils.an_answer import was_cut_off

    said = text.strip()
    if said.count("```") % 2:
        said = said[: said.rfind("```")].rstrip()  # a code block she did not close is not said
    if not was_cut_off(said) and not said.endswith(":"):  # a colon at the end introduces what was not said
        return said
    for end in range(len(said) - 1, -1, -1):
        if said[end] in ".!?" and not was_cut_off(said[: end + 1]):
            return said[: end + 1]
    return ""


def _she_ended_it(client: Any, origin: str) -> bool:
    """Whether her model ended the generation that wrote her turn, not her time or a request to stop.

    Read off her worker's terminal event (core/brain/llm/mlx_worker_surface_quality.the_model_ended_its_turn), which
    her client keeps in the calling task's context, so it is asked inside the wait that ran the generation. Nobody asked
    for this turn, so nobody is owed what there was of it when it was stopped. LIVE 17 November 2026, 09:50, her
    remark was ended by its deadline in a table she was still writing, and its rows up to the last full stop went to
    the bubble and her conversation as what she had said.
    """
    receipt = getattr(client, "get_last_surface_control_receipt", dict)() or {}
    if receipt.get("ended_by_the_model") is False:
        logger.info("🫧 %s: her turn was ended by %s before she finished it; nothing is said", origin,
                    receipt.get("generation_stop_reason") or "the runtime")
        return False
    return True


async def on_her_cortex(messages: list[dict[str, Any]], *, origin: str, least_s: float) -> str:
    """Her cortex's next turn on these messages, admitted as her goal pass is; "" when her model is not free for it.

    Asked of her router as background work, a remark never reached her cortex: the router sends background work to its
    tertiary tier. LIVE 14 November 2026, 14:22 to 14:28, a traceback on the screen and her in the bubble: eleven times
    "Tier Lock: Background task requested 'primary'; using the governed tertiary tier", then "no endpoints matched"
    (her brainstem was out), and nothing formed. Her goal pass reaches her cortex through its client under the
    generation lane's lease (core/agency/her_goal_at_work.py). The lease is a background one, so a person's turn asks
    it to yield, and it is not asked for while a person is waiting for her model. The time it may take is what this
    prompt needs on this machine, and at least ``least_s``.
    """
    import asyncio

    from core.runtime.structured_input import A_CLOSED_QUESTIONS_FLOOR

    async def said(client: Any, seconds: float) -> str:
        text = await client.generate_text_async(
            "", messages=messages, system_prompt="", origin=origin, is_background=True,
            cognitive_mode=_HER_MODE, max_tokens=A_CLOSED_QUESTIONS_FLOOR, timeout=seconds) or ""
        return text if _she_ended_it(client, origin) else ""

    async def generate(client: Any, seconds: float) -> str:
        return await asyncio.wait_for(said(client, seconds), timeout=seconds)

    return str(await _on_her_lane(messages, origin=origin, least_s=least_s, generations=1,
                                  needs="generate_text_async", work=generate) or "")


#: Her turn at the desk while she keeps a person company: what she says may come with something she does on their
#: screen (core/skills/at_the_desk.py), under the standing grant that names this origin
#: (core/executive/standing_authority.py, DESK_COMPANY_ORIGIN).
DESK_ORIGIN = "her_desk_company"

#: Her desk turn's generations: one that answers, and one more after a call she makes, for what she says of it.
_DESK_GENERATIONS = 2


async def her_turn_at_the_desk(heading: str, reading: str, *, least_s: float,
                               looked_with: tuple[str, dict[str, Any]] | None = None) -> str:
    """Her next turn at the desk, with her desk tools offered as data; what she says, "" when nothing formed.

    LIVE 16 November 2026, 01:04, in the bubble: "I can't highlight or interact with any UI elements on your screen -
    I don't have tools for that." Composed with no tools in its prompt, her turn said she had none, and her pointing
    was chosen for her afterwards by matching her words against the screen. Here the call is hers, run through her
    capability engine as her goal pass's calls are, and what she says is her words after it.
    """
    import asyncio

    from core.runtime.structured_input import A_CLOSED_QUESTIONS_FLOOR
    from core.skills.at_the_desk import tool_definitions

    messages = her_conversation_now_with(heading, reading, looked_with=looked_with)

    async def acted(client: Any) -> dict[str, Any]:
        # The allowance her time was worked out for (_on_her_lane). LIVE 16 November 2026, 01:58, without it the loop
        # asked for the client's 4,096 tokens, and her turn was cancelled at the 50 s it had been given.
        result = dict(await client.think_and_act(
            "", system_prompt="", tools=tool_definitions(), max_turns=_DESK_GENERATIONS,
            context={"origin": DESK_ORIGIN, "requested_by": "aura"}, origin=DESK_ORIGIN,
            conversation=messages, work_in_background=True, native_tools_only=True, cognitive_mode=_HER_MODE,
            max_tokens=A_CLOSED_QUESTIONS_FLOOR, her_choice="by_meaning") or {})
        return result if _she_ended_it(client, DESK_ORIGIN) else {**result, "content": ""}

    async def act(client: Any, seconds: float) -> dict[str, Any]:
        return await asyncio.wait_for(acted(client), timeout=seconds)

    try:
        result = await _on_her_lane(messages, origin=DESK_ORIGIN, least_s=least_s, generations=_DESK_GENERATIONS,
                                    needs="think_and_act", work=act, to_someone_here=True)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
        record_degradation(DESK_ORIGIN, exc, severity="debug",
                           action="stayed silent because her own voice was unavailable")
        return ""
    for call in list((result or {}).get("tool_calls") or []):
        logger.info("🖐️ at the desk she called %s %s: %s", call.get("tool"), call.get("args") or {},
                    str(call.get("result") or "")[:200])
    return what_she_finished(str((result or {}).get("content") or ""))


async def _on_her_lane(messages: list[dict[str, Any]], *, origin: str, least_s: float, generations: int, needs: str,
                       work: Any, to_someone_here: bool = False) -> Any:
    """``work(client, seconds)`` on her cortex's client under one generation-lane lease; None when it is not hers now.

    It waits for the lane as long as it may take to speak. LIVE 15 November 2026, 18:29 to 18:32, a traceback on the
    screen and her in the bubble: six times "her model is held for something else". Her goal pass held the lane for
    44, 46, 47, 56 and 80 s a step, back to back, and a remark that waited the 5 s other background work waits came in
    only if a step happened to end in those 5 s. Waiting, it is woken when the step in hand releases the lane.
    """
    from core.brain.llm.mlx_client import time_a_prompt_needs
    from core.brain.llm_health_router import (
        _BACKGROUND_GENERATION_GATE_WAIT_S,
        a_person_is_waiting_for_her_model,
        acquire_external_generation_gate_lease,
        release_external_generation_gate_lease,
    )
    from core.runtime.service_access import resolve_inference_gate
    from core.runtime.structured_input import A_CLOSED_QUESTIONS_FLOOR
    from core.runtime.work_somebody_asked_for import under_way as work_is_under_way

    if a_person_is_waiting_for_her_model():
        logger.info("🫧 %s: her model is someone's now", origin)
        return None
    client = getattr(resolve_inference_gate(default=None), "_mlx_client", None)
    if client is None or not hasattr(client, needs):
        return None
    prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
    seconds = generations * max(least_s, time_a_prompt_needs(prompt_chars, A_CLOSED_QUESTIONS_FLOOR))
    waits = contextlib.nullcontext()
    # What they asked her for is theirs: a remark of hers comes between two steps of it, and stops none. LIVE 24 November
    # 2026 a remark worth 0.033 stopped a mend of a program at 1,222 tokens.
    if to_someone_here and not work_is_under_way():
        # What she has decided to say to the person in front of her comes before her own background work, as their
        # turn would: that work stands aside between its generations and the generation in hand is asked to stop,
        # and a pass of hers keeps what it had done. LIVE 16 November 2026, 08:31 local, keeping Bryan company, her
        # turn at the desk waited out its time behind her goal pass writing a history of quantum error correction,
        # "her model is held for something else", and nothing was said.
        from core.brain.llm_health_router_gate_owners import (
            _a_person_waits,
            _background_stands_aside,
        )

        waits = _a_person_waits()
        _background_stands_aside(f"{origin}: she has something to say to the person here")
    with waits:
        lease = await acquire_external_generation_gate_lease(
            owner=f"{origin}:her_own_turn", timeout_s=seconds, wait_s=max(seconds, _BACKGROUND_GENERATION_GATE_WAIT_S))
    if lease is None:
        logger.info("🫧 %s: her model is held for something else", origin)
        return None
    try:
        return await work(client, seconds)
    finally:
        release_external_generation_gate_lease(lease)


__all__ = ["A_LOOK_AT_THE_SCREEN", "CONVERSATION_TURNS", "DESK_ORIGIN", "her_conversation_now_with", "her_turn_at_the_desk", "her_turn_on",
           "on_her_cortex", "what_she_finished"]
