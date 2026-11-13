"""When a request's words leave open which capability it needs, her model's reading of it settles it.

The tool hand-off (inference_gate_living_context.py) reads a request's words
against what each capability declares, and calls the one it plainly asks for.
A request in other words reached nothing, or the file reader, and was answered
in words (core/intent/what_her_model_reads_it_needs.py says how that went
live). Here, when the words do not plainly name one capability and the turn is
an order about something real, her model reads it beside the catalogue.

One capability read as the whole of it is called as the plainly named one is.
Consent follows what was asked, not the reading: a request that names a real
file or asks for a thing to exist has asked for that effect, so a capability
that writes such things may run, and nothing reaches further than that.
"""
from __future__ import annotations

import contextvars
import logging
from typing import Any

logger = logging.getLogger("Aura.InferenceGate")

__all__ = ["her_reading", "her_reading_chose"]

#: What her reading chose for which words, for the dispatch a moment later.
_CHOSE: contextvars.ContextVar[tuple[str, str]] = contextvars.ContextVar("aura_her_reading_chose", default=("", ""))


def _asking(client: Any) -> Any:
    """Asking the model client the turn already holds: inside a turn the router's lane is the turn's own,
    and a second request through it is refused at once (LIVE 2026-10-05, answered in 0.18 seconds)."""
    if client is None or not hasattr(client, "generate_text_async"):
        return None
    import json
    import re

    from core.intent.what_her_model_reads_it_needs import _Needs

    async def ask(prompt: str) -> Any:
        asked = f'{prompt}\n\nAnswer with one JSON object: {{"capabilities": ["name", ...]}}'
        raw = await client.generate_text_async(
            asked, messages=[{"role": "user", "content": asked}], max_tokens=160, temperature=0.0,
            schema=_Needs.model_json_schema(), output_shape="json_object", origin="her_reading_of_a_request",
            internal_inference=True, foreground_request=True,
        )
        found = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.S), re.S)
        return _Needs.model_validate(json.loads(found.group(0))) if found else None

    return ask


async def her_reading(text: str, required: list[str], ceiling: str, scopes: Any, *, client: Any = None) -> tuple[list[str], str, Any]:
    """The capabilities ``text`` needs, its words' or her model's reading of it, and the ceiling it asked for."""
    from core.container import ServiceContainer
    from core.intent.capability_selection import the_one_asked_for
    from core.intent.what_her_model_reads_it_needs import (
        capabilities_her_model_reads,
        worth_reading,
    )

    if required and the_one_asked_for(text, {required[0]: {}}) is not None:
        return required, ceiling, scopes
    # A capability is for something she was asked to do. Live, 13 October,
    # "There is a wren nesting in the old boot by my back door" spent ten
    # seconds of her large model on which capability it needed.
    from core.together.taking_it_up import asked_to_do_something

    if not await asked_to_do_something(text):
        return required, ceiling, scopes
    import asyncio

    from core.brain.requests_by_meaning import meant_capability, meant_shortlist

    engine = ServiceContainer.get("capability_engine", default=None)
    skills = getattr(engine, "skills", None) or {}
    # By meaning first: her embedding of the request beside each capability's
    # meanings (core/intent/capability_by_meaning.py), measured, not guessed.
    meant = await asyncio.to_thread(meant_capability, text, skills)
    # When meaning narrows it to a few and no one is clearly ahead, her model chooses among those few.
    shortlist = [] if meant else await asyncio.to_thread(meant_shortlist, text, skills)
    # The words' own gate guards only her large model's reading, which costs a pass over the catalogue. A reading by
    # meaning costs an embedding and decides by its own margins. LIVE 13 November 2026: "Could you send me a desktop
    # notification that says the tea is ready?" never reached it, because no word of it is in that gate's verb list,
    # and she answered that she could not send notifications, with notify_user ready.
    if not meant and not shortlist and not worth_reading(text):
        return required, ceiling, scopes
    among = {name: skills[name] for name in shortlist if name in skills} or skills
    read = [meant] if meant else await capabilities_her_model_reads(text, among, ask=_asking(client))
    # A reading that lands on the reply in words asks for no tool: the turn is
    # answered in words (LIVE 2026-10-05, native_chat called with no message).
    from core.intent.what_her_model_reads_it_needs import answers_in_words

    read = [name for name in read if not answers_in_words(skills.get(name))]
    if not read:
        return required, ceiling, scopes
    logger.info("🔧 Tool handoff: the words left it open; %s needs %s.", "its meaning" if meant else "her reading", ", ".join(read))
    first = (getattr(engine, "skills", None) or {}).get(read[0])
    # The capability that does most of it is called with the request when it
    # needs nothing else to begin (no required arguments): it reads what it
    # needs from the person's words, as the plainly named one is called.
    if len(read) > 1 and _needs_only_the_request(first):
        read = read[:1]
    if len(read) > 1:
        return [*read, *(r for r in required if r not in read)], ceiling, scopes
    _CHOSE.set((text, read[0]))
    from core.intent.artifact_request import asks_for_an_artifact
    from core.intent.capability_selection import points_at_something_real
    from core.phases.response_contract import (
        _REQUESTABLE_SCOPES,
        _REQUESTED_ARTIFACT_CEILING,
        _REQUESTED_ARTIFACT_SCOPES,
    )
    from core.skills.catalog_policy import SKILL_EFFECT_SCOPES

    scope = SKILL_EFFECT_SCOPES.get(read[0])
    if (meant or read[0] in shortlist) and scope in _REQUESTABLE_SCOPES and scope not in scopes:
        # Asking for what a capability does is asking for that effect (requested_effect_ceiling's own principle), so
        # the capability meant sets how far the turn reaches, among the scopes a person's request may open; the
        # dispatch still judges the act.
        ceiling, scopes = scope, frozenset(scopes) | {scope}
    elif scope in _REQUESTED_ARTIFACT_SCOPES and (points_at_something_real(text) or asks_for_an_artifact(text)):
        ceiling, scopes = _REQUESTED_ARTIFACT_CEILING, frozenset(_REQUESTED_ARTIFACT_SCOPES)
    return [read[0]], ceiling, scopes


def _needs_only_the_request(meta: Any) -> bool:
    """Whether a capability can begin from the request alone: its input declares nothing required."""
    model = getattr(meta, "input_model", None)
    fields = getattr(model, "model_fields", None)
    return isinstance(fields, dict) and not any(field.is_required() for field in fields.values())


def her_reading_chose(text: str, offered: Any) -> str | None:
    """The one capability her reading chose for these words, when it is the one offered."""
    said, name = _CHOSE.get()
    return name if said == text and list(offered or ()) == [name] else None
