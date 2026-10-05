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
    if not worth_reading(text):
        return required, ceiling, scopes
    engine = ServiceContainer.get("capability_engine", default=None)
    read = await capabilities_her_model_reads(text, getattr(engine, "skills", None) or {}, ask=_asking(client))
    if not read:
        return required, ceiling, scopes
    logger.info("🔧 Tool handoff: the words left it open; her reading needs %s.", ", ".join(read))
    if len(read) > 1:
        return [*read, *(r for r in required if r not in read)], ceiling, scopes
    _CHOSE.set((text, read[0]))
    from core.intent.artifact_request import asks_for_an_artifact
    from core.intent.capability_selection import points_at_something_real
    from core.phases.response_contract import (
        _REQUESTED_ARTIFACT_CEILING,
        _REQUESTED_ARTIFACT_SCOPES,
    )
    from core.skills.catalog_policy import SKILL_EFFECT_SCOPES

    if SKILL_EFFECT_SCOPES.get(read[0]) in _REQUESTED_ARTIFACT_SCOPES and (points_at_something_real(text) or asks_for_an_artifact(text)):
        ceiling, scopes = _REQUESTED_ARTIFACT_CEILING, frozenset(_REQUESTED_ARTIFACT_SCOPES)
    return [read[0]], ceiling, scopes


def her_reading_chose(text: str, offered: Any) -> str | None:
    """The one capability her reading chose for these words, when it is the one offered."""
    said, name = _CHOSE.get()
    return name if said == text and list(offered or ()) == [name] else None
