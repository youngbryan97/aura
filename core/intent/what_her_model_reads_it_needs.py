"""Which capabilities a request needs, as her own model reads it, when its words alone do not settle it.

The words a capability declares are a floor: "fix this game" names the act, and
the selection reads it (capability_selection.py). People do not ask in the
words anyone declared. LIVE 2026-10-05: "pong at ~/aura-demos/pong/pong.html
doesn't work right. can you sort it out and then beat the computer at it?",
"my pong game (...) is messed up - get it working and win a round" and three
more ways of asking for the same repair reached the file reader alone, so the
turn would read the code and talk about it.

So when the words leave it open, and the turn is an order about something
real (a file or address it names, a thing it asks to exist, an act it names),
her model reads the request beside the catalogue of what she can do and says
which of them doing it needs. It is a reading, not an authority: what it names
must be in the catalogue, and the dispatch that follows applies the same
governance as any other.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("Intent.HerReading")

__all__ = ["capabilities_her_model_reads", "worth_reading"]

#: The most capabilities one reading names.
MOST = 3


class _Needs(BaseModel):
    capabilities: list[str] = Field(default_factory=list, description="capability names, the one that does most of it first")


def worth_reading(text: str) -> bool:
    """Whether a turn is an order about something real: worth her model's reading when the words leave it open."""
    from core.intent.artifact_request import asks_for_an_artifact
    from core.intent.capability_selection import points_at_something_real
    from core.intent.declared_capability import verb_class_of

    if points_at_something_real(text) or asks_for_an_artifact(text):
        return True
    acts = verb_class_of("fix") | verb_class_of("rebuild") | verb_class_of("run") | verb_class_of("open")
    return bool(acts & set(re.findall(r"[a-z]+", str(text or "").lower())))


def answers_in_words(meta: Any) -> bool:
    """Whether a skill is the reply in words itself (native_chat), which a reading never hands off to."""
    return bool(getattr(getattr(meta, "skill_class", None), "answers_in_words", False))


def _first_sentence(description: str) -> str:
    said = " ".join(str(description or "").split())
    return re.split(r"(?<=[.:;])\s", said, maxsplit=1)[0][:180]


async def capabilities_her_model_reads(
    text: str,
    skills: Mapping[str, Any],
    *,
    ask: Callable[[str], Awaitable[Any]] | None = None,
) -> list[str]:
    """The capabilities doing what ``text`` asks needs, as her model reads it beside the catalogue; at most three."""
    catalogue = {
        name: _first_sentence(getattr(meta, "description", "") or "")
        for name, meta in skills.items()
        if getattr(meta, "enabled", True) and getattr(meta, "description", "") and not answers_in_words(meta)
    }
    if not catalogue:
        return []
    prompt = (
        f"A person said: {text}\n\n"
        "Which of these capabilities does doing what they asked need? Name the one that does most of it first, "
        f"at most {MOST}, and none if what they want is only words in a reply.\n"
        + "\n".join(f"- {name}: {said}" for name, said in sorted(catalogue.items()))
    )
    if ask is None:
        from core.brain.llm.structured_llm import StructuredLLM

        reader = StructuredLLM(_Needs, max_retries=1)

        async def ask(asked: str) -> Any:
            return await reader.generate(asked, is_background=False, deadline_s=120.0)

    try:
        read = await ask(prompt)
    except (RuntimeError, TimeoutError, ValueError, TypeError, OSError) as why:
        logger.info("her model could not read what the request needs: %s", why)
        return []
    named = read.capabilities if isinstance(read, _Needs) else list(getattr(read, "capabilities", []) or [])
    found = [n.strip() for n in named if str(n).strip() in catalogue][:MOST]
    logger.info("her model reads the request as needing: %s", ", ".join(found) or "nothing but a reply")
    return found
