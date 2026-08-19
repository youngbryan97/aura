"""A question that needs a fact about the person, and whether she has it.

LIVE 2026-08-18: "what's the population of the town I grew up in?"

    I couldn't get to an answer I'd stand behind on that one, and I won't send
    you a thinner one and pass it off as the real thing.

The guard behind that was working perfectly. The draft had been rejected for
`fabricated_shared_history` — the model had supplied a home town it was never
told — and the retries then ran out, so correct behaviour produced the worst
possible answer.

The honest reply is one sentence: I do not know where you grew up. That is not
a failure of the turn, it is the answer. What was missing is the reading that
makes it sayable — whether the fact the question depends on is one she holds.

Nothing here enumerates biographical facts. The question names the fact
itself ("the town I grew up in", "my sister", "where I work"), so the subject
comes from the sentence and the answer comes from the store.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "PERSON_FACT_HEADER",
    "needed_person_fact",
    "person_fact_block",
]

PERSON_FACT_HEADER = "## WHETHER YOU ACTUALLY KNOW THIS ABOUT THEM"

#: "the town I grew up in", "the company I work for", "the school I went to"
#: The noun immediately before the relative pronoun is the fact, not the
#: whole phrase around it: "the population of the town I grew up in" needs
#: the TOWN, and the population is what is being asked about it.
_RELATIVE_FACT_RE = re.compile(
    r"\b(?:the|that|which|my)\s+(?P<thing>[\w-]+(?:\s+[\w-]+)?)\s+"
    r"(?:i|we)\s+(?P<verb>[\w']+(?:\s+[\w']+){0,2}?)"
    r"(?:\s+(?P<tail>in|at|for|to|from|with|on))?\s*(?=[?.,]|$)",
    re.IGNORECASE,
)

#: "my home town", "my sister's name", "my birthday"
_POSSESSIVE_FACT_RE = re.compile(
    r"\bmy\s+(?P<thing>(?:[\w-]+\s+){0,2}[\w-]+)",
    re.IGNORECASE,
)

#: Possessions that are about the conversation or her, not biography.
_NOT_BIOGRAPHY = frozenset(
    """
    question questions point message reply answer request last first previous
    screen clipboard desktop file files folder computer machine laptop
    understanding meaning words word sentence example examples
    """.split()
)


def needed_person_fact(prompt: Any) -> str:
    """The biographical fact this question depends on, or ""."""
    text = " ".join(str(prompt or "").split())
    if not text:
        return ""
    matches = list(_RELATIVE_FACT_RE.finditer(text))
    match = matches[-1] if matches else None
    if match:
        thing = match.group("thing").strip().lower()
        verb = match.group("verb").strip().lower()
        if thing and thing.split()[-1] not in _NOT_BIOGRAPHY:
            tail = (match.group("tail") or "").strip()
            phrase = f"the {thing} you {verb}"
            return f"{phrase} {tail}".strip() if tail else phrase.strip()
    match = _POSSESSIVE_FACT_RE.search(text)
    if match:
        thing = match.group("thing").strip().lower()
        if thing and thing.split()[-1] not in _NOT_BIOGRAPHY:
            return f"your {thing}"
    return ""


def _known_about_person() -> list[str]:
    """Everything the runtime holds about the person, as plain lines."""
    lines: list[str] = []
    try:
        from core.container import ServiceContainer

        graph = ServiceContainer.get("belief_graph", default=None) or (
            ServiceContainer.get("world_model", default=None)
        )
        if graph is not None and hasattr(graph, "get_beliefs"):
            for key, value in list(dict(graph.get_beliefs() or {}).items())[:40]:
                lines.append(f"{key}: {value}")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return lines


def person_fact_block(prompt: Any) -> str:
    """Whether the fact this question needs is one she holds."""
    needed = needed_person_fact(prompt)
    if not needed:
        return ""
    known = _known_about_person()
    # Match on the content words of the phrase, not its last token: the last
    # token of "the town you grew up in" is "in", which appears in almost
    # every line ever written.
    ignore = {"the", "you", "your", "a", "an", "in", "at", "for", "to", "from",
              "with", "on", "up", "went", "grew", "work", "works"}
    subjects = [
        word for word in re.findall(r"[a-z']+", needed.lower()) if word not in ignore
    ]
    matching = [
        line for line in known if any(word in line.lower() for word in subjects)
    ]
    if matching:
        return "\n".join(
            [f"You do hold something about {needed}:", *(f"- {line}" for line in matching[:6])]
        )
    return (
        f"You do NOT know {needed}. Nothing in what you hold about them says it. "
        f"Say that plainly — one sentence, and offer to use it if they tell you. "
        f"Do not supply a plausible one, and do not refuse the whole turn over it."
    )
