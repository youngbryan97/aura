"""What of an assembled context a request actually implicates.

The runtime has two prompt builders. The gate has a budget table with a
per-role limit for every profile and trims to it: a user-facing system
message is allowed 5,200 characters. The other builder is the one every
desktop conversation goes through, and it has no budget at all. Measured
live on 2026-08-28, a turn whose question was 213 characters carried a
96,430-character system message — 28,147 tokens of prefill — into the same
model, eighteen times what the table allows.

The asymmetry is the defect, and a shorter prompt is not by itself the fix.
A section with nothing to do with the question is not free information
sitting harmlessly in the window. It competes for the same attention as the
question, and the sections are numerous enough that the question is
outnumbered.

So what survives is decided by the request rather than by what happened to
be assembled. Every section is scored on the words it shares with what was
asked, weighted by how much each word narrows the field: a word occurring
in every section separates nothing and is worth nothing, a word occurring
in one section and in the request is the whole signal. The weighting is
computed from the sections in front of it, so there is no authored table of
important words and nothing to keep in step with the prompt as it changes.

Sections a caller names as always-kept are kept whatever they score. That
list is the caller's existing contract rather than a second one invented
here: grounding a turn cannot depend on the turn happening to mention it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "CRITICAL_FOREGROUND_HEADERS",
    "FOREGROUND_SECTION_VOLATILITY",
    "Section",
    "budget_for_answer",
    "fit_to_budget",
    "section_volatility",
    "sections_of",
]

#: The sections a foreground turn is grounded by, kept whatever else is
#: trimmed, and how volatile each one is. Both prompt builders read these,
#: because two copies of a list like this drift apart quietly and the
#: symptom is a turn that loses its grounding on one lane only.
CRITICAL_FOREGROUND_HEADERS: tuple[str, ...] = (
        "## PRESENT MOMENT",
        "## YOUR OWN INSTRUMENTS",
        "## WHAT YOU ACTUALLY JUST DID",
        "[LIVE MIND CONTEXT]",
        "## DERIVED RUNTIME SIGNALS",
        "[LIVE SPEECH GROUNDING]",
        "## LIVE TONE",
        "## UNITY",
        "## FUNCTIONAL STATE SIGNALS",
        "## GOALS",
        "## HELD POSITION",
        "## SOMATIC STATE",
        "## STATE",
        "## CONTINUITY SUMMARY",
        "## TEMPORAL OBLIGATIONS",
        "## CONVERSATIONAL INTENT",
        "## IMAGINATION WORKSPACE",
        "## BICAMERAL ADVISORY",
        "## LIVE DESKTOP RESPONSE CONTRACT",
        "## USER-FACING CONVERSATION RELIABILITY CONTRACT",
    )

FOREGROUND_SECTION_VOLATILITY: tuple[tuple[str, int], ...] = (
    ("## USER-FACING CONVERSATION RELIABILITY CONTRACT", 0),
    ("## LIVE DESKTOP RESPONSE CONTRACT", 0),
    ("## CONTINUITY SUMMARY", 1),
    ("## GOALS", 1),
    ("## HELD POSITION", 1),
    ("## TEMPORAL OBLIGATIONS", 1),
    ("## CONVERSATIONAL INTENT", 1),
    ("[LIVE MIND CONTEXT]", 1),
    ("## IMAGINATION WORKSPACE", 1),
    ("## BICAMERAL ADVISORY", 1),
    ("[LIVE SPEECH GROUNDING]", 2),
    ("## DERIVED RUNTIME SIGNALS", 2),
    ("## FUNCTIONAL STATE SIGNALS", 2),
    ("## PRESENT MOMENT", 2),
    ("## YOUR OWN INSTRUMENTS", 2),
    ("## WHAT YOU ACTUALLY JUST DID", 2),
    ("## SOMATIC STATE", 2),
    ("## STATE", 2),
    ("## LIVE TONE", 2),
    ("## UNITY", 2),
)



#: A section header at the start of its line, in either of the two shapes the
#: assembled prompt uses. Anchored at a line start because a header written
#: inside a sentence — in recalled memory, a fetched page, a tool result — is
#: text somebody wrote, not a section of the prompt.
_HEADER = re.compile(r"^(##[^\n]*|\[[A-Z][^\n\]]*\])$", re.MULTILINE)

#: Word tokens. Deliberately crude: what makes a word worth anything here is
#: measured below, not decided by a list.
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Section:
    """One headed part of an assembled prompt, and where it started."""

    header: str
    text: str
    order: int

    def __len__(self) -> int:
        return len(self.text)


def sections_of(prompt: str) -> list[Section]:
    """The prompt split at its headers, the head kept as its own section.

    The text before the first header is the identity the whole prompt hangs
    off. It has no header, and it is not optional, so it comes back as a
    section with an empty one.
    """

    body = str(prompt or "")
    if not body:
        return []
    marks = [(hit.start(), hit.group(0)) for hit in _HEADER.finditer(body)]
    if not marks:
        return [Section(header="", text=body.strip(), order=0)]

    found: list[Section] = []
    head = body[: marks[0][0]].strip()
    if head:
        found.append(Section(header="", text=head, order=0))
    for index, (start, header) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else len(body)
        text = body[start:end].strip()
        if text:
            found.append(Section(header=header.strip(), text=text, order=len(found)))
    return found


def _weights(sections: Sequence[Section]) -> dict[str, float]:
    """How much each word narrows the field, measured on these sections.

    A word in every section tells a reader nothing about which section to
    keep and is worth zero. A word in one is worth the most. Nothing is
    authored: the numbers come from the prompt being trimmed.
    """

    if not sections:
        return {}
    seen: dict[str, int] = {}
    for section in sections:
        for word in set(_WORD.findall(section.text.lower())):
            seen[word] = seen.get(word, 0) + 1
    total = len(sections)
    return {
        word: math.log(total / count)
        for word, count in seen.items()
        if count < total
    }


def _asked_for(request: str, weights: dict[str, float], section: Section) -> float:
    """What this section is worth to this request.

    The score is the weight of the request's own words that appear in it,
    divided by the section's length, because a long section shares words with
    everything and would otherwise win by size alone.
    """

    wanted = set(_WORD.findall(str(request or "").lower()))
    if not wanted:
        return 0.0
    holds = set(_WORD.findall(section.text.lower()))
    earned = sum(weights.get(word, 0.0) for word in wanted & holds)
    return earned / math.sqrt(max(1, len(section.text)))


def fit_to_budget(
    prompt: str,
    request: str,
    *,
    budget: int,
    always: Iterable[str] = (),
    volatility: Callable[[str], int] | None = None,
) -> str:
    """The part of ``prompt`` this ``request`` implicates, inside ``budget``.

    ``always`` names headers kept whatever they score — grounding a turn
    cannot wait for the turn to mention it. ``volatility`` orders what
    survives: a cached prompt is reuse of a byte-identical prefix, so
    anything changing every turn is emitted last and the stable identity in
    front of it stays reusable.

    A prompt already inside the budget comes back untouched. Trimming one
    that fits would be spending the risk for nothing.
    """

    body = str(prompt or "")
    budget = int(budget)
    if budget <= 0 or len(body) <= budget:
        return body

    parts = sections_of(body)
    if not parts:
        return body[:budget]

    kept_headers = {str(header).strip() for header in always}
    weights = _weights(parts)

    def priority(section: Section) -> tuple[int, float]:
        # The head and the named sections come first, in that order, and the
        # rest compete on what the request asked for.
        if not section.header:
            return (2, 0.0)
        if any(section.header.startswith(header) for header in kept_headers):
            return (1, _asked_for(request, weights, section))
        return (0, _asked_for(request, weights, section))

    ranked = sorted(parts, key=priority, reverse=True)

    taken: list[Section] = []
    spent = 0
    for section in ranked:
        if spent >= budget:
            break
        room = budget - spent
        if len(section.text) > room:
            # A section too big for what is left is cut rather than dropped
            # only while the cut still says something. A header and an
            # ellipsis is not a section.
            if room <= len(section.header) + 2:
                continue
            section = Section(
                header=section.header,
                text=section.text[: room - 1].rstrip() + "…",
                order=section.order,
            )
        taken.append(section)
        spent += len(section.text) + 2

    if volatility is not None:
        taken.sort(key=lambda section: (volatility(section.text), section.order))
    else:
        taken.sort(key=lambda section: section.order)
    return "\n\n".join(section.text for section in taken).strip()


def budget_for_answer(max_tokens: int) -> int:
    """How long a prompt may be, given what the answer it serves will cost.

    Reading is instrumental and writing is the deliverable, so a turn that
    spends longer reading its context than saying its answer has its effort
    the wrong way round. Both sides are measured rather than assumed: the
    reserve times decoding and reading from live turns and keeps the rates
    across restarts.

    This bites where the defect is. The lanes carrying the largest assembled
    prompts are the reflex ones whose answers are fifty tokens, so on the
    rates measured on 2026-08-28 a 96,430-character prompt costing ~46s of
    reading served ~5s of writing. A conversation turn allowed a thousand
    tokens is buying two minutes of writing and is not trimmed at all: a long
    answer earns a long prompt, and nothing here needs a number chosen by
    hand to say so.
    """

    try:
        wanted = int(max_tokens)
    except (TypeError, ValueError):
        return 0
    if wanted <= 0:
        return 0
    from core.brain.llm.thinking_reserve import chars_readable_in, seconds_to_decode

    writing = seconds_to_decode(wanted)
    if writing <= 0:
        # Nothing timed, so nothing to be proportionate to. A budget invented
        # here would be the authored number this exists to avoid.
        return 0
    return int(chars_readable_in(writing))


def section_volatility(section: str) -> int:
    """How much of this section changes from one turn to the next.

    Emission order decides whether a cached prompt can be reused at all: a
    cached entry is the KV for a byte-identical prefix, so a turn-volatile
    byte placed early throws away the reuse of everything after it. Unranked
    sections sort as the most volatile, which is the safe assumption for
    something nobody has looked at.
    """

    text = str(section or "")
    for header, rank in FOREGROUND_SECTION_VOLATILITY:
        if text.startswith(header):
            return rank
    return max(rank for _header, rank in FOREGROUND_SECTION_VOLATILITY) + 1
