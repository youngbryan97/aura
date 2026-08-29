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

import asyncio
import atexit
import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.runtime.errors import record_degradation

__all__ = [
    "CRITICAL_FOREGROUND_HEADERS",
    "FOREGROUND_SECTION_VOLATILITY",
    "Section",
    "budget_for_answer",
    "fit_to_budget",
    "observe_sections",
    "save_volatility",
    "section_volatility",
    "stable_prefix_first",
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


def prefill_ceiling(room_taken_by_the_rest: int = 0) -> int:
    """The most a system message may be before the worker starts cutting it.

    Not a budget invented here. The client bounds any prompt over its prefill
    ceiling by keeping the head and the tail and dropping the middle, and the
    comment above that ceiling says no legitimate turn is near it. Twenty-seven
    turns in this log reached it, each one recorded as a fault and each one
    feeding the runtime's own affect: "frustration=0.26 depletion=0.10
    state=friction".

    What the middle of an assembled prompt holds is the mind context. So the
    ceiling is worth meeting deliberately, with the request deciding what
    survives, rather than by amputation at a byte offset.
    """

    try:
        from core.brain.llm.mlx_client import _PREFILL_CEILING_CHARS
    except (ImportError, AttributeError):
        return 0
    return max(0, int(_PREFILL_CEILING_CHARS) - max(0, int(room_taken_by_the_rest)))


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


# --------------------------------------------------------------- what changes

#: How often each section's text has differed from the last turn that carried
#: it: header -> (times seen again, times it had changed). Volatility is a fact
#: about a running system, and the table above is a prior over twenty headers
#: an assembled prompt of forty does not fit inside.
_CHANGED: dict[str, list[int]] = {}
_LAST_SEEN: dict[str, str] = {}
_VOLATILITY_STORE = "section_volatility.json"

#: Below this a change-rate is one or two turns of coincidence. The authored
#: prior answers until there is enough to beat it.
_ENOUGH_TURNS_TO_RANK = 6

#: Turns between writes. Every turn would be a write on the answer path;
#: never would be a measurement that does not survive a restart.
_WRITE_EVERY = 20
_since_write = 0


_taken_back = False


def _take_back_what_earlier_runs_measured() -> None:
    """Read the store once, the first time anything asks about a section.

    ``load_volatility`` was written to do this and nothing called it, so every
    boot began with an empty table and answered from the authored prior — a
    measurement that could not survive a restart, which is the same as one
    that was never taken. LIVE 2026-08-29: the store did not exist at all,
    because a session shorter than twenty observed prompts never reached the
    write and the restarts here are far more frequent than that.

    Here rather than at boot, because this module is imported by the thing
    that assembles prompts and that is the only caller who needs the table.
    """

    global _taken_back
    if _taken_back:
        return
    _taken_back = True
    try:
        load_volatility()
    except (OSError, ValueError, RuntimeError):
        pass


def observe_sections(prompt: str) -> None:
    """Note which sections of this prompt differ from the last one carrying them.

    Called with each assembled prompt. What it learns is which parts of the
    context are the same turn after turn, which is the only thing that decides
    whether a cached prefix is worth anything: a cache entry is the KV for a
    byte-identical prefix, and measured live on 2026-08-28 the runtime was
    reusing 558 tokens of 27,298 — two per cent — and prefilling the rest at a
    cost of about forty-six seconds a turn.

    Nobody has to keep a list in step with the assembler for this to work, and
    a section nothing has been learned about yet falls back to the prior.
    """

    _take_back_what_earlier_runs_measured()
    for section in sections_of(prompt):
        header = section.header or "<identity>"
            # A digest rather than the text, and a stable one: the built-in
        # hash is salted per process, which is invisible while nothing
        # crosses a restart and wrong the moment something does.
        digest = hashlib.blake2b(
            section.text.encode("utf-8", "replace"), digest_size=8
        ).hexdigest()
        seen_before = _LAST_SEEN.get(header)
        if seen_before is not None:
            counts = _CHANGED.setdefault(header, [0, 0])
            counts[0] += 1
            counts[1] += int(seen_before != digest)
        _LAST_SEEN[header] = digest
    _written_down()



def _keep_what_this_session_measured() -> None:
    """Write on the way out, whatever the cadence has reached.

    Twenty turns between writes keeps this off the answer path, and a session
    shorter than twenty turns then measures for nothing. Restarts here are
    minutes apart, so that was every session.
    """

    try:
        save_volatility()
    except (OSError, ValueError, RuntimeError):
        pass


atexit.register(_keep_what_this_session_measured)


def _written_down() -> None:
    """Persist every so often, off the event loop when there is one.

    An fsync on the loop froze the live runtime for twenty minutes once, and
    this runs on the path that assembles every turn's prompt. Where a loop is
    running the write is handed to it as a task; where there is not one — a
    test, a tool — it is written straight out.
    """

    global _since_write
    _since_write += 1
    if _since_write < _WRITE_EVERY:
        return
    _since_write = 0
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        save_volatility()
        return
    loop.create_task(save_volatility_async())


def _volatility_payload() -> tuple[Path, str] | None:
    try:
        from core.runtime.state_ownership import state_root

        return Path(state_root()) / _VOLATILITY_STORE, json.dumps({"changed": _CHANGED})
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return None


async def save_volatility_async() -> bool:
    """:func:`save_volatility` for the lane that has an event loop to protect."""

    made = _volatility_payload()
    if made is None:
        return False
    target, payload = made
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "brain.context_budget", domain="state_mutation"
        ):
            await get_file_write_gateway().write_text_async(
                target, payload, source="brain.context_budget"
            )
        return True
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "context_budget",
            exc,
            action="kept this run's section volatility in memory only",
        )
        return False


def measured_volatility(header: str) -> float | None:
    """How often this section has changed, or None while that is not known."""

    counts = _CHANGED.get(str(header or "").strip())
    if not counts or counts[0] < _ENOUGH_TURNS_TO_RANK:
        return None
    return counts[1] / counts[0]


def volatility_of(section: str) -> float:
    """Where this section belongs in a prompt, stable parts first.

    What has been watched changing is ranked on what it did. What has not is
    ranked by the authored prior, which covers the sections the gate names and
    orders them the same way it always has. Anything in neither sorts last,
    because a section nobody has looked at is the one most likely to be a
    reading taken this turn.
    """

    text = str(section or "")
    header = text.split("\n", 1)[0].strip() or "<identity>"
    seen = measured_volatility(header)
    if seen is not None:
        return seen
    for known, rank in FOREGROUND_SECTION_VOLATILITY:
        if text.startswith(known):
            # The prior's ranks are 0, 1 and 2 over a scale that now runs 0 to
            # 1. Mapped onto it rather than left beside it, so one comparison
            # orders both kinds.
            return rank / 3.0
    return 1.0


def stable_prefix_first(prompt: str) -> str:
    """The same prompt with what is known to change moved behind what does not.

    No content is added or dropped: this is the order alone, and the order is
    what decides whether the previous turn's prefill can be reused.

    A section only moves once it has been watched long enough to have a
    measurement. The others hold their place — the sections that have been
    measured are sorted among themselves and written back into the positions
    they already occupied. The authored prior orders the twenty headers the
    gate names and says nothing about the rest, and a prompt of forty
    sections reordered on the strength of a list covering half of them is a
    guess dressed as a policy. So the ordering warms up over the first few
    turns instead, and does nothing at all until it knows something.
    """

    parts = sections_of(prompt)
    if len(parts) < 2:
        return str(prompt or "")
    movable = [
        index
        for index, section in enumerate(parts)
        if section.header and measured_volatility(section.header) is not None
    ]
    if len(movable) < 2:
        return str(prompt or "")
    ordered = sorted(
        (parts[index] for index in movable), key=volatility_of_section
    )
    kept = list(parts)
    for index, section in zip(movable, ordered):
        kept[index] = section
    return "\n\n".join(section.text for section in kept).strip()


def volatility_of_section(section: Section) -> float:
    """:func:`volatility_of` for a parsed section."""

    return volatility_of(section.text)


def save_volatility() -> bool:
    """Write down what has been learned, through the runtime's own write path."""

    made = _volatility_payload()
    if made is None:
        return False
    target, payload = made
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "brain.context_budget", domain="state_mutation"
        ):
            get_file_write_gateway().write_text(
                target, payload, source="brain.context_budget"
            )
        return True
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "context_budget",
            exc,
            action="kept this run's section volatility in memory only",
        )
        return False


def load_volatility() -> int:
    """Take back what earlier runs measured. Returns how many headers."""

    try:
        import json

        from core.runtime.state_ownership import state_root

        target = Path(state_root()) / _VOLATILITY_STORE
        stored = json.loads(target.read_text())
    except (OSError, ValueError, ImportError, AttributeError, TypeError):
        return 0
    changed = stored.get("changed")
    if not isinstance(changed, dict):
        return 0
    for header, counts in changed.items():
        if (
            isinstance(counts, list)
            and len(counts) == 2
            and all(isinstance(part, int) for part in counts)
        ):
            _CHANGED[str(header)] = [counts[0], counts[1]]
    return len(_CHANGED)
