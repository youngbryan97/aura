"""One working memory, and the capacity everything reads it against.

What Aura is currently thinking about lives in one place:
``state.cognition.working_memory``, a list of exchanges bounded by
``MAX_WORKING_MEMORY``. That much was already true. What was not true is that
the readers agreed about how big it is.

Four numbers claimed to be the capacity of the same list:

* ``MAX_WORKING_MEMORY`` = 150, the one the trimmer actually enforces.
* ``StateBoundsConfig.MAX_WORKING_MEMORY_ITEMS`` = 100, in the sandbox's
  memory-bomb guard — which read ``state.working_memory``, an attribute
  ``AuraState`` does not have, so it passed a 5,000-item bomb.
* ``working_memory_cap=40``, passed by the context assembler into the
  finitude model, whose ``context_usage`` therefore read 1.0 from the
  fortieth exchange onward and went into a live self-report block.
* ``/ 20.0`` in the executive closure state vector, saturating at 20.

A reader that normalises against the wrong capacity is not slightly off. It
is pinned at its ceiling for most of a conversation, which makes it a
constant wearing a measurement's clothes.

So: ask ``the_capacity()`` and ``how_full()`` here rather than writing a
number down. ``the_caps_that_disagree()`` is the gate — it reads the source
for a second opinion about the same list, and the baseline only shrinks.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = [
    "THE_STORE",
    "how_full",
    "the_capacity",
    "the_caps_that_disagree",
    "the_working_memory",
    "who_else_holds_it",
]

#: The one address. A dotted path rather than an object, because the point is
#: that everything names the same field of the same state.
THE_STORE = "cognition.working_memory"


def the_capacity() -> int:
    """How many exchanges the working memory holds before it is trimmed.

    Read from the trimmer's own constant, so there is one number and it is
    the enforced one.
    """
    from core.state.aura_state import MAX_WORKING_MEMORY

    return int(MAX_WORKING_MEMORY)


def the_working_memory(state: Any) -> list[dict]:
    """The canonical list, from whatever is handed in.

    Accepts a full ``AuraState`` or a bare ``CognitiveContext``: the guard
    this replaces read ``state.working_memory`` on an ``AuraState`` and
    therefore never fired, and a helper that only works on one of the two
    shapes reproduces exactly that.
    """
    held = getattr(state, "working_memory", None)
    if held is None:
        held = getattr(getattr(state, "cognition", None), "working_memory", None)
    if not isinstance(held, list):
        return []
    return held


def how_full(state: Any) -> float:
    """How full it is, against the capacity that is actually enforced.

    Not clamped at 1.0 by a smaller ceiling somebody wrote down. Over-full is
    a real condition — the trimmer runs after the append, so a reader can see
    151 — and reporting it as exactly full hides the one moment worth seeing.
    """
    cap = max(1, the_capacity())
    return len(the_working_memory(state)) / float(cap)


#: The other places the same content is held, and what makes each of them a
#: projection rather than a second copy of the truth. A holder that cannot
#: name its derivation is not a projection, it is a fork.
_PROJECTIONS: tuple[dict[str, str], ...] = (
    {
        "who": "core.memory.stratified.working.WorkingMemoryTier",
        "holds": "active modal, current goal, hazards, plan, last actions",
        "derived": "written by the environment run; never read back into "
        "cognition.working_memory",
        "fresh": "per environment step",
    },
    {
        "who": "core.unity.mind_moment",
        "holds": "the last few exchanges as bound content",
        "derived": "read-only slice of the canonical list",
        "fresh": "per tick",
    },
    {
        "who": "core.unity.runtime._working_memory_contents",
        "holds": "the last six items as BoundContent",
        "derived": "read-only slice of the canonical list",
        "fresh": "per binding pass",
    },
    {
        "who": "core.cognitive.spiking_active_inference.BoundedWorkingMemoryQueueModel",
        "holds": "a queue load and an admission decision",
        "derived": "a model of pressure, not of contents; shares the name only",
        "fresh": "per inference step",
    },
    {
        "who": "core.state.aura_state.CognitiveContext.history",
        "holds": "the canonical list itself",
        "derived": "alias property, same object",
        "fresh": "always",
    },
)


def who_else_holds_it() -> tuple[dict[str, str], ...]:
    """Every other holder of this content, with its derivation and freshness."""
    return _PROJECTIONS


# --------------------------------------------------------------- the gate

#: A number written next to the words that name this list. Deliberately narrow:
#: a window like ``[-8:]`` is a reader looking at recent items, which is fine,
#: and only a divisor or a declared cap claims to be the capacity.
_A_SECOND_OPINION = re.compile(
    r"working_memory[^\n]{0,40}?/\s*([0-9]+(?:\.[0-9]+)?)"
    r"|MAX_WORKING_MEMORY_ITEMS[^\n]*?=\s*([0-9]+)"
    r"|working_memory_cap\s*[=:]\s*([0-9]+)"
)

#: Where the one number lives, so the file that defines it does not report
#: itself, and this file's own prose does not either.
_ALLOWED = {
    "core/state/aura_state.py",
    "core/state/one_working_memory.py",
}


def the_caps_that_disagree(root: Path | None = None) -> list[str]:
    """Sites that normalise the working memory against their own number.

    Returns ``path:line: number`` for each. The right fix is always to ask
    ``the_capacity()``; a reader that genuinely wants a different denominator
    wants a different quantity and should name it.
    """
    here = root or Path(__file__).resolve().parents[2]
    cap = float(the_capacity())
    found: list[str] = []
    for path in sorted((here / "core").rglob("*.py")):
        rel = str(path.relative_to(here))
        if rel in _ALLOWED or "__pycache__" in rel:
            continue
        try:
            text = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            match = _A_SECOND_OPINION.search(line)
            if not match:
                continue
            said = next((g for g in match.groups() if g), None)
            if said is None:
                continue
            try:
                number = float(said)
            except ValueError:
                continue
            if number == cap:
                continue
            found.append(f"{rel}:{line_no}: {said}")
    return found
