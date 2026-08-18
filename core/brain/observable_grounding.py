"""Take the reading. One registry, so an observable is an entry and not a wire.

Every one of these was the same defect wearing different clothes, and each was
fixed separately before the pattern was obvious:

    "how many .py files in core/introspection"  ->  "There are 3"      (ten)
    "read CONTRIBUTING.md"                      ->  "I tried and failed"
                                                    (nothing ran)
    "what's on my clipboard right now?"         ->  "I can only work with the
                                                     information you provide"
                                                    (it held BUILD-7741-verify)
    "explain correlation vs causation"          ->  a wrong definition, with a
                                                    correct one on disk

The capability was registered every time. The reading was cheap every time. The
answer came from the model every time, because nothing had taken the reading
before the answer was composed, and a model asked about a fact it does not hold
will produce something fact-shaped.

The general statement: a question about observable local state must be answered
from an observation, and the observation has to happen BEFORE generation. That
is not a property of files, or of the clipboard — it is a property of every
observable this runtime can reach.

So observables live in one registry. Each entry says what question it answers
and how to take the reading; adding a new one is a few lines here rather than a
new branch threaded through the inference gate, which is how the previous four
ended up in four different places with four different bugs.

Deliberate constraints:

  * A reader runs only when the turn actually asks. Pulling someone's clipboard
    into every prompt because it might come up is a different product.
  * Every reader is bounded in time and size. A turn that stalls on a reading
    is worse than one that answers without it.
  * A reader that fails yields nothing rather than a guess, and a reading that
    is EMPTY says so — "it's empty" and "I couldn't look" are different
    answers and only one of them is ever true.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Observable",
    "OBSERVABLES",
    "observable_blocks",
    "register_observable",
]

#: No single reading may hold the turn. Measured against the slowest reader in
#: the set (a corpus search over ~7M pages answers in 8-80ms).
DEFAULT_READ_TIMEOUT_S = 2.5


@dataclass(frozen=True, slots=True)
class Observable:
    """One thing the runtime can look at, and when it should."""

    name: str
    header: str
    #: Does this turn ask about it? Cheap, synchronous, no side effects.
    matches: Callable[[str], bool]
    #: Take the reading. Returns the block body, or "" for nothing to say.
    read: Callable[[str], Awaitable[str]]
    timeout_s: float = DEFAULT_READ_TIMEOUT_S


OBSERVABLES: list[Observable] = []


def register_observable(observable: Observable) -> None:
    """Add an observable. Later registrations win on name collision."""

    global OBSERVABLES
    OBSERVABLES = [item for item in OBSERVABLES if item.name != observable.name]
    OBSERVABLES.append(observable)


async def _read_one(observable: Observable, prompt: str) -> tuple[str, str]:
    try:
        if not observable.matches(prompt):
            return observable.name, ""
    except Exception:  # noqa: BLE001 - a matcher must never break a turn
        return observable.name, ""
    try:
        body = await asyncio.wait_for(
            observable.read(prompt), timeout=max(0.1, float(observable.timeout_s))
        )
    except (TimeoutError, asyncio.CancelledError):
        return observable.name, ""
    except Exception:  # noqa: BLE001 - a failed reading is not a failed turn
        return observable.name, ""
    text = str(body or "").strip()
    if not text:
        return observable.name, ""
    return observable.name, f"{observable.header}\n{text}"


async def observable_blocks(user_prompt: Any) -> list[str]:
    """Grounding blocks for every observable this turn asks about.

    Readings run concurrently: two observables in one question ("what's on my
    clipboard and what does config.py say") should cost one wait, not two.
    """

    prompt = str(user_prompt or "")
    if not prompt.strip() or not OBSERVABLES:
        return []
    results = await asyncio.gather(
        *(_read_one(observable, prompt) for observable in OBSERVABLES),
        return_exceptions=False,
    )
    return [block for _name, block in results if block]
