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
import contextvars
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Observable",
    "OBSERVABLES",
    "observable_blocks",
    "readings_delivered_this_turn",
    "register_observable",
]

#: No single reading may hold the turn. Measured against the slowest reader in
#: the set (over 6.5M pages, the slowest of six cold lookups took 243ms).
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

    # ── the examples are part of the definition, not of the tests ────────────
    #
    # Seven matchers in this codebase were fixed in one session for the same
    # reason: each was a hand-written regex covering the phrasings its author
    # imagined, and each met a real question it did not recognise.
    #
    #   "what did I just copy?"                    -> clipboard, missed
    #   "the first THING I said"                   -> transcript, missed
    #   "are you planning to do anything later?"   -> queued work, missed
    #   "how many python files LIVE in X"          -> count, missed
    #   "what does X.md SAY about Y"               -> file read, missed
    #   "put BUILD-42 on my clipboard"             -> screen, matched wrongly
    #   "why do you think PEOPLE lie"              -> provenance, matched wrongly
    #
    # Every one was found by a person asking, live. Declaring the phrasings
    # beside the matcher moves that discovery to the test suite: the registry
    # test below asserts every example matches and every counter-example does
    # not, so an observable cannot be registered with a matcher nobody probed.
    #: Questions this observable MUST recognise.
    examples: tuple[str, ...] = ()
    #: Questions it must NOT claim — usually neighbours that belong to another
    #: reading, which is where over-matching does its damage.
    counter_examples: tuple[str, ...] = ()

    def example_failures(self) -> list[str]:
        """Examples the matcher gets wrong. Empty means it agrees with itself."""

        failures: list[str] = []
        for prompt in self.examples:
            try:
                if not self.matches(prompt):
                    failures.append(f"{self.name}: should match {prompt!r}")
            except Exception as exc:  # noqa: BLE001 - report, never raise
                failures.append(f"{self.name}: matcher raised on {prompt!r}: {exc}")
        for prompt in self.counter_examples:
            try:
                if self.matches(prompt):
                    failures.append(f"{self.name}: should NOT match {prompt!r}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{self.name}: matcher raised on {prompt!r}: {exc}")
        return failures


OBSERVABLES: list[Observable] = []

#: What was actually read for the turn in progress.
_READINGS_DELIVERED: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "aura_readings_delivered", default=()
)


def register_observable(observable: Observable) -> None:
    """Add an observable. Later registrations win on name collision.

    In place, never rebinding. Assigning a new list to the module global left
    every existing reference pointing at the old one, so a holder went on
    reading a registry that had stopped changing — and a caller removing its
    own registration edited an orphan while the real list kept it. Found as a
    test that passed alone and failed under a different order, with a
    registration from another test still answering.
    """

    OBSERVABLES[:] = [item for item in OBSERVABLES if item.name != observable.name]
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
    delivered = tuple(name for name, block in results if block)
    _READINGS_DELIVERED.set(delivered)
    try:
        from core.conversation.session_scope import record_evidence_delivered

        for name in delivered:
            record_evidence_delivered(name)
    except ImportError:  # pragma: no cover - grounding must never fail a turn
        pass
    return [block for _name, block in results if block]


def readings_delivered_this_turn() -> tuple[str, ...]:
    """Which observables actually produced a block for the turn in progress.

    Recorded because a reply can only be checked against evidence if something
    knows the evidence was handed over. LIVE 2026-08-19: the file reading was
    taken and delivered, and the reply said "the file path and contents are
    fictional for this example" — a disclaimer of evidence in hand, which
    nothing was in a position to notice.
    """
    return tuple(_READINGS_DELIVERED.get())
