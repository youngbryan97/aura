"""A matcher states the questions it must recognise, or it has not been probed.

Nine matchers in this codebase were fixed in one session for the same reason:
each was a hand-written regex covering the phrasings its author imagined, and
each met a real question it did not recognise.

    "what did I just copy?"                    clipboard, missed
    "the first THING I said"                   transcript, missed
    "are you planning to do anything later?"   queued work, missed
    "how many python files LIVE in X"          count, missed
    "what does X.md SAY about Y"               file read, missed
    "put BUILD-42 on my clipboard"             screen, matched wrongly
    "why do you think PEOPLE lie"              provenance, matched wrongly
    "put BUILD-42 on my clipboard"             clipboard, matched wrongly
    "how many python files live in X"          capability denial, cascaded

Widening a regex after each one only ever fixes the phrasing that was tried.
The generalisation is not a better regex; it is that a matcher which decides
whether a READING applies must carry the phrasings it claims and the
neighbours it disclaims, next to the matcher, where the person changing it
will see them.

Observable already does this for the grounding registry. This is the same
contract for the matchers that live outside it, so one test covers both and
adding either kind forces the same question: what did you actually try?
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "INTENT_MATCHERS",
    "IntentMatcher",
    "matcher_failures",
    "register_intent_matcher",
]


@dataclass(frozen=True, slots=True)
class IntentMatcher:
    """A predicate that decides whether some reading or branch applies."""

    name: str
    #: Truthy means "this turn is asking for it". A matcher may return a rich
    #: object rather than a bool; truthiness is what the callers test.
    predicate: Callable[[str], Any]
    #: Questions it MUST claim.
    examples: tuple[str, ...] = ()
    #: Questions it must NOT claim — usually neighbours belonging to another
    #: reading, which is where over-matching does its damage.
    counter_examples: tuple[str, ...] = ()
    #: Where it lives, for a failure message that points somewhere.
    where: str = ""

    def failures(self) -> list[str]:
        found: list[str] = []
        for prompt in self.examples:
            try:
                if not self.predicate(prompt):
                    found.append(f"{self.name} ({self.where}): should match {prompt!r}")
            except Exception as exc:  # noqa: BLE001 - report, never raise
                found.append(f"{self.name}: raised on {prompt!r}: {exc}")
        for prompt in self.counter_examples:
            try:
                if self.predicate(prompt):
                    found.append(
                        f"{self.name} ({self.where}): should NOT match {prompt!r}"
                    )
            except Exception as exc:  # noqa: BLE001
                found.append(f"{self.name}: raised on {prompt!r}: {exc}")
        return found


INTENT_MATCHERS: list[IntentMatcher] = []


def register_intent_matcher(matcher: IntentMatcher) -> None:
    """Add a matcher. Later registrations win on name collision."""

    global INTENT_MATCHERS
    INTENT_MATCHERS = [m for m in INTENT_MATCHERS if m.name != matcher.name]
    INTENT_MATCHERS.append(matcher)


def matcher_failures() -> list[str]:
    """Every registered matcher that disagrees with its own examples."""

    return [failure for matcher in INTENT_MATCHERS for failure in matcher.failures()]
