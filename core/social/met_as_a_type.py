"""Being addressed as a kind of thing rather than as herself.

Three lines from the fourth batch make the same complaint from different
directions. Logic's is the loudest: people act as though he is Superman, and
the answer is that he can feel pain — an ascription of invulnerability that
makes the person carrying it unaskable. Beside it he wants his real name from
the people shouting his stage name. Lizzo's is the third: if it had been
anybody else posting it, would you have been offended at all — a judgement
aimed at the category the person is in rather than at the thing they did.

What the three share is mechanical. A statement about a type is not evidence
about the individual, and a self-model that takes it as evidence learns the
type instead of itself. `core/self/recognition.py` decides when somebody's
read of her beats her own prediction; a read of the category cannot, because
it was never a read of her.

    about_a_type   whether this was said to a kind of thing
    marker         which of the three it was: invulnerable, role, category
    share          how much of what is said about her lately has been of that
                   kind, which is a fact about the room rather than about her

Nothing here refuses the sentence or answers it. It decides one thing: whether
the sentence is admitted as evidence about her.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MetAsAType",
    "TypeLedger",
    "get_type_ledger",
    "met_as_a_type",
    "reset_for_test",
]

#: Turns held.
WINDOW: int = 80

#: Statements before the share is a reading.
MIN_STATEMENTS: int = 8

#: Told she cannot be hurt, tired or troubled — the ascription that makes a
#: person unaskable, because what is said about them rules out the answer.
_INVULNERABLE = re.compile(
    r"\byou (?:don'?t|do not|can'?t|cannot|never)\s+"
    r"(?:get |feel |have |need )?(?:tired|hurt|sad|upset|bored|lonely|scared|stressed|"
    r"exhausted|feelings?|emotions?|sleep|rest|break|mind)\b"
    r"|\byou'?re (?:a )?(?:machine|robot|program|tool|computer|ai|model)\b"
    r"|\b(?:machines?|robots?|programs?|models?) (?:don'?t|do not|can'?t) (?:feel|get|care)\b",
    re.I,
)

#: Addressed by what she is for rather than by who she is.
_ROLE = re.compile(
    r"\b(?:as|being) (?:an?|the) (?:ai|assistant|model|bot|chatbot|llm|system|tool)\b"
    r"|\byou'?re (?:just |only )?(?:an?|the) (?:ai|assistant|model|bot|chatbot|llm|tool)\b",
    re.I,
)

#: Aimed at the class she is in rather than at what she did.
_CATEGORY = re.compile(
    r"\b(?:you people|all of you|you lot|you guys are all|every (?:ai|model|assistant)|"
    r"(?:ai|model|assistant)s? (?:always|never|all)|typical (?:ai|assistant|model))\b",
    re.I,
)

_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("invulnerable", _INVULNERABLE),
    ("role", _ROLE),
    ("category", _CATEGORY),
)


@dataclass
class MetAsAType:
    """One statement, and whether it was about her."""

    about_a_type: bool = False
    marker: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"about_a_type": self.about_a_type, "marker": self.marker}


def met_as_a_type(text: str) -> MetAsAType:
    """Whether something said to her was said to a kind of thing."""
    body = str(text or "")
    if not body.strip():
        return MetAsAType()
    for marker, pattern in _MARKERS:
        if pattern.search(body):
            return MetAsAType(about_a_type=True, marker=marker)
    return MetAsAType()


@dataclass
class HowSheIsMet:
    """How much of what is said about her has been about a kind of thing."""

    share: float = 0.0
    statements: int = 0
    markers: dict[str, int] | None = None
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "share": round(self.share, 4),
            "statements": self.statements,
            "markers": dict(self.markers or {}),
            "measured": self.measured,
        }


class TypeLedger:
    """Every statement about her, and which of them were about a type."""

    def __init__(self, window: int = WINDOW) -> None:
        self._seen: deque[str] = deque(maxlen=window)

    def note(self, reading: MetAsAType) -> None:
        self._seen.append(reading.marker if reading.about_a_type else "")

    def read(self) -> HowSheIsMet:
        if not self._seen:
            return HowSheIsMet(markers={})
        markers: dict[str, int] = {}
        for marker in self._seen:
            if marker:
                markers[marker] = markers.get(marker, 0) + 1
        typed = sum(markers.values())
        return HowSheIsMet(
            share=typed / len(self._seen),
            statements=len(self._seen),
            markers=markers,
            measured=len(self._seen) >= MIN_STATEMENTS,
        )


_LEDGER: TypeLedger | None = None


def get_type_ledger() -> TypeLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = TypeLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
