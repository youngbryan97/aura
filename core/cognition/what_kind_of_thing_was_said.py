"""What somebody meant, learned from what turned out to answer them.

What she had was a list of words. A turn starting "please" or "build" or
"open" was a request; one ending in a question mark was a question; anything
else was a statement at a confidence of 0.55. Seven ways of asking for the
same thing:

    open the 2048 app                  request
    can you get 2048 up on screen      question
    I'd like to see 2048 running       statement
    let's play some 2048               statement
    2048. now.                         statement
    mind firing up that tile game      statement
    put 2048 in front of me            statement

Five of the seven come back as remarks, and that label goes into what she is
told about the person before she answers — so she is reasoning about somebody
who made an observation when they asked her to do something. Adding words to
the list does not fix this. It is the wrong kind of thing to be doing.

What a turn IS cannot be read off its surface, but it does not have to be: it
shows up in what turned out to answer it. A turn that was answered well by
doing something was a request, whatever words it used. A turn answered well by
saying something was a question. She has both halves already — she takes turns
and she finds out how they went — so the kind of thing somebody said is
learnable from her own record rather than declared in advance.

Nothing here is a list of words she was given. The words are the ones she has
heard, and their weight is how often each has ended in one kind of answer
rather than another. With nothing heard yet it says so and offers nothing,
which is worth more than a wrong label at 0.55: the label reaches her
reasoning, and a confident wrong one is worse there than none.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhatSheHasHeard", "WhatKindItIs"]

#: Words carry the signal; punctuation and case do not.
_WORDS = re.compile(r"[a-z0-9']+")


def _words_in(said: str) -> list[str]:
    return _WORDS.findall(str(said or "").casefold())


@dataclass(frozen=True)
class WhatKindItIs:
    """What kind of thing a turn was, and how much she has to say so on."""

    kind: str
    how_sure: float
    #: How many turns like this one she is going on.
    from_turns: int

    @property
    def worked_out(self) -> bool:
        return bool(self.kind) and self.from_turns > 0

    def describe(self) -> str:
        if not self.worked_out:
            return "what kind of thing that was is not worked out yet"
        return f"{self.kind} ({self.how_sure:.0%}, from {self.from_turns} like it)"


@dataclass
class WhatSheHasHeard:
    """Turns she has taken, and what turned out to answer each one."""

    #: word -> kind of answer -> how often that answer went well after it.
    after: dict[str, dict[str, int]] = field(default_factory=dict)
    #: kind of answer -> how often it has gone well at all.
    ever: dict[str, int] = field(default_factory=dict)
    turns: int = 0

    def it_was_answered_by(self, said: str, doing: str, *, went_well: bool) -> None:
        """One turn, what she did about it, and whether that was right.

        Only what went well is counted. A response that did not work says
        nothing about what the person meant — it says something about her, and
        this is not a record of her.
        """
        kind = str(doing or "").strip()
        if not kind or not went_well:
            return
        self.turns += 1
        self.ever[kind] = self.ever.get(kind, 0) + 1
        for word in set(_words_in(said)):
            self.after.setdefault(word, {})[kind] = (
                self.after.setdefault(word, {}).get(kind, 0) + 1
            )

    def what_kind(self, said: str) -> WhatKindItIs:
        """What kind of thing this is, from what has answered turns like it.

        Every word she has heard before votes with the weight of what followed
        it, by Laplace's rule so that a word heard once does not decide. A word
        she has never heard says nothing rather than counting as evidence for
        the commonest answer, which is how a list of words gets rebuilt by
        accident.
        """
        kinds = sorted(self.ever)
        if not kinds:
            return WhatKindItIs("", 0.0, 0)
        weight = {kind: 0.0 for kind in kinds}
        heard = 0
        for word in set(_words_in(said)):
            seen = self.after.get(word)
            if not seen:
                continue
            heard += 1
            total = sum(seen.values())
            for kind in kinds:
                weight[kind] += (seen.get(kind, 0) + 1) / (total + len(kinds))
        if not heard:
            return WhatKindItIs("", 0.0, 0)
        best = max(kinds, key=lambda one: (weight[one], one))
        spread = sum(weight.values())
        return WhatKindItIs(
            kind=best,
            how_sure=(weight[best] / spread) if spread else 0.0,
            from_turns=heard,
        )

    def what_she_has_heard_of(self, said: str) -> int:
        """How many of these words she has heard before."""
        return sum(1 for word in set(_words_in(said)) if self.after.get(word))

    def kinds_she_knows(self) -> Sequence[str]:
        return sorted(self.ever)

    def as_memory(self) -> dict[str, Any]:
        return {
            "after": {word: dict(kinds) for word, kinds in self.after.items()},
            "ever": dict(self.ever),
            "turns": self.turns,
        }

    @classmethod
    def from_memory(cls, held: Any) -> WhatSheHasHeard:
        if not isinstance(held, dict):
            return cls()
        after: dict[str, dict[str, int]] = {}
        for word, kinds in (held.get("after") or {}).items():
            if not isinstance(kinds, dict):
                continue
            kept = {}
            for kind, many in kinds.items():
                try:
                    kept[str(kind)] = int(many)
                except (TypeError, ValueError):
                    # not a failure: a count that is not a number is not one.
                    continue
            if kept:
                after[str(word)] = kept
        ever: dict[str, int] = {}
        for kind, many in (held.get("ever") or {}).items():
            try:
                ever[str(kind)] = int(many)
            except (TypeError, ValueError):
                continue
        try:
            turns = int(held.get("turns") or 0)
        except (TypeError, ValueError):
            turns = 0
        return cls(after=after, ever=ever, turns=max(0, turns))
