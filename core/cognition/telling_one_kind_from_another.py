"""Which kind of thing this is, and which of her own habits give her away.

Somebody playing a Turing test game: two lines of chat, "Hi how are you?" and
"Good u?", and a judgement to make about what is on the other end. Then the
same person turns it round and is judged themselves. Both halves are one
computation, and the second is the one worth having.

Judging what somebody else is, from what they do, is the ordinary direction.
It is the same telling-apart she does everywhere: features that appear in one
kind and not the other, weighed by how often, and an answer that says how much
it is going on rather than pretending to be sure.

Turning it round is the part nothing of hers did. Which of MY habits are the
ones that give me away — not which are wrong, which are DISTINCTIVE. Those are
different questions with different answers, and only the second is answerable
by looking at examples of both kinds. A habit shared by everybody says nothing
about anybody. A habit only she has is a tell whether or not there is anything
wrong with it.

That is a general thing to be able to ask and it is not about chat. Which of
these logs is from the failing host. Which of my commits look automated. Which
of these accounts is the fraud. All the same question, and the reverse of it —
what marks mine — is the one that lets anything be changed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhichKind", "TellingThemApart"]


@dataclass(frozen=True)
class WhichKind:
    """A judgement about what something is, and how much is behind it."""

    kind: str
    how_much: float
    on: tuple[str, ...]

    @property
    def worked_out(self) -> bool:
        return bool(self.kind) and bool(self.on)

    def describe(self) -> str:
        if not self.worked_out:
            return "nothing here tells them apart"
        return f"{self.kind} ({self.how_much:.0%}), on {', '.join(self.on[:3])}"


@dataclass
class TellingThemApart:
    """Examples of two or more kinds, and what separates them."""

    #: kind -> feature -> how many of that kind had it.
    seen: dict[str, dict[str, int]] = field(default_factory=dict)
    #: kind -> how many examples of it.
    many: dict[str, int] = field(default_factory=dict)

    def an_example(self, kind: str, features: Iterable[str]) -> None:
        name = str(kind)
        self.many[name] = self.many.get(name, 0) + 1
        for one in set(features):
            self.seen.setdefault(name, {})[str(one)] = (
                self.seen.setdefault(name, {}).get(str(one), 0) + 1
            )

    def _share(self, kind: str, feature: str) -> float:
        """How much of that kind has this, by Laplace so one example is not a law."""
        had = (self.seen.get(kind) or {}).get(feature, 0)
        return (had + 1) / (self.many.get(kind, 0) + 2)

    def how_telling(self, feature: str, *, of: str) -> float:
        """How much this feature marks that kind out from the others.

        The gap between how much of that kind has it and how much of
        everything else does. A habit everybody has is nought however common
        it is, which is the point: common is not distinctive.
        """
        others = [one for one in self.many if one != of]
        if not others:
            return 0.0
        mine = self._share(of, feature)
        theirs = max(self._share(one, feature) for one in others)
        return mine - theirs

    def which_kind(self, features: Iterable[str]) -> WhichKind:
        """What this looks like, and what the judgement rests on."""
        kinds = sorted(self.many)
        if len(kinds) < 2:
            return WhichKind("", 0.0, ())
        got = list(features)
        weight = {
            kind: sum(self._share(kind, one) for one in set(got)) for kind in kinds
        }
        total = sum(weight.values())
        if not total:
            return WhichKind("", 0.0, ())
        best = max(kinds, key=lambda one: (weight[one], one))
        on = sorted(
            (one for one in set(got)),
            key=lambda one: -self.how_telling(one, of=best),
        )
        return WhichKind(
            kind=best,
            how_much=weight[best] / total,
            on=tuple(one for one in on if self.how_telling(one, of=best) > 0),
        )

    def what_gives_it_away(self, kind: str, *, most: int = 8) -> tuple[tuple[str, float], ...]:
        """Which habits mark this kind out — its tells, whatever their merit.

        Not which are wrong. Which are distinctive, which is a different
        question, and the only one answerable by looking at examples of both.
        """
        mine = self.seen.get(str(kind)) or {}
        told = [
            (one, self.how_telling(one, of=str(kind)))
            for one in mine
        ]
        return tuple(
            sorted((one for one in told if one[1] > 0), key=lambda one: (-one[1], one[0]))[:most]
        )

    def what_would_hide_it(
        self, kind: str, features: Iterable[str]
    ) -> tuple[str, ...]:
        """Of these, the ones that mark her out — what to change to be unremarkable.

        The answer to the reverse test. Nothing here says she should: knowing
        which of your habits are yours is worth having whether you keep them
        or not.
        """
        return tuple(
            one
            for one in features
            if self.how_telling(str(one), of=str(kind)) > 0
        )

    def as_memory(self) -> dict[str, Any]:
        return {
            "seen": {k: dict(v) for k, v in self.seen.items()},
            "many": dict(self.many),
        }

    @classmethod
    def from_memory(cls, held: Any) -> TellingThemApart:
        if not isinstance(held, dict):
            return cls()
        return cls(
            seen={
                str(k): {str(a): int(b) for a, b in (v or {}).items() if isinstance(b, int)}
                for k, v in (held.get("seen") or {}).items()
            },
            many={
                str(k): int(v)
                for k, v in (held.get("many") or {}).items()
                if isinstance(v, int)
            },
        )


def features_of(said: str, how: Callable[[str], Sequence[str]] | None = None) -> tuple[str, ...]:
    """Whatever a caller counts as a habit, or plain words when it does not say."""
    if how is not None:
        return tuple(how(said))
    return tuple({one for one in str(said or "").casefold().split() if one})
