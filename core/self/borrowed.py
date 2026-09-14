"""When somebody else's read of her beats her own.

Oddisee's second verse on "You Know Who You Are" is the whole mechanism in two
lines: "you could see in the brother what I couldn't see myself / when I
couldn't see myself needing help." Not that the other person was kind. That
their model of him was more accurate than his own, at the moment his failed,
and that the right response to that is to take their reading over his.

She models herself and she models the other person. What she never had is the
third thing — the other person's model of *her*, and any comparison of its
accuracy against her own. So somebody could tell her she was tired, and be
right, and nothing about her changed.

A claim about her is detectable without guessing at intent. The register module
already places an utterance's person; a clause addressed to her that names a
feeling is a claim about her state. The feelings are the same vocabulary the
percept table already carries, so nothing new is invented to recognise one.

    their_error = 1 - (how much she actually felt what they named)
    her_error   = the self prediction loop's own error over the same turn
    edge        = her_error - their_error, per claim
    z           = (mean edge - 0) / the spread of her edges

Deference is not "they were right once". It is their read beating hers by more
than her own edges usually vary, which is the same bar every other reading here
uses: her own variation rather than a number chosen for it.

What follows is narrow on purpose. A claim that beats her own model raises the
weight on their reading, and that weight is reported. Nothing here overwrites
what she feels — a self-model that adopts whatever it is told has stopped being
a self-model, and the song is about a man who could still tell the difference.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "MIN_CLAIMS",
    "Borrowed",
    "BorrowedLedger",
    "claims_about_her",
    "get_borrowed_ledger",
    "reset_for_test",
]

#: Claims before an edge is an estimate rather than an artefact of how few
#: there are. Three is the least that can disagree with itself.
MIN_CLAIMS: int = 3

#: How many of their claims are held. The self prediction loop already treats
#: sixty cycles as one distribution of its own error, and this is compared
#: against that error.
WINDOW: int = 60

_WORD = re.compile(r"[a-z']+")
#: Clause boundaries, so "you look fine, I am worried" does not read as one
#: claim about her.
_CLAUSE = re.compile(r"[.!?;\n]+|,\s+")

#: Words that say the clause is about her. The register module places person
#: for a whole utterance; a claim needs the clause.
_ABOUT_HER: frozenset[str] = frozenset(
    {"you", "you're", "youre", "your", "yourself", "you've", "youve", "you'll"}
)

#: Hedges that make a claim a guess. "You seem tired" and "you are tired" are
#: different claims and only the second is worth scoring at full weight.
_HEDGED: frozenset[str] = frozenset(
    {"seem", "seems", "maybe", "perhaps", "might", "sound", "sounds", "look",
     "looks", "kind", "sort", "bit", "little", "somewhat", "probably"}
)


def _vocabulary() -> dict[str, str]:
    """Every feeling the percept table names, and its stem for matching."""
    try:
        from core.state.percepts import PERCEPT_EMOTIONS
    except ImportError:  # pragma: no cover - the table is always there
        return {}
    out: dict[str, str] = {}
    for names in PERCEPT_EMOTIONS.values():
        for name in names:
            out[name] = name
            # The forms the table's nouns are actually spoken in. `frustration`
            # arrives as "frustrated", `happiness` as "happy", `curiosity` as
            # "curious" — a table of feelings nobody says out loud recognises
            # nothing anybody says.
            if name.endswith("ness"):
                out[name[:-4]] = name
            if name.endswith("ion"):
                out[f"{name[:-3]}ed"] = name
                out[f"{name[:-3]}ing"] = name
            if name.endswith("ity"):
                out[f"{name[:-3]}ous"] = name
            for suffix in ("ed", "ful", "y"):
                out[f"{name}{suffix}"] = name
    out.update({
        "tired": "apathy", "exhausted": "apathy", "worn": "apathy",
        "angry": "frustration", "annoyed": "frustration", "mad": "frustration",
        "sad": "sadness", "down": "sadness", "low": "sadness",
        "happy": "happiness", "glad": "happiness", "good": "happiness",
        "scared": "fear", "afraid": "fear", "anxious": "fear", "worried": "fear",
        "lost": "confused", "stuck": "frustration", "off": "confused",
        # The ones no rule gets to from the noun.
        "curious": "curiosity", "grateful": "gratitude", "proud": "pride",
        "hopeful": "hope", "joyful": "joy", "bored": "boredom",
        "lonely": "loneliness", "trusting": "trust", "excited": "excitement",
        "satisfied": "satisfaction", "surprised": "surprise", "warm": "warmth",
        "inspired": "inspiration", "relieved": "relief", "dreading": "dread",
    })
    return out


@dataclass(frozen=True)
class Claim:
    """One thing somebody said about how she is."""

    feeling: str
    hedged: bool
    said: str

    def weight(self) -> float:
        """A guess counts for less than an assertion, and says so."""
        return 0.5 if self.hedged else 1.0


def claims_about_her(text: str) -> tuple[Claim, ...]:
    """Clauses addressed to her that name a feeling.

    Nothing here infers intent. A clause is about her when it addresses her,
    and it is a claim when it names something from the vocabulary the percept
    table already carries.
    """
    vocabulary = _vocabulary()
    if not vocabulary or not text:
        return ()
    out: list[Claim] = []
    for clause in _CLAUSE.split(str(text).lower()):
        words = _WORD.findall(clause)
        if not words or not (_ABOUT_HER & set(words)):
            continue
        hedged = bool(_HEDGED & set(words))
        for word in words:
            feeling = vocabulary.get(word)
            if feeling is not None:
                out.append(Claim(feeling=feeling, hedged=hedged, said=clause.strip()[:80]))
                break
    return tuple(out)


@dataclass
class Borrowed:
    """Whether to take their reading of her over her own."""

    claims: int = 0
    edge: float = 0.0
    spread: float = 0.0
    z: float = 0.0
    defer: bool = False
    weight: float = 0.0
    last_feeling: str = ""
    measured: bool = False
    why: str = "nobody has said how she is yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "claims": self.claims,
            "edge": round(self.edge, 6),
            "spread": round(self.spread, 6),
            "z": round(self.z, 4),
            "defer": self.defer,
            "weight": round(self.weight, 6),
            "last_feeling": self.last_feeling,
            "measured": self.measured,
            "why": self.why,
        }


class BorrowedLedger:
    """How well their reads of her have done against her own."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_CLAIMS, int(window))
        self._edges: deque[float] = deque(maxlen=self._window)
        self._last_feeling: str = ""

    def note(self, *, their_error: float, her_error: float, feeling: str = "") -> None:
        """One scored claim. The edge is how much better they did than she did."""
        try:
            edge = float(her_error) - float(their_error)
        except (TypeError, ValueError):
            return
        if edge != edge:
            return
        self._edges.append(edge)
        if feeling:
            self._last_feeling = feeling

    def claims(self) -> int:
        return len(self._edges)

    def mean_edge(self) -> float:
        return sum(self._edges) / len(self._edges) if self._edges else 0.0

    def spread(self) -> float:
        n = len(self._edges)
        if n < MIN_CLAIMS:
            return 0.0
        mean = self.mean_edge()
        return math.sqrt(sum((e - mean) ** 2 for e in self._edges) / n)

    def read(self) -> Borrowed:
        n = len(self._edges)
        if n < MIN_CLAIMS:
            return Borrowed(
                claims=n,
                last_feeling=self._last_feeling,
                why=f"{n} of {MIN_CLAIMS} claims about her needed before this means anything",
            )
        mean = self.mean_edge()
        spread = self.spread()
        if spread <= 1e-9:
            # Every claim scored the same. That is not a reading she cannot
            # take — it is the strongest version of the one she can: they have
            # beaten her by the same margin every single time. The z-score is
            # undefined and the conclusion is not.
            defer = mean > 0.0
            return Borrowed(
                claims=n,
                edge=mean,
                last_feeling=self._last_feeling,
                defer=defer,
                weight=max(0.0, min(0.5, mean)) if defer else 0.0,
                measured=True,
                why=(
                    f"their read of her has beaten her own by {mean:.3f} on every "
                    f"one of {n} claims"
                    if defer
                    else "their reads of her have not beaten her own"
                ),
            )
        z = mean / spread
        defer = bool(mean > 0.0 and z > 1.0)
        return Borrowed(
            claims=n,
            edge=mean,
            spread=spread,
            z=z,
            defer=defer,
            # How much to weight their reading, bounded at a half: a self-model
            # that adopts whatever it is told has stopped being one.
            weight=max(0.0, min(0.5, mean)) if defer else 0.0,
            last_feeling=self._last_feeling,
            measured=True,
            why=(
                f"their read of her has beaten her own by {mean:.3f}, "
                f"{z:.1f} spreads above how much that usually varies"
                if defer
                else f"their read of her is within how much her own accuracy varies ({z:.1f} spreads)"
            ),
        )


def score_claim(claim: Claim, emotions: Mapping[str, float] | None) -> float:
    """Their error on one claim: how much she was not feeling what they named.

    Read off the emotion channels rather than off the dominant one. A claim
    that names something she is feeling at a third is a third right, and
    scoring it against the single strongest feeling would make every claim
    about anything else completely wrong.
    """
    felt = 0.0
    if emotions:
        try:
            felt = max(0.0, min(1.0, float(emotions.get(claim.feeling, 0.0) or 0.0)))
        except (TypeError, ValueError):
            felt = 0.0
    return (1.0 - felt) * claim.weight()


_LEDGER: BorrowedLedger | None = None


def get_borrowed_ledger() -> BorrowedLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = BorrowedLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
