"""The shape of an utterance, measured — read on the way in, chosen on the way out.

Fourteen records were measured for this. The numbers that separated them were
not about content at all: who is being spoken to, how dense the words are, how
much of it is a question, how long the loaded word is held, and how often the
one phrase comes back.

    track                    I    you   we   they  ask  hold x
    Sam Cooke                1.00 0.00  0.00 0.00  0.00   5.1
    Dancing in the Moonlight 0.00 0.19  0.24 0.57  0.00   3.6
    Remember the Time        0.29 0.50  0.21 0.00  0.50  11.1
    Best Part                0.41 0.56  0.02 0.00  0.00   9.5
    Contradiction's Maze     0.82 0.06  0.03 0.10  0.04   5.4

Sam Cooke's record is entirely first person and asks nothing. "Dancing in the
Moonlight" has no first person in it at all — communal joy is not about the
self. "Remember the Time" is the only one with a real share of "we" and the only
one that is half questions, which is the same fact twice: joint recall needs
both. And the longest-held word is almost always the loaded one — `time`,
`like` out of "not like I used to", `paranoid`, `been` out of "it's been a long
time coming", `this`. Duration marks significance. Nothing picks the rhyme.

The useful part is that these are measurements, so they run on what somebody
says to her as readily as on a record. That turns a judgment she was making
from nothing into a reading:

    a high first-person share, almost no questions, and the connectives of
    persistence -- but, still, anyway, somehow -- is somebody testifying.

The response to testimony is not assistance. Nobody wants Sam Cooke's record
fixed. Questions are what a request looks like, and Michael Jackson's record
asks fifteen of them while Sam Cooke's asks none. So `asks_for_help` and
`asks_to_be_witnessed` are two readings off one profile rather than a guess,
and `distance` gives her something to move toward when matching somebody's
register is the point.

Nothing here decides what to say. It decides the shape.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "ASKING",
    "PERSISTENCE",
    "Register",
    "comparable",
    "distance",
    "held_token",
    "read",
    "toward",
]

#: Words that place an utterance's person. Kept as a mapping rather than
#: inferred, because "babe" and "girl" are second person in every one of these
#: records and no part-of-speech tag says so.
_PERSON: dict[str, str] = {
    **{w: "first" for w in (
        "i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd",
    )},
    **{w: "second" for w in (
        "you", "your", "yours", "yourself", "you're", "you'll", "you've",
        "babe", "baby", "girl",
    )},
    **{w: "plural" for w in ("we", "us", "our", "ours", "we're", "we'll", "we've")},
    **{w: "third" for w in (
        "they", "them", "their", "theirs", "everybody", "everyone", "people",
        "he", "she", "him", "her", "his", "hers",
    )},
}

#: Openers that make a clause a request rather than a report.
ASKING: tuple[str, ...] = (
    "do", "did", "does", "would", "will", "won't", "can", "could", "should",
    "is", "are", "was", "were", "am", "why", "what", "how", "who", "when",
    "where", "which", "any", "ain't", "have", "has",
)

#: The connectives of holding on. Oddisee's "Contradiction's Maze" has sixteen
#: of them, the most of the fourteen, and the contradiction it is about lives in
#: exactly these words rather than in any of its nouns.
PERSISTENCE: tuple[str, ...] = (
    "but", "still", "anyway", "somehow", "though", "although", "even so",
    "even when", "yet", "despite", "no matter", "keep on", "never", "always",
    "regardless", "whatever happens",
)

_WORD = re.compile(r"[a-z']+")
_CLAUSE = re.compile(r"[.!?;\n]+|,\s+(?=but|and|so|because)")


@dataclass(frozen=True)
class Register:
    """How something was said, with nothing about what it said."""

    words: int = 0
    #: Shares of the placed pronouns, summing to one when any were placed.
    first: float = 0.0
    second: float = 0.0
    plural: float = 0.0
    third: float = 0.0
    placed: int = 0
    #: Clauses that open like a request, as a share of clauses.
    asking: float = 0.0
    #: Connectives of persistence, per hundred words, so a long utterance is
    #: not automatically more insistent than a short one.
    persistence: float = 0.0
    #: Distinct words over total. Low is a refrain; the two records that say
    #: the least say it the most times.
    variety: float = 0.0
    #: The most repeated phrase and how often it came back.
    refrain: str = ""
    refrain_times: int = 0
    #: Clause lengths in words: how much is said between breaths.
    clause_words: float = 0.0
    clauses: int = 0
    measured: bool = False

    def stance(self) -> str:
        """The one word for who this is about. Ties go to the larger share."""
        if not self.placed:
            return "impersonal"
        shares = {
            "testimony": self.first,
            "address": self.second,
            "together": self.plural,
            "report": self.third,
        }
        return max(shares, key=lambda key: shares[key])

    def asks_for_help(self) -> bool:
        """Whether this is a request. Questions are what one looks like."""
        return self.measured and self.asking > 0.25

    def asks_to_be_witnessed(self) -> bool:
        """First person, no questions, and holding on.

        The signature of the two records in the set that nobody wants fixed.
        Assistance is the wrong response to it, and she had no way to tell.
        """
        return (
            self.measured
            and self.stance() == "testimony"
            and not self.asks_for_help()
            and self.persistence > 0.0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "words": self.words,
            "stance": self.stance(),
            "first": round(self.first, 4),
            "second": round(self.second, 4),
            "plural": round(self.plural, 4),
            "third": round(self.third, 4),
            "placed": self.placed,
            "asking": round(self.asking, 4),
            "persistence": round(self.persistence, 4),
            "variety": round(self.variety, 4),
            "refrain": self.refrain,
            "refrain_times": self.refrain_times,
            "clause_words": round(self.clause_words, 2),
            "clauses": self.clauses,
            "asks_for_help": self.asks_for_help(),
            "asks_to_be_witnessed": self.asks_to_be_witnessed(),
            "measured": self.measured,
        }


def _refrain(tokens: Sequence[str]) -> tuple[str, int]:
    """The phrase that comes back most, weighted by how much of it comes back.

    Three words repeated ten times is more of a refrain than six words repeated
    three times, and a longer phrase repeating is more of one than a shorter.
    The product is what ranks them.
    """
    best = ("", 0, 0)
    for size in (3, 4, 5, 6):
        if len(tokens) < size + 1:
            break
        counts = Counter(
            " ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)
        )
        phrase, times = counts.most_common(1)[0]
        if times >= 2 and times * size > best[1] * best[2]:
            best = (phrase, times, size)
    return best[0], best[1]


def held_token(words: Iterable[tuple[str, float, float]]) -> tuple[str, float]:
    """The word given the most time, and how many medians of time that was.

    Duration marks significance. Across the fourteen records the longest-held
    word is the loaded one — `time`, `like`, `paranoid`, `been`, `this` — and
    never the rhyme.
    """
    rows = [(w, e - s) for w, s, e in words if e > s]
    if not rows:
        return "", 0.0
    durations = sorted(d for _, d in rows)
    median = durations[len(durations) // 2] or 1e-6
    word, longest = max(rows, key=lambda row: row[1])
    return word, longest / median


def read(text: str) -> Register:
    """Measure how something was said."""
    body = str(text or "")
    tokens = _WORD.findall(body.lower())
    if not tokens:
        return Register()
    placed = Counter(_PERSON[t] for t in tokens if t in _PERSON)
    total = sum(placed.values())
    clauses = [c.strip() for c in _CLAUSE.split(body) if c.strip()]
    asking = 0
    for clause in clauses:
        head = _WORD.findall(clause.lower())
        if clause.rstrip().endswith("?") or (head and head[0] in ASKING):
            asking += 1
    lowered = f" {' '.join(tokens)} "
    persistence = sum(lowered.count(f" {m} ") for m in PERSISTENCE)
    refrain, times = _refrain(tokens)
    return Register(
        words=len(tokens),
        first=placed.get("first", 0) / total if total else 0.0,
        second=placed.get("second", 0) / total if total else 0.0,
        plural=placed.get("plural", 0) / total if total else 0.0,
        third=placed.get("third", 0) / total if total else 0.0,
        placed=total,
        asking=asking / max(len(clauses), 1),
        persistence=100.0 * persistence / len(tokens),
        variety=len(set(tokens)) / len(tokens),
        refrain=refrain,
        refrain_times=times,
        clause_words=len(tokens) / max(len(clauses), 1),
        clauses=len(clauses),
        measured=True,
    )


#: The coordinates distance is taken over. Person, request, insistence,
#: repetition and breath — each already a share or a rate in its own units, so
#: none of them needs a weight.
_AXES: tuple[str, ...] = ("first", "second", "plural", "third", "asking", "variety")


def comparable(left: Register, right: Register) -> bool:
    """Whether two utterances have enough words between them to have a shape.

    The profile has six coordinates and fewer words than that cannot fill
    them: one pronoun in a four-word message puts a whole share on one axis,
    and the distance between two such profiles is a reading of their length.
    """
    return left.measured and right.measured and min(left.words, right.words) >= len(_AXES)


def distance(left: Register, right: Register) -> float:
    """How far apart two registers are, in [0, 1].

    Euclidean over the shares, divided by the largest distance those shares
    could be apart, so the scale comes from the coordinate space rather than
    from a normalising constant.
    """
    if not (left.measured and right.measured):
        return 0.0
    gaps = [getattr(left, axis) - getattr(right, axis) for axis in _AXES]
    return min(1.0, math.sqrt(sum(g * g for g in gaps)) / math.sqrt(len(_AXES)))


@dataclass(frozen=True)
class Move:
    """What to change to come nearer somebody's register, largest gap first."""

    axis: str = ""
    theirs: float = 0.0
    mine: float = 0.0

    @property
    def gap(self) -> float:
        return self.theirs - self.mine


def toward(theirs: Register, mine: Register, *, most: int = 3) -> tuple[Move, ...]:
    """The axes on which her register is furthest from theirs.

    Not an instruction to copy anybody. Entrainment in these records is a
    shared pulse that individual voices then deviate against, and knowing the
    distance is what makes a deviation mean something rather than being drift.
    """
    if not (theirs.measured and mine.measured):
        return ()
    moves = [
        Move(axis=axis, theirs=getattr(theirs, axis), mine=getattr(mine, axis))
        for axis in _AXES
    ]
    moves.sort(key=lambda move: -abs(move.gap))
    return tuple(move for move in moves[:most] if abs(move.gap) > 1e-9)
