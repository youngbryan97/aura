"""core/perception/how_she_finds_out.py — the loop, over the real inventory.

`expected_information_gain.py` computes what one observation is worth. It is
correct and it is a calculator: somebody has to hand it a hypothesis set, a
model of the instrument, and a cost, and then do something with the answer.
Nobody did. There is information-gain reasoning elsewhere, experiment
proposal and selection, active sensing loops, and cost machinery relating gain
to spend — and no single controller that goes

    uncertainty -> candidate observations -> P(o|h) -> EIG - C
        -> take the best one -> update the belief

over the things she can actually do. So she could seek information, and had no
general policy for deciding what to seek.

This is that controller, and the design constraint is that it must not know
what a screen is. A controller holding a list of sensors is a controller that
covers the sensors somebody thought of; the inventory is registered by the
subsystems that own each way of finding out, and the controller only knows
that a way has a cost, a reliability and something it can be asked.

Where the instrument model comes from
-------------------------------------
Writing down P(outcome | hypothesis) is most of the work, and inventing one is
worse than having none. So the model here has exactly one parameter — how
often this way of finding out has actually been right — and that parameter is
measured rather than declared. A way used forty times and right thirty-two of
them has a reliability with an interval around it; a way used never has a
uniform one, which is the honest statement that nothing is known about it.

That creates the usual cold start: a way with reliability one-half
discriminates nothing, has zero expected gain, is never taken, and never
learns. The fix is not a constant nudging it upward. It is to score on a
reliability DRAWN from the posterior rather than its mean — an untried way is
sometimes optimistic and gets its trial, a way that has failed forty times
almost never is, and neither behaviour is written down anywhere. The draw is
seeded, because a ranking that changes per process is a ranking that cannot be
debugged.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from contextlib import contextmanager
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.perception.expected_information_gain import (
    Observation,
    Recommendation,
    Score,
    choose,
)

logger = logging.getLogger("Aura.Perception.HowSheFindsOut")

__all__ = [
    "Finding",
    "WayOfFindingOut",
    "clear_the_inventory",
    "find_out",
    "how_it_went",
    "register_a_way",
    "the_inventory",
    "what_to_look_at",
]

#: Uniform. Nothing is known about a way nobody has used, and a prior that
#: says otherwise is a claim about an instrument nobody has held.
_PRIOR = (1.0, 1.0)


@dataclass
class WayOfFindingOut:
    """One thing she can do to find something out.

    ``take`` returns the outcome it observed, as one of ``outcomes``, or None
    when it could not be made. A way that cannot say what it saw is a way that
    saw nothing, and the belief is left alone.
    """

    name: str
    #: What this way is for. A subject it does not cover is not a candidate.
    about: tuple[str, ...]
    #: What it costs, in the same unit as the value of the question.
    cost: float
    #: What it can come back with. Two at minimum: a way with one outcome
    #: cannot discriminate anything.
    outcomes: tuple[str, ...]
    take: Callable[[str], str | None]
    description: str = ""
    #: Times it was right, and times it was wrong, counted from use.
    right: int = 0
    wrong: int = 0
    #: Times it could not be made at all. Kept apart from being wrong: a
    #: sensor that is not running is not an inaccurate sensor.
    unavailable: int = 0
    last_used: float = 0.0

    @property
    def used(self) -> int:
        return self.right + self.wrong

    @property
    def reliability(self) -> float:
        """The posterior mean. What to report; not what to rank on."""
        a, b = _PRIOR
        return (a + self.right) / (a + b + self.used)

    def drawn_reliability(self, draw: Callable[[float, float], float]) -> float:
        a, b = _PRIOR
        return draw(a + self.right, b + self.wrong)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "about": list(self.about),
            "cost": self.cost,
            "outcomes": list(self.outcomes),
            "used": self.used,
            "right": self.right,
            "unavailable": self.unavailable,
            "reliability": round(self.reliability, 4),
            "description": self.description,
        }


@dataclass(frozen=True)
class Finding:
    """What happened when she went and looked."""

    subject: str
    #: The way taken, or "" when nothing was worth taking.
    way: str = ""
    #: What it came back with.
    saw: str = ""
    #: The scores, best first, so a refusal can be read as well as a choice.
    considered: tuple[Score, ...] = ()
    before: Mapping[str, float] = field(default_factory=dict)
    after: Mapping[str, float] = field(default_factory=dict)
    because: str = ""

    @property
    def looked(self) -> bool:
        return bool(self.way)

    @property
    def bits_gained(self) -> float:
        """Entropy that actually fell, which is not what was expected to."""
        from core.perception.expected_information_gain import entropy

        if not self.before or not self.after:
            return 0.0
        return entropy(self.before) - entropy(self.after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "way": self.way,
            "saw": self.saw,
            "looked": self.looked,
            "bits_gained": round(self.bits_gained, 5),
            "considered": [one.to_dict() for one in self.considered],
            "because": self.because,
        }


_INVENTORY: dict[str, WayOfFindingOut] = {}
_LOCK = threading.RLock()


def register_a_way(way: WayOfFindingOut) -> WayOfFindingOut:
    """Declare something she can do to find out. Owned by its own subsystem.

    Registration rather than a list here, so the controller never has to know
    what a screen is, and so a new sense arrives by being declared rather than
    by an edit to this file.
    """

    if len(way.outcomes) < 2:
        raise ValueError(
            f"{way.name} has {len(way.outcomes)} outcome(s); a way of finding "
            "out that can only come back one way discriminates nothing"
        )
    with _LOCK:
        held = _INVENTORY.get(way.name)
        if held is not None:
            # Re-registration keeps what was learned. A boot that forgot the
            # track record would make every way look untried every morning.
            way.right, way.wrong = held.right, held.wrong
            way.unavailable = held.unavailable
            way.last_used = held.last_used
        _INVENTORY[way.name] = way
    return way


def the_inventory(about: str = "") -> tuple[WayOfFindingOut, ...]:
    """Every way she has of finding out, or every one that covers this."""

    with _LOCK:
        ways = tuple(_INVENTORY.values())
    if not about:
        return tuple(sorted(ways, key=lambda one: one.name))
    return tuple(
        sorted(
            (one for one in ways if not one.about or about in one.about),
            key=lambda one: one.name,
        )
    )


def clear_the_inventory() -> None:
    with _LOCK:
        _INVENTORY.clear()


def _instrument(way: WayOfFindingOut, hypotheses: Sequence[str], reliability: float) -> Observation:
    """P(outcome | hypothesis), from the one thing that is measured about it.

    A way of finding out is a test: asked about a hypothesis, it comes back
    supporting it with probability r when it is true, and with probability
    (1 - r) spread over the others when it is not. That is the weakest
    instrument model that still says something, and its single parameter is
    counted rather than chosen.
    """

    others = max(1, len(way.outcomes) - 1)
    likelihoods: dict[str, dict[str, float]] = {}
    for index, outcome in enumerate(way.outcomes):
        row: dict[str, float] = {}
        for place, hypothesis in enumerate(hypotheses):
            #: Outcome i is the one this way returns when hypothesis i holds.
            agrees = (place % len(way.outcomes)) == index
            row[hypothesis] = reliability if agrees else (1.0 - reliability) / others
        likelihoods[outcome] = row
    return Observation(
        name=way.name,
        likelihoods=likelihoods,
        cost=way.cost,
        description=way.description,
    )


def what_to_look_at(
    beliefs: Mapping[str, float],
    *,
    about: str = "",
    value_per_bit: float = 1.0,
    draw: Callable[[float, float], float] | None = None,
) -> tuple[Score, ...]:
    """Rank the ways of finding out, best first. Empty when there are none."""

    ways = the_inventory(about)
    if not ways or len(beliefs) < 2:
        return ()
    sample = draw or random.Random(_seed_for(about, beliefs)).betavariate
    hypotheses = sorted(beliefs)
    observations = [
        _instrument(way, hypotheses, way.drawn_reliability(sample)) for way in ways
    ]
    return choose(beliefs, observations, value_per_bit=value_per_bit)


def _seed_for(about: str, beliefs: Mapping[str, float]) -> int:
    """The draw is random and it is not arbitrary.

    Seeded on the question, so asking the same question twice in one state
    ranks the same way twice. A ranking that changes per process is a ranking
    where one sample never settles whether a failure was yours.
    """

    return hash((str(about), tuple(sorted(beliefs)))) & 0x7FFFFFFF


def find_out(
    subject: str,
    beliefs: Mapping[str, float],
    *,
    value_per_bit: float = 1.0,
    draw: Callable[[float, float], float] | None = None,
) -> Finding:
    """The whole loop. Rank, take the best if it is worth it, update.

    Returns without looking when nothing discriminates, when the best costs
    more than the question is worth, or when there is nothing left to be
    unsure about — and says which, because "I did not look" and "there was
    nothing worth looking at" are different answers.
    """

    with _reported(subject) as spent:
        return _find_out(
            subject, beliefs, value_per_bit=value_per_bit, draw=draw, spent=spent
        )


@contextmanager
def _reported(subject: str) -> Iterator[dict[str, Any]]:
    """What this cost, told to the developmental record.

    Perception is one of the parts of her the developmental policy had no
    evidence about, so a slow or useless way of finding out could never be
    the thing it chose to fix.
    """

    try:
        from core.cognition.what_the_whole_organism_costs import while_doing
    except ImportError:
        yield {}
        return
    with while_doing("perception", f"finding out about {subject}") as said:
        yield said


def _find_out(
    subject: str,
    beliefs: Mapping[str, float],
    *,
    value_per_bit: float,
    draw: Callable[[float, float], float] | None,
    spent: dict[str, Any],
) -> Finding:
    ranked = what_to_look_at(
        beliefs, about=subject, value_per_bit=value_per_bit, draw=draw
    )
    if not ranked:
        return Finding(
            subject=subject,
            before=dict(beliefs),
            after=dict(beliefs),
            because=(
                "nothing registered can find this out"
                if len(beliefs) >= 2
                else "fewer than two hypotheses; there is nothing to tell apart"
            ),
        )
    best = next((one for one in ranked if one.take), None)
    if best is None:
        return Finding(
            subject=subject,
            considered=ranked,
            before=dict(beliefs),
            after=dict(beliefs),
            because=ranked[0].because,
        )
    with _LOCK:
        way = _INVENTORY.get(best.observation)
    if way is None:
        return Finding(
            subject=subject,
            considered=ranked,
            before=dict(beliefs),
            after=dict(beliefs),
            because=f"{best.observation} left the inventory between ranking and taking",
        )
    try:
        saw = way.take(subject)
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        # Unavailable, not wrong. Counting this against the instrument would
        # teach her that a camera nobody plugged in is an unreliable camera,
        # and the reliability she ranks on would then be measuring whether
        # the subsystem was up rather than whether the reading was right.
        logger.info("%s could not be taken: %s", way.name, exc)
        with _LOCK:
            way.unavailable += 1
        return Finding(
            subject=subject,
            considered=ranked,
            before=dict(beliefs),
            after=dict(beliefs),
            because=f"{way.name} could not be made: {exc}",
        )
    with _LOCK:
        way.last_used = time.time()
    if saw is None or saw not in way.outcomes:
        return Finding(
            subject=subject,
            way=way.name,
            considered=ranked,
            before=dict(beliefs),
            after=dict(beliefs),
            because=(
                f"{way.name} came back with {saw!r}, which is not one of the "
                "outcomes it declared; a way that cannot say what it saw saw nothing"
            ),
        )
    hypotheses = sorted(beliefs)
    updated = _after_seeing(beliefs, hypotheses, way, saw)
    spent["admitted"] = f"{way.name} said {saw}"
    return Finding(
        subject=subject,
        way=way.name,
        saw=saw,
        considered=ranked,
        before=dict(beliefs),
        after=updated,
        because=best.because,
    )


def _after_seeing(
    beliefs: Mapping[str, float],
    hypotheses: Sequence[str],
    way: WayOfFindingOut,
    saw: str,
) -> dict[str, float]:
    from core.perception.expected_information_gain import posterior

    return posterior(
        beliefs, _instrument(way, hypotheses, way.reliability).likelihoods[saw]
    )


def how_it_went(name: str, *, right: bool) -> WayOfFindingOut | None:
    """Record whether a way of finding out was actually right.

    The one number the instrument model has, and the only place it comes
    from. Called when the question later settles, not when the observation is
    made: an observation is not right because it was taken.
    """

    with _LOCK:
        way = _INVENTORY.get(str(name))
        if way is None:
            return None
        if right:
            way.right += 1
        else:
            way.wrong += 1
        return way


def snapshot() -> dict[str, Any]:
    """What she can find out, and how well each way has worked."""

    ways = the_inventory()
    return {
        "ways": len(ways),
        "measured": sum(1 for one in ways if one.used),
        "subjects": sorted({one for way in ways for one in way.about}),
        "inventory": [one.to_dict() for one in ways],
    }
