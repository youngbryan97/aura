"""core/verify/coalition_credit.py — what one mechanism is worth in company.

:mod:`core.verify.causal_influence` answers "did this faculty change the
output?" by lesioning it alone and comparing. That is the right question and
it has one blind spot, and the blind spot is not small: **one-at-a-time
lesioning cannot tell redundancy from irrelevance.**

Two mechanisms that back each other up each read UNMEASURED when lesioned
alone — the other one covers — and the pair is essential. Two mechanisms that
only work together each read UNMEASURED alone as well, for the opposite
reason. A system with thousands of mechanisms and a one-at-a-time protocol
will report most of them as doing nothing, and will be wrong about which ones.

Coalitions fix it. A mechanism's marginal contribution is measured against
many different backgrounds — some with its partners present, some without —
and averaged. That is a Shapley value, and the three numbers it makes
available are the ones that matter:

    leave_one_out   what removing it from the whole system costs
    marginal        what it adds averaged over every background
    interaction     marginal minus leave_one_out

Interaction is the finding, and its sign has to be read carefully, because it
is the opposite of what the words suggest at first.

A mechanism with a *duplicate* costs nothing to remove — the twin covers — so
its leave-one-out is zero while its marginal is real. Interaction is positive,
and the mechanism is REDUNDANT. This is the case a one-at-a-time protocol gets
exactly wrong: both twins read as doing nothing and the pair is essential.

A mechanism that only works *with a partner* costs everything to remove, since
taking it out destroys the joint effect, while on an average background it is
worth less than that. Interaction is negative, and the mechanism is
SYNERGISTIC.

Interaction near zero means the one-at-a-time verdict was telling the truth.

Exact Shapley is exponential and this is a system with thousands of parts, so
the estimate is by permutation sampling with a reported standard error, and a
value whose interval spans zero is reported as UNMEASURED rather than as a
small number. The sampling is seeded, because a credit assignment that changed
between processes would be a ranking of the random seed.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Verify.Coalition")

#: Permutations sampled per estimate. The standard error falls as 1/sqrt(n),
#: so this is the point where another doubling buys less than the noise in the
#: trials themselves.
DEFAULT_PERMUTATIONS = 128

#: Coalitions below this and the interval will span zero for everything, which
#: is a true statement that tells nobody anything.
MIN_PERMUTATIONS = 16

#: How many standard errors away from zero a value has to sit before it is a
#: finding rather than noise. Two, which is the ordinary bar and is stated
#: here rather than left implicit in a comparison.
SIGNIFICANCE_SIGMA = 2.0

#: Interaction beyond this fraction of the solo effect is a real difference in
#: kind rather than estimator noise around the same number.
INTERACTION_FRACTION = 0.25


class Role(StrEnum):
    """What a mechanism turns out to be doing in the system."""

    #: Matters alone and in company, by about the same amount.
    INDEPENDENT = "independent"
    #: Removing it costs more than it adds on an average background: it
    #: completes something, and taking it out destroys a joint effect.
    SYNERGISTIC = "synergistic"
    #: Removing it costs less than it adds on an average background: a
    #: duplicate covers, so the whole system barely notices while the
    #: mechanism is genuinely contributing.
    REDUNDANT = "redundant"
    #: Its marginal contribution cannot be told from zero at this sample size.
    UNMEASURED = "unmeasured"
    #: Measured, and it is not contributing.
    INERT = "inert"


@dataclass(frozen=True)
class Credit:
    """What one mechanism contributed, alone and in company."""

    channel: str
    #: What removing it from the complete system costs. This is the number a
    #: one-at-a-time lesion protocol measures, and the reason it is not
    #: enough on its own.
    leave_one_out: float
    marginal: float
    standard_error: float
    permutations: int
    role: Role
    because: str

    @property
    def interaction(self) -> float:
        return self.marginal - self.leave_one_out

    @property
    def significant(self) -> bool:
        return abs(self.marginal) > SIGNIFICANCE_SIGMA * self.standard_error

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "leave_one_out": round(self.leave_one_out, 5),
            "marginal": round(self.marginal, 5),
            "interaction": round(self.interaction, 5),
            "standard_error": round(self.standard_error, 5),
            "permutations": self.permutations,
            "role": str(self.role),
            "significant": self.significant,
            "because": self.because,
        }


@dataclass(frozen=True)
class Attribution:
    """Credit across every mechanism in one run."""

    credits: tuple[Credit, ...]
    trials: int
    at: float = field(default_factory=time.time)

    def of_role(self, role: Role) -> tuple[Credit, ...]:
        return tuple(c for c in self.credits if c.role is role)

    @property
    def hidden(self) -> tuple[Credit, ...]:
        """Mechanisms a one-at-a-time protocol would have called inert.

        The finding this module exists for: a solo effect indistinguishable
        from nothing, and a marginal contribution that is not.
        """
        return tuple(
            c
            for c in self.credits
            if c.significant
            and abs(c.leave_one_out) <= SIGNIFICANCE_SIGMA * c.standard_error
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "trials": self.trials,
            "credits": [c.to_dict() for c in self.credits],
            "roles": {
                str(role): [c.channel for c in self.of_role(role)]
                for role in Role
                if self.of_role(role)
            },
            "hidden_by_one_at_a_time": [c.channel for c in self.hidden],
        }


def _classify(leave_one_out: float, marginal: float, error: float) -> tuple[Role, str]:
    if abs(marginal) <= SIGNIFICANCE_SIGMA * error:
        if error <= 0.0:
            return (Role.INERT, "no contribution, and the trials agreed exactly")
        return (
            Role.UNMEASURED,
            f"marginal {marginal:+.4f} is inside {SIGNIFICANCE_SIGMA:.0f} standard "
            f"errors of zero ({error:.4f}); more coalitions would be needed",
        )
    interaction = marginal - leave_one_out
    scale = max(abs(leave_one_out), abs(marginal))
    if scale <= 0.0 or abs(interaction) <= INTERACTION_FRACTION * scale:
        return (
            Role.INDEPENDENT,
            f"removing it costs {leave_one_out:+.4f} and it adds {marginal:+.4f} "
            "on an average background; the two agree",
        )
    if interaction > 0.0:
        return (
            Role.REDUNDANT,
            f"removing it costs only {leave_one_out:+.4f} while it adds "
            f"{marginal:+.4f} on an average background; something else covers, "
            "and a one-at-a-time lesion would call it inert",
        )
    return (
        Role.SYNERGISTIC,
        f"removing it costs {leave_one_out:+.4f} while it adds {marginal:+.4f} "
        "on an average background; it completes something",
    )


def attribute(
    channels: Sequence[str],
    value_of: Callable[[frozenset[str]], float],
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0xC0A1,
) -> Attribution:
    """Credit each mechanism by its marginal contribution over coalitions.

    ``value_of`` takes the set of channels left INTACT and returns how well the
    system did. It is called once per coalition and the results are cached, so
    the trial count is what it says rather than what the permutations imply.
    """
    names = list(dict.fromkeys(str(c) for c in channels))
    if not names:
        return Attribution(credits=(), trials=0)

    cache: dict[frozenset[str], float] = {}
    trials = 0

    def value(active: frozenset[str]) -> float:
        nonlocal trials
        if active not in cache:
            cache[active] = float(value_of(active))
            trials += 1
        return cache[active]

    everything = frozenset(names)
    nothing: frozenset[str] = frozenset()
    baseline = value(nothing)

    leave_one_out = {
        # What removing it from the complete system costs — the comparison a
        # one-at-a-time lesion protocol makes, kept so the two can be
        # contrasted rather than replaced.
        name: value(everything) - value(everything - {name})
        for name in names
    }

    rounds = max(MIN_PERMUTATIONS, int(permutations))
    contributions: dict[str, list[float]] = {name: [] for name in names}
    rng = random.Random(seed)
    order = list(names)
    exact = math.factorial(len(names)) <= rounds if len(names) <= 8 else False
    sequences: Iterable[Sequence[str]]
    if exact:
        # Small enough to enumerate. An exact answer beats a sampled one and
        # removes the standard error from the verdict entirely.
        sequences = list(itertools.permutations(names))
    else:
        sequences = []
        for _ in range(rounds):
            rng.shuffle(order)
            sequences.append(tuple(order))

    for sequence in sequences:
        running: set[str] = set()
        previous = baseline
        for name in sequence:
            running.add(name)
            current = value(frozenset(running))
            contributions[name].append(current - previous)
            previous = current

    credits: list[Credit] = []
    for name in names:
        samples = contributions[name]
        marginal = sum(samples) / len(samples) if samples else 0.0
        if len(samples) > 1:
            spread = math.sqrt(
                sum((s - marginal) ** 2 for s in samples) / (len(samples) - 1)
            )
            error = 0.0 if exact else spread / math.sqrt(len(samples))
        else:
            error = 0.0
        role, because = _classify(leave_one_out[name], marginal, error)
        credits.append(
            Credit(
                channel=name,
                leave_one_out=leave_one_out[name],
                marginal=marginal,
                standard_error=error,
                permutations=len(samples),
                role=role,
                because=because,
            )
        )
    credits.sort(key=lambda c: (-abs(c.marginal), c.channel))
    return Attribution(credits=tuple(credits), trials=trials)


def attribute_registered(
    channels: Sequence[str],
    measure: Callable[[], float],
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0xC0A1,
) -> Attribution:
    """Credit real faculties, using the lesion registry to build coalitions.

    ``measure`` runs one trial and returns how well the system did. This
    lesions everything NOT in the coalition before each call, so the value
    function is "the system with exactly these faculties intact" — which is
    what the maths needs and what a one-at-a-time protocol never assembles.

    The registry already knows how to neutralise a faculty; nothing here needs
    to know anything about the subsystems it is measuring, which is the
    property that lets this run over thousands of them.
    """
    from core.verify.lesion_registry import get_lesion_registry, lesioned

    known = set(get_lesion_registry().channels())
    usable = [c for c in channels if c in known]
    missing = [c for c in channels if c not in known]
    if missing:
        logger.warning(
            "Coalition credit skipping %d channel(s) with no registered lesion: %s",
            len(missing),
            sorted(missing)[:8],
        )
    if not usable:
        return Attribution(credits=(), trials=0)

    def value_of(active: frozenset[str]) -> float:
        off = [c for c in usable if c not in active]
        if not off:
            return float(measure())
        with lesioned(*off):
            return float(measure())

    return attribute(usable, value_of, permutations=permutations, seed=seed)


__all__ = [
    "DEFAULT_PERMUTATIONS",
    "INTERACTION_FRACTION",
    "MIN_PERMUTATIONS",
    "SIGNIFICANCE_SIGMA",
    "Attribution",
    "Credit",
    "Role",
    "attribute",
    "attribute_registered",
]
