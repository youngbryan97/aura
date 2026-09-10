"""core/connectome/island.py — a piece of human cortex, wired the way it was measured.

Operationally: takes the three numbers H01 published about how strongly two
connected cells are joined, fits the candidate laws that could have produced
them, tests each against the number it was not fitted to, and hands the
surviving law to whatever needs to wire something.

What H01 measured
-----------------
A cubic millimetre of human temporal cortex, 57,000 cells and 150 million
synapses, and within it every pair of connected cells and how many contacts
join them. Three numbers come out of that and they are the whole of what this
module rests on:

    96.5%     of connected pairs are joined by exactly one contact
    0.092%    by four or more
    about 50  contacts in the heaviest pair observed

Everything else here is derived, and the derivation is falsifiable.

Why an island rather than a mesh
---------------------------------
Her mesh wires 4,096 units by drawing a Gaussian weight for every pair a coin
flip connects. That is a modelling convention: it says a connection's strength
is a continuous quantity centred on zero, symmetric, with no heavy tail. What
H01 says is that strength is a COUNT — how many times one cell touches another
— that almost always comes out at one and occasionally comes out at fifty.
Those are different objects, and only one of them was measured in a person.

An island is a piece of the mesh wired from the measurement instead of the
convention. Naming it an island rather than converting the whole mesh is
deliberate: what H01 measured is local wiring in one cortical volume, and
claiming it for a network whose long-range structure came from nowhere near a
microscope would be borrowing its authority.

What the two laws disagree about
--------------------------------
Fit a truncated power law to the single-contact fraction alone and it predicts
0.146% of pairs at four or more, against the measured 0.092% — too heavy by
half. Add an exponential cutoff and both fractions come out exactly right,
because two parameters against two numbers is not a prediction.

The third number separates them. Under the cutoff, a fifty-contact pair is
impossible: about three in a hundred million of them expected across the whole
volume. Under the plain power law about one such pair is expected, which is what
H01 saw. So the tail is scale-free out to the heaviest pair, the cutoff that fit
the middle of the distribution is refuted by its end, and the law this module
hands out is the one that was fitted to less and survived more.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import H01_REFERENCE

logger = logging.getLogger("Aura.Connectome.Island")

__all__ = [
    "MAX_CONTACTS",
    "REFUTED_BELOW",
    "MultiplicityLaw",
    "candidate_laws",
    "contacts_per_pair",
    "expected_heaviest_pair",
    "probability_of_the_observed_maximum",
    "surviving_law",
    "wire_island",
]

#: The largest contact count the fit ranges over. H01's heaviest observed pair,
#: so the support is the support that was seen rather than a round number.
MAX_CONTACTS = int(H01_REFERENCE.get("max_observed_contacts") or 50)

#: How many connected pairs the volume holds, from the synapse count and the
#: mean contacts per pair. Needed to ask how often the heaviest pair should
#: turn up.
_SYNAPSES = float(H01_REFERENCE.get("synapses") or 150_000_000.0)

#: How improbable the observed maximum has to be under a law before that law is
#: refuted by it. One in a hundred: a volume is one sample, and a law that makes
#: what was sampled rarer than this did not produce it.
REFUTED_BELOW = 0.01


@dataclass(frozen=True, slots=True)
class MultiplicityLaw:
    """A candidate answer to "how many contacts join two connected cells?"."""

    name: str
    #: What it was fitted to. A law fitted to everything predicts nothing.
    fitted_to: tuple[str, ...]
    parameters: dict[str, float]
    #: P(k) for k = 1 .. MAX_CONTACTS.
    probabilities: tuple[float, ...]

    def p(self, contacts: int) -> float:
        if contacts < 1 or contacts > len(self.probabilities):
            return 0.0
        return self.probabilities[contacts - 1]

    def p_at_least(self, contacts: int) -> float:
        return sum(self.probabilities[max(0, contacts - 1) :])

    def mean(self) -> float:
        return sum(
            (index + 1) * probability for index, probability in enumerate(self.probabilities)
        )


def _normalise(weights: Sequence[float]) -> tuple[float, ...]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("multiplicity_weights_do_not_sum")
    return tuple(weight / total for weight in weights)


def _power_law(exponent: float) -> tuple[float, ...]:
    return _normalise([k ** (-exponent) for k in range(1, MAX_CONTACTS + 1)])


def _power_law_with_cutoff(exponent: float, cutoff: float) -> tuple[float, ...]:
    return _normalise(
        [k ** (-exponent) * math.exp(-k / cutoff) for k in range(1, MAX_CONTACTS + 1)]
    )


def _solve(f, low: float, high: float, *, tolerance: float = 1e-12, steps: int = 200) -> float:
    """Bisection. Written out rather than imported: this package must not need
    a numerical stack to say what it measured."""
    f_low = f(low)
    if f_low == 0.0:
        return low
    for _ in range(steps):
        middle = (low + high) / 2.0
        f_middle = f(middle)
        if f_middle == 0.0 or (high - low) / 2.0 < tolerance:
            return middle
        if (f_middle > 0) == (f_low > 0):
            low, f_low = middle, f_middle
        else:
            high = middle
    return (low + high) / 2.0


def candidate_laws() -> tuple[MultiplicityLaw, ...]:
    """Every law that could have produced the single-contact fraction.

    The first is fitted to one number and predicts the second. The second is
    fitted to both and predicts neither, which is the point of listing them
    together.
    """
    single = float(H01_REFERENCE.get("single_contact_fraction") or 0.965)
    four_plus = float(H01_REFERENCE.get("four_or_more_contact_fraction") or 0.00092)

    exponent = _solve(lambda s: _power_law(s)[0] - single, 1.0, 40.0)
    plain = MultiplicityLaw(
        name="power law",
        fitted_to=("single_contact_fraction",),
        parameters={"exponent": exponent},
        probabilities=_power_law(exponent),
    )

    # Two constraints, two parameters. Solve the cutoff for the four-or-more
    # fraction with the exponent re-solved for the single-contact fraction at
    # each step, so both hold exactly rather than approximately.
    def _with_cutoff(cutoff: float) -> tuple[float, ...]:
        inner = _solve(
            lambda s: _power_law_with_cutoff(s, cutoff)[0] - single, 0.1, 40.0
        )
        return _power_law_with_cutoff(inner, cutoff)

    cutoff = _solve(lambda c: sum(_with_cutoff(c)[3:]) - four_plus, 1.2, 40.0)
    inner_exponent = _solve(
        lambda s: _power_law_with_cutoff(s, cutoff)[0] - single, 0.1, 40.0
    )
    damped = MultiplicityLaw(
        name="power law with an exponential cutoff",
        fitted_to=("single_contact_fraction", "four_or_more_contact_fraction"),
        parameters={"exponent": inner_exponent, "cutoff": cutoff},
        probabilities=_power_law_with_cutoff(inner_exponent, cutoff),
    )
    return (plain, damped)


def connected_pairs(law: MultiplicityLaw, synapses: float = _SYNAPSES) -> float:
    """How many connected pairs the volume holds, under this law."""
    average = law.mean()
    return synapses / average if average > 0 else 0.0


def expected_heaviest_pair(law: MultiplicityLaw) -> float:
    """How many pairs as heavy as the heaviest one H01 saw this law expects.

    The test neither law was fitted to. A law that expects none of them cannot
    have produced a volume containing one.
    """
    return connected_pairs(law) * law.p_at_least(MAX_CONTACTS)


def probability_of_the_observed_maximum(law: MultiplicityLaw) -> float:
    """How often a volume drawn from this law would contain a pair that heavy.

    Poisson in the expected count, which is the right shape: the pairs are many
    and each is individually unlikely to be the heavy one.
    """
    expected = expected_heaviest_pair(law)
    if expected <= 0.0:
        return 0.0
    return 1.0 - math.exp(-expected)


def surviving_law() -> MultiplicityLaw:
    """The law under which the heaviest pair H01 saw is not a freak.

    Neither law was fitted to the observed maximum, so both are exposed to it.
    Under the plain power law a volume this size contains a fifty-contact pair
    a third of the time; under the cutoff, once in twenty-three million. The
    second is refuted by the volume it was fitted to.
    """
    survivors = [
        law
        for law in candidate_laws()
        if probability_of_the_observed_maximum(law) >= REFUTED_BELOW
    ]
    if not survivors:
        # Never silently pick one. If nothing expects what was seen, the family
        # is wrong and saying so is the finding.
        raise ValueError("no_multiplicity_law_expects_the_observed_maximum")
    # Fewest fitted constraints first: a law that predicted what it was not
    # given is worth more than one that was handed the answer.
    return min(survivors, key=lambda law: len(law.fitted_to))


def contacts_per_pair(law: MultiplicityLaw, size: int, rng: Any) -> Any:
    """Draw a contact count for each of `size` connected pairs."""
    import numpy as np

    return rng.choice(
        np.arange(1, len(law.probabilities) + 1),
        size=size,
        p=np.asarray(law.probabilities, dtype=np.float64),
    )


def wire_island(
    units: int,
    density: float,
    rng: Any,
    *,
    law: MultiplicityLaw | None = None,
    contact_strength: float = 0.1,
) -> Any:
    """A weight matrix whose strengths are contact counts, not Gaussian draws.

    `density` decides which pairs are connected at all — H01 says nothing about
    that, and pretending otherwise would be the borrowing this module exists to
    avoid. What it decides is the strength of the pairs that are connected: a
    count drawn from the measured law, scaled so that a single contact is worth
    `contact_strength`.

    Magnitudes only, all positive. Whether a connection excites or inhibits is
    a fact about the cell sending it and not about how many times it touches
    what it sends to, so the sign belongs to whatever applies Dale's law. An
    earlier version signed each connection with a coin flip here, which put
    both signs in one cell's output and left the caller with nothing to apply
    Dale's law to.

    The visible difference from the mesh's own wiring is the tail. A Gaussian
    gives a connection thirty times the median about never; this gives it to
    roughly one connection in a hundred thousand, which is what was measured in
    a person.
    """
    import numpy as np

    if units <= 0:
        raise ValueError("island_needs_units")
    law = law or surviving_law()
    connected = rng.random((units, units)) < density
    np.fill_diagonal(connected, False)
    weights = np.zeros((units, units), dtype=np.float32)
    count = int(connected.sum())
    if count:
        contacts = contacts_per_pair(law, count, rng).astype(np.float32)
        weights[connected] = contacts * float(contact_strength)
    return weights
