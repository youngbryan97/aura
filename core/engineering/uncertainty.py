"""How wrong a number could be, carried alongside the number.

A design figure quoted to three decimal places from inputs known to ten per
cent is a lie told with a straight face, and it is the single most common
way an analysis misleads the person reading it. Aerospace and subsea
practice is to state the uncertainty with the value, and to say how the
uncertainty was arrived at.

Three methods are here because three are needed. First-order propagation
follows the ISO/IEC Guide 98-3 (the GUM): partial derivatives times input
uncertainties, combined in quadrature, which is exact for a linear function
and close enough for a mildly curved one. Monte Carlo follows GUM Supplement
1 and is the honest answer when the function is strongly nonlinear or an
input is far from Gaussian. Worst case is interval arithmetic, and it is
what a safety argument uses, because "unlikely" is not a defence.

A tolerance stack-up is the same machinery: the root-sum-square of
independent tolerances is what a production run actually produces, and the
arithmetic sum is what the one bad assembly looks like. Both are reported,
because designing to RSS and inspecting to worst case is how parts that
passed every check still fail to go together.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.engineering.units import Q, Quantity

__all__ = [
    "Uncertain",
    "UncertaintyBudget",
    "propagate",
    "monte_carlo",
    "worst_case",
    "rss_stack",
    "arithmetic_stack",
    "coverage_factor",
    "from_tolerance",
    "from_percentage",
    "significant_text",
]

#: Coverage factors for a normal distribution, per the GUM. k = 2 is the
#: 95% interval reported by convention in almost every engineering context.
_COVERAGE: dict[int, float] = {68: 1.0, 90: 1.645, 95: 1.960, 99: 2.576}

#: Dividing a half-width by these turns a stated limit into a standard
#: uncertainty, per GUM section 4.3: rectangular for a tolerance band with
#: no other information, triangular when the middle is more likely, and
#: normal when the limit was quoted as a confidence interval.
_DISTRIBUTION_DIVISOR: dict[str, float] = {
    "rectangular": math.sqrt(3.0),
    "uniform": math.sqrt(3.0),
    "triangular": math.sqrt(6.0),
    "normal": 1.960,
    "gaussian": 1.960,
    "u_shaped": math.sqrt(2.0),
}


def coverage_factor(confidence: int = 95) -> float:
    """The multiplier that turns a standard uncertainty into an interval."""
    return _COVERAGE.get(int(confidence), 1.960)


def _si(value: float, dimension: Any, unit: str) -> Quantity:
    """Wrap a value that is ALREADY in SI, without converting it again."""
    from core.engineering.units import Quantity as _Quantity
    from core.engineering.units import dimension_of

    return _Quantity(float(value), dimension or dimension_of(unit or "m"), unit or "m")


@dataclass(frozen=True, slots=True)
class Uncertain:
    """A value and its standard uncertainty, in SI units.

    ``standard`` is one standard uncertainty, the GUM's u(x). An expanded
    uncertainty for reporting is ``k * standard``, and :meth:`interval`
    applies the coverage factor for a stated confidence.
    """

    value: float
    standard: float = 0.0
    unit: str = ""
    source: str = ""
    distribution: str = "normal"

    @staticmethod
    def exact(value: Any, unit: str = "") -> Uncertain:
        quantity = value if isinstance(value, Quantity) else Q(value, unit)
        return Uncertain(float(quantity.value), 0.0, quantity.unit, "exact by definition")

    @property
    def relative(self) -> float:
        return abs(self.standard / self.value) if self.value else 0.0

    def quantity(self) -> Quantity:
        return Q(self.value, self.unit) if self.unit else Q(self.value)

    def interval(self, confidence: int = 95) -> tuple[float, float]:
        k = coverage_factor(confidence)
        return (self.value - k * self.standard, self.value + k * self.standard)

    def text(self, confidence: int = 95) -> str:
        """The value written the way a report writes it."""
        if self.standard <= 0:
            return self.quantity().text()
        k = coverage_factor(confidence)
        expanded = k * self.standard
        return (
            f"{self.quantity().text()} plus or minus "
            f"{Q(expanded, self.unit).text()} ({confidence}% confidence)"
        )

    def plain(self, confidence: int = 95) -> str:
        """What the uncertainty means, for a reader who is not a metrologist."""
        if self.standard <= 0:
            return "This figure is exact: it follows from the definitions, not a measurement."
        low, high = self.interval(confidence)
        return (
            f"The real value is very likely between {Q(low, self.unit).text()} and "
            f"{Q(high, self.unit).text()}. Anything more precise than that is "
            "false confidence."
        )

    def to_dict(self, confidence: int = 95) -> dict[str, Any]:
        low, high = self.interval(confidence)
        return {
            "value": self.value,
            "standard_uncertainty": self.standard,
            "relative": self.relative,
            "unit": self.unit,
            "confidence": confidence,
            "low": low,
            "high": high,
            "text": self.text(confidence),
            "plain": self.plain(confidence),
            "source": self.source,
            "distribution": self.distribution,
        }


def from_tolerance(
    value: Any,
    half_width: Any,
    *,
    unit: str = "",
    distribution: str = "rectangular",
    source: str = "",
) -> Uncertain:
    """Turn a plus-or-minus tolerance into a standard uncertainty.

    A drawing that says 10.00 plus or minus 0.05 mm has told you the limits
    and nothing about the shape inside them, so the rectangular divisor is
    the honest reading, per GUM 4.3.7.
    """
    centre = value if isinstance(value, Quantity) else Q(value, unit)
    band = half_width if isinstance(half_width, Quantity) else Q(half_width, unit or centre.unit)
    divisor = _DISTRIBUTION_DIVISOR.get(distribution.lower(), math.sqrt(3.0))
    return Uncertain(
        float(centre.value),
        abs(float(band.value)) / divisor,
        centre.unit,
        source or f"stated tolerance, treated as {distribution}",
        distribution,
    )


def from_percentage(
    value: Any, percent: float, *, unit: str = "", source: str = ""
) -> Uncertain:
    """A value known to within a percentage, the usual form for a datasheet."""
    centre = value if isinstance(value, Quantity) else Q(value, unit)
    half_width = abs(float(centre.value)) * percent / 100.0
    return Uncertain(
        float(centre.value),
        half_width / math.sqrt(3.0),
        centre.unit,
        source or f"datasheet tolerance of {percent:g}%",
        "rectangular",
    )


@dataclass(frozen=True, slots=True)
class UncertaintyBudget:
    """Every input's contribution to the uncertainty in a result.

    This is the table an audit asks for: which input dominates, and by how
    much. It is the difference between "the answer is uncertain" and "the
    answer is uncertain because the wall thickness tolerance is loose".
    """

    result: Uncertain
    contributions: tuple[tuple[str, float, float], ...] = ()
    method: str = ""

    def dominant(self) -> str:
        if not self.contributions:
            return ""
        return max(self.contributions, key=lambda row: abs(row[2]))[0]

    def plain(self) -> str:
        if not self.contributions or self.result.standard <= 0:
            return self.result.plain()
        name = self.dominant()
        share = 0.0
        total = sum(row[2] ** 2 for row in self.contributions)
        if total > 0:
            share = max(row[2] ** 2 for row in self.contributions) / total
        return (
            f"{self.result.plain()} Most of that spread — {share * 100:.0f}% of it — "
            f"comes from how well {name} is known, so that is the one to pin down first."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result.to_dict(),
            "method": self.method,
            "dominant_input": self.dominant(),
            "contributions": [
                {
                    "input": name,
                    "sensitivity": sensitivity,
                    "contribution": contribution,
                }
                for name, sensitivity, contribution in self.contributions
            ],
        }


def propagate(
    function: Callable[..., float],
    inputs: Mapping[str, Uncertain],
    *,
    unit: str = "",
    relative_step: float = 1e-6,
) -> UncertaintyBudget:
    """First-order uncertainty propagation, the GUM's law of propagation.

    Partial derivatives are taken by central difference, so any function of
    the inputs works without anybody writing derivatives by hand. The
    contributions are kept per input, which is what makes the result
    actionable rather than merely honest.
    """
    names = list(inputs)
    centre = {name: inputs[name].value for name in names}
    base = float(function(**centre))
    contributions: list[tuple[str, float, float]] = []
    variance = 0.0
    for name in names:
        entry = inputs[name]
        if entry.standard == 0.0:
            contributions.append((name, 0.0, 0.0))
            continue
        step = abs(entry.value) * relative_step or relative_step
        high = dict(centre)
        low = dict(centre)
        high[name] = entry.value + step
        low[name] = entry.value - step
        try:
            sensitivity = (float(function(**high)) - float(function(**low))) / (2.0 * step)
        except (ValueError, ZeroDivisionError, ArithmeticError):
            sensitivity = 0.0
        contribution = sensitivity * entry.standard
        contributions.append((name, sensitivity, contribution))
        variance += contribution * contribution
    result = Uncertain(
        base,
        math.sqrt(variance),
        unit,
        "first-order propagation, ISO/IEC Guide 98-3 (GUM)",
    )
    return UncertaintyBudget(result, tuple(contributions), "GUM first-order")


def monte_carlo(
    function: Callable[..., float],
    inputs: Mapping[str, Uncertain],
    *,
    unit: str = "",
    trials: int = 20000,
    seed: int = 20260824,
) -> UncertaintyBudget:
    """Monte Carlo propagation, GUM Supplement 1.

    Used when the function bends enough that a first-order estimate is
    wrong, which is any time a term is squared, inverted or raised to a
    fractional power over a wide input range. The seed is fixed so two runs
    of the same design give the same interval; an uncertainty that moves
    between runs cannot be reviewed.
    """
    rng = random.Random(seed)
    names = list(inputs)
    samples: list[float] = []
    for _ in range(int(trials)):
        draw = {}
        for name in names:
            entry = inputs[name]
            if entry.standard == 0.0:
                draw[name] = entry.value
            elif entry.distribution in {"rectangular", "uniform"}:
                half = entry.standard * math.sqrt(3.0)
                draw[name] = rng.uniform(entry.value - half, entry.value + half)
            elif entry.distribution == "triangular":
                half = entry.standard * math.sqrt(6.0)
                draw[name] = rng.triangular(entry.value - half, entry.value + half, entry.value)
            else:
                draw[name] = rng.gauss(entry.value, entry.standard)
        try:
            samples.append(float(function(**draw)))
        except (ValueError, ZeroDivisionError, ArithmeticError):
            continue
    if not samples:
        return UncertaintyBudget(Uncertain(0.0, 0.0, unit, "no valid samples"), (), "monte carlo")
    mean = sum(samples) / len(samples)
    variance = sum((value - mean) ** 2 for value in samples) / max(len(samples) - 1, 1)
    return UncertaintyBudget(
        Uncertain(mean, math.sqrt(variance), unit, f"Monte Carlo, {len(samples)} trials, GUM S1"),
        (),
        "monte carlo",
    )


def worst_case(
    function: Callable[..., float],
    inputs: Mapping[str, Uncertain],
    *,
    confidence: int = 95,
) -> tuple[float, float]:
    """The interval the result cannot leave, from the input limits.

    Every combination of input extremes is evaluated, which is exponential
    in the input count and therefore capped: past twelve inputs the corner
    enumeration is replaced by a sensitivity-signed bound, which is exact
    for a monotonic function and conservative otherwise.
    """
    names = list(inputs)
    k = coverage_factor(confidence)
    limits = {
        name: (inputs[name].value - k * inputs[name].standard,
               inputs[name].value + k * inputs[name].standard)
        for name in names
    }
    if len(names) <= 12:
        low = math.inf
        high = -math.inf
        for corner in range(2 ** len(names)):
            draw = {
                name: limits[name][(corner >> index) & 1]
                for index, name in enumerate(names)
            }
            try:
                value = float(function(**draw))
            except (ValueError, ZeroDivisionError, ArithmeticError):
                continue
            low = min(low, value)
            high = max(high, value)
        if low <= high:
            return (low, high)
    budget = propagate(function, inputs)
    span = sum(abs(contribution) * k for _n, _s, contribution in budget.contributions)
    return (budget.result.value - span, budget.result.value + span)


def rss_stack(tolerances: Sequence[Any], *, unit: str = "") -> Quantity:
    """The statistical stack-up: what a production run actually produces.

    Independent tolerances combine in quadrature, so five parts each held to
    a tenth of a millimetre stack to about a quarter, not half. Designing to
    this figure is standard practice and is why assemblies fit.
    """
    total = 0.0
    dimension = None
    for entry in tolerances:
        quantity = entry if isinstance(entry, Quantity) else Q(entry, unit)
        dimension = quantity.dimension
        total += float(quantity.value) ** 2
    # Built from the SI magnitude directly. Passing it back through Q with
    # the display unit converted it a second time, so a 0.03 and a 0.04
    # millimetre tolerance stacked to 5e-8 metres instead of 0.05.
    return _si(math.sqrt(total), dimension, unit)


def arithmetic_stack(tolerances: Sequence[Any], *, unit: str = "") -> Quantity:
    """The worst-case stack-up: the one assembly where every part is at its limit.

    Larger than the RSS figure and much rarer, and it is the number a safety
    argument or an interference check has to use, because a rare assembly is
    still an assembly somebody receives.
    """
    total = 0.0
    dimension = None
    for entry in tolerances:
        quantity = entry if isinstance(entry, Quantity) else Q(entry, unit)
        dimension = quantity.dimension
        total += abs(float(quantity.value))
    return _si(total, dimension, unit)


def significant_text(value: Uncertain) -> str:
    """The value rounded to the digits its uncertainty can support.

    Quoting more digits than the uncertainty justifies is the visible symptom
    of a number nobody checked, so this trims to two significant figures of
    uncertainty, which is what the GUM recommends for reporting.
    """
    if value.standard <= 0 or value.value == 0:
        return value.quantity().text()
    magnitude = math.floor(math.log10(abs(value.standard)))
    places = max(0, -(magnitude - 1))
    rounded = round(value.value, places)
    uncertainty = round(value.standard, places)
    unit = f" {value.unit}" if value.unit else ""
    return f"{rounded:.{places}f} plus or minus {uncertainty:.{places}f}{unit}"
