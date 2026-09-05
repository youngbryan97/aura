"""core/science/parameter_registry.py — which constants were measured, and which were chosen.

``ActrParameters`` gets this right for fifteen numbers: each carries where it
came from, and the module records that ACT-R's latency law does not transfer to
Aura's recall rather than shipping a fitted-looking constant that was not fitted.
The rest of the repository has thousands of constants and no way to tell a
measurement from a decision.

The distinction is not pedantic. A **fitted** parameter is an estimate of
something about the world, so it has a dataset, an uncertainty, and a
sensitivity — and a claim that depends on it is only as strong as the fit. A
**policy** parameter is a choice, so it has a rationale and a cost of being
wrong, and no amount of data makes it more correct. A **derived** parameter is
neither: it follows from others, and changing it independently is a bug.

The failure this prevents has a committed instance. Three encoder widths were
run, twelve scored best, and twelve became the default described in the source
as "the measured optimum". With three arms and one campaign, best-of-three is
the expected shape of noise. Registered here, that constant is FITTED with
``n=1`` campaign and no interval, and :meth:`ParameterRegistry.unidentifiable`
lists it — which is the sentence the source should have carried.

Identifiability
---------------
A fitted parameter whose interval spans a range where the model behaves the
same way is not determined by the data. :meth:`Parameter.identifiable` asks
for an interval and a sensitivity: an interval alone says the estimate is
imprecise, and sensitivity alone says the model cares. Both together say
whether the campaign could have found a different answer.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "Kind",
    "Parameter",
    "ParameterRegistry",
    "get_parameter_registry",
    "reset_parameter_registry_for_test",
    "parameter_registry_reset",
    "UnprovenanceError",
]


class Kind(StrEnum):
    #: An estimate of something about the world. Has a dataset and an interval.
    FITTED = "fitted"
    #: A choice. Has a rationale and a cost of being wrong.
    POLICY = "policy"
    #: Follows from other parameters. Changing it alone is a bug.
    DERIVED = "derived"
    #: A hard limit that exists to bound a resource, not to be right.
    BOUND = "bound"


class UnprovenanceError(ValueError):
    """A parameter was registered without what its kind requires."""


@dataclass(frozen=True, slots=True)
class Parameter:
    """One constant, and what kind of thing it is."""

    name: str
    value: float
    kind: Kind
    owner: str
    #: FITTED: what it was fitted to, and how much of it.
    dataset: str = ""
    n: int = 0
    interval: tuple[float, float] | None = None
    #: How much the outcome moves when this moves. Without it, an interval is
    #: just imprecision, not identifiability.
    sensitivity: float | None = None
    #: POLICY: why this value, and what it costs to be wrong.
    rationale: str = ""
    cost_if_wrong: str = ""
    #: DERIVED: the parameters it follows from.
    derived_from: tuple[str, ...] = ()
    unit: str = ""

    def __post_init__(self) -> None:
        if self.kind is Kind.FITTED:
            if not self.dataset or self.n <= 0:
                raise UnprovenanceError(
                    f"{self.name!r} is FITTED and names no dataset or no sample size; "
                    "a fitted-looking constant that was not fitted is the worst kind"
                )
        elif self.kind is Kind.POLICY:
            if not self.rationale.strip():
                raise UnprovenanceError(
                    f"{self.name!r} is POLICY and gives no rationale; a chosen number "
                    "with no reason is a number nobody can argue with"
                )
        elif self.kind is Kind.DERIVED and not self.derived_from:
            raise UnprovenanceError(f"{self.name!r} is DERIVED and names nothing to derive from")

    @property
    def identifiable(self) -> bool:
        """Whether the data could have produced a different answer."""
        if self.kind is not Kind.FITTED:
            return True
        if self.interval is None or self.sensitivity is None:
            return False
        width = self.interval[1] - self.interval[0]
        scale = abs(self.value) or 1.0
        return (width / scale) < 1.0 and self.sensitivity > 0.0

    @property
    def supports_a_calibration_claim(self) -> bool:
        """Whether a claim of calibration may rest on this parameter."""
        return self.kind is Kind.FITTED and self.identifiable and self.n > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "kind": self.kind.value,
            "owner": self.owner,
            "unit": self.unit,
            "dataset": self.dataset,
            "n": self.n,
            "interval": list(self.interval) if self.interval else None,
            "sensitivity": self.sensitivity,
            "rationale": self.rationale,
            "cost_if_wrong": self.cost_if_wrong,
            "derived_from": list(self.derived_from),
            "identifiable": self.identifiable,
            "supports_a_calibration_claim": self.supports_a_calibration_claim,
        }


class ParameterRegistry:
    """Every declared constant, and whether a claim may lean on it."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.science.parameter_registry.ParameterRegistry", reentrant=True)
        self._parameters: dict[str, Parameter] = {}

    def declare(self, parameter: Parameter) -> Parameter:
        with self._lock:
            self._parameters[parameter.name] = parameter
            return parameter

    def fitted(
        self,
        name: str,
        value: float,
        *,
        owner: str,
        dataset: str,
        n: int,
        interval: tuple[float, float] | None = None,
        sensitivity: float | None = None,
        unit: str = "",
    ) -> Parameter:
        return self.declare(
            Parameter(
                name=name, value=value, kind=Kind.FITTED, owner=owner,
                dataset=dataset, n=n, interval=interval, sensitivity=sensitivity, unit=unit,
            )
        )

    def policy(
        self, name: str, value: float, *, owner: str, rationale: str,
        cost_if_wrong: str = "", unit: str = "",
    ) -> Parameter:
        return self.declare(
            Parameter(
                name=name, value=value, kind=Kind.POLICY, owner=owner,
                rationale=rationale, cost_if_wrong=cost_if_wrong, unit=unit,
            )
        )

    def get(self, name: str) -> Parameter | None:
        with self._lock:
            return self._parameters.get(name)

    def unidentifiable(self) -> list[Parameter]:
        """Fitted parameters the data could not have chosen between."""
        with self._lock:
            return sorted(
                (p for p in self._parameters.values() if p.kind is Kind.FITTED and not p.identifiable),
                key=lambda p: p.name,
            )

    def check_calibration_claim(self, parameters: Sequence[str]) -> dict[str, Any]:
        """Whether a claim of calibration may rest on these parameters.

        The gate card 010 asks for. A claim naming an unregistered parameter,
        or one that cannot support a calibration claim, does not pass.
        """
        problems: list[str] = []
        with self._lock:
            for name in parameters:
                parameter = self._parameters.get(name)
                if parameter is None:
                    problems.append(f"{name}: not registered")
                elif not parameter.supports_a_calibration_claim:
                    if parameter.kind is not Kind.FITTED:
                        problems.append(f"{name}: {parameter.kind.value}, not fitted to anything")
                    elif parameter.n <= 1:
                        problems.append(f"{name}: fitted on n={parameter.n}; best-of-n is noise")
                    else:
                        problems.append(f"{name}: fitted but not identifiable")
        return {"ok": not problems, "problems": problems}

    def report(self) -> dict[str, Any]:
        with self._lock:
            parameters = list(self._parameters.values())
        by_kind: dict[str, int] = {}
        for parameter in parameters:
            by_kind[parameter.kind.value] = by_kind.get(parameter.kind.value, 0) + 1
        return {
            "parameters": len(parameters),
            "by_kind": dict(sorted(by_kind.items())),
            "unidentifiable": [p.name for p in self.unidentifiable()],
            "can_support_calibration": sum(
                1 for p in parameters if p.supports_a_calibration_claim
            ),
        }

    def parameters(self) -> list[Parameter]:
        with self._lock:
            return sorted(self._parameters.values(), key=lambda p: p.name)


_lock = checked_lock("core.science.parameter_registry.singleton")
_registry: ParameterRegistry | None = None


def get_parameter_registry() -> ParameterRegistry:
    global _registry
    with _lock:
        if _registry is None:
            _registry = ParameterRegistry()
            _install_known(_registry)
        return _registry


def reset_parameter_registry_for_test(*, known: bool = False) -> ParameterRegistry:
    """Replace the process-wide registry. Prefer :func:`parameter_registry_reset`.

    A bare reset is permanent for the process; see core/science/singletons.py
    for the two times that has already cost a debugging session here.
    """
    global _registry
    with _lock:
        _registry = ParameterRegistry()
        if known:
            _install_known(_registry)
        return _registry


@contextlib.contextmanager
def parameter_registry_reset(*, known: bool = False) -> Iterator[ParameterRegistry]:
    """A fresh registry for the body, and the real one back afterwards."""
    import sys

    from core.science.singletons import scoped_singleton

    def _fresh() -> ParameterRegistry:
        registry = ParameterRegistry()
        if known:
            _install_known(registry)
        return registry

    with scoped_singleton(sys.modules[__name__], "_registry", _fresh, _lock) as registry:
        yield registry


def _install_known(registry: ParameterRegistry) -> None:
    """Constants this session touched, declared at what they actually are."""
    registry.policy(
        "evidence.LOOKAHEAD", 1.0,
        owner="core/evidence/packet.py",
        rationale=(
            "PLN's confidence lookahead, matched to atomspace._LOOKAHEAD so a packet and "
            "a truth value report the same confidence for the same mass. A second value "
            "here would silently fork the two scales."
        ),
        cost_if_wrong="confidence readings diverge between the two representations",
    )
    registry.policy(
        "concept_handle.LABEL_CEILING", 0.4,
        owner="core/cognition/concept_handle.py",
        rationale=(
            "A binding made by comparing two strings cannot report high confidence. The "
            "value is a ceiling rather than an estimate: it exists to make a guess "
            "readable as a guess, and any value below the SIMILARITY ceiling does that."
        ),
        cost_if_wrong="label matches either look like measurements or are ignored entirely",
    )
    registry.policy(
        "entity_track.AMBIGUITY_RATIO", 1.25,
        owner="core/cognition/entity_track.py",
        rationale=(
            "A runner-up within 25 percent of the best match makes the association a coin "
            "flip. Chosen to refuse rather than guess; a lower value guesses more often "
            "and a higher one starts more tracks than it should."
        ),
        cost_if_wrong="too low writes false histories; too high fragments real tracks",
    )
    registry.policy(
        "entity_track.PERSISTENCE_PER_SIGHTING", 2.0,
        owner="core/cognition/entity_track.py",
        unit="frames per sighting",
        rationale=(
            "Occlusion budget scales with support so a thing seen sixty times survives a "
            "longer absence than a thing seen twice. The scaling is the claim; the "
            "constant is a chosen slope and has not been fitted to any tracking data."
        ),
        cost_if_wrong="tracks are dropped through short occlusions, or kept after they leave",
    )
    registry.policy(
        "substate.DEFAULT_DEPTH", 3.0,
        owner="core/cognition/substate.py",
        unit="levels",
        rationale=(
            "Recursion needs a bound and any bound is a choice. Three is deep enough for "
            "a deadlock inside a deadlock inside a deadlock and shallow enough that an "
            "exhausted budget is reported in the same turn."
        ),
        cost_if_wrong="deeper problems report EXHAUSTED that a larger budget would solve",
    )
    registry.policy(
        "automaticity.PER_SECOND", 1.0,
        owner="core/cognition/automaticity.py",
        unit="cost per second",
        rationale=(
            "Wall clock outweighs tokens and planner expansions because the scarcest "
            "resource is the user's time. A caller measuring something else passes its own "
            "weights, and every reading carries the weights it was taken under."
        ),
        cost_if_wrong="the automaticity curve measures token thrift instead of speed",
    )
