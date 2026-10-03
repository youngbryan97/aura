"""Whether a loop settles, read from its equations before it runs.

:mod:`core.verify.dynamics` runs a loop for hundreds of ticks and looks at the
trajectory. That finds a defect after it has happened, on the inputs the run
happened to use. A loop whose update is a contraction can be judged from the
update alone: every step shrinks the largest difference between two states by
a factor below one, so there is exactly one resting point, and how far a state
is from it is bounded by how far the last step moved it (the Banach
fixed-point theorem). The idea is adapted from the settling certificate of
Cadence (muellerberndt/cadence, docs/certificate.md); nothing of its code is
used.

Aura has had this defect. On 16 September the mood baseline moved toward the
feeling and the feeling decayed toward the baseline. As a matrix the pair had
an eigenvalue of exactly one, so a steady input climbed for 1,200 turns
without levelling. A certificate would have read rate 1.0 off the two
constants.

Two forms are certified here:

* a linear update ``x <- A x + b``: the rate is the spectral radius of ``A``.
  Below one the loop converges from anywhere; the rate is asymptotic;
* a leaky recurrent update ``x <- (1 - leak dt) x + dt g(W x + c)`` with ``g``
  of slope at most ``L``: the rate is ``1 - dt (leak - L rho)``, ``rho`` the
  largest absolute row sum of ``W``. Clipping the state to a box does not
  enlarge it. Noise added to a step leaves the rate as it is.

An organ registers a provider that builds its certificate from its live
parameters, and says whether it must settle. A reservoir kept near the edge of
chaos on purpose is reported and not required to contract; a feeling that is
meant to return to rest is required to. The invariant fails only for an organ
that must settle and does not.

Pure and offline: no imports from the organs certified (see core/verify/DEPS).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.verify.invariants import Violation, invariant

__all__ = [
    "SettlingCertificate",
    "leaky_recurrent_certificate",
    "linear_certificate",
    "register_settling",
    "settling_report",
]


@dataclass(frozen=True)
class SettlingCertificate:
    """How fast one loop forgets where it started."""

    name: str
    #: Factor by which a step shrinks the distance to the resting point.
    rate: float
    #: "spectral" (asymptotic, any norm) or "row_mass" (every step, max norm).
    basis: str
    must_settle: bool
    detail: str = ""

    @property
    def certified(self) -> bool:
        return math.isfinite(self.rate) and self.rate < 1.0

    @property
    def half_life_steps(self) -> float:
        """Steps for the distance to the resting point to halve."""
        if not self.certified:
            return math.inf
        if self.rate <= 0.0:
            return 0.0
        return math.log(2.0) / -math.log(self.rate)

    def remaining_distance(self, last_step_movement: float) -> float:
        """Bound on the distance to the resting point after a step that moved this much."""
        if not self.certified or self.basis != "row_mass":
            return math.inf
        return abs(float(last_step_movement)) * self.rate / (1.0 - self.rate)

    def steps_within(self, distance: float, tolerance: float) -> int:
        """Steps until a state ``distance`` from rest is within ``tolerance`` of it."""
        if not self.certified:
            raise ValueError(f"{self.name} is not certified to settle")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if distance <= tolerance or self.rate <= 0.0:
            return 0
        return math.ceil(math.log(distance / tolerance) / -math.log(self.rate))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rate": self.rate,
            "basis": self.basis,
            "certified": self.certified,
            "must_settle": self.must_settle,
            "half_life_steps": self.half_life_steps,
            "detail": self.detail,
        }


def linear_certificate(name: str, matrix: Any, *, must_settle: bool) -> SettlingCertificate:
    """Certificate for ``x <- A x + b`` from the spectral radius of ``A``."""
    a = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    if a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        return SettlingCertificate(name, math.inf, "spectral", must_settle, "matrix not square and finite")
    radius = float(np.max(np.abs(np.linalg.eigvals(a)))) if a.size else 0.0
    return SettlingCertificate(name, radius, "spectral", must_settle, f"spectral radius of a {a.shape[0]}x{a.shape[0]} update")


def leaky_recurrent_certificate(
    name: str,
    weights: Any,
    *,
    leak: float,
    dt: float,
    slope: float = 1.0,
    must_settle: bool,
) -> SettlingCertificate:
    """Certificate for ``x <- (1 - leak dt) x + dt g(W x + c)`` with ``|g'| <= slope``."""
    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    if not np.all(np.isfinite(w)) or not 0.0 < leak * dt <= 1.0 or slope < 0:
        return SettlingCertificate(name, math.inf, "row_mass", must_settle, "leak, step or weights out of range")
    row_mass = float(np.max(np.sum(np.abs(w), axis=1))) if w.size else 0.0
    rate = 1.0 - dt * (leak - slope * row_mass)
    limit = leak / slope if slope > 0 else math.inf
    return SettlingCertificate(
        name, rate, "row_mass", must_settle, f"row mass {row_mass:.4g} against a limit of {limit:.4g}"
    )


_providers: dict[str, Callable[[], SettlingCertificate | None]] = {}
_providers_lock = threading.Lock()


def register_settling(name: str, provider: Callable[[], SettlingCertificate | None]) -> None:
    """Register an organ's certificate, built from its live parameters when asked for."""
    with _providers_lock:
        _providers[name] = provider


def settling_report() -> list[SettlingCertificate]:
    """Every registered loop's certificate; a provider that fails reports an uncertified one."""
    with _providers_lock:
        providers = dict(_providers)
    report = []
    for name, provider in sorted(providers.items()):
        try:
            certificate = provider()
        except (ArithmeticError, AttributeError, LookupError, TypeError, ValueError) as exc:
            certificate = SettlingCertificate(name, math.inf, "unknown", True, f"provider failed: {exc}")
        if certificate is not None:
            report.append(certificate)
    return report


def _uncertified(certificates: Sequence[SettlingCertificate]) -> Iterator[Violation]:
    for certificate in certificates:
        if certificate.must_settle and not certificate.certified:
            yield Violation(
                subject=certificate.name,
                message=(
                    f"must return to rest, but a step shrinks its distance from rest by "
                    f"{certificate.rate:.6g} ({certificate.basis}; {certificate.detail})"
                ),
                remedy="add a leak toward the declared rest, or lower the gain that feeds the loop back",
            )


@invariant(
    "dynamics.declared_settling",
    scope="dynamics",
    owner="core/verify/settling.py",
    description="A loop declared to return to rest has an update that contracts.",
)
def _declared_loops_settle() -> Iterator[Violation]:
    yield from _uncertified(settling_report())
