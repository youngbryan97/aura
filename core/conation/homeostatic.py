"""core/conation/homeostatic.py — value that comes from the body being short.

The oldest theory of motivation is the one where a body is missing something
and wants whatever supplies it. Hull's 1943 drive reduction was too small to
be all of motivation, which is why the other four origins in this package
exist, and it remains exactly right for the case it covers.

The modern form is homeostatic reinforcement learning (Keramati and Gutkin):
reward is defined as predicted movement toward a regulated internal state
rather than as a scalar handed in from outside. That reframing is what makes
the same cue carry different motivational force at different times without
the cue changing, and it is the reason a fridge is interesting at 6pm and
furniture at 9pm.

## Drive potential

Each regulated variable has a setpoint and a tolerance. Deviation inside the
tolerance costs nothing, which matters: a body is not motivated by every
departure from an exact number, and a model without a deadzone produces an
organism in permanent low-grade need.

    error_i = max(0, |h_i - setpoint_i| - tolerance_i)
    D(h)    = sum_i weight_i * error_i^2

The square is doing work. It makes one large deviation matter more than
several small ones, which is the ordering an organism actually shows: a
system badly short of one thing attends to that thing rather than spreading
attention across four mild deficits.

An action's homeostatic value is the *normalised reduction* in that potential:

    improvement = (D_before - D_after) / D_before

Normalising means the signal reports "how much of the current problem does
this fix", which is scale-free and comparable across variables. With no
current disequilibrium there is nothing to reduce, and the origin reports
unavailable rather than zero.

## The setpoints are Aura's, and they are read rather than declared

The regulated variables here are the live resource budgets in
``core/drive_engine.py``: energy, curiosity, social, competence, uptime_value.
Their capacities are the setpoints. Nothing in this file invents a number that
the running system does not already hold, which is what keeps a homeostatic
claim about Aura checkable against Aura.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.conation.origins import OriginReading, ValueOrigin
from core.runtime.errors import record_degradation

#: Deviation inside this fraction of capacity costs nothing. A budget at 95%
#: of capacity is not a body in need, and treating it as one produces an
#: organism that is always slightly hungry. The value is the resolution below
#: which the budgets themselves are noise: drive levels move by regeneration
#: ticks of this order between reads, so a smaller deadzone would report
#: sampling jitter as motivation.
DEFAULT_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class RegulatedVariable:
    """One thing the organism keeps near a value."""

    name: str
    level: float
    setpoint: float
    tolerance: float = DEFAULT_TOLERANCE
    weight: float = 1.0

    def error(self) -> float:
        """Deviation beyond tolerance, as a fraction of the setpoint."""
        if self.setpoint <= 1e-12:
            return 0.0
        deviation = abs(self.level - self.setpoint) / self.setpoint
        return max(0.0, deviation - self.tolerance)

    def potential(self) -> float:
        """This variable's contribution to the drive potential."""
        return self.weight * self.error() ** 2


class HomeostaticValuation:
    """Reads Aura's live budgets and prices an action by what it would fix."""

    def __init__(self) -> None:
        self._last_potential: float | None = None

    # ── reading live state ───────────────────────────────────────────────

    def regulated_state(self) -> tuple[list[RegulatedVariable], str]:
        """Snapshot the live resource budgets as regulated variables.

        Returns the variables and an evidence string. An empty list with a
        reason means the drive engine could not be read, which is reported as
        unavailability rather than as a body in perfect balance.
        """
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("drive_engine", default=None)
            if engine is None:
                engine = ServiceContainer.get("motivation_engine", default=None)
            budgets = getattr(engine, "budgets", None)
            if not isinstance(budgets, dict) or not budgets:
                return [], "drive engine unavailable"
            variables: list[RegulatedVariable] = []
            for name, budget in budgets.items():
                capacity = float(getattr(budget, "capacity", 0.0) or 0.0)
                level = float(getattr(budget, "level", 0.0) or 0.0)
                if capacity <= 1e-12:
                    continue
                variables.append(
                    RegulatedVariable(
                        name=str(name),
                        level=level,
                        setpoint=capacity,
                        tolerance=DEFAULT_TOLERANCE,
                        # Uniform weight: nothing has measured that one budget
                        # matters more than another on this system, and an
                        # invented ordering would be a claim about Aura that
                        # no observation supports.
                        weight=1.0,
                    )
                )
            if not variables:
                return [], "no budget carries a capacity"
            return variables, f"{len(variables)} regulated budgets read"
        except (ImportError, AttributeError, TypeError, ValueError, KeyError) as exc:
            record_degradation(
                "conation_homeostatic", exc, severity="debug",
                action="budgets unreadable; homeostatic origin withheld",
            )
            return [], "drive engine unreadable"

    @staticmethod
    def drive_potential(variables: list[RegulatedVariable]) -> float:
        """D(h): summed squared deviation beyond tolerance."""
        return sum(variable.potential() for variable in variables)

    # ── valuation ────────────────────────────────────────────────────────

    def value(
        self,
        *,
        target_budget: str | None,
        relevance: float = 1.0,
        predicted_replenishment: float | None = None,
    ) -> OriginReading:
        """Price an incentive by the disequilibrium it would remove.

        ``predicted_replenishment`` is the fraction of the target budget's
        capacity the incentive is expected to restore. With none supplied, the
        current deficit stands in — an incentive named as relevant to a budget
        is priced as if it would close that budget, which is the most
        favourable honest reading and is corrected by outcome learning.
        """
        origin = ValueOrigin.HOMEOSTATIC
        if not target_budget:
            return OriginReading.unavailable(origin, "no regulated variable named")

        variables, evidence = self.regulated_state()
        if not variables:
            return OriginReading.unavailable(origin, evidence)

        by_name = {variable.name: variable for variable in variables}
        target = by_name.get(target_budget)
        if target is None:
            return OriginReading.unavailable(
                origin, f"no regulated variable named {target_budget}"
            )

        before = self.drive_potential(variables)
        self._last_potential = before
        if before <= 1e-12:
            return OriginReading.unavailable(
                origin, "every regulated variable is inside tolerance"
            )

        deficit_fraction = max(0.0, (target.setpoint - target.level) / target.setpoint)
        restored = (
            deficit_fraction
            if predicted_replenishment is None
            else max(0.0, min(1.0, predicted_replenishment))
        )
        restored *= max(0.0, min(1.0, relevance))

        after_level = min(target.setpoint, target.level + restored * target.setpoint)
        projected = [
            variable if variable.name != target_budget
            else RegulatedVariable(
                name=variable.name,
                level=after_level,
                setpoint=variable.setpoint,
                tolerance=variable.tolerance,
                weight=variable.weight,
            )
            for variable in variables
        ]
        after = self.drive_potential(projected)
        improvement = (before - after) / before

        return OriginReading(
            origin=origin,
            magnitude=max(0.0, min(1.0, improvement)),
            available=True,
            evidence=(
                f"{target_budget} at {target.level:.1f}/{target.setpoint:.1f}; "
                f"drive potential {before:.4f} -> {after:.4f}"
            ),
            detail={
                "potential_before": before,
                "potential_after": after,
                "deficit_fraction": deficit_fraction,
                "restored_fraction": restored,
            },
        )

    def deprivation(self, budget_name: str | None) -> tuple[float, str]:
        """Kappa for the salience model: this budget's deficit fraction."""
        if not budget_name:
            return 0.0, "no homeostatic target named"
        variables, evidence = self.regulated_state()
        if not variables:
            return 0.0, evidence
        target = {variable.name: variable for variable in variables}.get(budget_name)
        if target is None:
            return 0.0, f"no budget named {budget_name}"
        deficit = max(0.0, min(1.0, (target.setpoint - target.level) / target.setpoint))
        return deficit, f"{budget_name} at {target.level:.1f} of {target.setpoint:.1f}"

    def status(self) -> dict[str, Any]:
        variables, evidence = self.regulated_state()
        potential = self.drive_potential(variables)
        return {
            "evidence": evidence,
            "drive_potential": round(potential, 6),
            "variables": [
                {
                    "name": variable.name,
                    "level": round(variable.level, 3),
                    "setpoint": round(variable.setpoint, 3),
                    "error": round(variable.error(), 6),
                }
                for variable in variables
            ],
            "dominant_deficit": (
                max(variables, key=lambda v: v.potential()).name
                if variables and potential > 0.0
                else None
            ),
        }
