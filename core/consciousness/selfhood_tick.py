"""Drive the selfhood layers from the state a tick actually holds.

An external reader went through the consciousness layers and found the same
shape six times: instantiated, registered, exposed in a snapshot, and never
driven. `MinimalSelfhood.update()` had no caller anywhere in the tree, so
`current_state()` returned None for the life of the process while
`get_priority_bias()` handed out a zero vector that read like a measurement.
`RecursiveSelfKnowingKernel.observe_claim()` had no caller either, and
`cognitive_engine.py` reads `second_order_strength` off it — a reader with no
writer, which is how a subsystem reports healthy forever.

This is the caller. It runs on the background tick, reads the fields the tick
already carries, and drives the layers from them.

It refuses to run on made-up numbers. Every input is either read from state or
absent, and when a layer's inputs are absent the layer is not ticked — an
update assembled entirely from defaults is a fabricated observation, and those
are worse than a gap because they look like data. `readings` says which inputs
were real, so a caller can tell a quiet tick from a fictional one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SelfhoodReading", "drive_selfhood", "read_body_budget", "read_affect",
           "read_cognition"]

logger = logging.getLogger("Consciousness.SelfhoodTick")

#: Below this many real readings, a tick would be mostly defaults.
_MINIMUM_REAL_READINGS = 2


@dataclass(slots=True)
class SelfhoodReading:
    """What one drive produced, and what it was built from."""

    readings: dict[str, float] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    selfhood: dict[str, Any] = field(default_factory=dict)
    self_knowing: dict[str, Any] = field(default_factory=dict)
    skipped: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "readings": dict(self.readings),
            "missing": list(self.missing),
            "selfhood": dict(self.selfhood),
            "self_knowing": dict(self.self_knowing),
            "skipped": self.skipped,
        }


def _number(value: Any) -> float | None:
    """A float, or None when there is nothing to read."""
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN reads as absent


def _fraction(level: Any, capacity: Any) -> float | None:
    """A budget as a fraction of its own capacity."""
    have, most = _number(level), _number(capacity)
    if have is None or most is None or most <= 0.0:
        return None
    return max(0.0, min(1.0, have / most))


def read_body_budget(state: Any) -> dict[str, float]:
    """Energy, resource pressure and heat, as the tick measured them."""
    found: dict[str, float] = {}
    budgets = getattr(getattr(state, "motivation", None), "budgets", None) or {}
    energy = budgets.get("energy") if isinstance(budgets, dict) else None
    if isinstance(energy, dict):
        reserve = _fraction(energy.get("level"), energy.get("capacity"))
        if reserve is not None:
            found["energy_reserves"] = reserve
    hardware = getattr(getattr(state, "soma", None), "hardware", None) or {}
    if isinstance(hardware, dict):
        # Whichever of the two is under more pressure is the pressure.
        pressures = [
            value / 100.0
            for key in ("cpu_usage", "vram_usage", "ram_usage")
            if (value := _number(hardware.get(key))) is not None
        ]
        if pressures:
            found["resource_pressure"] = max(0.0, min(1.0, max(pressures)))
        heat = _number(hardware.get("temperature"))
        if heat is not None and heat > 0.0:
            # Reported in degrees; 90C is where a machine starts throttling.
            found["thermal_stress"] = max(0.0, min(1.0, heat / 90.0))
    return found


def read_affect(state: Any) -> dict[str, float]:
    """Coherence and curiosity, as the tick measured them."""
    found: dict[str, float] = {}
    coherence = _number(getattr(getattr(state, "cognition", None), "coherence_score", None))
    if coherence is not None:
        found["coherence"] = max(0.0, min(1.0, coherence))
    curiosity = _number(getattr(getattr(state, "affect", None), "curiosity", None))
    if curiosity is not None:
        found["curiosity"] = max(0.0, min(1.0, curiosity))
    return found


def read_cognition(state: Any) -> dict[str, float]:
    """Social hunger and prediction error, as the tick measured them."""
    found: dict[str, float] = {}
    hunger = _number(getattr(getattr(state, "affect", None), "social_hunger", None))
    if hunger is not None:
        found["social_hunger"] = max(0.0, min(1.0, hunger))
    unity = getattr(getattr(state, "cognition", None), "unity_state", None)
    error = _number(getattr(unity, "prediction_error", None))
    if error is not None:
        found["prediction_error"] = max(0.0, min(1.0, error))
    # agency_score is deliberately absent: there is no reading of it on the
    # state, and a default dressed as a measurement is the defect this module
    # exists to stop.
    return found


def _drive_minimal_selfhood(
    body: dict[str, float], affect: dict[str, float], cognitive: dict[str, float]
) -> dict[str, Any]:
    """Advance the chemotaxis layer and report what it moved to."""
    try:
        from core.consciousness.minimal_selfhood import get_minimal_selfhood

        selfhood = get_minimal_selfhood()
        state = selfhood.update(body, affect, cognitive)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation("consciousness.selfhood_tick", exc, action="selfhood not advanced")
        return {}
    return {
        "mode": getattr(getattr(state, "mode", None), "name", ""),
        "speed_scalar": round(float(getattr(state, "speed_scalar", 0.0)), 4),
        "dominant_deficit": str(getattr(state, "dominant_deficit", "")),
        "n_updates": int(getattr(state, "n_updates", 0)),
    }


def _drive_self_knowing(state: Any, readings: dict[str, float]) -> dict[str, Any]:
    """Give the second-order kernels something that actually happened.

    The claim is the phenomenal claim this tick assembled and the confidence is
    the coherence it was assembled at, both read off the tick. Nothing here is
    authored: with no claim on the state there is nothing to observe, and the
    kernels are left alone.
    """
    phenomenal = getattr(getattr(state, "cognition", None), "phenomenal_state", None)
    # The phenomenal state is a field carrying a claim, or the claim itself.
    claim = str(getattr(phenomenal, "claim", None) or phenomenal or "").strip()
    if not claim:
        return {}
    confidence = readings.get("coherence")
    if confidence is None:
        return {}
    moved: dict[str, Any] = {}
    try:
        from core.runtime.service_registry import get_runtime_service

        recursive = get_runtime_service("recursive_self_knowing", default=None)
        if recursive is not None:
            frame = recursive.observe_claim(
                claim[:400],
                confidence=float(confidence),
                evidence=tuple(f"{name}={value:.3f}" for name, value in sorted(readings.items())),
            )
            moved["recursive"] = frame.as_dict() if hasattr(frame, "as_dict") else {}
        automatic = get_runtime_service("automatic_self_knowing", default=None)
        if automatic is not None:
            frame = automatic.tick()
            moved["automatic"] = frame.as_dict() if hasattr(frame, "as_dict") else {}
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation("consciousness.selfhood_tick", exc, action="self-knowing not fed")
    return moved


def drive_selfhood(state: Any) -> SelfhoodReading:
    """Advance the selfhood layers from this tick, or say why nothing moved."""
    body = read_body_budget(state)
    affect = read_affect(state)
    cognitive = read_cognition(state)
    readings = {**body, **affect, **cognitive}
    wanted = (
        "energy_reserves", "resource_pressure", "thermal_stress", "coherence",
        "curiosity", "social_hunger", "prediction_error",
    )
    missing = tuple(name for name in wanted if name not in readings)
    if len(readings) < _MINIMUM_REAL_READINGS:
        return SelfhoodReading(
            readings=readings,
            missing=missing,
            skipped=f"only {len(readings)} of {len(wanted)} inputs were readable",
        )
    return SelfhoodReading(
        readings=readings,
        missing=missing,
        selfhood=_drive_minimal_selfhood(body, affect, cognitive),
        self_knowing=_drive_self_knowing(state, readings),
    )
