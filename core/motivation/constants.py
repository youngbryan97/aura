from __future__ import annotations

from typing import Dict


SECONDS_PER_DAY = 86400.0


def _per_day(value: float) -> float:
    return float(value) / SECONDS_PER_DAY


MOTIVATION_BUDGET_DEFAULTS: Dict[str, Dict[str, float]] = {
    "energy": {"level": 100.0, "capacity": 100.0, "decay": 0.0},
    "curiosity": {"level": 80.0, "capacity": 100.0, "decay": _per_day(2.0)},
    "social": {"level": 90.0, "capacity": 100.0, "decay": _per_day(1.5)},
    "integrity": {"level": 95.0, "capacity": 100.0, "decay": _per_day(0.5)},
    "growth": {"level": 50.0, "capacity": 100.0, "decay": _per_day(0.75)},
}


def clone_motivation_budget_defaults() -> Dict[str, Dict[str, float]]:
    return {name: dict(values) for name, values in MOTIVATION_BUDGET_DEFAULTS.items()}
