"""core/cognition/drive_sensitivity.py — how much each drive should matter, learned.

Aura's drives are richer than most architectures': curiosity, welfare,
coherence, care, and the rest, each with a weight that decides how strongly it
pulls on goal generation. Every one of those weights was chosen by hand. That
is fine as a starting point and wrong as a permanent arrangement, because the
right weight depends on what the drive has actually been buying, and that
changes with what she is doing.

The sensitivity of a drive is the derivative nobody computes: how much outcome
moves per unit of drive. It is estimated the only honest way available, from
occasions where the drive's level varied and the outcome was observed. A drive
whose level never varies has no measurable sensitivity, and the estimator says
so rather than returning zero - "no effect" and "never tested" are different
findings and collapsing them is how a drive gets quietly turned off.

The lesion arm
--------------
:meth:`DriveModel.lesion_effect` compares outcomes with the drive suppressed
against outcomes with it free. A sensitivity computed from natural variation
can be confounded by whatever moved the drive; the lesion cannot. Both are
reported, and when they disagree the natural estimate is the suspect one.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["Observation", "DriveModel", "DriveSensitivities"]

#: Distinct drive levels needed before a sensitivity is estimated. Two points
#: fit a line through anything.
MIN_LEVELS = 4


@dataclass(frozen=True, slots=True)
class Observation:
    """One occasion: what the drive was at, and how it went."""

    level: float
    outcome: float
    lesioned: bool = False


@dataclass
class DriveModel:
    """One drive, its observed levels, and what they bought."""

    name: str
    hand_weight: float
    observations: list[Observation] = field(default_factory=list)

    def observe(self, level: float, outcome: float, *, lesioned: bool = False) -> None:
        self.observations.append(Observation(level, outcome, lesioned))

    @property
    def free(self) -> list[Observation]:
        return [o for o in self.observations if not o.lesioned]

    def sensitivity(self) -> dict[str, Any]:
        """Outcome per unit of drive, from occasions where the level varied."""
        free = self.free
        levels = {round(o.level, 6) for o in free}
        if len(levels) < MIN_LEVELS:
            return {
                "measurable": False,
                "distinct_levels": len(levels),
                "reason": (
                    "this drive's level has not varied enough to estimate a slope; "
                    "no effect and never tested are different findings"
                ),
            }
        xs = [o.level for o in free]
        ys = [o.outcome for o in free]
        mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        slope = 0.0 if denominator == 0 else sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
        ) / denominator
        return {
            "measurable": True,
            "slope": slope,
            "distinct_levels": len(levels),
            "n": len(free),
            "direction": "helps" if slope > 0 else "hurts" if slope < 0 else "flat",
        }

    def lesion_effect(self) -> dict[str, Any]:
        """Outcome with the drive suppressed against outcome with it free."""
        lesioned = [o.outcome for o in self.observations if o.lesioned]
        free = [o.outcome for o in self.free]
        if not lesioned or not free:
            return {"measurable": False, "reason": "both arms are needed"}
        effect = statistics.fmean(free) - statistics.fmean(lesioned)
        return {
            "measurable": True,
            "effect": effect,
            "free_mean": statistics.fmean(free),
            "lesioned_mean": statistics.fmean(lesioned),
            "n_free": len(free),
            "n_lesioned": len(lesioned),
        }

    def learned_weight(self) -> dict[str, Any]:
        """What the weight should be, and whether the two estimates agree."""
        natural = self.sensitivity()
        lesion = self.lesion_effect()
        if not natural["measurable"]:
            return {
                "name": self.name, "hand_weight": self.hand_weight,
                "learned_weight": self.hand_weight, "source": "unmeasured",
                "reason": natural["reason"],
            }
        agree = (
            not lesion["measurable"]
            or (natural["slope"] > 0) == (lesion["effect"] > 0)
        )
        return {
            "name": self.name,
            "hand_weight": self.hand_weight,
            "learned_weight": max(0.0, natural["slope"]),
            "source": "natural_variation" if not lesion["measurable"] else "confirmed_by_lesion",
            "estimates_agree": agree,
            "reason": (
                ""
                if agree
                else "the natural estimate and the lesion disagree; the natural one is "
                "confounded by whatever moved the drive and should not be trusted"
            ),
        }


class DriveSensitivities:
    """Every drive, its hand weight, and what the evidence says it should be."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.cognition.drive_sensitivity.DriveSensitivities", reentrant=True)
        self._drives: dict[str, DriveModel] = {}

    def drive(self, name: str, *, hand_weight: float = 1.0) -> DriveModel:
        with self._lock:
            return self._drives.setdefault(name, DriveModel(name=name, hand_weight=hand_weight))

    def report(self) -> dict[str, Any]:
        with self._lock:
            drives = list(self._drives.values())
        rows = [d.learned_weight() for d in drives]
        return {
            "drives": len(rows),
            "measured": sum(1 for r in rows if r["source"] != "unmeasured"),
            "never_varied": [r["name"] for r in rows if r["source"] == "unmeasured"],
            "disagreeing": [r["name"] for r in rows if r.get("estimates_agree") is False],
            "would_reweight": [
                r["name"] for r in rows
                if r["source"] != "unmeasured" and abs(r["learned_weight"] - r["hand_weight"]) > 0.1
            ],
            "by_drive": {r["name"]: r for r in rows},
        }
