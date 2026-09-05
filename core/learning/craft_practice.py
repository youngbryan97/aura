"""core/learning/craft_practice.py — getting better at something that pushes back.

Making things by hand is not a hobby version of manufacturing. Richard Sennett's
account of it turns on two words: resistance and ambiguity. The material does
not do what you meant. It does something adjacent, and the difference is the
only information you get about what you actually did. Learning a craft is
learning to read that difference, and it cannot be learned any other way,
because the map from what you do to how it comes out is not written down
anywhere.

That is a precise statement about an optimisation problem. There is a quality
surface over the things you could do. You cannot see it, you cannot
differentiate it, each evaluation costs real time and material, and the
evaluation is noisy. Under exactly those conditions the correct method is
simultaneous perturbation: try it slightly differently in every respect at
once, twice, and take the difference. Spall's algorithm needs two evaluations
per step no matter how many parameters there are, which is why a potter
learning fifteen things about their hands at once is not doing something
mysterious.

Three properties separate this from executing a known procedure, and all three
are here because taking any of them out leaves something that still runs.

**There is a model of the material, and outcomes correct it.** Without one
this is repetition. The estimate that moves is the resistance — how hard the
material pushes back per unit of what you did — and it scales the step, so
that a material that fights is approached carefully and a forgiving one is
explored fast.

**Practice is scheduled by what is improving, not by what is needed.** This is
the whole of "for its own sake" as a mechanism. A scheduler that picks the
skill the current task requires stops at sufficiency and never gets past
competent. One that picks the skill whose quality is rising fastest keeps
going after the requirement is met, which is the behaviour, and it is not
sentiment — the derivative is a real quantity and the ranking on it is a real
ranking. ``practice_target`` does the second thing, and ``required_target``
does the first, so an ablation can swap them and measure what changes.

**Difficulty is chosen to sit near the edge of what can be done.**
Csikszentmihalyi's condition for flow is a challenge close to the skill:
comfortably under it is boring, far over it is not practice but failure. The
ratio is computed, the next difficulty is set from it, and both are reported.
"""

from __future__ import annotations

import logging
import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Learning.Craft")

#: Spall's recommended decay exponents for simultaneous perturbation. They are
#: his, from the asymptotic convergence conditions, and not fitted here.
STEP_DECAY = 0.602
PERTURBATION_DECAY = 0.101

#: How many attempts the improvement rate is measured over. Two points give a
#: slope that is mostly noise; a handful gives a usable one.
RATE_WINDOW = 8

#: Challenge-to-skill band that counts as flow. Under it the work is easy,
#: over it the attempt is not practice.
FLOW_LOW = 0.9
FLOW_HIGH = 1.3

MAX_ATTEMPTS_KEPT = 512


@dataclass(frozen=True)
class Attempt:
    """One go at making the thing, and what the material did about it."""

    parameters: tuple[float, ...]
    quality: float
    resistance: float
    """How hard the material pushed back. Never negative."""

    difficulty: float = 1.0
    note: str = ""


@dataclass
class Skill:
    """One thing being got better at, on one material.

    The parameters are whatever the caller's ``attempt`` function takes. This
    object never interprets them; it only moves them, which is what keeps the
    module usable for glaze chemistry, a sanding motion, a prompt, or a set of
    compiler flags.
    """

    name: str
    parameters: list[float]
    step_size: float = 0.1
    perturbation: float = 0.1
    attempts: list[Attempt] = field(default_factory=list)
    resistance_estimate: float = 0.0
    iterations: int = 0

    def record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)
        if len(self.attempts) > MAX_ATTEMPTS_KEPT:
            del self.attempts[: len(self.attempts) - MAX_ATTEMPTS_KEPT]
        # A running estimate of the material rather than the last reading,
        # because one stubborn piece of wood is not a fact about wood.
        weight = 1.0 / min(len(self.attempts), 32)
        self.resistance_estimate += weight * (attempt.resistance - self.resistance_estimate)

    def competence(self, window: int = RATE_WINDOW) -> float | None:
        """Recent typical quality. The skill half of the flow ratio."""
        recent = [a.quality for a in self.attempts[-window:]]
        if not recent:
            return None
        return float(statistics.median(recent))

    def improvement_rate(self, window: int = RATE_WINDOW) -> float | None:
        """Quality gained per attempt, over the recent window.

        Least squares on the last few attempts. The scheduler ranks on this,
        so it has to be a slope rather than a difference: the difference
        between two noisy evaluations is mostly the noise.
        """
        recent = [a.quality for a in self.attempts[-window:]]
        n = len(recent)
        if n < 3:
            return None
        mean_x = (n - 1) / 2.0
        mean_y = sum(recent) / n
        denominator = sum((i - mean_x) ** 2 for i in range(n))
        if denominator <= 0:
            return None
        numerator = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(recent))
        return float(numerator / denominator)

    def flow(self, difficulty: float) -> dict[str, Any]:
        """Where this difficulty sits against what can currently be done."""
        skill = self.competence()
        if skill is None or skill <= 0:
            return {"ratio": None, "state": "unknown"}
        ratio = float(difficulty) / skill
        if ratio < FLOW_LOW:
            state = "under"
        elif ratio > FLOW_HIGH:
            state = "over"
        else:
            state = "flow"
        return {"ratio": round(ratio, 4), "state": state}

    def next_difficulty(self) -> float | None:
        """A difficulty that would sit in the band, given what can be done now."""
        skill = self.competence()
        if skill is None or skill <= 0:
            return None
        return float(skill * (FLOW_LOW + FLOW_HIGH) / 2.0)


class CraftPractice:
    """Skills, the materials they meet, and what to work on next.

    Nothing here knows what is being made. The caller supplies a function that
    takes parameters and returns a quality and a resistance, and everything
    else follows from those two numbers.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        self._random = random.Random(seed)
        self._sufficiency: dict[str, float] = {}

    def add_skill(self, name: str, parameters: Sequence[float], *,
                  step_size: float = 0.1, perturbation: float = 0.1) -> Skill:
        skill = self._skills.get(name)
        if skill is None:
            skill = Skill(
                name=name, parameters=[float(p) for p in parameters],
                step_size=float(step_size), perturbation=float(perturbation),
            )
            self._skills[name] = skill
        return skill

    def set_sufficiency(self, name: str, quality: float) -> None:
        """The quality a task needs. Recorded, and kept out of the scheduler."""
        self._sufficiency[name] = float(quality)

    # ------------------------------------------------------------- practise

    def practise(
        self,
        name: str,
        attempt: Callable[[Sequence[float]], tuple[float, float]],
        *,
        difficulty: float | None = None,
    ) -> dict[str, Any]:
        """One round: two tries either side, then a step along the difference.

        The two evaluations are the whole cost, whatever the number of
        parameters. That is the property that makes the method the right one
        for a craft: a person adjusting grip, angle, speed and pressure at
        once is not learning four things separately and does not have four
        times as many attempts to spend.
        """
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(f"no such skill: {name}")
        k = skill.iterations
        # Decaying gains, so early attempts move far and later ones settle.
        step = skill.step_size / ((k + 1) ** STEP_DECAY)
        spread = skill.perturbation / ((k + 1) ** PERTURBATION_DECAY)
        # A material that fights is approached carefully. This is the only
        # place the estimate of the material enters the update, and taking it
        # out is the ablation that turns craft into repetition.
        step /= 1.0 + max(0.0, skill.resistance_estimate)

        delta = [1.0 if self._random.random() < 0.5 else -1.0 for _ in skill.parameters]
        up = [p + spread * d for p, d in zip(skill.parameters, delta, strict=True)]
        down = [p - spread * d for p, d in zip(skill.parameters, delta, strict=True)]
        quality_up, resistance_up = attempt(up)
        quality_down, resistance_down = attempt(down)

        gradient = [
            (quality_up - quality_down) / (2.0 * spread * d) for d in delta
        ]
        skill.parameters = [
            p + step * g for p, g in zip(skill.parameters, gradient, strict=True)
        ]
        skill.iterations += 1

        chosen_difficulty = (
            difficulty if difficulty is not None else (skill.next_difficulty() or 1.0)
        )
        quality = (quality_up + quality_down) / 2.0
        resistance = (resistance_up + resistance_down) / 2.0
        skill.record(
            Attempt(
                parameters=tuple(skill.parameters), quality=quality,
                resistance=resistance, difficulty=chosen_difficulty,
            )
        )
        return {
            "skill": name,
            "quality": round(quality, 6),
            "resistance": round(resistance, 6),
            "step": round(step, 6),
            "gradient_norm": round(math.sqrt(sum(g * g for g in gradient)), 6),
            "flow": skill.flow(chosen_difficulty),
            "iterations": skill.iterations,
        }

    # ------------------------------------------------------------ what next

    def practice_target(self) -> str | None:
        """The skill improving fastest. What a craft practice actually picks.

        No reference to what any task needs. A skill that has stopped
        improving is not chosen even when it is the one being asked for, and a
        skill nothing is asking for is chosen when it is moving.
        """
        rates = {
            name: skill.improvement_rate()
            for name, skill in self._skills.items()
        }
        measured = {k: v for k, v in rates.items() if v is not None}
        if not measured:
            # Nothing has enough history to have a slope, so practise whatever
            # has been practised least. Falling back on the task would let the
            # requirement decide, which is the thing being avoided.
            unmeasured = sorted(self._skills.values(), key=lambda s: s.iterations)
            return unmeasured[0].name if unmeasured else None
        return max(measured, key=lambda k: measured[k])

    def required_target(self) -> str | None:
        """The skill furthest below what a task needs.

        The other scheduler, present so the two can be swapped and the
        difference measured. It stops as soon as everything is sufficient.
        """
        shortfalls: dict[str, float] = {}
        for name, needed in self._sufficiency.items():
            skill = self._skills.get(name)
            if skill is None:
                continue
            have = skill.competence()
            if have is None or have < needed:
                shortfalls[name] = needed - (have or 0.0)
        if not shortfalls:
            return None
        return max(shortfalls, key=lambda k: shortfalls[k])

    def past_sufficiency(self) -> list[str]:
        """Skills still being worked on after the task stopped needing it."""
        out: list[str] = []
        for name, needed in self._sufficiency.items():
            skill = self._skills.get(name)
            if skill is None:
                continue
            have = skill.competence()
            if have is not None and have >= needed and (skill.improvement_rate() or 0) > 0:
                out.append(name)
        return sorted(out)

    def status(self) -> dict[str, Any]:
        return {
            "skills": {
                name: {
                    "iterations": s.iterations,
                    "competence": None if s.competence() is None else round(s.competence(), 4),
                    "improvement_rate": (
                        None if s.improvement_rate() is None
                        else round(s.improvement_rate(), 6)
                    ),
                    "resistance": round(s.resistance_estimate, 4),
                    "flow": s.flow(s.next_difficulty() or 1.0),
                    "sufficiency": self._sufficiency.get(name),
                }
                for name, s in sorted(self._skills.items())
            },
            "practice_target": self.practice_target(),
            "required_target": self.required_target(),
            "past_sufficiency": self.past_sufficiency(),
        }


_PRACTICE: CraftPractice | None = None


def get_craft_practice() -> CraftPractice:
    global _PRACTICE
    if _PRACTICE is None:
        _PRACTICE = CraftPractice()
    return _PRACTICE


def reset_craft_practice_for_test() -> None:
    global _PRACTICE
    _PRACTICE = None
