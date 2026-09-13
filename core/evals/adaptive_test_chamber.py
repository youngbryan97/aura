"""core/evals/adaptive_test_chamber.py

Adaptive Test Chamber  (lineage: GLaDOS — Portal)
================================================
GLaDOS designs test chambers, runs a subject through them, measures, and adapts
the difficulty to sit right at the edge of capability. The real science is
automated curriculum / adaptive testing (item-response theory): keep the
challenge near the frontier where learning signal is highest.

This designs challenges about Aura's *own* capabilities, tracks a difficulty
level per capability, and moves it toward the frontier as results come in — up on
a pass, down on a fail. It lives in evals/ beside eval_arena.py and can feed
results into it. (GLaDOS's removable "morality core" is, in our system, the
immutable conscience — which is *not* removable.)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.TestChamber")

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 10


@dataclass
class Challenge:
    challenge_id: str
    capability: str
    difficulty: int
    prompt: str
    success_criterion: str
    timestamp: float = field(default_factory=lambda: time.time())


class AdaptiveTestChamber:
    STEP_UP = 1
    STEP_DOWN = 1
    FRONTIER_BAND = (0.55, 0.80)   # target pass-rate window

    def __init__(self):
        self._difficulty: dict[str, int] = {}
        self._history: dict[str, deque] = {}
        self._issued = 0
        logger.info("🧪 AdaptiveTestChamber initialized (GLaDOS lineage)")

    def _diff(self, capability: str) -> int:
        return self._difficulty.setdefault(capability, 3)

    def design_challenge(self, capability: str, *, hint: str = "") -> Challenge:
        self._issued += 1
        difficulty = self._diff(capability)
        scale = {
            1: "a trivial warm-up", 2: "an easy", 3: "a moderate", 4: "a moderate-plus",
            5: "a solid", 6: "a hard", 7: "a very hard", 8: "an expert", 9: "a brutal",
            10: "a near-impossible",
        }.get(difficulty, "a moderate")
        prompt = (
            f"{scale.capitalize()} {capability} challenge"
            + (f" focused on {hint}" if hint else "")
            + f" (difficulty {difficulty}/10). Demonstrate the capability end to end."
        )
        return Challenge(
            challenge_id=f"chal_{capability}_{self._issued}",
            capability=capability,
            difficulty=difficulty,
            prompt=prompt,
            success_criterion=f"Passes an objective check of '{capability}' at difficulty {difficulty}.",
        )

    def record_result(self, capability: str, passed: bool) -> int:
        """Update difficulty toward the capability frontier. Returns new difficulty."""
        hist = self._history.setdefault(capability, deque(maxlen=20))
        hist.append(1 if passed else 0)
        current = self._diff(capability)

        if passed and current < MAX_DIFFICULTY:
            current += self.STEP_UP
        elif not passed and current > MIN_DIFFICULTY:
            current -= self.STEP_DOWN

        # Nudge by recent pass-rate so we settle in the frontier band.
        if len(hist) >= 5:
            rate = sum(hist) / len(hist)
            if rate > self.FRONTIER_BAND[1] and current < MAX_DIFFICULTY:
                current = min(MAX_DIFFICULTY, current + 1)
            elif rate < self.FRONTIER_BAND[0] and current > MIN_DIFFICULTY:
                current = max(MIN_DIFFICULTY, current - 1)

        self._difficulty[capability] = current
        return current

    def frontier(self, capability: str) -> dict[str, Any]:
        hist = self._history.get(capability, deque())
        rate = (sum(hist) / len(hist)) if hist else None
        return {
            "capability": capability,
            "difficulty": self._diff(capability),
            "recent_pass_rate": round(rate, 3) if rate is not None else None,
            "samples": len(hist),
        }

    def get_status(self) -> dict[str, Any]:
        """Report the chamber's own condition from what it has actually seen.

        ``"healthy": True`` used to be a literal. A health field that cannot
        return False is not a health field — it is a decoration on a dictionary,
        and it reads as a passed check to everything that consumes it.

        The chamber is healthy when it is doing its job: it has results to learn
        from, and the difficulties it chose are sitting near the frontier band
        rather than pinned at either wall (pinned means every challenge is
        trivial or every challenge is impossible, and neither produces signal).
        Before any results exist there is nothing to be healthy or unhealthy
        about, so it says so.
        """
        scored = {cap: hist for cap, hist in self._history.items() if hist}
        if not scored:
            return {
                "capabilities_tracked": len(self._difficulty),
                "challenges_issued": self._issued,
                "healthy": None,
                "reason": "no results recorded yet",
            }
        pinned = []
        for cap, hist in scored.items():
            if len(hist) < 5 or self._diff(cap) not in (MIN_DIFFICULTY, MAX_DIFFICULTY):
                continue
            rate = sum(hist) / len(hist)
            if not (self.FRONTIER_BAND[0] <= rate <= self.FRONTIER_BAND[1]):
                pinned.append(cap)
        return {
            "capabilities_tracked": len(self._difficulty),
            "challenges_issued": self._issued,
            "capabilities_scored": len(scored),
            "healthy": not pinned,
            "pinned_at_a_wall": pinned,
            "reason": (
                "adapting within the frontier band"
                if not pinned
                else "difficulty pinned at a wall, so challenges carry no signal"
            ),
        }


_INSTANCE: AdaptiveTestChamber | None = None


def get_test_chamber() -> AdaptiveTestChamber:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = AdaptiveTestChamber()
    return _INSTANCE


def register_test_chamber(orchestrator: Any = None) -> AdaptiveTestChamber:
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.GLADOS, default=None) or get_test_chamber()
    register_runtime_service(
        ServiceNames.GLADOS,
        inst,
        required=False,
        owner="core/evals/adaptive_test_chamber.py",
        registered_by="register_test_chamber",
    )
    register_runtime_service(
        "glados",
        inst,
        required=False,
        owner="core/evals/adaptive_test_chamber.py",
        registered_by="register_test_chamber",
    )
    return inst


__all__ = ["AdaptiveTestChamber", "Challenge", "get_test_chamber", "register_test_chamber"]
