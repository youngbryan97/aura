"""No catastrophic regressions -- the guard that makes change reversible (CP245).

Every capability change in this arc -- enabling the cognitive-loop pathway,
promoting a GRPO adapter, fusing weights -- carries the same risk: a gain in
one place bought with a silent loss somewhere else. Anima Rationis line 662
names the bar precisely: no more than a small decline in language, social
reasoning, instruction following, or factual reliability. Averages hide
exactly this -- a math gain can mask a language regression and the aggregate
still rises.

So this guard protects EACH capability family independently. It measures a
baseline before a change, measures a candidate after, and flags any family
that dropped more than the allowed margin. A change passes only if NO
protected family regressed past the margin -- the min over families, not the
mean, which is the honest reading of "wholesale, without regressions."

Distinct from the two existing RegressionGuards (the factory patch guard and
the orphan live-error-rate stub): this is a per-CAPABILITY battery, not a
single error signal. Model-agnostic -- probes are (prompt, grader) pairs and
the solver is a callback -- so it runs against the live model, a trained
adapter, or a stub in a test. It produces the evidence a governed promotion
gate needs; it does not itself decide to keep or revert.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

CAPABILITY_REGRESSION_SCHEMA = "aura.capability_regression_battery.v1"

# The protected families. A change may improve a target capability, but it
# must not quietly degrade any of these -- they are what "still herself"
# means after a change.
PROTECTED_FAMILIES = (
    "language",
    "instruction_following",
    "factual",
    "social_reasoning",
    "math",
    "code",
)


@dataclass(frozen=True)
class Probe:
    """One graded item in a capability family."""

    family: str
    prompt: str
    grader: Callable[[str], bool]

    def __post_init__(self) -> None:
        if self.family not in PROTECTED_FAMILIES:
            raise ValueError(
                f"{self.family} is not a protected family {PROTECTED_FAMILIES}"
            )
        if not self.prompt.strip():
            raise ValueError("probe needs a prompt")
        if not callable(self.grader):
            raise ValueError("probe grader must be callable")


def score_family(
    probes: list[Probe], solve: Callable[[str], str]
) -> dict[str, float]:
    """Accuracy per family for a solver. Never raises on bad model output."""
    by_family: dict[str, list[bool]] = {}
    for probe in probes:
        try:
            answer = solve(probe.prompt)
            ok = bool(probe.grader(str(answer)))
        except Exception as exc:  # noqa: BLE001 - a crashed grader is a failed probe
            # Counting it as a failure is right; doing so silently is not.
            # A grader that crashes on every probe and a model that fails
            # every probe produce the same regression number.
            logger.warning(
                "Capability probe grader crashed for family %s: %s",
                probe.family,
                exc,
            )
            ok = False
        by_family.setdefault(probe.family, []).append(ok)
    return {family: round(sum(v) / len(v), 4) for family, v in by_family.items()}


@dataclass
class CapabilityRegressionGuard:
    """Baseline a set of probes, then judge a candidate against it."""

    probes: list[Probe]
    # The most a protected family may drop before it counts as a regression.
    # Anima Rationis' bar is ~2 points; kept configurable, with a small floor
    # that absorbs sampling jitter rather than real loss.
    max_drop: float = 0.02
    baseline: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.probes:
            raise ValueError("a guard with no probes protects nothing")
        if not 0.0 <= float(self.max_drop) <= 0.5:
            raise ValueError("max_drop must be inside [0, 0.5]")

    def measure_baseline(self, solve: Callable[[str], str]) -> dict[str, float]:
        """Score the pre-change model. Must be called before evaluate()."""
        self.baseline = score_family(self.probes, solve)
        return dict(self.baseline)

    def evaluate(self, solve: Callable[[str], str]) -> dict[str, Any]:
        """Score the post-change model and flag every family that regressed.

        The verdict is the AND over families: safe only if none dropped past
        the margin. Reports every delta, so a governed gate sees the full
        picture, not just a pass/fail bit.
        """
        if not self.baseline:
            raise ValueError(
                "no baseline; call measure_baseline on the pre-change model first"
            )
        candidate = score_family(self.probes, solve)
        deltas: dict[str, dict[str, float]] = {}
        regressions: list[str] = []
        improvements: list[str] = []
        for family, before in self.baseline.items():
            after = candidate.get(family, 0.0)
            delta = round(after - before, 4)
            deltas[family] = {"before": before, "after": after, "delta": delta}
            if delta < -self.max_drop:
                regressions.append(family)
            elif delta > self.max_drop:
                improvements.append(family)
        return {
            "schema": CAPABILITY_REGRESSION_SCHEMA,
            "max_drop": self.max_drop,
            "families": deltas,
            "regressions": regressions,
            "improvements": improvements,
            # Safe ONLY if nothing regressed. A change that helps math and
            # hurts language is not "net positive" -- it is a regression that
            # must be caught before it goes live.
            "safe": not regressions,
            "verdict": (
                "no catastrophic regression"
                if not regressions
                else f"REGRESSION in {regressions}: change must not go live as-is"
            ),
        }


def compose_from_battery(tasks: list[Any], *, family: str) -> list[Probe]:
    """Turn a heldout/verifiable task list into probes for one family.

    Reuses the existing sealed batteries (heldout_battery, verifiable_tasks)
    rather than inventing new items, so the guard measures the same
    exact-checkable capabilities the rest of the arc trained and evaluated.
    Each task must expose ``.prompt`` and ``.grade(response)->{"correct":bool}``.
    """
    probes: list[Probe] = []
    for task in tasks:
        prompt = getattr(task, "prompt", None)
        grade = getattr(task, "grade", None)
        if not prompt or not callable(grade):
            continue
        probes.append(
            Probe(
                family=family,
                prompt=prompt,
                grader=lambda answer, _g=grade: bool(_g(answer).get("correct")),
            )
        )
    if not probes:
        raise ValueError("no gradeable tasks produced probes")
    return probes


__all__ = [
    "CAPABILITY_REGRESSION_SCHEMA",
    "PROTECTED_FAMILIES",
    "CapabilityRegressionGuard",
    "Probe",
    "compose_from_battery",
    "score_family",
]
