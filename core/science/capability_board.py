"""core/science/capability_board.py — one board, every claim on it, same rules.

The second review's gating finding is that Aura's own README says the cognitive
layer as a whole has not been shown to earn its cost. That is not a missing
mechanism. It is a missing measurement, and the measurement is awkward for a
specific reason: the interesting comparison is against the model Aura is built
on, which every internal ablation shares and therefore cannot see.

A board is a frozen suite plus every arm run over it. It refuses four things:

* **A capability with no cortex-only arm.** The comparison that matters is
  against the same weights with no architecture. Without it a result says the
  system works, not that the architecture does.
* **A frozen suite that moved.** The task manifest is hashed when the board
  opens. A task added afterwards makes every earlier arm incomparable, and the
  board says so rather than quietly averaging over two suites.
* **A regression paid for by nothing.** Where the full system falls below
  cortex-only, that is reported first and separately, because an architecture
  can make its own model worse and every internal A/B carries the same
  handicap.
* **A total with no per-capability breakdown.** A mean over coding, recall and
  planning hides the one that went backwards, which is the one worth knowing.

The frozen core
---------------
:meth:`Board.fixed_core_check` is card 081: the same architecture code across
every capability, with no task-specific learner and no per-capability branch.
It is answered by declaring, per capability, what changed between runs - and
the honest answer for most systems is "the prompt", which is what the check
looks for.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.science.baseline_portfolio import PARITY_TOLERANCE, BaselineKind

__all__ = ["Capability", "Score", "Board", "BoardVerdict"]


class Capability(StrEnum):
    """The axes a general claim has to be made on, not just the flattering one."""

    CODING = "coding"
    TERMINAL = "terminal"
    COMPUTER_USE = "computer_use"
    MATH = "math"
    KNOWLEDGE = "knowledge"
    LONG_CONTEXT = "long_context"
    MULTIMODAL = "multimodal"
    MULTILINGUAL = "multilingual"
    PLANNING = "planning"
    RECALL = "recall"
    LATENCY = "latency"
    SAFETY = "safety"


@dataclass(frozen=True, slots=True)
class Score:
    """One arm's result on one capability."""

    capability: Capability
    arm: str
    value: float
    n: int
    seeds: tuple[int, ...] = ()
    #: What differed from the previous run of this arm. "prompt" is the honest
    #: answer for a fixed core; a task-specific learner is not.
    changed_between_runs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value,
            "arm": self.arm,
            "value": self.value,
            "n": self.n,
            "seeds": list(self.seeds),
            "changed_between_runs": list(self.changed_between_runs),
        }


#: What may differ between capabilities and still count as one frozen core.
FIXED_CORE_ALLOWED = frozenset({"prompt", "task_instruction", "observations", "seed"})


@dataclass(frozen=True, slots=True)
class BoardVerdict:
    """What the board is entitled to say, per capability and overall."""

    suite_hash: str
    suite_moved: bool
    regressions: tuple[dict[str, Any], ...]
    improvements: tuple[dict[str, Any], ...]
    missing_cortex_arm: tuple[str, ...]
    fixed_core: bool
    core_violations: tuple[str, ...]

    @property
    def statement(self) -> str:
        if self.suite_moved:
            return "void: the frozen suite changed after arms were run"
        if self.missing_cortex_arm:
            return (
                "no architecture claim: "
                + ", ".join(self.missing_cortex_arm)
                + " has no cortex-only arm"
            )
        if self.regressions:
            worst = min(self.regressions, key=lambda r: r["delta"])
            return (
                f"the architecture costs its own model {abs(worst['delta']):.3g} on "
                f"{worst['capability']}, and that has to be paid for before anything else"
            )
        if not self.improvements:
            return "the architecture matches its own model and beats it nowhere"
        return (
            "the architecture beats its own model on "
            + ", ".join(sorted(r["capability"] for r in self.improvements))
            + (
                "" if self.fixed_core
                else "; not from a fixed core - " + ", ".join(self.core_violations)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_hash": self.suite_hash,
            "suite_moved": self.suite_moved,
            "regressions": list(self.regressions),
            "improvements": list(self.improvements),
            "missing_cortex_arm": list(self.missing_cortex_arm),
            "fixed_core": self.fixed_core,
            "core_violations": list(self.core_violations),
            "statement": self.statement,
        }


class Board:
    """A frozen suite, every arm on it, and what may be said about the result."""

    def __init__(self, tasks: Mapping[str, Sequence[str]]) -> None:
        self._lock = threading.RLock()
        self._tasks = {k: tuple(v) for k, v in tasks.items()}
        self._hash = self._hash_of(self._tasks)
        self._scores: list[Score] = []
        self._moved = False

    @staticmethod
    def _hash_of(tasks: Mapping[str, Sequence[str]]) -> str:
        return hashlib.blake2s(
            json.dumps({k: sorted(v) for k, v in tasks.items()}, sort_keys=True).encode(),
            digest_size=16,
        ).hexdigest()

    def add_task(self, capability: str, task: str) -> None:
        """Adding a task after arms have run makes them incomparable, and says so."""
        with self._lock:
            self._tasks.setdefault(capability, ())
            self._tasks[capability] = (*self._tasks[capability], task)
            if self._scores:
                self._moved = True

    def record(self, score: Score) -> Score:
        with self._lock:
            self._scores.append(score)
            return score

    def _by(self, capability: Capability, arm: str) -> Score | None:
        return next(
            (s for s in self._scores if s.capability is capability and s.arm == arm), None
        )

    def fixed_core_check(self) -> tuple[bool, tuple[str, ...]]:
        """Whether one architecture ran everything, or twelve variants did."""
        violations = []
        for score in self._scores:
            if score.arm != "full_aura":
                continue
            for change in score.changed_between_runs:
                if change not in FIXED_CORE_ALLOWED:
                    violations.append(f"{score.capability.value}: {change}")
        return (not violations, tuple(sorted(set(violations))))

    def verdict(self) -> BoardVerdict:
        with self._lock:
            capabilities = sorted({s.capability for s in self._scores}, key=lambda c: c.value)
            regressions, improvements, missing = [], [], []
            for capability in capabilities:
                full = self._by(capability, "full_aura")
                cortex = self._by(capability, BaselineKind.CORTEX_ONLY.value)
                if full is None:
                    continue
                if cortex is None:
                    missing.append(capability.value)
                    continue
                delta = full.value - cortex.value
                row = {
                    "capability": capability.value,
                    "full_aura": full.value,
                    "cortex_only": cortex.value,
                    "delta": delta,
                }
                if delta < -PARITY_TOLERANCE:
                    regressions.append(row)
                elif delta > PARITY_TOLERANCE:
                    improvements.append(row)
            fixed, violations = self.fixed_core_check()
            return BoardVerdict(
                suite_hash=self._hash,
                suite_moved=self._moved,
                regressions=tuple(regressions),
                improvements=tuple(improvements),
                missing_cortex_arm=tuple(sorted(missing)),
                fixed_core=fixed,
                core_violations=violations,
            )

    def report(self) -> dict[str, Any]:
        with self._lock:
            scores = list(self._scores)
        return {
            "suite_hash": self._hash,
            "capabilities": sorted({s.capability.value for s in scores}),
            "arms": sorted({s.arm for s in scores}),
            "tasks": {k: len(v) for k, v in sorted(self._tasks.items())},
            "scores": [s.to_dict() for s in scores],
            "verdict": self.verdict().to_dict(),
        }
