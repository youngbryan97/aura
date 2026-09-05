"""core/science/continual_metrics.py — what learning the new thing cost the old ones.

Aura's learning systems report what they gained. None of them reports what it
broke. A promotion gate that checks the new capability and not the old ones
will, over a long enough life, trade away everything it once knew one
improvement at a time, and every individual trade will have looked like a win.

Two numbers, both standard in continual learning and neither computed here
before:

* **Backward transfer.** After learning task N, re-run every earlier task. The
  mean change is BWT. Negative BWT is forgetting, and it is the number a
  promotion gate has to look at.
* **Forward transfer.** Before learning task N, how well does the system
  already do it compared to a fresh instance. Positive FWT is the developmental
  claim measured from the other side.

:meth:`ContinualLedger.promotion_verdict` is the gate. A learning block that
improves its own task and costs more than a declared budget of backward
transfer does not promote. The budget is explicit and small, because "a little
forgetting is fine" with no number attached is how a lifetime of little
forgettings happens.

The ecology
-----------
:class:`ArtifactEcology` is the same accounting for stored things rather than
for tasks. Every learned artifact - a procedure, a rule, a concept, a memory -
carries usage, benefit, interference, storage and match cost, and its retention
value is what it earns minus what it costs to keep. That is the arithmetic
``impasse.ChunkStore`` already applies to chunks and ``an_ecology_of_words``
applies to vocabulary; this is the shape both share, so a store that has not
implemented its own can use it, and so the total across stores is computable.

Retiring is not deleting
------------------------
An artifact below its retention value is retired, and a retired artifact keeps
its record. A store that deletes cannot tell you what it forgot, and "what did
she stop being able to do" is exactly the question a lifetime run has to answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "TaskScore",
    "ContinualLedger",
    "Artifact",
    "ArtifactEcology",
    "DEFAULT_FORGETTING_BUDGET",
]

#: How much mean backward transfer a learning block may cost before it is
#: refused. Small and explicit: "a little forgetting is fine" with no number
#: attached is how a lifetime of little forgettings happens.
DEFAULT_FORGETTING_BUDGET = 0.02


@dataclass(frozen=True, slots=True)
class TaskScore:
    """One task's score at one point in the training sequence."""

    task: str
    after_block: int
    score: float
    #: Score a fresh instance gets. Needed for forward transfer.
    naive_score: float | None = None


class ContinualLedger:
    """Every task's score after every block, and what each block cost."""

    def __init__(self, *, forgetting_budget: float = DEFAULT_FORGETTING_BUDGET) -> None:
        self._lock = checked_lock("core.science.continual_metrics.ContinualLedger", reentrant=True)
        self._scores: dict[tuple[str, int], TaskScore] = {}
        self._trained_at: dict[str, int] = {}
        self._budget = float(forgetting_budget)

    def record(
        self, task: str, after_block: int, score: float, *, naive_score: float | None = None
    ) -> None:
        with self._lock:
            self._scores[(task, after_block)] = TaskScore(task, after_block, score, naive_score)

    def trained(self, task: str, block: int) -> None:
        """Declare which block taught this task."""
        with self._lock:
            self._trained_at[task] = block

    def backward_transfer(self, after_block: int) -> dict[str, Any]:
        """How the tasks learned before this block changed after it.

        Compares each earlier task's score right after it was learned with its
        score now. Negative is forgetting.
        """
        with self._lock:
            earlier = {t: b for t, b in self._trained_at.items() if b < after_block}
            deltas = {}
            for task, learned_at in earlier.items():
                then = self._scores.get((task, learned_at))
                now = self._scores.get((task, after_block))
                if then is not None and now is not None:
                    deltas[task] = now.score - then.score
        mean = sum(deltas.values()) / len(deltas) if deltas else 0.0
        return {
            "after_block": after_block,
            "tasks_compared": len(deltas),
            "bwt": mean,
            "per_task": dict(sorted(deltas.items())),
            "worst": min(deltas.items(), key=lambda kv: kv[1]) if deltas else None,
            "forgetting": mean < 0,
        }

    def forward_transfer(self, task: str) -> dict[str, Any]:
        """How much better than naive the system was before it learned this task."""
        with self._lock:
            learned_at = self._trained_at.get(task)
            if learned_at is None or learned_at == 0:
                return {"task": task, "measurable": False}
            before = self._scores.get((task, learned_at - 1))
        if before is None or before.naive_score is None:
            return {"task": task, "measurable": False, "reason": "no naive comparison recorded"}
        return {
            "task": task,
            "measurable": True,
            "fwt": before.score - before.naive_score,
            "before": before.score,
            "naive": before.naive_score,
        }

    def promotion_verdict(self, after_block: int, *, own_task_gain: float) -> dict[str, Any]:
        """Whether this block may promote, given what it cost the old tasks."""
        backward = self.backward_transfer(after_block)
        cost = -min(0.0, backward["bwt"])
        allowed = cost <= self._budget
        return {
            "block": after_block,
            "own_task_gain": own_task_gain,
            "bwt": backward["bwt"],
            "forgetting_cost": cost,
            "budget": self._budget,
            "promote": allowed and own_task_gain > 0,
            "reason": (
                f"gained {own_task_gain:+.4g} and cost {cost:.4g} backward transfer "
                f"(budget {self._budget})"
                if allowed
                else f"cost {cost:.4g} backward transfer against a budget of {self._budget}"
                + (f"; worst was {backward['worst'][0]} at {backward['worst'][1]:+.4g}"
                   if backward["worst"] else "")
            ),
        }

    def report(self) -> dict[str, Any]:
        with self._lock:
            blocks = sorted({b for _, b in self._scores})
        return {
            "blocks": len(blocks),
            "tasks": len({t for t, _ in self._scores}),
            "bwt_by_block": {b: self.backward_transfer(b)["bwt"] for b in blocks},
            "blocks_that_forgot": [b for b in blocks if self.backward_transfer(b)["forgetting"]],
        }


@dataclass
class Artifact:
    """One learned thing, and what keeping it is worth."""

    artifact_id: str
    kind: str
    uses: int = 0
    benefit_per_use: float = 0.0
    #: How much this artifact costs OTHER artifacts by existing: match cost it
    #: adds to every decision, and outcomes it got wrong that something else
    #: would have got right.
    interference: float = 0.0
    storage_cost: float = 0.0
    match_cost: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    retired: bool = False
    retired_because: str = ""

    @property
    def retention_value(self) -> float:
        """What it earns minus what it costs to keep. Below zero, it goes."""
        return (
            self.uses * self.benefit_per_use
            - self.interference
            - self.storage_cost
            - self.match_cost * max(1, self.uses)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "uses": self.uses,
            "retention_value": self.retention_value,
            "interference": self.interference,
            "retired": self.retired,
            "retired_because": self.retired_because,
        }


class ArtifactEcology:
    """Every learned artifact, priced for keeping, across every store."""

    def __init__(self, *, min_uses_before_retiring: int = 3) -> None:
        self._lock = checked_lock("core.science.continual_metrics.ArtifactEcology", reentrant=True)
        self._artifacts: dict[str, Artifact] = {}
        self._min_uses = int(min_uses_before_retiring)

    def add(self, artifact: Artifact) -> Artifact:
        with self._lock:
            self._artifacts[artifact.artifact_id] = artifact
            return artifact

    def use(self, artifact_id: str, *, benefit: float | None = None) -> None:
        with self._lock:
            artifact = self._artifacts.get(artifact_id)
            if artifact is None:
                return
            artifact.uses += 1
            artifact.last_used = time.time()
            if benefit is not None:
                artifact.benefit_per_use = benefit

    def retire_what_does_not_pay(self) -> list[Artifact]:
        """Retire, never delete: a store that deletes cannot say what it forgot."""
        with self._lock:
            gone = []
            for artifact in self._artifacts.values():
                if artifact.retired or artifact.uses < self._min_uses:
                    continue
                if artifact.retention_value >= 0:
                    continue
                artifact.retired = True
                artifact.retired_because = (
                    f"retention value {artifact.retention_value:.4g} after {artifact.uses} uses"
                )
                gone.append(artifact)
            return gone

    def report(self) -> dict[str, Any]:
        with self._lock:
            artifacts = list(self._artifacts.values())
        live = [a for a in artifacts if not a.retired]
        by_kind: dict[str, int] = {}
        for artifact in live:
            by_kind[artifact.kind] = by_kind.get(artifact.kind, 0) + 1
        return {
            "artifacts": len(artifacts),
            "live": len(live),
            "retired": len(artifacts) - len(live),
            "by_kind": dict(sorted(by_kind.items())),
            "total_retention_value": sum(a.retention_value for a in live),
            "interfering": sorted(a.artifact_id for a in live if a.interference > 0),
            "what_she_stopped_being_able_to_do": [
                {"artifact_id": a.artifact_id, "kind": a.kind, "why": a.retired_because}
                for a in artifacts
                if a.retired
            ],
        }
