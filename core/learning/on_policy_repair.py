"""On-policy first-error repair: fix Aura from the state she actually reached.

Distillation's classic failure: the teacher provides a perfect solution whose
intermediate states the student never visits, so the student learns nothing
about recovering from its OWN mistakes. This protocol trains the correction
from the exact reached state instead:

    1. capture the actual trajectory (transition_grading.Transition stream);
    2. locate the EARLIEST causally important error by replaying prefixes —
       causal means "repairing here flips the downstream outcome", not
       "looks wrong";
    3. generate corrected transitions from that exact state (the teacher
       federation plugs in here as the corrector pool);
    4. rerun from before the error and require the repaired run to succeed;
    5. retain ONLY corrections that also transfer to fresh, structurally
       related tasks — a repair that fixes one instance is a patch, not
       learning.

Every retained repair emits the spec's training unit —
(cognitive state, possible operations, best operation, verified outcome) —
with full provenance, and every rejection is receipted with the gate that
refused it. Replays are injected callables: this module orchestrates and
accounts; it never talks to a model directly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.learning.transition_grading import (
    Transition,
    grade_trajectory,
    state_digest,
)

logger = logging.getLogger("Aura.Learning.OnPolicyRepair")

REPAIR_SCHEMA = "aura.on_policy_repair.v1"
TRAINING_UNIT_SCHEMA = "aura.repair_training_unit.v1"

MAX_REPLAYS_PER_TRAJECTORY = 24
MIN_TRANSFER_TASKS = 2


@dataclass(frozen=True)
class TrajectoryRecord:
    """What actually happened, captured from the agent's own run."""

    task_id: str
    family: str
    transitions: tuple[Transition, ...]
    final_success: bool
    latent_receipt_sha256: str = ""

    def validated(self) -> "TrajectoryRecord":
        if not self.task_id.strip() or not self.family.strip():
            raise ValueError("trajectory requires task_id and family")
        if not self.transitions:
            raise ValueError("trajectory requires transitions")
        for transition in self.transitions:
            transition.validated()
        return self


@dataclass
class RepairCandidate:
    """One corrected transition proposed from the exact reached state."""

    error_index: int
    corrected_action: str
    corrector: str  # which teacher/strategy produced it
    rationale: str = ""

    def validated(self) -> "RepairCandidate":
        if type(self.error_index) is not int or self.error_index < 0:
            raise ValueError("repair error_index must be a non-negative integer")
        if not self.corrected_action.strip():
            raise ValueError("repair requires a corrected action")
        if not self.corrector.strip():
            raise ValueError("repair requires a corrector identity")
        return self


@dataclass
class RepairOutcome:
    accepted: bool
    stage: str  # located | reran | transferred | retained — the furthest stage passed
    error_index: int | None
    training_unit: dict[str, Any] | None
    receipt: dict[str, Any]


def locate_first_causal_error(
    trajectory: TrajectoryRecord,
    replay_prefix_fn: Callable[[int], bool],
    *,
    max_replays: int = MAX_REPLAYS_PER_TRAJECTORY,
) -> tuple[int | None, dict[str, Any]]:
    """Earliest transition whose repair flips the downstream outcome.

    ``replay_prefix_fn(k)`` reruns the task keeping transitions [0, k) exactly
    as recorded and letting the agent continue freshly from there; it returns
    the rerun's final success. If continuing fresh from before transition k
    succeeds while the recorded run failed, some transition at index >= k-1
    boundary is causally implicated; binary search finds the earliest such
    boundary within the replay budget.
    """
    record = trajectory.validated()
    if record.final_success:
        return None, {"reason": "trajectory_succeeded", "replays": 0}
    replays = 0

    def replay(prefix_length: int) -> bool:
        nonlocal replays
        if replays >= max_replays:
            raise RuntimeError("replay budget exhausted during error localization")
        replays += 1
        return bool(replay_prefix_fn(prefix_length))

    total = len(record.transitions)
    # If even a full fresh rerun (empty prefix) fails, the defect precedes the
    # recorded transitions (task setup / capability), not a specific step.
    if not replay(0):
        return None, {
            "reason": "fresh_rerun_fails_no_single_step_causal",
            "replays": replays,
        }
    # Invariant: replay(low) succeeds, replay at full recorded prefix fails
    # (that IS the recorded failed run). Binary search the flip boundary.
    low, high = 0, total
    while high - low > 1:
        mid = (low + high) // 2
        if replay(mid):
            low = mid
        else:
            high = mid
    error_index = record.transitions[high - 1].index
    return error_index, {
        "reason": "causal_flip_located",
        "replays": replays,
        "prefix_succeeds": low,
        "prefix_fails": high,
    }


def validate_repair(
    trajectory: TrajectoryRecord,
    candidate: RepairCandidate,
    *,
    rerun_with_repair_fn: Callable[[RepairCandidate], bool],
    transfer_tasks: Sequence[str],
    run_transfer_fn: Callable[[str, RepairCandidate], bool],
    min_transfer_tasks: int = MIN_TRANSFER_TASKS,
) -> RepairOutcome:
    """Gate a proposed correction: rerun must succeed, then transfer must hold.

    ``transfer_tasks`` are FRESH, structurally related task ids the recorded
    trajectory never touched; ``run_transfer_fn`` applies the corrected
    strategy there. A repair is retained only when the rerun succeeds AND a
    strict majority of at least ``min_transfer_tasks`` transfer runs succeed.
    """
    record = trajectory.validated()
    repair = candidate.validated()
    receipt: dict[str, Any] = {
        "schema": REPAIR_SCHEMA,
        "task_id": record.task_id,
        "family": record.family,
        "error_index": repair.error_index,
        "corrector": repair.corrector,
        "created_at": time.time(),
    }
    if len(transfer_tasks) < min_transfer_tasks:
        receipt["refusal"] = "insufficient_transfer_tasks"
        return RepairOutcome(False, "located", repair.error_index, None, receipt)

    reran = bool(rerun_with_repair_fn(repair))
    receipt["rerun_success"] = reran
    if not reran:
        receipt["refusal"] = "rerun_still_fails"
        return RepairOutcome(False, "located", repair.error_index, None, receipt)

    transfer_results = {
        str(task): bool(run_transfer_fn(str(task), repair))
        for task in transfer_tasks
    }
    receipt["transfer_results"] = transfer_results
    passed = sum(1 for ok in transfer_results.values() if ok)
    receipt["transfer_passed"] = passed
    if passed * 2 <= len(transfer_results):
        receipt["refusal"] = "no_transfer_majority"
        return RepairOutcome(False, "reran", repair.error_index, None, receipt)

    failed = next(
        (
            t
            for t in record.transitions
            if t.index == repair.error_index
        ),
        None,
    )
    unit = {
        "schema": TRAINING_UNIT_SCHEMA,
        "cognitive_state_digest": (
            failed.state_digest if failed is not None and failed.state_digest else
            state_digest(
                {
                    "task_id": record.task_id,
                    "prefix": [t.action for t in record.transitions if t.index < repair.error_index],
                }
            )
        ),
        "possible_operations": sorted(
            {
                repair.corrected_action,
                *(
                    [failed.action] if failed is not None else []
                ),
            }
        ),
        "best_operation": repair.corrected_action,
        "verified_outcome": True,
        "family": record.family,
        "corrector": repair.corrector,
        "transfer_evidence": transfer_results,
        "source_trajectory": record.task_id,
        "latent_receipt_sha256": record.latent_receipt_sha256,
    }
    receipt["retained"] = True
    return RepairOutcome(True, "retained", repair.error_index, unit, receipt)


def repair_intake(trajectory: TrajectoryRecord) -> list[int]:
    """Which transitions deserve repair attention, per the grading spine."""
    record = trajectory.validated()
    graded = grade_trajectory(
        record.task_id,
        list(record.transitions),
        final_success=record.final_success,
    )
    return graded.repair_queue()


__all__ = [
    "MAX_REPLAYS_PER_TRAJECTORY",
    "MIN_TRANSFER_TASKS",
    "REPAIR_SCHEMA",
    "RepairCandidate",
    "RepairOutcome",
    "TRAINING_UNIT_SCHEMA",
    "TrajectoryRecord",
    "locate_first_causal_error",
    "repair_intake",
    "validate_repair",
]
