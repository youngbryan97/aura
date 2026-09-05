"""Teacher federation: learn from the strongest VERIFIED trajectory.

Distilling one frontier model inherits that one model's topology of
strengths and blind spots. The federation gathers competing solutions from
independent teachers — frontier generalists, coding/math specialists, formal
solvers, simulators, tool executions, human demonstrations, Aura's own full
system — and selects what to learn from by VERIFICATION, never by prestige:

    Teacher = VerifiedConsensus(T1, ..., Tn, tools, world)

Authority rules, in order:
1. Where an objective verifier exists, it has the last word. A verified
   candidate from the humblest teacher beats an unverified one from the
   strongest. Ties break on the teachers' measured reliability ledgers
   (Wilson lower bounds — the Verifier Foundry's math), never on kind.
2. Where no objective check exists, agreement across INDEPENDENT teachers
   yields a consensus candidate that is explicitly tiered
   ``consensus_unverified`` — it may train lower-stakes behavior, and the
   receipt says exactly what it is.
3. Verified FAILURES are kept as negative examples: where a proposed
   solution provably broke is itself supervision.

Teachers are injected callables; this module never talks to a network. Every
selection emits a full receipt: who proposed what, what the verifier said,
which rule decided, and each teacher's updated reliability ledger.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.brain.verifiers.foundry import wilson_lower_bound

logger = logging.getLogger("Aura.Learning.TeacherFederation")

FEDERATION_SCHEMA = "aura.teacher_federation.v1"

TEACHER_KINDS = (
    "frontier_generalist",
    "coding_specialist",
    "math_solver",
    "formal_solver",
    "simulator",
    "tool_execution",
    "human_demonstration",
    "aura_full_system",
)

MAX_TEACHERS = 12
MAX_CANDIDATE_CHARS = 8000


@dataclass(frozen=True)
class Teacher:
    """One registered solution source. ``propose`` returns candidate text or None."""

    name: str
    kind: str
    propose: Callable[[str], str | None]

    def validated(self) -> "Teacher":
        if not self.name.strip():
            raise ValueError("teacher requires a name")
        if self.kind not in TEACHER_KINDS:
            raise ValueError(f"teacher kind must be one of {TEACHER_KINDS}")
        if not callable(self.propose):
            raise ValueError("teacher requires a propose callable")
        return self


@dataclass
class TeacherLedger:
    """Verified-outcome track record per teacher (the anti-prestige device)."""

    successes: int = 0
    trials: int = 0

    @property
    def reliability_lower_bound(self) -> float:
        if self.trials <= 0:
            return 0.0
        return float(wilson_lower_bound(self.successes, self.trials))

    def record(self, verified_success: bool) -> None:
        self.trials += 1
        if verified_success:
            self.successes += 1

    def to_receipt(self) -> dict[str, Any]:
        return {
            "successes": self.successes,
            "trials": self.trials,
            "wilson_lb": round(self.reliability_lower_bound, 4),
        }


@dataclass
class FederatedSelection:
    tier: str  # verified | consensus_unverified | none
    selected_teacher: str
    candidate: str
    receipt: dict[str, Any]
    negative_examples: list[dict[str, Any]] = field(default_factory=list)


class TeacherFederation:
    """Registry + selection engine with persistent per-teacher ledgers."""

    def __init__(self, teachers: Sequence[Teacher]) -> None:
        validated = [teacher.validated() for teacher in teachers]
        if not validated:
            raise ValueError("federation requires at least one teacher")
        if len(validated) > MAX_TEACHERS:
            raise ValueError(f"federation supports at most {MAX_TEACHERS} teachers")
        names = [teacher.name for teacher in validated]
        if len(set(names)) != len(names):
            raise ValueError("teacher names must be unique")
        self.teachers = list(validated)
        self.ledgers: dict[str, TeacherLedger] = {
            teacher.name: TeacherLedger() for teacher in validated
        }

    # ── Selection ───────────────────────────────────────────────────────
    def select(
        self,
        task: str,
        *,
        verifier: Callable[[str], bool] | None = None,
    ) -> FederatedSelection:
        """Gather candidates, verify where possible, select by the rules."""
        if not isinstance(task, str) or not task.strip():
            raise ValueError("federation selection requires a task")
        proposals: list[dict[str, Any]] = []
        for teacher in self.teachers:
            try:
                raw = teacher.propose(task)
            except Exception as exc:  # noqa: BLE001 - teacher contract unknown; absence is data
                proposals.append(
                    {
                        "teacher": teacher.name,
                        "kind": teacher.kind,
                        "status": f"proposal_error:{type(exc).__name__}",
                    }
                )
                continue
            if raw is None or not str(raw).strip():
                proposals.append(
                    {
                        "teacher": teacher.name,
                        "kind": teacher.kind,
                        "status": "abstained",
                    }
                )
                continue
            proposals.append(
                {
                    "teacher": teacher.name,
                    "kind": teacher.kind,
                    "status": "proposed",
                    "candidate": str(raw).strip()[:MAX_CANDIDATE_CHARS],
                }
            )

        receipt: dict[str, Any] = {
            "schema": FEDERATION_SCHEMA,
            "selected_at": time.time(),
            "verifier_present": verifier is not None,
            "proposals": [
                {k: v for k, v in row.items() if k != "candidate"}
                | {"candidate_chars": len(row.get("candidate", ""))}
                for row in proposals
            ],
        }
        candidates = [row for row in proposals if row["status"] == "proposed"]
        if not candidates:
            receipt["decision"] = "no_candidates"
            return FederatedSelection("none", "", "", receipt)

        if verifier is not None:
            return self._select_verified(candidates, verifier, receipt)
        return self._select_consensus(candidates, receipt)

    def _select_verified(
        self,
        candidates: list[dict[str, Any]],
        verifier: Callable[[str], bool],
        receipt: dict[str, Any],
    ) -> FederatedSelection:
        verified: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []
        verdicts: dict[str, bool] = {}
        for row in candidates:
            try:
                passed = bool(verifier(row["candidate"]))
            except Exception as exc:  # noqa: BLE001 - verifier fault must not crown anyone
                receipt["decision"] = f"verifier_error:{type(exc).__name__}"
                return FederatedSelection("none", "", "", receipt)
            verdicts[row["teacher"]] = passed
            self.ledgers[row["teacher"]].record(passed)
            if passed:
                verified.append(row)
            else:
                # A verified failure is supervision too.
                negatives.append(
                    {
                        "teacher": row["teacher"],
                        "kind": row["kind"],
                        "candidate": row["candidate"],
                        "verified_outcome": False,
                    }
                )
        receipt["verifier_verdicts"] = verdicts
        receipt["ledgers"] = {
            name: ledger.to_receipt() for name, ledger in self.ledgers.items()
        }
        if not verified:
            receipt["decision"] = "all_candidates_failed_verification"
            return FederatedSelection(
                "none", "", "", receipt, negative_examples=negatives
            )
        # Verifier had the last word; among the verified, measured reliability
        # (then stable name order) breaks the tie — never teacher kind.
        winner = max(
            verified,
            key=lambda row: (
                self.ledgers[row["teacher"]].reliability_lower_bound,
                row["teacher"],
            ),
        )
        receipt["decision"] = "verified_winner"
        receipt["selected_teacher"] = winner["teacher"]
        return FederatedSelection(
            "verified",
            winner["teacher"],
            winner["candidate"],
            receipt,
            negative_examples=negatives,
        )

    def _select_consensus(
        self,
        candidates: list[dict[str, Any]],
        receipt: dict[str, Any],
    ) -> FederatedSelection:
        """No objective check exists: independent agreement, honestly tiered."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            groups.setdefault(_normalize(row["candidate"]), []).append(row)
        best_key = max(
            groups,
            key=lambda key: (
                len({row["teacher"] for row in groups[key]}),
                max(
                    self.ledgers[row["teacher"]].reliability_lower_bound
                    for row in groups[key]
                ),
                key,
            ),
        )
        agreeing = groups[best_key]
        independent = len({row["teacher"] for row in agreeing})
        receipt["consensus_group_size"] = independent
        receipt["consensus_groups"] = len(groups)
        if independent < 2:
            receipt["decision"] = "no_consensus_without_verifier"
            return FederatedSelection("none", "", "", receipt)
        representative = max(
            agreeing,
            key=lambda row: (
                self.ledgers[row["teacher"]].reliability_lower_bound,
                row["teacher"],
            ),
        )
        receipt["decision"] = "consensus_unverified"
        receipt["selected_teacher"] = representative["teacher"]
        return FederatedSelection(
            "consensus_unverified",
            representative["teacher"],
            representative["candidate"],
            receipt,
        )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


__all__ = [
    "FEDERATION_SCHEMA",
    "FederatedSelection",
    "MAX_TEACHERS",
    "TEACHER_KINDS",
    "Teacher",
    "TeacherFederation",
    "TeacherLedger",
]
