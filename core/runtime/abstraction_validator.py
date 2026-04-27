"""AbstractionEngine validation framework.

Audit constraint: principles must validate against held-out cases,
retire on failure, score transfer, and detect contradictions before
the engine claims to "abstract first principles."
"""
from __future__ import annotations


import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class HeldOutEpisode:
    episode_id: str
    description: str
    expected_outcome: Any


@dataclass
class PrincipleCandidate:
    principle_id: str
    text: str
    proposed_at: float = field(default_factory=time.time)


@dataclass
class ValidationResult:
    principle_id: str
    passed: int = 0
    failed: int = 0
    transfer_score: float = 0.0


@dataclass
class PrincipleRecord:
    candidate: PrincipleCandidate
    application_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    retired: bool = False
    retired_reason: Optional[str] = None
    last_used_at: Optional[float] = None


PrincipleApplicator = Callable[[PrincipleCandidate, HeldOutEpisode], bool]


class PrincipleStore:
    def __init__(self):
        self._records: Dict[str, PrincipleRecord] = {}

    def register(self, candidate: PrincipleCandidate) -> PrincipleRecord:
        rec = PrincipleRecord(candidate=candidate)
        self._records[candidate.principle_id] = rec
        return rec

    def get(self, principle_id: str) -> Optional[PrincipleRecord]:
        return self._records.get(principle_id)

    def all(self) -> List[PrincipleRecord]:
        return list(self._records.values())

    def active(self) -> List[PrincipleRecord]:
        return [r for r in self._records.values() if not r.retired]


class PrincipleValidator:
    def __init__(self, store: PrincipleStore):
        self.store = store

    def validate(
        self,
        candidate: PrincipleCandidate,
        episodes: List[HeldOutEpisode],
        applicator: PrincipleApplicator,
    ) -> ValidationResult:
        result = ValidationResult(principle_id=candidate.principle_id)
        for ep in episodes:
            try:
                ok = bool(applicator(candidate, ep))
            except Exception:
                ok = False
            if ok:
                result.passed += 1
            else:
                result.failed += 1
        total = result.passed + result.failed
        result.transfer_score = (result.passed / total) if total else 0.0
        return result


class RetirementPolicy:
    def __init__(
        self,
        store: PrincipleStore,
        *,
        min_applications: int = 5,
        max_failure_ratio: float = 0.5,
    ):
        self.store = store
        self.min_applications = min_applications
        self.max_failure_ratio = max_failure_ratio

    def review(self) -> List[PrincipleRecord]:
        retired: List[PrincipleRecord] = []
        for rec in self.store.all():
            total = rec.application_count
            if total < self.min_applications or rec.retired:
                continue
            failure_ratio = rec.failure_count / total
            if failure_ratio > self.max_failure_ratio:
                rec.retired = True
                rec.retired_reason = (
                    f"failure_ratio={failure_ratio:.2f} > {self.max_failure_ratio}"
                )
                retired.append(rec)
        return retired


class ContradictionDetector:
    def __init__(self, store: PrincipleStore):
        self.store = store

    def detect(self) -> List[Dict[str, str]]:
        contradictions: List[Dict[str, str]] = []
        active = self.store.active()
        # Simple textual contradiction sniff: a principle saying "X" and
        # another saying "not X" or "never X". Real impl would semantic.
        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                a_text = a.candidate.text.lower()
                b_text = b.candidate.text.lower()
                if (
                    f"never {a_text.split()[-1]}" in b_text
                    or f"do not {a_text.split()[-1]}" in b_text
                ):
                    contradictions.append(
                        {"a": a.candidate.principle_id, "b": b.candidate.principle_id}
                    )
        return contradictions
