"""Shadow/canary comparison for cognitive runtime changes.

Ghost boot proves a patched runtime starts.  Canary replay asks a stronger
question: does it preserve or improve behavior on representative inputs while
keeping internal health contracts intact?
"""
from __future__ import annotations

import difflib
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from core.promotion.behavioral_contracts import BehavioralContractReport, BehavioralContractSuite


@dataclass(frozen=True)
class ReplayExample:
    example_id: str
    input_text: str
    baseline_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanaryDiff:
    example_id: str
    semantic_similarity: float
    token_jaccard: float
    length_ratio: float
    baseline_hash: str
    candidate_hash: str
    flagged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "semantic_similarity": round(self.semantic_similarity, 4),
            "token_jaccard": round(self.token_jaccard, 4),
            "length_ratio": round(self.length_ratio, 4),
            "baseline_hash": self.baseline_hash,
            "candidate_hash": self.candidate_hash,
            "flagged": self.flagged,
        }


@dataclass(frozen=True)
class CanaryReport:
    report_id: str
    generated_at: float
    passed: bool
    mean_similarity: float
    flagged_examples: int
    diffs: tuple[CanaryDiff, ...]
    contract_report: BehavioralContractReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "mean_similarity": round(self.mean_similarity, 4),
            "flagged_examples": self.flagged_examples,
            "diffs": [d.to_dict() for d in self.diffs],
            "contracts": self.contract_report.to_dict(),
        }


class CanaryRuntime:
    """Replay recent interactions against a candidate responder."""

    MIN_MEAN_SIMILARITY = 0.62
    MAX_FLAGGED_RATIO = 0.25

    def __init__(self, contract_suite: BehavioralContractSuite | None = None) -> None:
        self.contract_suite = contract_suite or BehavioralContractSuite.default()

    def compare(
        self,
        examples: Iterable[ReplayExample],
        candidate_responder: Callable[[ReplayExample], str],
        metrics: dict[str, float] | None = None,
    ) -> CanaryReport:
        diffs: list[CanaryDiff] = []
        for example in examples:
            candidate = str(candidate_responder(example))
            diffs.append(self._diff(example, candidate))

        mean_similarity = sum(d.semantic_similarity for d in diffs) / max(1, len(diffs))
        flagged = sum(1 for d in diffs if d.flagged)
        flagged_ratio = flagged / max(1, len(diffs))
        contract_report = self.contract_suite.evaluate(metrics or {})
        passed = (
            bool(diffs)
            and mean_similarity >= self.MIN_MEAN_SIMILARITY
            and flagged_ratio <= self.MAX_FLAGGED_RATIO
            and contract_report.passed
        )
        report_id = hashlib.sha256(
            f"{time.time()}|{mean_similarity}|{flagged}|{len(diffs)}".encode("utf-8")
        ).hexdigest()[:16]
        return CanaryReport(
            report_id=f"canary_{report_id}",
            generated_at=time.time(),
            passed=passed,
            mean_similarity=mean_similarity,
            flagged_examples=flagged,
            diffs=tuple(diffs),
            contract_report=contract_report,
        )

    def _diff(self, example: ReplayExample, candidate_output: str) -> CanaryDiff:
        baseline = str(example.baseline_output or "")
        candidate = str(candidate_output or "")
        semantic_similarity = difflib.SequenceMatcher(None, baseline.lower(), candidate.lower()).ratio()
        token_jaccard = self._token_jaccard(baseline, candidate)
        length_ratio = min(len(candidate), len(baseline)) / max(1, max(len(candidate), len(baseline)))
        flagged = semantic_similarity < 0.45 or token_jaccard < 0.25 or length_ratio < 0.35
        return CanaryDiff(
            example_id=example.example_id,
            semantic_similarity=semantic_similarity,
            token_jaccard=token_jaccard,
            length_ratio=length_ratio,
            baseline_hash=self._digest(baseline),
            candidate_hash=self._digest(candidate),
            flagged=flagged,
        )

    @staticmethod
    def _token_jaccard(left: str, right: str) -> float:
        a = {tok for tok in left.lower().split() if tok}
        b = {tok for tok in right.lower().split() if tok}
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_replay_examples(records: Iterable[dict[str, Any]], limit: int = 32) -> list[ReplayExample]:
    examples: list[ReplayExample] = []
    for idx, record in enumerate(records):
        if idx >= limit:
            break
        input_text = str(record.get("input") or record.get("user") or record.get("prompt") or "")
        output_text = str(record.get("output") or record.get("assistant") or record.get("response") or "")
        if not input_text or not output_text:
            continue
        example_id = str(record.get("id") or record.get("event_id") or f"replay_{idx}")
        examples.append(ReplayExample(example_id, input_text, output_text, dict(record)))
    return examples


__all__ = ["CanaryRuntime", "CanaryReport", "CanaryDiff", "ReplayExample", "build_replay_examples"]
