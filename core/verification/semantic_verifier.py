"""SemanticVerifier — three independent channels of semantic agreement.

Channels:
  1. self-consistency — sample-level mean cosine over an embedding set.
  2. paraphrase invariance — answers to paraphrased prompts must
     remain close to the answer to the original prompt.
  3. proof-carrying code — generated code must include an assertion
     and pass the F4 / F19 sandbox.

Each channel returns a typed result; ``SemanticVerifier.combined``
aggregates them.  Failure on *any* required channel rejects the
candidate, which is the entire point — semantic improvement that
only one channel believes in is exactly the failure mode this
module is built to catch.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.discovery.code_eval import DiscoveryEvaluation, SafeCodeEvaluator
from core.verification.embedder import HashEmbedder


@dataclass
class SelfConsistencyResult:
    ok: bool
    mean_cosine: float
    pairs: int

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "mean_cosine": self.mean_cosine, "pairs": self.pairs}


@dataclass
class InvarianceResult:
    ok: bool
    similarities: List[float]
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "similarities": list(self.similarities),
            "threshold": self.threshold,
        }


@dataclass
class ProofCarryingResult:
    ok: bool
    has_assertion: bool
    sandbox: DiscoveryEvaluation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "has_assertion": self.has_assertion,
            "sandbox": self.sandbox.to_dict(),
        }


@dataclass
class SemanticVerificationReport:
    accepted: bool
    self_consistency: Optional[SelfConsistencyResult] = None
    invariance: Optional[InvarianceResult] = None
    proof: Optional[ProofCarryingResult] = None
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "self_consistency": self.self_consistency.to_dict() if self.self_consistency else None,
            "invariance": self.invariance.to_dict() if self.invariance else None,
            "proof": self.proof.to_dict() if self.proof else None,
            "reasons": list(self.reasons),
        }


class SemanticVerifier:
    def __init__(
        self,
        *,
        embedder: Optional[HashEmbedder] = None,
        code_evaluator: Optional[SafeCodeEvaluator] = None,
        consistency_threshold: float = 0.72,
        invariance_threshold: float = 0.72,
    ):
        self.embedder = embedder or HashEmbedder()
        self.code_evaluator = code_evaluator or SafeCodeEvaluator()
        self.consistency_threshold = float(consistency_threshold)
        self.invariance_threshold = float(invariance_threshold)

    # ------------------------------------------------------------------
    def self_consistency(
        self,
        outputs: Sequence[str],
        *,
        threshold: Optional[float] = None,
    ) -> SelfConsistencyResult:
        thr = self.consistency_threshold if threshold is None else float(threshold)
        if len(outputs) < 2:
            return SelfConsistencyResult(ok=True, mean_cosine=1.0, pairs=0)
        embeddings = [self.embedder.embed(o) for o in outputs]
        sims: List[float] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(self.embedder.cosine(embeddings[i], embeddings[j]))
        mean = statistics.mean(sims) if sims else 1.0
        return SelfConsistencyResult(ok=mean >= thr, mean_cosine=mean, pairs=len(sims))

    def paraphrase_invariance(
        self,
        original_answer: str,
        paraphrase_answers: Sequence[str],
        *,
        threshold: Optional[float] = None,
    ) -> InvarianceResult:
        thr = self.invariance_threshold if threshold is None else float(threshold)
        if not paraphrase_answers:
            return InvarianceResult(ok=True, similarities=[], threshold=thr)
        base = self.embedder.embed(original_answer)
        sims = [self.embedder.cosine(base, self.embedder.embed(p)) for p in paraphrase_answers]
        return InvarianceResult(ok=all(s >= thr for s in sims), similarities=sims, threshold=thr)

    def proof_carrying_code(
        self,
        code: str,
        fn_name: str,
        tests: Sequence[Tuple[Tuple[Any, ...], Any]],
    ) -> ProofCarryingResult:
        has_assertion = "assert " in code or "assert(" in code
        sandbox = self.code_evaluator.evaluate(code, fn_name, tests)
        ok = has_assertion and sandbox.ok
        return ProofCarryingResult(ok=ok, has_assertion=has_assertion, sandbox=sandbox)

    # ------------------------------------------------------------------
    def verify(
        self,
        *,
        consistency_outputs: Optional[Sequence[str]] = None,
        invariance: Optional[Tuple[str, Sequence[str]]] = None,
        proof: Optional[Tuple[str, str, Sequence[Tuple[Tuple[Any, ...], Any]]]] = None,
        require: Sequence[str] = ("consistency", "invariance", "proof"),
    ) -> SemanticVerificationReport:
        """Run any subset of channels and combine.

        ``require`` lists which channels must pass when present.  A
        channel that wasn't supplied does not block.
        """
        report = SemanticVerificationReport(accepted=True)
        require_set = {r.strip() for r in require}

        if consistency_outputs is not None:
            report.self_consistency = self.self_consistency(consistency_outputs)
            if "consistency" in require_set and not report.self_consistency.ok:
                report.accepted = False
                report.reasons.append(
                    f"self_consistency mean={report.self_consistency.mean_cosine:.3f} "
                    f"below threshold {self.consistency_threshold:.3f}"
                )

        if invariance is not None:
            original_answer, paraphrase_answers = invariance
            report.invariance = self.paraphrase_invariance(original_answer, paraphrase_answers)
            if "invariance" in require_set and not report.invariance.ok:
                report.accepted = False
                report.reasons.append(
                    f"paraphrase_invariance min={min(report.invariance.similarities, default=0.0):.3f} "
                    f"below threshold {report.invariance.threshold:.3f}"
                )

        if proof is not None:
            code, fn_name, tests = proof
            report.proof = self.proof_carrying_code(code, fn_name, tests)
            if "proof" in require_set and not report.proof.ok:
                report.accepted = False
                missing = []
                if not report.proof.has_assertion:
                    missing.append("missing_assertion")
                if not report.proof.sandbox.ok:
                    missing.append(f"sandbox_outcome={report.proof.sandbox.outcome}")
                report.reasons.append("proof_carrying:" + ",".join(missing))

        if not report.reasons:
            report.reasons.append("all required channels passed")
        return report
