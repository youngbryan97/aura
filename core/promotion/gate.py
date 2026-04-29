"""PromotionGate — confidence-interval-based monotone checkpoint gate.

This is the structurally honest version of "monotonic improvement":
the gate compares a candidate score vector against a stored baseline
and refuses to promote unless the candidate **statistically dominates**
on every critical metric.  It does not claim universal intelligence
monotonicity — what it claims is that promoted checkpoints never
silently regress on locked critical slices.

Each PromotionDecision is also emitted to the F1 hash-chained audit
log as a ``GovernanceReceipt`` so the chain carries the full
provenance: which metrics were compared, what the deltas were, what
the gate decided, and what reasons it gave.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class ScoreEstimate:
    """A single metric's mean estimate plus optional stderr + n.

    ``higher_is_better`` lets the gate compare loss-style and
    accuracy-style metrics through one interface.
    """

    mean: float
    stderr: float = 0.0
    n: int = 1
    higher_is_better: bool = True

    def lower_confidence(self, z: float = 1.96) -> float:
        if self.higher_is_better:
            return self.mean - z * self.stderr
        return self.mean + z * self.stderr

    def upper_confidence(self, z: float = 1.96) -> float:
        if self.higher_is_better:
            return self.mean + z * self.stderr
        return self.mean - z * self.stderr


@dataclass
class PromotionDecision:
    accepted: bool
    reasons: List[str]
    deltas: Dict[str, float]
    candidate: Dict[str, ScoreEstimate]
    baseline: Dict[str, ScoreEstimate]
    receipt_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "deltas": dict(self.deltas),
            "candidate": {
                k: {
                    "mean": v.mean,
                    "stderr": v.stderr,
                    "n": v.n,
                    "higher_is_better": v.higher_is_better,
                }
                for k, v in self.candidate.items()
            },
            "baseline": {
                k: {
                    "mean": v.mean,
                    "stderr": v.stderr,
                    "n": v.n,
                    "higher_is_better": v.higher_is_better,
                }
                for k, v in self.baseline.items()
            },
            "receipt_id": self.receipt_id,
            "metadata": dict(self.metadata),
        }


# Optional Will-side decider — keeps this module decoupled from core.will.
WillDecideCallable = Callable[[Dict[str, Any]], Dict[str, Any]]


class PromotionGate:
    """Componentwise statistical monotone-acceptance gate.

    * ``critical_metrics`` must satisfy ``conservative_delta >= delta``.
    * non-critical metrics may regress up to ``max_regression``.
    * any direction mismatch (higher-is-better flipped) is a failure.
    * baseline updates only on acceptance.

    When ``emit_receipts=True`` and ``core.runtime.receipts`` is
    importable, every decision becomes a ``GovernanceReceipt`` on the
    F1 audit chain.
    """

    def __init__(
        self,
        critical_metrics: Sequence[str],
        *,
        delta: float = 0.0,
        z: float = 1.96,
        max_regression: float = 0.0,
        emit_receipts: bool = True,
        will_decide_fn: Optional[WillDecideCallable] = None,
    ):
        self.critical = set(critical_metrics)
        self.delta = float(delta)
        self.z = float(z)
        self.max_regression = float(max_regression)
        self.emit_receipts = bool(emit_receipts)
        self.will_decide_fn = will_decide_fn
        self.baseline: Optional[Dict[str, ScoreEstimate]] = None
        self.history: List[PromotionDecision] = []

    def set_baseline(self, scores: Dict[str, ScoreEstimate]) -> None:
        self.baseline = dict(scores)

    def _evaluate(
        self, candidate: Dict[str, ScoreEstimate]
    ) -> tuple[bool, List[str], Dict[str, float]]:
        """Pure comparison: no receipts, no Will, no baseline mutation."""
        if self.baseline is None:
            return True, ["No prior baseline; candidate becomes baseline."], {}
        ok = True
        reasons: List[str] = []
        deltas: Dict[str, float] = {}
        for name, base in self.baseline.items():
            if name not in candidate:
                if name in self.critical:
                    ok = False
                    reasons.append(f"missing critical metric: {name}")
                continue
            cand = candidate[name]
            if cand.higher_is_better != base.higher_is_better:
                ok = False
                reasons.append(f"direction mismatch on {name}")
                continue
            if cand.higher_is_better:
                conservative = cand.lower_confidence(self.z) - base.upper_confidence(self.z)
                raw = cand.mean - base.mean
            else:
                conservative = base.lower_confidence(self.z) - cand.upper_confidence(self.z)
                raw = base.mean - cand.mean
            required = self.delta if name in self.critical else -self.max_regression
            deltas[name] = raw
            if conservative < required:
                ok = False
                reasons.append(
                    f"{name} failed: conservative_delta={conservative:.4g} "
                    f"< required={required:.4g}"
                )
        if ok:
            reasons.append("accepted")
        return ok, reasons, deltas

    def compare(
        self,
        candidate: Dict[str, ScoreEstimate],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromotionDecision:
        if not candidate:
            raise ValueError("candidate score vector must not be empty")
        prior_baseline = (
            None if self.baseline is None else dict(self.baseline)
        )
        accepted, reasons, deltas = self._evaluate(candidate)
        is_initial_baseline = prior_baseline is None

        # Optional Will-side veto — even an internally-passing candidate
        # is rejected if the Will refuses.  Skip on the very first
        # compare() since that's just initial baseline registration,
        # not a real promotion decision.
        if accepted and self.will_decide_fn is not None and not is_initial_baseline:
            try:
                payload = {
                    "domain": "checkpoint_promotion",
                    "deltas": deltas,
                    "metadata": metadata or {},
                }
                will_out = self.will_decide_fn(payload)
                outcome = str(will_out.get("outcome", "proceed")).lower()
                if outcome != "proceed":
                    accepted = False
                    reasons.append(
                        f"will_{outcome}:{will_out.get('reason', 'no_reason')}"
                    )
            except Exception as exc:  # noqa: BLE001 — fail-closed
                accepted = False
                reasons.append(f"will_decide_raised:{type(exc).__name__}")

        receipt_id: Optional[str] = None
        if self.emit_receipts:
            receipt_id = self._emit_receipt(
                accepted=accepted,
                reasons=reasons,
                deltas=deltas,
                candidate=candidate,
                metadata=metadata,
            )

        decision = PromotionDecision(
            accepted=accepted,
            reasons=reasons,
            deltas=deltas,
            candidate=dict(candidate),
            baseline=prior_baseline if prior_baseline is not None else dict(candidate),
            receipt_id=receipt_id,
            metadata=dict(metadata or {}),
        )
        self.history.append(decision)
        if accepted:
            self.baseline = dict(candidate)
        return decision

    def _emit_receipt(
        self,
        *,
        accepted: bool,
        reasons: List[str],
        deltas: Dict[str, float],
        candidate: Dict[str, ScoreEstimate],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        try:
            from core.runtime.receipts import GovernanceReceipt, get_receipt_store

            store = get_receipt_store()
            receipt = store.emit(
                GovernanceReceipt(
                    cause="promotion_gate.compare",
                    domain="checkpoint_promotion",
                    action="promote" if accepted else "reject",
                    approved=bool(accepted),
                    reason="; ".join(reasons)[:1024],
                    metadata={
                        "deltas": deltas,
                        "candidate": {k: v.mean for k, v in candidate.items()},
                        "metadata": metadata or {},
                    },
                )
            )
            return receipt.receipt_id
        except Exception:
            return None
