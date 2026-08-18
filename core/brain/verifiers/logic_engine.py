"""Logic truth engine — catch provable non-sequiturs in any prose answer.

Routes a candidate through :meth:`SymbolicBridge.audit_reasoning`, which runs the
natural-deduction prover over the reasoning's claims. ``domains = ("*",)`` so it
runs on every amplified answer: a deductive contradiction is a hard fail
regardless of task type.
"""
from __future__ import annotations

from typing import Any

from core.runtime.errors import record_degradation

from .base import VerificationResult


class LogicTruthEngine:
    name = "logic"
    domains = ("*",)

    def handles(self, task_type: str) -> bool:  # noqa: ARG002 - always-on
        return True

    async def verify(self, candidate: str, *, context: dict[str, Any] | None = None) -> VerificationResult:
        text = str(candidate or "")
        if len(text.split()) < 6:
            return VerificationResult(domain="logic", ok=True, checked=False, engine=self.name)
        try:
            from core.reasoning.symbolic_bridge import SymbolicBridge

            audit = SymbolicBridge().audit_reasoning(text)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("logic_truth_engine", exc)
            return VerificationResult(domain="logic", ok=True, checked=False, engine=self.name)

        non_sequiturs = audit.get("non_sequiturs", []) or []
        arithmetic = audit.get("arithmetic_errors", []) or []
        issues = [f"non-sequitur: {ns.get('conclusion')}" for ns in non_sequiturs]
        issues += [f"arithmetic: {ae.get('claim')}" for ae in arithmetic]
        checked = bool(audit.get("checked", False))
        ok = not non_sequiturs and not arithmetic
        score = 0.9 if ok else max(0.05, 0.5 - 0.15 * len(issues))
        return VerificationResult(
            domain="logic",
            ok=ok,
            checked=checked,
            score=round(score, 4),
            engine=self.name,
            issues=issues,
            detail={
                "non_sequiturs": len(non_sequiturs),
                "checked_inferences": int(audit.get("checked_inferences", 0) or 0),
                "checked_arithmetic_claims": int(
                    audit.get("checked_arithmetic_claims", 0) or 0
                ),
            },
        )
