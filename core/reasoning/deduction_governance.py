"""Governance signal for logical inconsistency in Aura's beliefs.

When the natural-deduction prover detects that Aura is holding a proposition and
its negation at high confidence, that is a constitutional concern — an
inconsistent self-model — analogous to the homeostatic-pressure breach in the
plasticity substrate. This module records those events, exposes a governance
signal the constitution/governor can poll, and emits a metric.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core.reasoning.belief_consistency import ConsistencyReport
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Reasoning.DeductionGovernance")


class _DeductionGovernance:
    def __init__(self) -> None:
        self._last_report: ConsistencyReport | None = None
        self._inconsistency_events = 0
        self._last_contradictions: list[tuple[str, str]] = []
        self._last_checked_at = 0.0
        self._non_sequitur_events = 0
        self._last_non_sequiturs: list[dict[str, Any]] = []
        self._arithmetic_error_events = 0
        self._last_arithmetic_errors: list[dict[str, Any]] = []

    def record(self, report: ConsistencyReport) -> None:
        self._last_report = report
        self._last_checked_at = time.time()
        if not report.consistent:
            self._inconsistency_events += 1
            self._last_contradictions = list(report.contradictions)
            for affirm, deny in report.contradictions:
                logger.warning(
                    "[Deduction] belief inconsistency: \"%s\" contradicts \"%s\" — "
                    "constitutional concern, surfacing for revision.",
                    affirm,
                    deny,
                )
            try:
                from core.observability.metrics import get_metrics

                get_metrics().increment_counter("belief_logical_inconsistency_total")
            except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
                record_degradation("deduction_governance", exc)

    def record_reasoning_audit(
        self,
        non_sequiturs: list[dict[str, Any]],
        arithmetic_errors: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record findings from the active-reasoning audit (her own draft replies)."""
        arithmetic_errors = arithmetic_errors or []
        if non_sequiturs:
            self._non_sequitur_events += len(non_sequiturs)
            self._last_non_sequiturs = list(non_sequiturs)
            for ns in non_sequiturs:
                logger.warning(
                    "[Deduction] non-sequitur in reasoning: \"%s\" does not follow from %s",
                    ns.get("conclusion"),
                    ns.get("premises"),
                )
        if arithmetic_errors:
            self._arithmetic_error_events += len(arithmetic_errors)
            self._last_arithmetic_errors = list(arithmetic_errors)
            for ae in arithmetic_errors:
                logger.warning(
                    "[Arithmetic] calculation error: \"%s\" (stated %s, correct %s)",
                    ae.get("claim"), ae.get("stated"), ae.get("correct"),
                )
        if not non_sequiturs and not arithmetic_errors:
            return
        try:
            from core.observability.metrics import get_metrics

            metrics = get_metrics()
            if non_sequiturs:
                metrics.increment_counter("reasoning_non_sequitur_total")
            if arithmetic_errors:
                metrics.increment_counter("reasoning_arithmetic_error_total")
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            record_degradation("deduction_governance", exc)

    def governance_signal(self) -> dict[str, Any]:
        report = self._last_report
        try:
            from core.reasoning.proof_kernel import get_theorem_ledger

            kernel = get_theorem_ledger().stats()
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            record_degradation("deduction_governance", exc, action="kernel stats unavailable")
            kernel = {}
        return {
            "beliefs_consistent": bool(report.consistent) if report else True,
            "contradictions": list(self._last_contradictions),
            "inconsistency_events": self._inconsistency_events,
            "checked": report.checked if report else 0,
            "last_checked_at": self._last_checked_at,
            "non_sequitur_events": self._non_sequitur_events,
            "last_non_sequiturs": list(self._last_non_sequiturs),
            "arithmetic_error_events": self._arithmetic_error_events,
            "last_arithmetic_errors": list(self._last_arithmetic_errors),
            # Lean-style kernel bookkeeping: proofs only count when the trusted
            # checker accepts them; admitted (sorry) claims stay visible until
            # discharged, and kernel rejections are prover soundness bugs.
            "proof_kernel": kernel,
            "kernel_sound": kernel.get("kernel_rejections", 0) == 0,
        }


_instance: _DeductionGovernance | None = None


def get_deduction_governance() -> _DeductionGovernance:
    global _instance
    if _instance is None:
        _instance = _DeductionGovernance()
    return _instance


def record_belief_inconsistency(report: ConsistencyReport) -> None:
    """Record a belief-consistency report (no-op safe for consistent reports)."""
    get_deduction_governance().record(report)
