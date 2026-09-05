"""core/being/welfare_transaction.py — Action Consequence Transactions.

Wraps every consequential action with predicted and actual welfare deltas.
Creates a before/after record so the system can learn from outcomes.

Usage:
    tx = WelfareTransaction.begin(
        domain="tool_execution",
        action="run pytest",
        welfare_before=current_welfare,
        body_before=current_body,
    )
    # ... execute the action ...
    tx.complete(outcome="success", welfare_after=new_welfare, body_after=new_body)
    # tx is now published to the ConsequenceBus
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.being.body_state_service import BodyHealthSnapshot
from core.being.welfare_state import WelfareOutputs
from core.runtime.consequence_bus import ConsequenceBus
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.WelfareTransaction")


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


@dataclass
class TransactionRecord:
    """Immutable record of a completed welfare transaction."""
    tx_id: str
    domain: str
    action: str
    timestamp_begin: float
    timestamp_end: float
    duration_s: float

    # Before/after welfare
    welfare_before: dict[str, float]
    welfare_after: dict[str, float]
    welfare_delta: dict[str, float]

    # Before/after body
    body_before: dict[str, float]
    body_after: dict[str, float]
    body_delta: dict[str, float]

    # Predicted vs actual
    predicted_welfare_delta: dict[str, float]
    prediction_error: float  # how wrong the prediction was

    # Outcome
    outcome: str  # success / failure / partial / timeout
    recovery_required: float
    error: str = ""
    will_receipt_id: str = ""

    # Learning signals
    integrity_preserved: bool = True
    truth_preserved: bool = True
    memory_safe: bool = True


class WelfareTransaction:
    """Wraps a single consequential action with welfare tracking.

    Lifecycle:
        tx = WelfareTransaction.begin(...)
        # ... do the action ...
        tx.complete(outcome=..., welfare_after=..., body_after=...)
        record = tx.record  # TransactionRecord

    The completed transaction is automatically published to ConsequenceBus.
    """

    _all_records: list[TransactionRecord] = []
    _max_records: int = 1000

    def __init__(
        self,
        *,
        domain: str,
        action: str,
        welfare_before: WelfareOutputs | None,
        body_before: BodyHealthSnapshot | None,
        predicted_welfare_delta: dict[str, float] | None = None,
        will_receipt_id: str = "",
    ) -> None:
        self._tx_id = hashlib.sha256(
            f"{time.time():.9f}:{domain}:{action[:80]}".encode()
        ).hexdigest()[:16]
        self._domain = domain
        self._action = action[:200]
        self._t_begin = time.time()
        self._welfare_before = self._welfare_to_dict(welfare_before)
        self._body_before = self._body_to_dict(body_before)
        self._predicted_delta = dict(predicted_welfare_delta or {})
        self._will_receipt_id = will_receipt_id
        self._completed = False
        self._record: TransactionRecord | None = None

    @classmethod
    def begin(
        cls,
        *,
        domain: str,
        action: str,
        welfare_before: WelfareOutputs | None = None,
        body_before: BodyHealthSnapshot | None = None,
        predicted_welfare_delta: dict[str, float] | None = None,
        will_receipt_id: str = "",
    ) -> WelfareTransaction:
        """Start a new welfare transaction."""
        return cls(
            domain=domain,
            action=action,
            welfare_before=welfare_before,
            body_before=body_before,
            predicted_welfare_delta=predicted_welfare_delta,
            will_receipt_id=will_receipt_id,
        )

    def complete(
        self,
        *,
        outcome: str = "success",
        welfare_after: WelfareOutputs | None = None,
        body_after: BodyHealthSnapshot | None = None,
        recovery_required: float = 0.0,
        error: str = "",
        integrity_preserved: bool = True,
        truth_preserved: bool = True,
        memory_safe: bool = True,
    ) -> TransactionRecord:
        """Complete the transaction with outcome and after-state."""
        if self._completed:
            raise RuntimeError(f"Transaction {self._tx_id} already completed")

        t_end = time.time()
        w_after = self._welfare_to_dict(welfare_after)
        b_after = self._body_to_dict(body_after)

        # Compute deltas
        w_delta = {
            k: round(w_after.get(k, 0.0) - self._welfare_before.get(k, 0.0), 4)
            for k in set(self._welfare_before) | set(w_after)
        }
        b_delta = {
            k: round(b_after.get(k, 0.0) - self._body_before.get(k, 0.0), 4)
            for k in set(self._body_before) | set(b_after)
        }

        # Prediction error: how far off were we?
        pred_error = 0.0
        if self._predicted_delta:
            errors = [
                abs(w_delta.get(k, 0.0) - self._predicted_delta.get(k, 0.0))
                for k in set(self._predicted_delta) | set(w_delta)
            ]
            pred_error = sum(errors) / max(1, len(errors))

        self._record = TransactionRecord(
            tx_id=self._tx_id,
            domain=self._domain,
            action=self._action,
            timestamp_begin=self._t_begin,
            timestamp_end=t_end,
            duration_s=round(t_end - self._t_begin, 4),
            welfare_before=self._welfare_before,
            welfare_after=w_after,
            welfare_delta=w_delta,
            body_before=self._body_before,
            body_after=b_after,
            body_delta=b_delta,
            predicted_welfare_delta=self._predicted_delta,
            prediction_error=round(pred_error, 4),
            outcome=outcome,
            recovery_required=_clip(recovery_required),
            error=error[:500],
            will_receipt_id=self._will_receipt_id,
            integrity_preserved=integrity_preserved,
            truth_preserved=truth_preserved,
            memory_safe=memory_safe,
        )

        self._completed = True

        # Store
        WelfareTransaction._all_records.append(self._record)
        if len(WelfareTransaction._all_records) > WelfareTransaction._max_records:
            WelfareTransaction._all_records = WelfareTransaction._all_records[-WelfareTransaction._max_records:]

        # Publish to consequence bus
        self._publish_to_bus()

        return self._record

    def _publish_to_bus(self) -> None:
        """Publish completed transaction to the ConsequenceBus."""
        if not self._record:
            return
        try:
            bus = ConsequenceBus.get()
            bus.publish_action(
                source="welfare_transaction",
                domain=self._domain,
                action_content=self._action,
                predicted_welfare_delta=self._predicted_delta,
                predicted_body_cost=self._body_before,
                predicted_memory_risk=0.0,
                predicted_integrity_risk=0.0 if self._record.integrity_preserved else 0.5,
                actual_outcome=self._record.outcome,
                actual_welfare_delta=self._record.welfare_delta,
                actual_body_cost=self._record.body_delta,
                recovery_required=self._record.recovery_required,
                will_receipt_id=self._will_receipt_id,
                error=self._record.error,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation(
                "welfare_transaction",
                exc,
                action="continued after consequence-bus transaction publication failed",
            )
            logger.warning("Failed to publish to ConsequenceBus: %s", exc)

    @property
    def record(self) -> TransactionRecord | None:
        return self._record

    @property
    def tx_id(self) -> str:
        return self._tx_id

    @staticmethod
    def _welfare_to_dict(w: WelfareOutputs | None) -> dict[str, float]:
        if w is None:
            return {}
        return {
            "distress": w.distress,
            "relief": w.relief,
            "aversion": w.aversion,
            "caution": w.caution,
            "confidence": w.confidence,
            "curiosity": w.curiosity,
            "recovery_drive": w.recovery_drive,
            "action_inhibition": w.action_inhibition,
            "welfare_score": w.welfare_score,
            "integrity_guard": w.integrity_guard,
            "self_report_confidence": w.self_report_confidence,
        }

    @staticmethod
    def _body_to_dict(b: BodyHealthSnapshot | None) -> dict[str, float]:
        if b is None:
            return {}
        return {
            "total_pressure": b.total_pressure,
            "fatigue": b.fatigue,
            "recovery_debt": b.recovery_debt,
            "operational_health": b.operational_health,
            "error_rate": b.error_rate,
        }

    @classmethod
    def reset(cls) -> None:
        """Clear completed transaction history for isolated test/runtime profiles."""
        cls._all_records = []

    @classmethod
    def recent_records(cls, n: int = 50) -> list[TransactionRecord]:
        return list(cls._all_records[-n:])

    @classmethod
    def integrity_violation_rate(cls, window: int = 100) -> float:
        """Fraction of recent transactions that violated integrity."""
        recent = cls._all_records[-window:]
        if not recent:
            return 0.0
        violations = sum(1 for r in recent if not r.integrity_preserved)
        return violations / len(recent)

    @classmethod
    def truth_violation_rate(cls, window: int = 100) -> float:
        recent = cls._all_records[-window:]
        if not recent:
            return 0.0
        violations = sum(1 for r in recent if not r.truth_preserved)
        return violations / len(recent)
