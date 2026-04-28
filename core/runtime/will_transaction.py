"""WillTransaction — transactional governance receipts for consequential actions.

Wraps the existing UnifiedWill.decide() flow in a context manager so callers
cannot bypass governance and cannot accidentally commit an action before
they have a receipt:

    async with WillTransaction(domain="memory", action="write", cause="user_msg") as txn:
        if not txn.approved:
            return
        await do_the_thing()
        txn.record_result({"bytes_written": 42})

    print(txn.receipt_id)

Behavior:

- entering the block calls UnifiedWill.decide()
- approved=False -> the block runs but txn.approved is False; callers
  must short-circuit. Recording a result on a denied transaction is a
  hard error.
- approved=True -> callers must call txn.record_result(...) before exit
  in strict mode, otherwise a degraded event is logged.
- the receipt is always returned via txn.receipt_id and is queryable
  through UnifiedWill.verify_receipt().

The transaction never silently fails open: if Will is unavailable or
raises, the transaction is treated as DENIED and a violation is logged.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.WillTransaction")


@dataclass
class WillTransactionRecord:
    """Audit-shaped record of one transaction. Persisted via the existing
    UnifiedWill receipt store."""

    domain: str
    action: str
    cause: str
    started_at: float
    decision: Any = None
    receipt_id: Optional[str] = None
    approved: bool = False
    result: Optional[Dict[str, Any]] = None
    finished_at: Optional[float] = None
    failure: Optional[str] = None


class WillTransactionError(RuntimeError):
    """Raised when a caller violates the transaction contract (e.g. records
    a result on a denied transaction)."""


class WillTransaction:
    def __init__(
        self,
        *,
        domain: str,
        action: str,
        cause: str,
        context: Optional[Dict[str, Any]] = None,
        will: Any = None,
    ):
        self.record = WillTransactionRecord(
            domain=domain,
            action=action,
            cause=cause,
            started_at=time.time(),
        )
        self._context = context or {}
        self._will = will  # injected for tests; resolved lazily otherwise
        self._entered = False
        self._committed = False

    @property
    def approved(self) -> bool:
        return self.record.approved

    @property
    def receipt_id(self) -> Optional[str]:
        return self.record.receipt_id

    @property
    def decision(self) -> Any:
        return self.record.decision

    # --- async context ------------------------------------------------------

    async def __aenter__(self) -> "WillTransaction":
        if self._entered:
            raise WillTransactionError("WillTransaction may not be re-entered")
        self._entered = True
        try:
            will = self._resolve_will()
            decision = await self._invoke_decide(will)
            self.record.decision = decision
            approved = self._is_approved(decision)
            self.record.approved = approved
            self.record.receipt_id = self._extract_receipt_id(decision)
            if not approved:
                logger.info(
                    "WillTransaction DENIED: domain=%s action=%s cause=%s",
                    self.record.domain,
                    self.record.action,
                    self.record.cause,
                )
        except Exception as exc:
            record_degradation('will_transaction', exc)
            self.record.approved = False
            self.record.failure = repr(exc)
            logger.error(
                "WillTransaction governance call failed; treating as DENIED: %s",
                exc,
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self.record.finished_at = time.time()
        if exc is not None:
            self.record.failure = self.record.failure or repr(exc)
            return False  # propagate
        if (
            self.record.approved
            and self.record.result is None
            and not self._committed
            and os.environ.get("AURA_STRICT_RUNTIME") == "1"
        ):
            logger.error(
                "WillTransaction approved but no result recorded; "
                "potential ungoverned-effect path. domain=%s action=%s",
                self.record.domain,
                self.record.action,
            )
        return False

    # --- caller surface -----------------------------------------------------

    def record_result(self, result: Dict[str, Any]) -> None:
        if not self.record.approved:
            raise WillTransactionError(
                "cannot record result on a denied WillTransaction"
            )
        if self._committed:
            raise WillTransactionError(
                "WillTransaction result already recorded"
            )
        self.record.result = dict(result or {})
        self._committed = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.record.domain,
            "action": self.record.action,
            "cause": self.record.cause,
            "started_at": self.record.started_at,
            "finished_at": self.record.finished_at,
            "approved": self.record.approved,
            "receipt_id": self.record.receipt_id,
            "result": self.record.result,
            "failure": self.record.failure,
        }

    # --- internal -----------------------------------------------------------

    def _resolve_will(self) -> Any:
        if self._will is not None:
            return self._will
        try:
            from core.will import get_will

            return get_will()
        except Exception as exc:
            record_degradation('will_transaction', exc)
            logger.debug("UnifiedWill unavailable: %s", exc)
            return None

    async def _invoke_decide(self, will: Any) -> Any:
        if will is None:
            return None
        decide = getattr(will, "decide", None)
        if decide is None:
            return None
        try:
            decision = decide(
                domain=self.record.domain,
                action=self.record.action,
                cause=self.record.cause,
                context=self._context,
            )
        except TypeError:
            # Some Will variants take positional args.
            decision = decide(
                self.record.domain,
                self.record.action,
                self.record.cause,
                self._context,
            )
        if asyncio.iscoroutine(decision):
            decision = await decision
        return decision

    @staticmethod
    def _is_approved(decision: Any) -> bool:
        if decision is None:
            return False
        approved = getattr(decision, "is_approved", None)
        if callable(approved):
            try:
                return bool(approved())
            except Exception:
                return False
        if isinstance(decision, dict):
            return bool(decision.get("approved", False))
        outcome = getattr(decision, "outcome", None)
        if outcome is not None:
            value = getattr(outcome, "value", str(outcome)).lower()
            return value in {"approved", "approve", "allow", "allowed"}
        return bool(decision)

    @staticmethod
    def _extract_receipt_id(decision: Any) -> Optional[str]:
        if decision is None:
            return None
        if isinstance(decision, dict):
            return decision.get("receipt_id") or decision.get("receipt")
        for attr in ("receipt_id", "receipt", "id"):
            value = getattr(decision, attr, None)
            if value:
                return str(value)
        return None
