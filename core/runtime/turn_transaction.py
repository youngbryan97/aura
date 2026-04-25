"""TurnTransaction — single-transaction wrapper for one user turn.

Audit-driven contract: a user turn must not partially commit. Every side
effect (history append, world state update, drive update, neurochemical
update, vector memory write, discourse tracker, ToM update, output emit)
is *staged* on the transaction. Once governance approves the turn, the
transaction commits the staged effects in a deterministic order. If any
critical effect fails, the transaction rolls back the others.

This object is intentionally small: the existing live code paths can
adopt it incrementally by replacing direct mutations / fire-and-forget
calls with tx.stage_*() and tx.commit().
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

logger = logging.getLogger("Aura.TurnTransaction")


EffectFn = Callable[[], Union[None, Awaitable[Any]]]


@dataclass
class StagedEffect:
    name: str
    apply: EffectFn
    rollback: Optional[EffectFn] = None
    criticality: str = "required"  # required | optional | telemetry
    payload: Dict[str, Any] = field(default_factory=dict)


class EffectCriticality:
    REQUIRED = "required"
    OPTIONAL = "optional"
    TELEMETRY = "telemetry"


@dataclass
class TurnReceipt:
    turn_id: str
    origin: str
    started_at: float
    finished_at: Optional[float] = None
    governance_receipt_id: Optional[str] = None
    approved: bool = False
    committed_effects: List[str] = field(default_factory=list)
    failed_effects: List[Dict[str, str]] = field(default_factory=list)
    rolled_back_effects: List[str] = field(default_factory=list)
    canceled: bool = False


class TurnTransactionError(RuntimeError):
    pass


class TurnTransaction:
    """Stage → approve → commit (or rollback) lifecycle for one user turn."""

    def __init__(
        self,
        *,
        origin: str,
        message: str,
        will: Any = None,
        governance_decide: Optional[Callable[..., Any]] = None,
    ):
        self.turn_id = f"turn-{uuid.uuid4()}"
        self.origin = origin
        self.message = message
        self._will = will
        self._governance_decide = governance_decide
        self._effects: List[StagedEffect] = []
        self.receipt = TurnReceipt(turn_id=self.turn_id, origin=origin, started_at=time.time())
        self._committed = False
        self._closed = False

    # --- staging ---------------------------------------------------------

    def stage(
        self,
        name: str,
        apply: EffectFn,
        *,
        rollback: Optional[EffectFn] = None,
        criticality: str = EffectCriticality.REQUIRED,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._committed:
            raise TurnTransactionError("cannot stage effect after commit")
        if criticality not in {EffectCriticality.REQUIRED, EffectCriticality.OPTIONAL, EffectCriticality.TELEMETRY}:
            raise ValueError(f"invalid criticality '{criticality}'")
        self._effects.append(
            StagedEffect(
                name=name,
                apply=apply,
                rollback=rollback,
                criticality=criticality,
                payload=dict(payload or {}),
            )
        )

    # --- approval --------------------------------------------------------

    async def approve(self) -> bool:
        decision = await self._invoke_governance()
        approved, receipt_id = self._extract_decision(decision)
        self.receipt.approved = approved
        self.receipt.governance_receipt_id = receipt_id
        if not approved:
            logger.info(
                "TurnTransaction %s denied by governance (origin=%s, receipt=%s)",
                self.turn_id, self.origin, receipt_id,
            )
        return approved

    async def _invoke_governance(self) -> Any:
        if self._governance_decide is not None:
            decision = self._governance_decide(
                domain="turn",
                action="commit",
                cause=self.origin,
                context={"turn_id": self.turn_id, "message": self.message[:120]},
            )
            if asyncio.iscoroutine(decision):
                decision = await decision
            return decision
        if self._will is not None:
            decide = getattr(self._will, "decide", None)
            if decide is None:
                return None
            try:
                decision = decide(
                    domain="turn",
                    action="commit",
                    cause=self.origin,
                    context={"turn_id": self.turn_id},
                )
            except TypeError:
                decision = decide("turn", "commit", self.origin, {"turn_id": self.turn_id})
            if asyncio.iscoroutine(decision):
                decision = await decision
            return decision
        # No governance configured: in strict mode this is fail-closed.
        if os.environ.get("AURA_STRICT_RUNTIME") == "1":
            raise TurnTransactionError("AURA_STRICT_RUNTIME: TurnTransaction needs a governance authority")
        return None

    @staticmethod
    def _extract_decision(decision: Any):
        if decision is None:
            return False, None
        if isinstance(decision, dict):
            return bool(decision.get("approved")), decision.get("receipt_id")
        approved = getattr(decision, "is_approved", None)
        if callable(approved):
            return bool(approved()), getattr(decision, "receipt_id", None)
        if hasattr(decision, "approved"):
            return bool(getattr(decision, "approved")), getattr(decision, "receipt_id", None)
        return bool(decision), None

    # --- commit / rollback ----------------------------------------------

    async def commit(self) -> TurnReceipt:
        if self._committed:
            raise TurnTransactionError("commit() called twice")
        self._committed = True
        if not self.receipt.approved:
            raise TurnTransactionError("cannot commit a denied transaction")
        applied: List[StagedEffect] = []
        for effect in self._effects:
            try:
                result = effect.apply()
                if asyncio.iscoroutine(result):
                    await result
                applied.append(effect)
                self.receipt.committed_effects.append(effect.name)
            except BaseException as exc:
                self.receipt.failed_effects.append({"name": effect.name, "error": repr(exc)})
                logger.error(
                    "TurnTransaction %s effect '%s' failed (criticality=%s): %s",
                    self.turn_id, effect.name, effect.criticality, exc,
                )
                if effect.criticality == EffectCriticality.REQUIRED:
                    await self._rollback(applied)
                    self.receipt.finished_at = time.time()
                    return self.receipt
        self.receipt.finished_at = time.time()
        return self.receipt

    async def cancel(self) -> TurnReceipt:
        """Rollback any already-applied effects and mark the receipt
        canceled. Used when an upstream gate (Will / capability) decides
        the turn should not commit even though stage() calls already ran."""
        self._committed = True
        self.receipt.canceled = True
        await self._rollback([])  # nothing applied yet by design
        self.receipt.finished_at = time.time()
        return self.receipt

    async def _rollback(self, applied: List[StagedEffect]) -> None:
        for effect in reversed(applied):
            if effect.rollback is None:
                continue
            try:
                result = effect.rollback()
                if asyncio.iscoroutine(result):
                    await result
                self.receipt.rolled_back_effects.append(effect.name)
            except BaseException as exc:
                logger.error(
                    "TurnTransaction %s rollback for '%s' failed: %s",
                    self.turn_id, effect.name, exc,
                )

    # --- context manager sugar ------------------------------------------

    async def __aenter__(self) -> "TurnTransaction":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if not self._committed:
            self._committed = True
            self.receipt.canceled = True
            await self._rollback([])
            self.receipt.finished_at = time.time()
        return False
