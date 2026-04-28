"""Concrete StateGateway adapter.

Implements `core.runtime.gateways.StateGateway`. Every state mutation
must pass through this gateway; mutations are durably committed via
atomic_writer with schema-versioned envelopes and recorded as
StateMutationReceipts.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.runtime.atomic_writer import atomic_write_json, read_json_envelope
from core.runtime.gateways import (
    StateGateway as StateGatewayBase,
    StateMutationReceipt as StateMutationReceiptDC,
    StateMutationRequest,
)
from core.runtime.receipts import StateMutationReceipt, get_receipt_store

logger = logging.getLogger("Aura.StateGateway")


SCHEMA_VERSIONS = {
    "world_state": 1,
    "drives": 1,
    "neurochemicals": 1,
    "discourse": 1,
    "default": 1,
}


class ConcreteStateGateway(StateGatewayBase):
    """Single canonical state mutation authority."""

    def __init__(
        self,
        *,
        root: Optional[Path] = None,
        governance_decide: Optional[Callable[..., Any]] = None,
    ):
        self.root = Path(root) if root else (Path.home() / ".aura" / "state")
        self.root.mkdir(parents=True, exist_ok=True)
        self._governance = governance_decide
        self._lock = threading.RLock()
        self._cache: Dict[str, Any] = {}

    async def mutate(self, request: StateMutationRequest) -> StateMutationReceiptDC:
        domain = (request.cause if "/" not in request.key else request.key.split("/", 1)[0]) or "world_state"
        approved, gov_receipt_id = await self._authorize(domain, request)
        if not approved:
            raise PermissionError(
                f"StateGateway: governance denied mutation of '{request.key}'"
            )
        with self._lock:
            old_value = self._cache.get(request.key)
            self._cache[request.key] = request.new_value
        target = self.root / domain / f"{self._safe(request.key)}.json"
        schema_version = SCHEMA_VERSIONS.get(domain, SCHEMA_VERSIONS["default"])
        atomic_write_json(
            target,
            {"key": request.key, "value": request.new_value, "cause": request.cause, "at": time.time()},
            schema_version=schema_version,
            schema_name=f"state.{domain}",
        )
        receipt = StateMutationReceipt(
            receipt_id=f"statemut-{uuid.uuid4()}",
            cause=request.cause,
            domain=domain,
            key=request.key,
            schema_version=schema_version,
            governance_receipt_id=gov_receipt_id or request.receipt_id,
            metadata={"path": str(target)},
        )
        get_receipt_store().emit(receipt)
        return StateMutationReceiptDC(
            key=request.key,
            old_value=old_value,
            new_value=request.new_value,
            receipt_id=gov_receipt_id or request.receipt_id or "rcpt-pending",
        )

    async def read(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        # Best-effort durable read.
        for domain_dir in self.root.iterdir():
            if not domain_dir.is_dir():
                continue
            candidate = domain_dir / f"{self._safe(key)}.json"
            if candidate.exists():
                try:
                    env = read_json_envelope(candidate)
                    payload = env.get("payload") or {}
                    value = payload.get("value", default)
                    with self._lock:
                        self._cache[key] = value
                    return value
                except Exception:
                    continue
        return default

    async def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._cache)

    async def _authorize(self, domain: str, request: StateMutationRequest):
        if self._governance is None:
            return True, request.receipt_id
        try:
            decision = self._governance(
                domain="state_mutation",
                action=domain,
                cause=request.cause,
                context={"key": request.key},
            )
            if asyncio.iscoroutine(decision):
                decision = await decision
        except Exception as exc:
            record_degradation('state_gateway', exc)
            logger.warning(
                "StateGateway governance call failed; denying mutation (fail-closed): %s",
                exc,
            )
            return False, None
        if isinstance(decision, dict):
            return bool(decision.get("approved")), decision.get("receipt_id")
        approved = getattr(decision, "is_approved", None)
        if callable(approved):
            return bool(approved()), getattr(decision, "receipt_id", None)
        return bool(decision), None

    @staticmethod
    def _safe(key: str) -> str:
        return key.replace("/", "_").replace(" ", "_")


_global: Optional[ConcreteStateGateway] = None


def get_state_gateway(*, root: Optional[Path] = None) -> ConcreteStateGateway:
    global _global
    if _global is None:
        _global = ConcreteStateGateway(root=root)
    return _global


def reset_state_gateway() -> None:
    global _global
    _global = None
