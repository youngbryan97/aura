"""Concrete StateGateway adapter.

Implements `core.runtime.gateways.StateGateway`. Every state mutation
must pass through this gateway; mutations are durably committed via
atomic_writer with schema-versioned envelopes and recorded as
StateMutationReceipts. Mutations fail closed unless a governance authority
explicitly approves them.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import async_durable_unlink, read_json_envelope
from core.runtime.errors import record_degradation
from core.runtime.gateways import (
    StateGateway as StateGatewayBase,
)
from core.runtime.gateways import (
    StateMutationReceipt as StateMutationReceiptDC,
)
from core.runtime.gateways import (
    StateMutationRequest,
)
from core.runtime.receipts import StateMutationReceipt, get_receipt_store
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.StateGateway")
_SAFE_DOMAIN = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")


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
        root: Path | None = None,
        governance_decide: Callable[..., Any] | None = None,
    ):
        self.root = Path(root) if root else (state_root() / "state")
        self.root.mkdir(parents=True, exist_ok=True)
        self._governance = governance_decide
        self._lock = threading.RLock()
        self._mutation_lock = asyncio.Lock()
        self._cache: dict[tuple[str, str], Any] = {}

    async def mutate(self, request: StateMutationRequest) -> StateMutationReceiptDC:
        domain = _safe_domain(request.domain)
        safe_key = self._safe(request.key)
        approved, gov_receipt_id = await self._authorize(domain, request)
        if not approved:
            raise PermissionError(
                f"StateGateway: governance denied mutation of '{request.key}'"
            )
        target = self.root / domain / f"{safe_key}.json"
        schema_version = SCHEMA_VERSIONS.get(domain, SCHEMA_VERSIONS["default"])
        from core.runtime.atomic_writer import async_atomic_write_json

        async with self._mutation_lock:
            existed, old_payload = await asyncio.to_thread(_read_state_payload, target)
            old_value = old_payload.get("value") if existed else None
            payload = {
                "key": request.key,
                "value": request.new_value,
                "cause": request.cause,
                "at": time.time(),
            }
            await async_atomic_write_json(
                target,
                payload,
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
            try:
                emitted = await asyncio.to_thread(get_receipt_store().emit, receipt)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                try:
                    await _restore_state_payload(
                        target,
                        existed=existed,
                        payload=old_payload,
                        schema_version=schema_version,
                        schema_name=f"state.{domain}",
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as rollback_exc:
                    record_degradation(
                        "state_gateway",
                        rollback_exc,
                        severity="critical",
                        action="state receipt failed and compensating rollback also failed",
                        enforce_failure_policy=False,
                        extra={"target": str(target), "domain": domain, "key": request.key},
                    )
                    raise RuntimeError(
                        "state_mutation_receipt_failed_and_rollback_failed:"
                        f"{exc}; rollback={rollback_exc}"
                    ) from rollback_exc
                raise RuntimeError(f"state_mutation_receipt_failed_rolled_back:{exc}") from exc
            with self._lock:
                self._cache[(domain, request.key)] = request.new_value
        return StateMutationReceiptDC(
            key=request.key,
            old_value=old_value,
            new_value=request.new_value,
            receipt_id=emitted.receipt_id,
        )

    async def read(
        self,
        key: str,
        default: Any = None,
        *,
        domain: str = "world_state",
        fresh: bool = False,
    ) -> Any:
        domain = _safe_domain(domain)
        safe_key = self._safe(key)
        cache_key = (domain, key)
        if not fresh:
            with self._lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]
        target = self.root / domain / f"{safe_key}.json"
        existed, payload = await asyncio.to_thread(_read_state_payload, target)
        if not existed:
            if fresh:
                with self._lock:
                    self._cache.pop(cache_key, None)
            return default
        value = payload.get("value", default)
        with self._lock:
            self._cache[cache_key] = value
        return value

    async def snapshot(self, *, domain: str = "world_state") -> dict[str, Any]:
        domain = _safe_domain(domain)
        def read_domain() -> dict[str, Any]:
            rows: dict[str, Any] = {}
            for path in sorted((self.root / domain).glob("*.json")):
                _, payload = _read_state_payload(path)
                key = payload.get("key")
                if not isinstance(key, str) or not key or key in rows:
                    raise ValueError("state snapshot contains an invalid or duplicate key")
                rows[key] = payload.get("value")
            return rows

        async with self._mutation_lock:
            return await asyncio.to_thread(read_domain)

    async def _authorize(
        self,
        domain: str,
        request: StateMutationRequest,
    ) -> tuple[bool, str | None]:
        from core.governance_context import get_active_governance, require_governance

        active = get_active_governance()
        if active is not None:
            token = require_governance(
                "state_gateway.mutate",
                strict=True,
                allowed_domains={"state_mutation"},
            )
            return True, token.receipt_id
        if self._governance is None:
            logger.warning(
                "StateGateway has no governance authority; denying mutation of '%s' (fail-closed).",
                request.key,
            )
            return False, None
        try:
            decision = self._governance(
                domain="state_mutation",
                action=domain,
                cause=request.cause,
                context={"key": request.key},
            )
            if asyncio.iscoroutine(decision):
                decision = await decision
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('state_gateway', exc)
            logger.warning(
                "StateGateway governance call failed; denying mutation (fail-closed): %s",
                exc,
            )
            return False, None
        if isinstance(decision, dict):
            receipt_id = decision.get("receipt_id")
            return bool(decision.get("approved")), str(receipt_id) if receipt_id else None
        approved = getattr(decision, "is_approved", None)
        if callable(approved):
            receipt_id = getattr(decision, "receipt_id", None)
            return bool(approved()), str(receipt_id) if receipt_id else None
        return bool(decision), None

    @staticmethod
    def _safe(key: str) -> str:
        text = str(key or "").strip()
        if not text or "\x00" in text:
            raise ValueError("state key must be non-empty and contain no NUL bytes")
        return text.replace("/", "_").replace("\\", "_").replace(" ", "_")[:240]


def _safe_domain(value: Any) -> str:
    text = str(value or "world_state").strip()
    if text in {".", ".."} or not _SAFE_DOMAIN.fullmatch(text):
        raise ValueError(
            "state domain must be 1-96 letters, digits, dot, dash, or underscore"
        )
    return text


def _read_state_payload(path: Path) -> tuple[bool, dict[str, Any]]:
    if not path.exists():
        return False, {}
    envelope = read_json_envelope(path)
    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    if not isinstance(payload, dict):
        raise ValueError(f"state envelope payload is not a mapping: {path}")
    return True, dict(payload)


async def _restore_state_payload(
    path: Path,
    *,
    existed: bool,
    payload: dict[str, Any],
    schema_version: int,
    schema_name: str,
) -> None:
    if existed:
        from core.runtime.atomic_writer import async_atomic_write_json

        await async_atomic_write_json(
            path,
            payload,
            schema_version=schema_version,
            schema_name=schema_name,
        )
        return
    await async_durable_unlink(path, missing_ok=True)


# Alias for compatibility and closeout-rubric checks
StateGateway = ConcreteStateGateway


_global: ConcreteStateGateway | None = None


async def _default_state_governance_decide(**kwargs: Any) -> dict[str, Any]:
    from core.governance.will_client import WillClient, WillRequest

    decision = await WillClient().decide_async(
        WillRequest(
            content=f"state mutation:{kwargs.get('action', 'unknown')}",
            source="state_gateway",
            domain="state_mutation",
            priority=0.7,
            context=dict(kwargs),
        )
    )
    return {
        "approved": WillClient.is_approved(decision),
        "receipt_id": getattr(decision, "receipt_id", None),
    }


def get_state_gateway(*, root: Path | None = None) -> ConcreteStateGateway:
    """Explicit roots are contracts — see get_memory_write_gateway.

    Same latent flaw as the memory gateway (singleton silently ignored
    a differing explicit root); it only ever passed scenarios by boot-
    order luck. Explicit-root callers get a dedicated instance.
    """
    global _global
    if root is not None:
        resolved = Path(root)
        if _global is not None and Path(_global.root) == resolved:
            return _global
        return ConcreteStateGateway(
            root=resolved, governance_decide=_default_state_governance_decide
        )
    if _global is None:
        _global = ConcreteStateGateway(root=None, governance_decide=_default_state_governance_decide)
    return _global


def reset_state_gateway() -> None:
    global _global
    _global = None
