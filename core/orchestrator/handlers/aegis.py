"""Aegis sentinel loop for mycelial integrity protection."""
from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from typing import TYPE_CHECKING, Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.service_registry import get_runtime_service, register_runtime_service
from core.runtime.wall_clock import monotonic_time

if TYPE_CHECKING:
    from core.orchestrator.main import RobustOrchestrator

logger = logging.getLogger("Aura.Core.Orchestrator.Aegis")

_AEGIS_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_DEFAULT_SENTINEL_INTERVAL_S = 10.0
_DEFAULT_VAULT_SYNC_INTERVAL_S = 60.0


def _record_aegis_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        record_degradation(
            "aegis",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation("aegis", error, severity=severity, action=action)
        except TypeError:
            logger.debug("AEGIS degradation could not be recorded: %s", signature_exc)


def _set_aegis_status(orch: RobustOrchestrator, **updates: Any) -> dict[str, Any]:
    status = dict(getattr(orch, "aegis_status", {}) or {})
    status.update(updates)
    status["updated_at_monotonic"] = time.monotonic()
    orch.aegis_status = status
    return status


def _is_stop_requested(orch: RobustOrchestrator) -> bool:
    stop_event = getattr(orch, "_stop_event", None)
    is_set = getattr(stop_event, "is_set", None)
    if callable(is_set):
        try:
            return bool(is_set())
        except _AEGIS_ERRORS as exc:
            _record_aegis_degradation(
                exc,
                action="treated unreadable stop event as not stopped for aegis loop",
                severity="warning",
            )
    return False


async def _sleep_or_stop(orch: RobustOrchestrator, seconds: float) -> None:
    deadline = monotonic_time() + max(0.0, seconds)
    while not _is_stop_requested(orch):
        remaining = deadline - monotonic_time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 1.0))


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


async def _aegis_pulse(
    orch: RobustOrchestrator,
    *,
    vault_sync_interval_s: float = _DEFAULT_VAULT_SYNC_INTERVAL_S,
) -> dict[str, Any]:
    """Run one integrity pulse and return an operational status record."""
    from core.mycelium import MycelialNetwork

    mycelium = get_runtime_service("mycelial_network", default=None)
    if mycelium is None:
        restored = MycelialNetwork()
        try:
            register_runtime_service("mycelial_network", restored, required=False, owner="core/orchestrator/handlers/aegis.py", registered_by="_run_aegis_integrity_pulse")
        except _AEGIS_ERRORS as exc:
            orch._aegis_integrity_failed = True
            _record_aegis_degradation(
                exc,
                action="failed closed after mycelial network restoration could not be registered",
                severity="critical",
            )
            return _set_aegis_status(
                orch,
                state="failed_closed",
                reason="mycelial_registration_failed",
                locked=bool(getattr(restored, "_aegis_locked", False)),
            )
        return _set_aegis_status(
            orch,
            state="restored",
            reason="mycelial_network_missing",
            locked=bool(getattr(restored, "_aegis_locked", False)),
        )

    if not getattr(mycelium, "_aegis_locked", False):
        restore_fn = getattr(MycelialNetwork, "restore_from_vault", None)
        restored_ok = False
        if callable(restore_fn):
            restored_ok = bool(await _maybe_await(restore_fn()))
        if restored_ok:
            return _set_aegis_status(
                orch,
                state="restored",
                reason="true_lock_restored_from_vault",
                locked=True,
            )
        orch._aegis_integrity_failed = True
        _record_aegis_degradation(
            RuntimeError("mycelial true-lock disabled and vault restore did not succeed"),
            action="marked aegis integrity failed after true-lock restoration failed",
            severity="critical",
            extra={"has_restore": callable(restore_fn)},
        )
        return _set_aegis_status(
            orch,
            state="failed_closed",
            reason="true_lock_restore_failed",
            locked=False,
        )

    now = time.monotonic()
    raw_last_sync = getattr(orch, "_last_vault_sync", 0.0)
    try:
        last_sync = float(raw_last_sync or 0.0)
    except (TypeError, ValueError) as exc:
        _record_aegis_degradation(
            exc,
            action="reset malformed root-vault sync clock for immediate retry",
        )
        last_sync = 0.0
    if not math.isfinite(last_sync) or last_sync < 0.0 or last_sync > now:
        clock_error = ValueError(
            "root-vault sync clock is non-finite or outside this process epoch"
        )
        _record_aegis_degradation(
            clock_error,
            action="reset invalid root-vault sync clock for immediate retry",
        )
        last_sync = 0.0
    synced = False
    if now - last_sync >= max(1.0, vault_sync_interval_s):
        vault_sync = getattr(mycelium, "vault_sync", None)
        if callable(vault_sync):
            synced = bool(await _maybe_await(vault_sync()))
            if not synced:
                error = RuntimeError("root-vault sync returned an unsuccessful receipt")
                _record_aegis_degradation(
                    error,
                    action=(
                        "retained immediate root-vault retry eligibility after "
                        "persistence failed"
                    ),
                    severity="degraded",
                )
                return _set_aegis_status(
                    orch,
                    state="degraded",
                    reason="root_vault_sync_failed",
                    locked=True,
                    vault_synced=False,
                    vault_sync_retry_eligible=True,
                )
            orch._last_vault_sync = time.monotonic()

    return _set_aegis_status(
        orch,
        state="healthy",
        reason="true_lock_verified",
        locked=True,
        vault_synced=synced,
    )


async def aegis_sentinel_loop(
    orch: RobustOrchestrator,
    *,
    interval_s: float = _DEFAULT_SENTINEL_INTERVAL_S,
    vault_sync_interval_s: float = _DEFAULT_VAULT_SYNC_INTERVAL_S,
) -> None:
    """Maintain mycelial integrity until orchestrator shutdown."""
    logger.info("AEGIS SENTINEL: Narrative integrity guard active")
    failures = 0
    interval = max(1.0, float(interval_s))
    _set_aegis_status(orch, state="starting", reason="sentinel_loop_entered")

    while not _is_stop_requested(orch):
        sleep_s = interval
        try:
            await _aegis_pulse(orch, vault_sync_interval_s=vault_sync_interval_s)
            failures = 0
        except _AEGIS_ERRORS as exc:
            failures += 1
            sleep_s = min(interval * (1 + failures), interval * 6)
            _set_aegis_status(
                orch,
                state="degraded",
                reason=f"{type(exc).__name__}: {exc}",
                consecutive_failures=failures,
            )
            _record_aegis_degradation(
                exc,
                action="kept aegis sentinel alive and backed off after pulse failure",
                severity="degraded",
                extra={"consecutive_failures": failures},
            )
            logger.debug("AEGIS Sentinel pulse failed: %s", exc)
        await _sleep_or_stop(orch, sleep_s)
