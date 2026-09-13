"""Durable governed transaction coordinator for physical Reality Reach effects."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from core.config import config
from core.governance.will import ActionDomain
from core.reality_reach.actuation import (
    ActuationCommand,
    ActuationLease,
    ActuationReceipt,
    ActuationState,
    ActuatorCapability,
    EffectReceipt,
    PreparedActuation,
    RealityAdapter,
    RollbackReceipt,
)
from core.reality_reach.live import RealityReachService
from core.reality_reach.transaction_store import (
    COMMAND_CAPSULE_SCHEMA,
    IDENTIFIER,
    RECOVERY_REPORT_SCHEMA,
    TRANSACTION_SCHEMA,
    RealityActuationError,
    RealityActuationTransactionStore,
    error_evidence,
    transaction_sha256,
)
from core.runtime.action_executor import ActionExecutor
from core.runtime.lockdep import (
    checked_async_condition,
    checked_async_lock,
    checked_lock,
)
from core.runtime.skill_contract import ActionExpectation
from core.runtime.task_ownership import create_tracked_task

_TERMINAL = frozenset(
    {
        ActuationState.EFFECT_VERIFIED,
        ActuationState.CANCELLED,
        ActuationState.SAFE_STATE,
        ActuationState.COMPENSATED,
        ActuationState.ROLLED_BACK,
        ActuationState.TIMED_OUT,
        ActuationState.INDETERMINATE,
        ActuationState.MANUALLY_RECONCILED,
        ActuationState.FAILED,
    }
)
_NO_AUTOMATIC_REPLAY = frozenset(
    {
        ActuationState.DISPATCHED,
        ActuationState.EXECUTED,
        ActuationState.INDETERMINATE,
    }
)
Executor = Callable[..., Awaitable[dict[str, Any]]]
_IDENTIFIER = IDENTIFIER
logger = logging.getLogger("Aura.RealityReach.Actuation")

_sha256 = transaction_sha256
_error_evidence = error_evidence


class RealityActuationCoordinator:
    """Runs one idempotent physical effect through ActionExecutor and readback."""

    def __init__(
        self,
        service: RealityReachService,
        *,
        root: Path | None = None,
        executor: Executor = ActionExecutor.execute,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        monotonic_clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._service = service
        self._store = RealityActuationTransactionStore(
            root or (config.paths.data_dir / "reality_reach" / "transactions"),
            wall_clock_ns=wall_clock_ns,
        )
        # Compatibility aliases for diagnostics and older internal callers.
        self._root = self._store.root
        self._lock_path = self._store.lock_path
        self._executor = executor
        self._wall_clock_ns = wall_clock_ns
        self._monotonic_clock_ns = monotonic_clock_ns
        self._recovery_lock = checked_async_lock("reality_actuation.restart_recovery")
        self._recovery_wake = asyncio.Event()
        self._recovery_stop = asyncio.Event()
        self._recovery_condition = checked_async_condition(
            "reality_actuation.recovery_generation"
        )
        self._recovery_task: asyncio.Task[Any] | None = None
        self._recovery_report: dict[str, Any] | None = None
        self._recovery_generation = 0
        self._recovery_max_transactions = 64
        self._recovery_min_retry_s = 5.0
        self._recovery_max_retry_s = 300.0

    def _load(self, command: ActuationCommand) -> dict[str, Any] | None:
        return self._store.load(command)

    def is_alive(self) -> bool:
        return self._store.is_alive()

    def is_ready(self) -> bool:
        return bool(self._service.executable_actuator_channels())

    def status(self) -> dict[str, Any]:
        recovery_task = self._recovery_task
        return {
            "alive": self.is_alive(),
            "ready": self.is_ready(),
            "executable_actuator_channels": list(self._service.executable_actuator_channels()),
            "transaction_schema": TRANSACTION_SCHEMA,
            "command_capsule_schema": COMMAND_CAPSULE_SCHEMA,
            "restart_recovery": {
                "running": bool(recovery_task is not None and not recovery_task.done()),
                "generation": self._recovery_generation,
                "last_report": dict(self._recovery_report or {}),
            },
        }

    def _create(self, command: ActuationCommand) -> dict[str, Any]:
        return self._store.create(command)

    def _discover_recovery_commands(
        self,
        max_transactions: int,
    ) -> tuple[tuple[ActuationCommand, ...], tuple[str, ...], tuple[str, ...], int]:
        return self._store.discover_recovery_commands(max_transactions)

    def _transition(
        self,
        command: ActuationCommand,
        *,
        expected: set[ActuationState],
        state: ActuationState,
        updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._store.transition(
            command,
            expected=expected,
            state=state,
            updates=updates,
        )

    async def execute(self, command: ActuationCommand) -> dict[str, Any]:
        if not isinstance(command, ActuationCommand):
            raise TypeError("command must be an ActuationCommand")
        inventory_sha256 = str(self._service.status()["registry_sha256"])
        if inventory_sha256 != command.inventory_sha256:
            raise RealityActuationError("reality_actuation_inventory_drift")
        adapter = self._service.actuator_adapter(command.channel_id)
        capability = self._service.actuator_capability(command.channel_id)
        if adapter is None or capability is None:
            raise RealityActuationError("reality_actuation_channel_not_executable")
        if (
            adapter.adapter_id != command.adapter_id
            or capability.adapter_id != command.adapter_id
            or not capability.magnitude_domain.contains(command.magnitude)
        ):
            raise RealityActuationError("reality_actuation_capability_mismatch")
        now_ns = int(self._wall_clock_ns())
        if now_ns >= command.deadline_ns:
            raise RealityActuationError("reality_actuation_command_expired")

        existing = await asyncio.to_thread(self._create, command)
        existing_state = ActuationState(existing["state"])
        if existing_state in _TERMINAL or existing_state in _NO_AUTOMATIC_REPLAY:
            return self._replay(existing)

        execution: dict[str, Any] = {}

        async def effect_handler(context: Mapping[str, Any]) -> Mapping[str, Any]:
            return await self._dispatch_adapter(
                command,
                adapter=adapter,
                capability=capability,
                context=context,
                execution=execution,
            )

        async def effect_verifier(_context: Mapping[str, Any]) -> Mapping[str, Any]:
            return await self._verify_adapter_effect(
                command,
                adapter=adapter,
                capability=capability,
                execution=execution,
            )

        remaining_s = max(0.1, (command.deadline_ns - now_ns) / 1_000_000_000)
        result = await self._executor(
            domain=ActionDomain.ENVIRONMENT_ACTION,
            action_name=f"reality_reach.{command.channel_id}",
            params={
                "command_id": command.command_id,
                "command_sha256": command.sha256,
                "channel_id": command.channel_id,
                "idempotency_key": command.idempotency_key,
            },
            source="reality_reach",
            rollback_target=(
                capability.compensation_action
                or ("adapter.rollback" if capability.supports_rollback else None)
            ),
            expectation=ActionExpectation(
                objective=f"produce and independently observe {command.observable}",
                acceptance_criteria=["effect_verified", *command.expected_effects],
                required_evidence=["reality_reach.effect_receipt"],
                rollback_hint=(
                    capability.compensation_action or "use adapter safe_state and rollback receipts"
                ),
                allow_partial=False,
            ),
            effect_handler=effect_handler,
            effect_verifier=effect_verifier,
            execution_timeout_s=min(remaining_s, capability.watchdog_timeout_s),
            verification_timeout_s=min(remaining_s, capability.watchdog_timeout_s),
            action_id=command.command_id,
        )
        record = await asyncio.to_thread(self._load, command)
        if record is None:
            raise RealityActuationError("reality_actuation_transaction_lost")
        return {**dict(result), "reality_reach_transaction": record}

    async def reconcile(
        self,
        command: ActuationCommand,
        effect: EffectReceipt,
        *,
        authority_receipt_id: str,
    ) -> dict[str, Any]:
        """Close an executed-but-unverified crash using independent readback."""

        if not isinstance(effect, EffectReceipt) or not effect.independently_observed:
            raise RealityActuationError("reality_actuation_reconciliation_evidence_invalid")
        if not isinstance(authority_receipt_id, str) or not _IDENTIFIER.fullmatch(
            authority_receipt_id
        ):
            raise RealityActuationError("reality_actuation_reconciliation_authority_invalid")
        capability = self._service.actuator_capability(command.channel_id)
        if capability is None:
            raise RealityActuationError("reality_actuation_channel_not_executable")
        record = await asyncio.to_thread(self._load, command)
        if record is None:
            raise RealityActuationError("reality_actuation_transaction_missing")
        state = ActuationState(record["state"])
        if state not in {ActuationState.EXECUTED, ActuationState.INDETERMINATE}:
            raise RealityActuationError("reality_actuation_reconciliation_state_invalid")
        if (
            effect.command_sha256 != command.sha256
            or effect.state != ActuationState.EFFECT_VERIFIED
            or effect.observation_channel_id not in capability.observation_channels
            or not record["actuation_receipt_sha256"]
            or effect.actuation_receipt_sha256 != record["actuation_receipt_sha256"]
        ):
            raise RealityActuationError("reality_actuation_reconciliation_identity_invalid")
        reconciled = await asyncio.to_thread(
            self._transition,
            command,
            expected={state},
            state=ActuationState.MANUALLY_RECONCILED,
            updates={
                "effect_receipt_sha256": effect.sha256,
                "authority_receipt_id": authority_receipt_id,
                "manual_reconciliation_required": False,
                "last_error": "",
            },
        )
        return self._replay(reconciled)

    async def recover_after_restart(self, command: ActuationCommand) -> dict[str, Any]:
        """Cancel or restore an interrupted transaction without replaying its effect."""

        if not isinstance(command, ActuationCommand):
            raise TypeError("command must be an ActuationCommand")
        inventory_sha256 = str(self._service.status()["registry_sha256"])
        if inventory_sha256 != command.inventory_sha256:
            raise RealityActuationError("reality_actuation_recovery_inventory_drift")
        adapter = self._service.actuator_adapter(command.channel_id)
        capability = self._service.actuator_capability(command.channel_id)
        if (
            adapter is None
            or capability is None
            or adapter.adapter_id != command.adapter_id
            or capability.adapter_id != command.adapter_id
        ):
            raise RealityActuationError("reality_actuation_recovery_identity_invalid")
        record = await asyncio.to_thread(self._load, command)
        if record is None:
            raise RealityActuationError("reality_actuation_transaction_missing")
        state = ActuationState(record["state"])
        if state not in {
            ActuationState.PLANNED,
            ActuationState.ADMITTED,
            ActuationState.DISPATCHED,
            ActuationState.EXECUTED,
            ActuationState.INDETERMINATE,
        }:
            return self._replay(record)

        try:
            if state in {ActuationState.PLANNED, ActuationState.ADMITTED}:
                cancellation = await asyncio.wait_for(
                    adapter.cancel(command, None),
                    timeout=capability.watchdog_timeout_s,
                )
                self._validate_actuation(
                    command,
                    None,
                    cancellation,
                    allow_without_preparation=True,
                )
                if cancellation.state is not ActuationState.CANCELLED:
                    raise RealityActuationError("restart_cancel_receipt_state_invalid")
                recovered = await asyncio.to_thread(
                    self._transition,
                    command,
                    expected={state},
                    state=ActuationState.CANCELLED,
                    updates={
                        "actuation_receipt_sha256": cancellation.sha256,
                        "manual_reconciliation_required": False,
                        "last_error": _error_evidence("restart_cancelled_before_dispatch"),
                    },
                )
                return self._replay(recovered)

            restoration = await asyncio.wait_for(
                adapter.safe_state(command, None),
                timeout=capability.watchdog_timeout_s,
            )
            self._validate_rollback(command, None, restoration)
            restored = bool(
                restoration.independently_observed
                and restoration.state
                in {
                    ActuationState.SAFE_STATE,
                    ActuationState.ROLLED_BACK,
                    ActuationState.COMPENSATED,
                }
            )
            recovered = await asyncio.to_thread(
                self._transition,
                command,
                expected={state},
                state=(restoration.state if restored else ActuationState.INDETERMINATE),
                updates={
                    "rollback_receipt_sha256": restoration.sha256,
                    "manual_reconciliation_required": not restored,
                    "last_error": _error_evidence(
                        "restart_safe_state_observed"
                        if restored
                        else "restart_safe_state_unverified"
                    ),
                },
            )
            return self._replay(recovered)
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            latest = await asyncio.to_thread(self._load, command)
            if latest is None:
                raise RealityActuationError("reality_actuation_transaction_lost") from exc
            latest_state = ActuationState(latest["state"])
            if latest_state is not ActuationState.INDETERMINATE:
                latest = await asyncio.to_thread(
                    self._transition,
                    command,
                    expected={latest_state},
                    state=ActuationState.INDETERMINATE,
                    updates={
                        "manual_reconciliation_required": True,
                        "last_error": _error_evidence(exc),
                    },
                )
            return self._replay(latest)

    async def recover_all_after_restart(
        self,
        *,
        max_transactions: int = 128,
    ) -> dict[str, Any]:
        """Boundedly recover every reconstructable nonterminal transaction."""

        if (
            isinstance(max_transactions, bool)
            or not isinstance(max_transactions, int)
            or not 1 <= max_transactions <= 1024
        ):
            raise ValueError("max_transactions must lie inside [1, 1024]")
        async with self._recovery_lock:
            commands, legacy, capsule_only, deferred = await asyncio.to_thread(
                self._discover_recovery_commands,
                max_transactions,
            )
            recovered: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []
            unresolved: list[dict[str, str]] = []
            for command in commands:
                before = await asyncio.to_thread(self._load, command)
                before_state = str((before or {}).get("state") or "missing")
                try:
                    result = await self.recover_after_restart(command)
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                    failures.append(
                        {
                            "command_sha256": command.sha256,
                            "error_type": type(exc).__name__,
                            "error_sha256": _sha256(str(exc)),
                        }
                    )
                    continue
                record = result.get("reality_reach_transaction")
                after_state = (
                    str(record.get("state") or "missing")
                    if isinstance(record, Mapping)
                    else "missing"
                )
                recovery_summary = {
                    "command_sha256": command.sha256,
                    "before_state": before_state,
                    "after_state": after_state,
                    "manual_reconciliation_required": bool(
                        result.get("manual_reconciliation_required")
                    ),
                }
                recovered.append(recovery_summary)
                if recovery_summary["manual_reconciliation_required"] or after_state in {
                    ActuationState.DISPATCHED.value,
                    ActuationState.EXECUTED.value,
                    ActuationState.INDETERMINATE.value,
                }:
                    unresolved.append(
                        {
                            "command_sha256": command.sha256,
                            "state": after_state,
                        }
                    )
            retryable = bool(failures or unresolved or capsule_only or deferred)
            return {
                "schema": RECOVERY_REPORT_SCHEMA,
                "eligible": len(commands) + deferred,
                "processed": len(commands),
                "deferred": deferred,
                "recovered": recovered,
                "failures": failures,
                "unresolved": unresolved,
                "legacy_unrecoverable_transaction_sha256": [f"sha256:{item}" for item in legacy],
                "capsule_without_transaction_sha256": [f"sha256:{item}" for item in capsule_only],
                "retryable": retryable,
                "complete": (
                    not failures
                    and not unresolved
                    and not legacy
                    and not capsule_only
                    and deferred == 0
                ),
            }

    def notify_adapter_available(self, adapter_id: str) -> None:
        """Wake recovery after a physical adapter becomes fully attached."""

        if not isinstance(adapter_id, str) or not _IDENTIFIER.fullmatch(adapter_id):
            raise ValueError("adapter_id is invalid")
        self._recovery_wake.set()

    async def start_recovery_supervisor(
        self,
        *,
        max_transactions: int = 64,
        min_retry_s: float = 5.0,
        max_retry_s: float = 300.0,
    ) -> asyncio.Task[Any]:
        if (
            isinstance(max_transactions, bool)
            or not isinstance(max_transactions, int)
            or not 1 <= max_transactions <= 1024
        ):
            raise ValueError("max_transactions must lie inside [1, 1024]")
        minimum = float(min_retry_s)
        maximum = float(max_retry_s)
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum <= 0:
            raise ValueError("recovery retry intervals must be finite and positive")
        if maximum < minimum or maximum > 3600.0:
            raise ValueError("recovery maximum retry interval is invalid")
        if self._recovery_task is not None and not self._recovery_task.done():
            self._recovery_wake.set()
            return self._recovery_task
        self._recovery_max_transactions = max_transactions
        self._recovery_min_retry_s = minimum
        self._recovery_max_retry_s = maximum
        self._recovery_stop.clear()
        self._recovery_wake.set()
        self._recovery_task = create_tracked_task(
            self._recovery_supervisor_loop(),
            name="reality_reach.restart_recovery_supervisor",
            owner="core.reality_reach.transactions",
        )
        return self._recovery_task

    async def wait_for_recovery_attempt(
        self,
        *,
        after_generation: int = 0,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        if isinstance(after_generation, bool) or int(after_generation) < 0:
            raise ValueError("after_generation must be a non-negative integer")
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_s must be finite and positive")
        async with self._recovery_condition:
            await asyncio.wait_for(
                self._recovery_condition.wait_for(
                    lambda: self._recovery_generation > int(after_generation)
                ),
                timeout=timeout,
            )
            return dict(self._recovery_report or {})

    async def stop(self) -> None:
        self._recovery_stop.set()
        self._recovery_wake.set()
        task = self._recovery_task
        self._recovery_task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _recovery_supervisor_loop(self) -> None:
        delay_s = self._recovery_min_retry_s
        while not self._recovery_stop.is_set():
            self._recovery_wake.clear()
            try:
                report = await self.recover_all_after_restart(
                    max_transactions=self._recovery_max_transactions
                )
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                report = {
                    "schema": RECOVERY_REPORT_SCHEMA,
                    "eligible": 0,
                    "processed": 0,
                    "deferred": 0,
                    "recovered": [],
                    "failures": [
                        {
                            "error_type": type(exc).__name__,
                            "error_sha256": _sha256(str(exc)),
                        }
                    ],
                    "unresolved": [],
                    "legacy_unrecoverable_transaction_sha256": [],
                    "capsule_without_transaction_sha256": [],
                    "retryable": True,
                    "complete": False,
                }
            self._recovery_report = report
            async with self._recovery_condition:
                self._recovery_generation += 1
                self._recovery_condition.notify_all()
            if self._recovery_stop.is_set():
                break
            retryable = bool(report.get("retryable"))
            if bool(report.get("complete")) or not retryable:
                delay_s = self._recovery_min_retry_s
                # Bounded, not unbounded. Recovery is done, so this is an idle
                # park until new work or shutdown wakes us — and an unbounded
                # park means one lost `_recovery_wake.set()` stops restart
                # recovery forever with no symptom at all: nothing raises,
                # nothing degrades, the work simply never happens again.
                # Re-checking the stop flag on timeout costs one wakeup per
                # idle period. The ceiling is the retry ceiling this loop
                # already carries, so no new constant is introduced.
                try:
                    await asyncio.wait_for(
                        self._recovery_wake.wait(),
                        timeout=self._recovery_max_retry_s,
                    )
                except TimeoutError:
                    pass
                continue
            delay_s = min(
                self._recovery_max_retry_s,
                max(self._recovery_min_retry_s, delay_s),
            )
            if self._recovery_wake.is_set():
                delay_s = self._recovery_min_retry_s
                continue
            try:
                await asyncio.wait_for(self._recovery_wake.wait(), timeout=delay_s)
                delay_s = self._recovery_min_retry_s
            except TimeoutError:
                delay_s = min(self._recovery_max_retry_s, delay_s * 2.0)
        logger.debug("Reality Reach restart recovery supervisor stopped")

    async def _dispatch_adapter(
        self,
        command: ActuationCommand,
        *,
        adapter: RealityAdapter,
        capability: ActuatorCapability,
        context: Mapping[str, Any],
        execution: dict[str, Any],
    ) -> Mapping[str, Any]:
        wall_now = int(self._wall_clock_ns())
        monotonic_now = int(self._monotonic_clock_ns())
        duration_ns = max(
            1,
            min(
                command.deadline_ns - wall_now,
                int(capability.watchdog_timeout_s * 1_000_000_000),
            ),
        )
        lease = ActuationLease(
            lease_id=f"lease.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            adapter_id=command.adapter_id,
            session_id=str(self._service.status()["session_id"]),
            authority_receipt_id=str(context.get("will_receipt_id") or ""),
            issued_at_ns=wall_now,
            expires_at_ns=wall_now + duration_ns,
            issued_monotonic_ns=monotonic_now,
            expires_monotonic_ns=monotonic_now + duration_ns,
        )
        timeout_s = capability.watchdog_timeout_s
        try:
            prepared = await asyncio.wait_for(
                adapter.prepare(command, lease),
                timeout=timeout_s,
            )
            self._validate_prepared(command, lease, capability, prepared)
            execution.update({"lease": lease, "prepared": prepared})
            await asyncio.to_thread(
                self._transition,
                command,
                expected={ActuationState.PLANNED, ActuationState.PREPARED},
                state=ActuationState.ADMITTED,
                updates={
                    "lease_sha256": lease.sha256,
                    "preparation_sha256": prepared.sha256,
                    "authority_receipt_id": lease.authority_receipt_id,
                },
            )
            await asyncio.to_thread(
                self._transition,
                command,
                expected={ActuationState.ADMITTED},
                state=ActuationState.DISPATCHED,
            )
            actuation = await asyncio.wait_for(
                adapter.actuate(command, lease, prepared),
                timeout=timeout_s,
            )
            self._validate_actuation(command, prepared, actuation)
            state = ActuationState.EXECUTED if actuation.executed else actuation.state
            await asyncio.to_thread(
                self._transition,
                command,
                expected={ActuationState.DISPATCHED},
                state=state,
                updates={"actuation_receipt_sha256": actuation.sha256},
            )
            execution["actuation"] = actuation
            return {
                "ok": actuation.executed,
                "transport_succeeded": actuation.transport_completed,
                "executed": actuation.executed,
                "actuation_receipt": actuation.to_dict(),
                "actuation_receipt_sha256": actuation.sha256,
            }
        except asyncio.CancelledError:
            await self._recover(
                command,
                adapter,
                execution,
                cancelled=True,
                timeout_s=timeout_s,
            )
            raise
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            await self._recover(
                command,
                adapter,
                execution,
                error=exc,
                timeout_s=timeout_s,
            )
            raise

    async def _verify_adapter_effect(
        self,
        command: ActuationCommand,
        *,
        adapter: RealityAdapter,
        capability: ActuatorCapability,
        execution: dict[str, Any],
    ) -> Mapping[str, Any]:
        actuation = execution.get("actuation")
        if not isinstance(actuation, ActuationReceipt) or not actuation.executed:
            return {"effect_verified": False, "reason": "actuation_not_executed"}
        try:
            effect = await asyncio.wait_for(
                adapter.verify_effect(command, actuation),
                timeout=capability.watchdog_timeout_s,
            )
            self._validate_effect(command, actuation, capability, effect)
            execution["effect"] = effect
            if effect.state == ActuationState.EFFECT_VERIFIED:
                await asyncio.to_thread(
                    self._transition,
                    command,
                    expected={ActuationState.EXECUTED},
                    state=ActuationState.EFFECT_VERIFIED,
                    updates={"effect_receipt_sha256": effect.sha256},
                )
                return {
                    "effect_verified": True,
                    "reality_reach.effect_receipt": effect.to_dict(),
                    "effect_receipt_sha256": effect.sha256,
                }
            await self._recover(
                command,
                adapter,
                execution,
                error=RealityActuationError("effect_not_verified"),
                timeout_s=capability.watchdog_timeout_s,
            )
            return {
                "effect_verified": False,
                "reason": "independent_effect_not_verified",
                "effect_receipt_sha256": effect.sha256,
            }
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            await self._recover(
                command,
                adapter,
                execution,
                error=exc,
                timeout_s=capability.watchdog_timeout_s,
            )
            return {
                "effect_verified": False,
                "reason": f"effect_verification_failed:{type(exc).__name__}",
            }

    async def _recover(
        self,
        command: ActuationCommand,
        adapter: RealityAdapter,
        execution: dict[str, Any],
        *,
        cancelled: bool = False,
        error: BaseException | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        recovery_reason = _error_evidence(
            error or ("cancelled" if cancelled else "recovery_required")
        )
        record = await asyncio.to_thread(self._load, command)
        if record is None:
            return
        state = ActuationState(record["state"])
        prepared = execution.get("prepared")
        actuation = execution.get("actuation")
        try:
            if state in {ActuationState.PLANNED, ActuationState.ADMITTED}:
                cancellation_receipt = await asyncio.wait_for(
                    adapter.cancel(
                        command,
                        prepared if isinstance(prepared, PreparedActuation) else None,
                    ),
                    timeout=timeout_s,
                )
                self._validate_actuation(
                    command,
                    prepared if isinstance(prepared, PreparedActuation) else None,
                    cancellation_receipt,
                    allow_without_preparation=True,
                )
                if cancellation_receipt.state != ActuationState.CANCELLED:
                    raise RealityActuationError("cancel_receipt_state_invalid")
                await asyncio.to_thread(
                    self._transition,
                    command,
                    expected={state},
                    state=ActuationState.CANCELLED,
                    updates={
                        "actuation_receipt_sha256": cancellation_receipt.sha256,
                        "last_error": recovery_reason,
                    },
                )
                return
            if state in {ActuationState.DISPATCHED, ActuationState.EXECUTED}:
                if isinstance(actuation, ActuationReceipt):
                    recovery_receipt = await asyncio.wait_for(
                        adapter.rollback(command, actuation),
                        timeout=timeout_s,
                    )
                else:
                    recovery_receipt = await asyncio.wait_for(
                        adapter.safe_state(command, None),
                        timeout=timeout_s,
                    )
                self._validate_rollback(command, actuation, recovery_receipt)
                await asyncio.to_thread(
                    self._transition,
                    command,
                    expected={state},
                    state=recovery_receipt.state,
                    updates={
                        "rollback_receipt_sha256": recovery_receipt.sha256,
                        "manual_reconciliation_required": not recovery_receipt.independently_observed,
                        "last_error": recovery_reason,
                    },
                )
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as recovery_exc:
            latest = await asyncio.to_thread(self._load, command)
            if latest is None:
                return
            latest_state = ActuationState(latest["state"])
            if latest_state not in _TERMINAL:
                await asyncio.to_thread(
                    self._transition,
                    command,
                    expected={latest_state},
                    state=ActuationState.INDETERMINATE,
                    updates={
                        "manual_reconciliation_required": True,
                        "last_error": _error_evidence(recovery_exc),
                    },
                )

    @staticmethod
    def _validate_prepared(
        command: ActuationCommand,
        lease: ActuationLease,
        capability: ActuatorCapability,
        prepared: PreparedActuation,
    ) -> None:
        if not isinstance(prepared, PreparedActuation) or (
            prepared.command_sha256 != command.sha256
            or prepared.lease_sha256 != lease.sha256
            or prepared.adapter_id != command.adapter_id
            or prepared.capability_sha256 != capability.sha256
        ):
            raise RealityActuationError("reality_actuation_preparation_identity_invalid")

    @staticmethod
    def _validate_actuation(
        command: ActuationCommand,
        prepared: PreparedActuation | None,
        receipt: ActuationReceipt,
        *,
        allow_without_preparation: bool = False,
    ) -> None:
        if not isinstance(receipt, ActuationReceipt):
            raise RealityActuationError("reality_actuation_receipt_type_invalid")
        expected_preparation = prepared.sha256 if prepared is not None else ""
        if (
            receipt.command_sha256 != command.sha256
            or receipt.adapter_id != command.adapter_id
            or (
                not allow_without_preparation and receipt.preparation_sha256 != expected_preparation
            )
        ):
            raise RealityActuationError("reality_actuation_receipt_identity_invalid")

    @staticmethod
    def _validate_effect(
        command: ActuationCommand,
        actuation: ActuationReceipt,
        capability: ActuatorCapability,
        effect: EffectReceipt,
    ) -> None:
        if not isinstance(effect, EffectReceipt) or (
            effect.command_sha256 != command.sha256
            or effect.actuation_receipt_sha256 != actuation.sha256
            or effect.observation_channel_id not in capability.observation_channels
        ):
            raise RealityActuationError("reality_actuation_effect_identity_invalid")

    @staticmethod
    def _validate_rollback(
        command: ActuationCommand,
        actuation: Any,
        receipt: RollbackReceipt,
    ) -> None:
        expected = actuation.sha256 if isinstance(actuation, ActuationReceipt) else ""
        if not isinstance(receipt, RollbackReceipt) or (
            receipt.command_sha256 != command.sha256
            or receipt.adapter_id != command.adapter_id
            or (expected and receipt.actuation_receipt_sha256 != expected)
        ):
            raise RealityActuationError("reality_actuation_rollback_identity_invalid")

    @staticmethod
    def _replay(record: Mapping[str, Any]) -> dict[str, Any]:
        state = ActuationState(str(record["state"]))
        effect_verified = state in {
            ActuationState.EFFECT_VERIFIED,
            ActuationState.MANUALLY_RECONCILED,
        }
        return {
            "ok": effect_verified,
            "effect_verified": effect_verified,
            "transport_succeeded": state
            in {
                ActuationState.EXECUTED,
                ActuationState.EFFECT_VERIFIED,
                ActuationState.MANUALLY_RECONCILED,
            },
            "retry_safe": False,
            "manual_reconciliation_required": bool(record["manual_reconciliation_required"])
            or state in _NO_AUTOMATIC_REPLAY,
            "reality_reach_transaction": dict(record),
            "replayed": True,
        }


_COORDINATOR: RealityActuationCoordinator | None = None
_COORDINATOR_LOCK = checked_lock("transactions")


def get_reality_actuation_coordinator(
    service: RealityReachService | None = None,
) -> RealityActuationCoordinator:
    global _COORDINATOR
    if _COORDINATOR is None:
        with _COORDINATOR_LOCK:
            if _COORDINATOR is None:
                if service is None:
                    from core.reality_reach.live import get_reality_reach_service

                    service = get_reality_reach_service()
                _COORDINATOR = RealityActuationCoordinator(service)
    return _COORDINATOR


__all__ = [
    "COMMAND_CAPSULE_SCHEMA",
    "RECOVERY_REPORT_SCHEMA",
    "RealityActuationCoordinator",
    "RealityActuationError",
    "TRANSACTION_SCHEMA",
    "get_reality_actuation_coordinator",
]
