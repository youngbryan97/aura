"""Protocol-neutral telemetry, service, and action lifecycles for physical adapters.

This is the ROS-shaped part of Reality Reach without a ROS dependency.  Device
connectors declare one managed node containing continuous telemetry, bounded
request/response services, and cancellable long-running actions.  The runtime
uses Aura's existing managed-organ and QoS foundations, fences every endpoint
to the live adapter identity, and persists action outcomes so a process restart
cannot silently replay an uncertain physical effect.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.bus.qos import QosBus
from core.governance_context import local_internal_governed_scope
from core.reality_reach.live import ChannelReading, RealityReachService
from core.reality_reach.middleware_contracts import (
    INFLIGHT_ACTION_STATES as _INFLIGHT_ACTION_STATES,
)
from core.reality_reach.middleware_contracts import (
    TERMINAL_ACTION_STATES as _TERMINAL_ACTION_STATES,
)
from core.reality_reach.middleware_contracts import (
    ActionContext,
    ActionEndpoint,
    ActionRecord,
    ActionState,
    ManagedAdapterDeclaration,
    ManagedRealityAdapter,
    PhysicalEffectIndeterminateError,
    RealityMiddlewareError,
    RestartPolicy,
    ServiceEndpoint,
    ServiceReceipt,
    TelemetryEndpoint,
    TelemetryMode,
)
from core.reality_reach.middleware_contracts import (
    bounded_payload as _bounded_payload,
)
from core.reality_reach.middleware_contracts import (
    bounded_seconds as _bounded_seconds,
)
from core.reality_reach.middleware_contracts import (
    canonical_identifier as _identifier,
)
from core.reality_reach.middleware_contracts import (
    sha256_digest as _digest,
)
from core.reality_reach.middleware_services import RealityServiceLane
from core.runtime.atomic_writer import read_json_envelope
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.lifecycle import ManagedOrgan, State
from core.runtime.lockdep import (
    CheckedAsyncLock,
    CheckedSemaphore,
    checked_async_lock,
    checked_semaphore,
)
from core.runtime.state_ownership import state_root
from core.utils.concurrency import cancel_and_join
from core.utils.task_tracker import get_task_tracker

_STATE_SCHEMA = "aura.reality_reach.middleware_state.v1"
_STATE_SCHEMA_VERSION = 1

@dataclass(slots=True)
class _Node:
    adapter: ManagedRealityAdapter
    declaration: ManagedAdapterDeclaration
    organ: ManagedOrgan
    telemetry: dict[str, TelemetryEndpoint]
    services: dict[str, ServiceEndpoint]
    actions: dict[str, ActionEndpoint]
    service_limits: dict[str, CheckedSemaphore]
    pull_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    telemetry_sequences: dict[str, int] = field(default_factory=dict)
    telemetry_failures: dict[str, int] = field(default_factory=dict)


class RealityMiddlewareRuntime(RealityServiceLane):
    """One protocol-neutral owner for managed physical endpoint lifecycles."""

    def __init__(
        self,
        service: RealityReachService,
        *,
        state_path: Path | None = None,
        qos_bus: QosBus | None = None,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        monotonic_clock: Callable[[], float] = time.monotonic,
        max_actions: int = 512,
        max_service_receipts: int = 512,
    ) -> None:
        if not isinstance(service, RealityReachService):
            raise TypeError("service must be a RealityReachService")
        if not 16 <= int(max_actions) <= 8192:
            raise ValueError("max_actions must lie inside [16, 8192]")
        if not 16 <= int(max_service_receipts) <= 8192:
            raise ValueError("max_service_receipts must lie inside [16, 8192]")
        self._service = service
        self._state_path = Path(state_path or (state_root() / "reality_middleware.json"))
        self._qos = qos_bus or QosBus()
        self._wall_clock_ns = wall_clock_ns
        self._monotonic_clock = monotonic_clock
        self._max_actions = int(max_actions)
        self._max_service_receipts = int(max_service_receipts)
        self._lock = checked_async_lock("reality_middleware.runtime")
        self._persist_lock = checked_async_lock("reality_middleware.persistence")
        self._nodes: dict[str, _Node] = {}
        self._endpoint_owner: dict[str, str] = {}
        self._actions: dict[str, ActionRecord] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._service_receipts: dict[str, ServiceReceipt] = {}
        self._service_inflight: dict[
            str,
            tuple[str, str, asyncio.Task[ServiceReceipt]],
        ] = {}
        self._action_admission_locks: dict[str, CheckedAsyncLock] = {}
        self._desired_active: dict[str, str] = {}
        self._running = False
        self._recovery_dirty = False
        self._load_state()

    def _load_state(self) -> None:
        if not os.path.lexists(self._state_path):
            return
        if self._state_path.is_symlink() or not self._state_path.is_file():
            raise RealityMiddlewareError("middleware state path must be a regular file")
        envelope = read_json_envelope(self._state_path)
        if envelope.get("schema_name") != _STATE_SCHEMA:
            raise RealityMiddlewareError("middleware state schema differs")
        if int(envelope.get("schema_version", 0)) != _STATE_SCHEMA_VERSION:
            raise RealityMiddlewareError("middleware state version differs")
        payload = dict(envelope.get("payload") or {})
        recorded = str(payload.pop("state_sha256", ""))
        if recorded != str(sha256_hex(canonical_json(payload))):
            raise RealityMiddlewareError("middleware state integrity check failed")
        if str(payload.get("service_session_id") or "") == self._service.session_id:
            raise RealityMiddlewareError("middleware restart state reused the current session id")
        actions = list(payload.get("actions") or [])
        if len(actions) > self._max_actions:
            raise RealityMiddlewareError("middleware state exceeds the action bound")
        for raw in actions:
            record = ActionRecord.from_dict(raw)
            if record.state.value in _INFLIGHT_ACTION_STATES:
                record.state = ActionState.INTERRUPTED
                record.error = "process_restart_interrupted_action"
                record.recovery_required = True
                record.updated_at_ns = int(self._wall_clock_ns())
                self._recovery_dirty = True
            self._actions[record.goal_id] = record
        service_receipts = list(payload.get("service_receipts") or [])
        if len(service_receipts) > self._max_service_receipts:
            raise RealityMiddlewareError("middleware state exceeds the service receipt bound")
        for raw in service_receipts:
            receipt = ServiceReceipt.from_dict(raw)
            self._service_receipts[receipt.request_id] = receipt
        desired = dict(payload.get("desired_active") or {})
        self._desired_active = {
            _identifier(str(node_id), name="node_id"): _digest(
                str(identity), name="adapter_identity_sha256"
            )
            for node_id, identity in desired.items()
        }

    async def _persist(self) -> None:
        async with self._persist_lock:
            actions = sorted(self._actions.values(), key=lambda item: item.updated_at_ns)
            if len(actions) > self._max_actions:
                terminal = [item for item in actions if item.state.terminal]
                keep_terminal = max(0, self._max_actions - (len(actions) - len(terminal)))
                keep_ids = {item.goal_id for item in terminal[-keep_terminal:]}
                self._actions = {
                    item.goal_id: item
                    for item in actions
                    if not item.state.terminal or item.goal_id in keep_ids
                }
                actions = sorted(self._actions.values(), key=lambda item: item.updated_at_ns)
            payload: dict[str, Any] = {
                "service_session_id": self._service.session_id,
                "saved_at_ns": int(self._wall_clock_ns()),
                "desired_active": dict(sorted(self._desired_active.items())),
                "actions": [item.to_dict() for item in actions],
                "service_receipts": [
                    item.to_dict()
                    for item in list(self._service_receipts.values())[
                        -self._max_service_receipts :
                    ]
                ],
            }
            payload["state_sha256"] = str(sha256_hex(canonical_json(payload)))
            with local_internal_governed_scope(
                "reality_reach.middleware.persist",
                domain="state_mutation",
            ):
                gateway = get_file_write_gateway()
                await gateway.ensure_directory_async(
                    self._state_path.parent,
                    source="reality_reach.middleware.persist",
                )
                await gateway.write_json_async(
                    self._state_path,
                    payload,
                    schema_version=_STATE_SCHEMA_VERSION,
                    schema_name=_STATE_SCHEMA,
                    source="reality_reach.middleware.persist",
                )

    async def start(self) -> None:
        async with self._lock:
            self._running = True
            dirty = self._recovery_dirty
            self._recovery_dirty = False
            nodes = tuple(self._nodes.values())
        for node in nodes:
            if node.organ.state is State.ACTIVE:
                self._start_pull_tasks(node)
        if dirty:
            await self._persist()

    async def register_adapter(
        self, adapter: ManagedRealityAdapter, *, activate: bool = True
    ) -> dict[str, Any]:
        if not isinstance(adapter, ManagedRealityAdapter):
            raise TypeError("adapter must satisfy ManagedRealityAdapter")
        declaration = adapter.lifecycle_declaration()
        if not isinstance(declaration, ManagedAdapterDeclaration):
            raise TypeError("lifecycle_declaration returned the wrong type")
        if declaration.adapter_id != adapter.adapter_id:
            raise RealityMiddlewareError("managed declaration adapter id differs")
        inventory = self._service.adapter_inventory().get(declaration.adapter_id)
        if inventory is None:
            raise RealityMiddlewareError("managed adapter is not in the live Reality Reach inventory")
        if inventory.identity_sha256 != declaration.adapter_identity_sha256:
            raise RealityMiddlewareError("managed adapter identity differs from live inventory")
        hooks = (
            "on_configure",
            "on_activate",
            "on_deactivate",
            "on_cleanup",
            "on_shutdown",
            "on_error",
            "read_telemetry",
            "handle_service",
            "execute_action",
            "cancel_action",
            "reconcile_action",
        )
        for name in hooks:
            if not inspect.iscoroutinefunction(getattr(adapter, name, None)):
                raise TypeError(f"managed adapter method {name} must be asynchronous")
        node = _Node(
            adapter=adapter,
            declaration=declaration,
            organ=ManagedOrgan(
                f"reality:{declaration.node_id}",
                on_configure=adapter.on_configure,
                on_activate=adapter.on_activate,
                on_deactivate=adapter.on_deactivate,
                on_cleanup=adapter.on_cleanup,
                on_shutdown=adapter.on_shutdown,
                on_error=adapter.on_error,
                transition_timeout_s=declaration.transition_timeout_s,
            ),
            telemetry={item.endpoint_id: item for item in declaration.telemetry},
            services={item.endpoint_id: item for item in declaration.services},
            actions={item.endpoint_id: item for item in declaration.actions},
            service_limits={
                item.endpoint_id: checked_semaphore(
                    f"reality_middleware.service.{declaration.node_id}.{item.endpoint_id}",
                    item.max_inflight,
                )
                for item in declaration.services
            },
        )
        endpoint_ids = set(node.telemetry) | set(node.services) | set(node.actions)
        async with self._lock:
            if declaration.node_id in self._nodes:
                raise RealityMiddlewareError("managed node is already registered")
            collisions = endpoint_ids & set(self._endpoint_owner)
            if collisions:
                raise RealityMiddlewareError(
                    f"managed endpoint already registered: {sorted(collisions)[0]}"
                )
            self._nodes[declaration.node_id] = node
            self._endpoint_owner.update(
                {endpoint_id: declaration.node_id for endpoint_id in endpoint_ids}
            )
            self._action_admission_locks.update(
                {
                    endpoint_id: checked_async_lock(
                        f"reality_middleware.action.{declaration.node_id}.{endpoint_id}"
                    )
                    for endpoint_id in node.actions
                }
            )
            restore_active = (
                self._desired_active.get(declaration.node_id)
                == declaration.adapter_identity_sha256
            )
        configured = await node.organ.configure()
        if not configured:
            await self._remove_failed_registration(node)
            raise RealityMiddlewareError("managed adapter configuration failed")
        should_activate = bool(activate or restore_active)
        if should_activate and not await self.activate_node(declaration.node_id):
            await self._remove_failed_registration(node)
            raise RealityMiddlewareError("managed adapter activation failed")
        await self._recover_node_actions(node)
        await self._persist()
        return self.node_status(declaration.node_id)

    async def _remove_failed_registration(self, node: _Node) -> None:
        with contextlib.suppress(Exception):
            await node.organ.shutdown()
        async with self._lock:
            self._nodes.pop(node.declaration.node_id, None)
            for endpoint_id in (
                set(node.telemetry) | set(node.services) | set(node.actions)
            ):
                self._endpoint_owner.pop(endpoint_id, None)
                self._action_admission_locks.pop(endpoint_id, None)

    async def activate_node(self, node_id: str) -> bool:
        node = self._node(node_id)
        if node.organ.state is State.ACTIVE:
            return True
        if node.organ.state is State.UNCONFIGURED and not await node.organ.configure():
            return False
        if node.organ.state is not State.INACTIVE:
            return False
        if not await node.organ.activate():
            return False
        async with self._lock:
            self._desired_active[node.declaration.node_id] = (
                node.declaration.adapter_identity_sha256
            )
        self._start_pull_tasks(node)
        await self._persist()
        return True

    async def deactivate_node(self, node_id: str) -> bool:
        node = self._node(node_id)
        await self._stop_pull_tasks(node)
        if node.organ.state is State.INACTIVE:
            ok = True
        elif node.organ.state is State.ACTIVE:
            ok = await node.organ.deactivate()
        else:
            ok = False
        if ok:
            async with self._lock:
                self._desired_active.pop(node.declaration.node_id, None)
            await self._persist()
        return ok

    async def unregister_adapter(self, node_id: str, *, forget_desired: bool = False) -> None:
        node = self._node(node_id)
        await self._stop_pull_tasks(node)
        active_goals = [
            item.goal_id
            for item in self._actions.values()
            if item.node_id == node.declaration.node_id and not item.state.terminal
        ]
        for goal_id in active_goals:
            await self.cancel_action(goal_id, reason="adapter_unregistered")
        await node.organ.shutdown()
        async with self._lock:
            self._nodes.pop(node.declaration.node_id, None)
            for endpoint_id in (
                set(node.telemetry) | set(node.services) | set(node.actions)
            ):
                self._endpoint_owner.pop(endpoint_id, None)
                self._action_admission_locks.pop(endpoint_id, None)
            if forget_desired:
                self._desired_active.pop(node.declaration.node_id, None)
        await self._persist()

    def _start_pull_tasks(self, node: _Node) -> None:
        if not self._running:
            return
        for endpoint in node.telemetry.values():
            if endpoint.mode is not TelemetryMode.PULL or endpoint.endpoint_id in node.pull_tasks:
                continue
            node.pull_tasks[endpoint.endpoint_id] = get_task_tracker().create_task(
                self._pull_loop(node, endpoint),
                name=f"RealityTelemetry:{endpoint.endpoint_id}",
            )

    async def _stop_pull_tasks(self, node: _Node) -> None:
        tasks = tuple(node.pull_tasks.values())
        node.pull_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _pull_loop(self, node: _Node, endpoint: TelemetryEndpoint) -> None:
        while self._running and node.organ.state is State.ACTIVE:
            started = self._monotonic_clock()
            try:
                sample = await asyncio.wait_for(
                    node.adapter.read_telemetry(endpoint.endpoint_id),
                    timeout=endpoint.sample_timeout_s,
                )
                await self.publish_telemetry(endpoint.endpoint_id, sample)
                node.telemetry_failures[endpoint.endpoint_id] = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - connector boundary must isolate failures
                streak = node.telemetry_failures.get(endpoint.endpoint_id, 0) + 1
                node.telemetry_failures[endpoint.endpoint_id] = streak
                if streak == 1 or streak % 10 == 0:
                    record_degradation(
                        f"reality_reach.telemetry.{endpoint.endpoint_id}",
                        exc,
                        severity="warning",
                        action="kept other physical endpoints live and scheduled the next bounded sample",
                        enforce_failure_policy=False,
                    )
            elapsed = self._monotonic_clock() - started
            await asyncio.sleep(max(0.0, endpoint.sample_period_s - elapsed))

    async def publish_telemetry(
        self,
        endpoint_id: str,
        payload: Mapping[str, Any] | tuple[ChannelReading, ...],
    ) -> dict[str, Any]:
        node, endpoint = self._telemetry_endpoint(endpoint_id)
        if node.organ.state is not State.ACTIVE:
            raise RealityMiddlewareError("telemetry publisher is not active")
        now_ns = int(self._wall_clock_ns())
        node.telemetry_sequences[endpoint.endpoint_id] = (
            node.telemetry_sequences.get(endpoint.endpoint_id, 0) + 1
        )
        sequence = node.telemetry_sequences[endpoint.endpoint_id]
        readings: tuple[ChannelReading, ...] = ()
        if isinstance(payload, tuple):
            readings = payload
            if any(not isinstance(item, ChannelReading) for item in readings):
                raise TypeError("telemetry tuple must contain ChannelReading values")
            if {item.channel_id for item in readings} - set(endpoint.channel_ids):
                raise RealityMiddlewareError("telemetry returned an undeclared channel")
            public_payload: dict[str, Any] = {
                "readings": [item.to_dict() for item in readings]
            }
        else:
            public_payload = _bounded_payload(
                payload, name="telemetry payload", maximum=endpoint.payload_bytes
            )
        envelope = {
            "schema": "aura.reality_reach.telemetry.v1",
            "endpoint_id": endpoint.endpoint_id,
            "node_id": node.declaration.node_id,
            "adapter_id": node.declaration.adapter_id,
            "adapter_identity_sha256": node.declaration.adapter_identity_sha256,
            "contract_sha256": node.declaration.sha256,
            "sequence": sequence,
            "published_at_ns": now_ns,
            "payload": public_payload,
        }
        if len(canonical_json(envelope)) > endpoint.payload_bytes:
            raise RealityMiddlewareError("telemetry envelope exceeds its byte bound")
        if readings:
            self._service.ingest_sensor_readings(node.declaration.adapter_id, readings)
        published = await self._qos.publish(
            endpoint.topic,
            envelope,
            profile=endpoint.qos,
        )
        return {
            "published": bool(published),
            "endpoint_id": endpoint.endpoint_id,
            "sequence": sequence,
            "envelope_sha256": str(sha256_hex(canonical_json(envelope))),
        }

    async def start_action(
        self,
        endpoint_id: str,
        request: Mapping[str, Any],
        *,
        goal_id: str | None = None,
        timeout_s: float | None = None,
        preempt: bool = False,
    ) -> dict[str, Any]:
        endpoint_id = _identifier(endpoint_id, name="endpoint_id")
        self._action_endpoint(endpoint_id)
        admission_lock = self._action_admission_locks.get(endpoint_id)
        if admission_lock is None:
            raise LookupError(endpoint_id)
        async with admission_lock:
            return await self._start_action_admitted(
                endpoint_id,
                request,
                goal_id=goal_id,
                timeout_s=timeout_s,
                preempt=preempt,
            )

    async def _start_action_admitted(
        self,
        endpoint_id: str,
        request: Mapping[str, Any],
        *,
        goal_id: str | None,
        timeout_s: float | None,
        preempt: bool,
    ) -> dict[str, Any]:
        node, endpoint = self._action_endpoint(endpoint_id)
        if node.organ.state is not State.ACTIVE:
            raise RealityMiddlewareError("action node is not active")
        goal_id = _identifier(goal_id or f"goal-{uuid.uuid4().hex}", name="goal_id")
        body = _bounded_payload(request, name="action request", maximum=endpoint.request_bytes)
        request_sha = str(sha256_hex(canonical_json(body)))
        unresolved = [
            item
            for item in self._actions.values()
            if item.endpoint_id == endpoint.endpoint_id and item.recovery_required
        ]
        if unresolved:
            raise RealityMiddlewareError(
                "action endpoint has an unresolved physical effect requiring reconciliation"
            )
        active = [
            item
            for item in self._actions.values()
            if item.endpoint_id == endpoint.endpoint_id and not item.state.terminal
        ]
        if active:
            if not preempt:
                raise RealityMiddlewareError("action endpoint already has an active goal")
            if not endpoint.preemptible:
                raise RealityMiddlewareError("action endpoint is not preemptible")
            for prior in active:
                await self.cancel_action(
                    prior.goal_id,
                    reason=f"preempted_by:{goal_id}",
                    preempted=True,
                )
                if self._actions[prior.goal_id].state is not ActionState.PREEMPTED:
                    raise RealityMiddlewareError(
                        "prior action did not prove safe preemption; replacement goal refused"
                    )
        async with self._lock:
            prior = self._actions.get(goal_id)
            if prior is not None:
                if prior.endpoint_id != endpoint.endpoint_id or prior.request_sha256 != request_sha:
                    raise RealityMiddlewareError("action goal id was reused with different content")
                return prior.to_dict()
            budget = endpoint.timeout_s
            if timeout_s is not None:
                budget = min(
                    budget,
                    _bounded_seconds(
                        timeout_s, name="timeout_s", minimum=0.1, maximum=86400.0
                    ),
                )
            now_ns = int(self._wall_clock_ns())
            record = ActionRecord(
                goal_id=goal_id,
                endpoint_id=endpoint.endpoint_id,
                node_id=node.declaration.node_id,
                adapter_id=node.declaration.adapter_id,
                adapter_identity_sha256=node.declaration.adapter_identity_sha256,
                request=body,
                request_sha256=request_sha,
                state=ActionState.ACCEPTED,
                created_at_ns=now_ns,
                updated_at_ns=now_ns,
                deadline_at_ns=now_ns + int(budget * 1_000_000_000),
            )
            self._actions[goal_id] = record
            cancel_event = asyncio.Event()
            self._cancel_events[goal_id] = cancel_event
        await self._persist()
        task = get_task_tracker().create_task(
            self._run_action(node, endpoint, record, cancel_event, budget),
            name=f"RealityAction:{endpoint.endpoint_id}:{goal_id}",
        )
        self._tasks[goal_id] = task
        return record.to_dict()

    async def _run_action(
        self,
        node: _Node,
        endpoint: ActionEndpoint,
        record: ActionRecord,
        cancel_event: asyncio.Event,
        budget: float,
    ) -> None:
        record.state = ActionState.EXECUTING
        record.updated_at_ns = int(self._wall_clock_ns())
        await self._persist()

        async def feedback(progress: float, payload: Mapping[str, Any]) -> None:
            if isinstance(progress, bool) or not 0.0 <= float(progress) <= 1.0:
                raise ValueError("action progress must lie inside [0, 1]")
            body = _bounded_payload(payload, name="action feedback", maximum=endpoint.result_bytes)
            previous = float(record.feedback[-1]["progress"]) if record.feedback else 0.0
            if float(progress) < previous:
                raise RealityMiddlewareError("action progress must be monotonic")
            item = {
                "sequence": len(record.feedback) + 1,
                "progress": float(progress),
                "payload": body,
                "recorded_at_ns": int(self._wall_clock_ns()),
            }
            record.feedback.append(item)
            if len(record.feedback) > endpoint.feedback_depth:
                del record.feedback[:-endpoint.feedback_depth]
            record.updated_at_ns = item["recorded_at_ns"]
            await self._persist()

        context = ActionContext(record.goal_id, cancel_event, feedback)
        try:
            raw = await asyncio.wait_for(
                node.adapter.execute_action(endpoint.endpoint_id, record.request, context),
                timeout=budget,
            )
            if cancel_event.is_set():
                record.state = (
                    ActionState.PREEMPTED
                    if record.state is ActionState.PREEMPTING
                    else ActionState.CANCELLED
                )
                record.error = "action_returned_after_cancellation_request"
            else:
                record.result = _bounded_payload(
                    raw, name="action result", maximum=endpoint.result_bytes
                )
                if endpoint.requires_effect_verification:
                    self._validate_effect_result(record.result)
                record.state = ActionState.SUCCEEDED
        except TimeoutError:
            cancel_event.set()
            acknowledged = await self._request_adapter_cancel(
                node,
                endpoint,
                record.goal_id,
                "deadline_exceeded",
            )
            record.state = (
                ActionState.TIMED_OUT if acknowledged else ActionState.INDETERMINATE
            )
            record.recovery_required = not acknowledged
            record.error = (
                f"action_deadline_exceeded:{budget:.3f}s"
                if acknowledged
                else f"deadline_exceeded_cancellation_unconfirmed:{budget:.3f}s"
            )
        except asyncio.CancelledError:
            if record.state is ActionState.PREEMPTING:
                record.state = ActionState.PREEMPTED
            elif record.state is ActionState.CANCEL_REQUESTED:
                record.state = ActionState.CANCELLED
            else:
                record.state = ActionState.INTERRUPTED
                record.recovery_required = True
            raise
        except PhysicalEffectIndeterminateError as exc:
            record.state = ActionState.INDETERMINATE
            record.recovery_required = True
            record.error = f"{type(exc).__name__}:{exc}"[:1024]
        except Exception as exc:  # noqa: BLE001 - adapter boundary becomes a durable result
            record.state = ActionState.ABORTED
            record.error = f"{type(exc).__name__}:{exc}"[:1024]
        finally:
            record.updated_at_ns = int(self._wall_clock_ns())
            async with self._lock:
                self._tasks.pop(record.goal_id, None)
                self._cancel_events.pop(record.goal_id, None)
            await self._persist()

    async def _request_adapter_cancel(
        self,
        node: _Node,
        endpoint: ActionEndpoint,
        goal_id: str,
        reason: str,
    ) -> bool:
        try:
            return bool(
                await asyncio.wait_for(
                    node.adapter.cancel_action(endpoint.endpoint_id, goal_id, reason),
                    timeout=endpoint.cancel_timeout_s,
                )
            )
        # not a failure: a cancel that does not land inside its own bound has not
        # cancelled, and False says so to the caller.
        except (TimeoutError, RuntimeError, OSError, ValueError, TypeError):
            return False

    async def cancel_action(
        self, goal_id: str, *, reason: str, preempted: bool = False
    ) -> dict[str, Any]:
        goal_id = _identifier(goal_id, name="goal_id")
        record = self._actions.get(goal_id)
        if record is None:
            raise LookupError(goal_id)
        if record.state.terminal:
            return record.to_dict()
        node, endpoint = self._action_endpoint(record.endpoint_id)
        if preempted and not endpoint.preemptible:
            raise RealityMiddlewareError("action endpoint is not preemptible")
        record.state = ActionState.PREEMPTING if preempted else ActionState.CANCEL_REQUESTED
        record.error = str(reason or "cancellation_requested")[:1024]
        record.updated_at_ns = int(self._wall_clock_ns())
        event = self._cancel_events.get(goal_id)
        if event is not None:
            event.set()
        await self._persist()
        acknowledged = await self._request_adapter_cancel(
            node, endpoint, goal_id, record.error
        )
        task = self._tasks.get(goal_id)
        if not acknowledged:
            record.state = ActionState.INDETERMINATE
            record.recovery_required = True
            record.error = f"cancellation_unconfirmed:{record.error}"[:1024]
            record.updated_at_ns = int(self._wall_clock_ns())
            await self._persist()
            return record.to_dict()
        if task is not None and not task.done():
            await cancel_and_join(
                task, owner="core.reality_reach.middleware", timeout=endpoint.cancel_timeout_s
            )
        if not acknowledged and record.state not in {
            ActionState.CANCELLED,
            ActionState.PREEMPTED,
        }:
            record.state = ActionState.INDETERMINATE
            record.recovery_required = True
            record.error = f"cancellation_unconfirmed:{record.error}"[:1024]
            record.updated_at_ns = int(self._wall_clock_ns())
            await self._persist()
        return record.to_dict()

    async def _recover_node_actions(self, node: _Node) -> None:
        candidates = [
            item
            for item in self._actions.values()
            if item.node_id == node.declaration.node_id and item.recovery_required
        ]
        for record in candidates:
            if record.adapter_identity_sha256 != node.declaration.adapter_identity_sha256:
                record.state = ActionState.INDETERMINATE
                record.error = "adapter_identity_changed_before_reconciliation"
                continue
            endpoint = node.actions.get(record.endpoint_id)
            if endpoint is None or endpoint.restart_policy is RestartPolicy.ABORT:
                record.state = ActionState.ABORTED
                record.error = "restart_policy_aborted_interrupted_action"
                record.recovery_required = False
                continue
            try:
                result = await asyncio.wait_for(
                    node.adapter.reconcile_action(record.endpoint_id, record.to_dict()),
                    timeout=endpoint.cancel_timeout_s,
                )
                claimed = ActionState(str(result.get("state") or "indeterminate"))
                if not claimed.terminal:
                    raise RealityMiddlewareError("reconciliation returned a non-terminal state")
                record.state = claimed
                record.result = _bounded_payload(
                    result.get("result") or {},
                    name="reconciled action result",
                    maximum=endpoint.result_bytes,
                )
                record.error = str(result.get("error") or "")[:1024]
                if claimed is ActionState.SUCCEEDED and endpoint.requires_effect_verification:
                    self._validate_effect_result(record.result)
                record.recovery_required = claimed is ActionState.INDETERMINATE
            except (TimeoutError, RuntimeError, OSError, ValueError, TypeError) as exc:
                record.state = ActionState.INDETERMINATE
                record.error = f"reconciliation_failed:{type(exc).__name__}:{exc}"[:1024]
                record.recovery_required = True
            record.updated_at_ns = int(self._wall_clock_ns())

    @staticmethod
    def _validate_effect_result(result: Mapping[str, Any]) -> None:
        if result.get("effect_verified") is not True:
            raise RealityMiddlewareError(
                "physical action success requires effect_verified=true"
            )
        _digest(
            str(result.get("effect_receipt_sha256") or ""),
            name="effect_receipt_sha256",
        )

    def action_status(self, goal_id: str) -> dict[str, Any]:
        record = self._actions.get(_identifier(goal_id, name="goal_id"))
        if record is None:
            raise LookupError(goal_id)
        return record.to_dict()

    def action_feedback(self, goal_id: str, *, after_sequence: int = 0) -> list[dict[str, Any]]:
        if isinstance(after_sequence, bool) or int(after_sequence) < 0:
            raise ValueError("after_sequence must be a non-negative integer")
        record = self._actions.get(_identifier(goal_id, name="goal_id"))
        if record is None:
            raise LookupError(goal_id)
        return [
            dict(item)
            for item in record.feedback
            if int(item.get("sequence", 0)) > int(after_sequence)
        ]

    async def wait_action(
        self,
        goal_id: str,
        *,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.05,
    ) -> dict[str, Any]:
        goal_id = _identifier(goal_id, name="goal_id")
        budget = _bounded_seconds(
            timeout_s,
            name="timeout_s",
            minimum=0.01,
            maximum=86400.0,
        )
        interval = _bounded_seconds(
            poll_interval_s,
            name="poll_interval_s",
            minimum=0.005,
            maximum=1.0,
        )
        deadline = self._monotonic_clock() + budget
        while True:
            record = self._actions.get(goal_id)
            if record is None:
                raise LookupError(goal_id)
            if record.state.terminal:
                return record.to_dict()
            remaining = deadline - self._monotonic_clock()
            if remaining <= 0:
                raise TimeoutError(f"action {goal_id} did not finish within {budget:.3f}s")
            await asyncio.sleep(min(interval, remaining))

    def actions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not 1 <= int(limit) <= 512:
            raise ValueError("limit must lie inside [1, 512]")
        values = sorted(self._actions.values(), key=lambda item: item.updated_at_ns)
        return [item.to_dict() for item in values[-int(limit) :]]

    def _node(self, node_id: str) -> _Node:
        normalized = _identifier(node_id, name="node_id")
        node = self._nodes.get(normalized)
        if node is None:
            raise LookupError(normalized)
        return node

    def _telemetry_endpoint(self, endpoint_id: str) -> tuple[_Node, TelemetryEndpoint]:
        endpoint_id = _identifier(endpoint_id, name="endpoint_id")
        owner = self._endpoint_owner.get(endpoint_id)
        if owner is None:
            raise LookupError(endpoint_id)
        node = self._node(owner)
        endpoint = node.telemetry.get(endpoint_id)
        if endpoint is None:
            raise LookupError(endpoint_id)
        return node, endpoint

    def _service_endpoint(self, endpoint_id: str) -> tuple[_Node, ServiceEndpoint]:
        endpoint_id = _identifier(endpoint_id, name="endpoint_id")
        owner = self._endpoint_owner.get(endpoint_id)
        if owner is None:
            raise LookupError(endpoint_id)
        node = self._node(owner)
        endpoint = node.services.get(endpoint_id)
        if endpoint is None:
            raise LookupError(endpoint_id)
        return node, endpoint

    def _action_endpoint(self, endpoint_id: str) -> tuple[_Node, ActionEndpoint]:
        endpoint_id = _identifier(endpoint_id, name="endpoint_id")
        owner = self._endpoint_owner.get(endpoint_id)
        if owner is None:
            raise LookupError(endpoint_id)
        node = self._node(owner)
        endpoint = node.actions.get(endpoint_id)
        if endpoint is None:
            raise LookupError(endpoint_id)
        return node, endpoint

    def node_status(self, node_id: str) -> dict[str, Any]:
        node = self._node(node_id)
        report = node.organ.report()
        return {
            **report,
            "node_id": node.declaration.node_id,
            "adapter_id": node.declaration.adapter_id,
            "adapter_identity_sha256": node.declaration.adapter_identity_sha256,
            "contract_sha256": node.declaration.sha256,
            "telemetry_endpoints": sorted(node.telemetry),
            "service_endpoints": sorted(node.services),
            "action_endpoints": sorted(node.actions),
            "pull_tasks": sorted(node.pull_tasks),
            "telemetry_failures": dict(sorted(node.telemetry_failures.items())),
        }

    def status(self) -> dict[str, Any]:
        nodes = [self.node_status(node_id) for node_id in sorted(self._nodes)]
        action_counts: dict[str, int] = {}
        for record in self._actions.values():
            action_counts[record.state.value] = action_counts.get(record.state.value, 0) + 1
        recovery_required_count = sum(
            1 for item in self._actions.values() if item.recovery_required
        )
        qos = self._qos.report()
        return {
            "alive": self._running,
            "ready": (
                self._running
                and recovery_required_count == 0
                and all(node["state"] == State.ACTIVE for node in nodes)
            ),
            "node_count": len(nodes),
            "nodes": nodes,
            "endpoint_count": len(self._endpoint_owner),
            "active_action_count": sum(
                count for state, count in action_counts.items() if state not in _TERMINAL_ACTION_STATES
            ),
            "recovery_required_count": recovery_required_count,
            "action_state_counts": dict(sorted(action_counts.items())),
            "qos": qos,
            "state_path": str(self._state_path),
        }

    def is_alive(self) -> bool:
        return self._running

    def is_ready(self) -> bool:
        return bool(self.status()["ready"])

    async def shutdown(self) -> None:
        self._running = False
        service_tasks = tuple(item[2] for item in self._service_inflight.values())
        for task in service_tasks:
            task.cancel()
        if service_tasks:
            await asyncio.gather(*service_tasks, return_exceptions=True)
        nodes = tuple(self._nodes.values())
        for node in nodes:
            await self._stop_pull_tasks(node)
        active = [item.goal_id for item in self._actions.values() if not item.state.terminal]
        for goal_id in active:
            with contextlib.suppress(LookupError, RealityMiddlewareError):
                await self.cancel_action(goal_id, reason="runtime_shutdown")
        for node in reversed(nodes):
            with contextlib.suppress(Exception):
                await node.organ.shutdown()
        await self._persist()


__all__ = [
    "ActionContext",
    "ActionEndpoint",
    "ActionRecord",
    "ActionState",
    "ManagedAdapterDeclaration",
    "ManagedRealityAdapter",
    "PhysicalEffectIndeterminateError",
    "RealityMiddlewareError",
    "RealityMiddlewareRuntime",
    "RestartPolicy",
    "ServiceEndpoint",
    "ServiceReceipt",
    "TelemetryEndpoint",
    "TelemetryMode",
]
