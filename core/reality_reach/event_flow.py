"""Durable typed event flows for sensing, reasoning, and physical effects.

The flow runtime is deliberately protocol-neutral. Intent planners and device
connectors compile work into typed graphs; this module owns bounded delivery,
replay, retries, and lifecycle. Consequential nodes delegate to
``RealityMiddlewareRuntime`` so a graph cannot become a second actuation plane.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.persistence.scoped_writes import write_private_state_async
from core.reality_reach.event_flow_contracts import (
    BackpressurePolicy,
    DeliveryState,
    FlowContractError,
    FlowDelivery,
    FlowEdge,
    FlowGraph,
    FlowNode,
    FlowNodeKind,
    FlowReceipt,
)
from core.reality_reach.middleware import RealityMiddlewareRuntime
from core.reality_reach.middleware_contracts import canonical_identifier
from core.runtime.atomic_writer import read_json_envelope
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_async_lock
from core.runtime.state_ownership import state_root
from core.utils.task_tracker import get_task_tracker

_STATE_SCHEMA = "aura.reality_reach.event_flow"
_STATE_SCHEMA_VERSION = 1
logger = logging.getLogger("Aura.RealityReach.EventFlow")

type FlowHandlerResult = Mapping[str, Any] | None
type FlowHandler = Callable[[FlowDelivery, FlowNode], Awaitable[FlowHandlerResult]]


class RealityEventFlowError(RuntimeError):
    """A flow lifecycle, execution, or durability contract failed."""


class RealityEventFlowDependencyUnavailableError(RealityEventFlowError):
    """A recoverable runtime binding is not available yet."""


class RealityEventFlowRuntime:
    """Persistent, deterministic executor for Reality Reach flow graphs."""

    def __init__(
        self,
        *,
        middleware: RealityMiddlewareRuntime | None = None,
        state_path: Path | None = None,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        max_deliveries: int = 8192,
        max_dead_letters: int = 2048,
        max_receipts: int = 8192,
        worker_enabled: bool = True,
    ) -> None:
        for name, value, minimum, maximum in (
            ("max_deliveries", max_deliveries, 64, 65_536),
            ("max_dead_letters", max_dead_letters, 16, 16_384),
            ("max_receipts", max_receipts, 64, 65_536),
        ):
            if isinstance(value, bool) or not minimum <= int(value) <= maximum:
                raise ValueError(f"{name} must lie inside [{minimum}, {maximum}]")
        self._middleware = middleware
        self._state_path = Path(state_path or (state_root() / "reality_event_flows.json"))
        self._wall_clock_ns = wall_clock_ns
        self._max_deliveries = int(max_deliveries)
        self._max_dead_letters = int(max_dead_letters)
        self._max_receipts = int(max_receipts)
        self._worker_enabled = bool(worker_enabled)
        self._lock = checked_async_lock("reality_event_flow.state")
        self._persist_lock = checked_async_lock("reality_event_flow.persist")
        self._graphs: dict[str, FlowGraph] = {}
        self._handlers: dict[tuple[str, str], FlowHandler] = {}
        self._queue: list[FlowDelivery] = []
        self._dead_letters: list[FlowDelivery] = []
        self._receipts: dict[str, FlowReceipt] = {}
        self._paused: set[str] = set()
        self._circuit_failures: dict[tuple[str, str], int] = {}
        self._circuit_open: set[tuple[str, str]] = set()
        self._dependency_blocks: dict[tuple[str, str], str] = {}
        self._sequence = 0
        self._running = False
        self._wake = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        self._worker_failures = 0
        self._last_worker_error = ""
        self._recovery_dirty = False
        self._counters = {
            "emitted": 0,
            "completed": 0,
            "retried": 0,
            "dead_lettered": 0,
            "cancelled": 0,
            "dropped_oldest": 0,
            "coalesced": 0,
            "backpressure_rejections": 0,
            "recovered_processing": 0,
        }
        self._load_state()

    def _load_state(self) -> None:
        if not os.path.lexists(self._state_path):
            return
        if self._state_path.is_symlink() or not self._state_path.is_file():
            raise RealityEventFlowError("event-flow state path must be a regular file")
        envelope = read_json_envelope(self._state_path)
        if envelope.get("schema_name") != _STATE_SCHEMA:
            raise RealityEventFlowError("event-flow state schema differs")
        if int(envelope.get("schema_version", 0)) != _STATE_SCHEMA_VERSION:
            raise RealityEventFlowError("event-flow state version differs")
        payload = dict(envelope.get("payload") or {})
        recorded = str(payload.pop("state_sha256", ""))
        if recorded != str(sha256_hex(canonical_json(payload))):
            raise RealityEventFlowError("event-flow state integrity check failed")
        graphs = tuple(FlowGraph.from_dict(item) for item in payload.get("graphs") or ())
        if len({item.graph_id for item in graphs}) != len(graphs):
            raise RealityEventFlowError("event-flow state contains duplicate graphs")
        self._graphs = {item.graph_id: item for item in graphs}
        queue = [FlowDelivery.from_dict(item) for item in payload.get("queue") or ()]
        dead_letters = [
            FlowDelivery.from_dict(item) for item in payload.get("dead_letters") or ()
        ]
        receipts = [FlowReceipt.from_dict(item) for item in payload.get("receipts") or ()]
        if len(queue) > self._max_deliveries:
            raise RealityEventFlowError("event-flow queue exceeds its configured bound")
        if len(dead_letters) > self._max_dead_letters:
            raise RealityEventFlowError("event-flow dead-letter queue exceeds its bound")
        if len(receipts) > self._max_receipts:
            raise RealityEventFlowError("event-flow receipt set exceeds its bound")
        for delivery in queue:
            graph = self._graphs.get(delivery.graph_id)
            if graph is None or graph.sha256 != delivery.graph_sha256:
                raise RealityEventFlowError("queued delivery graph identity is unavailable")
            if delivery.state is DeliveryState.PROCESSING:
                delivery.state = DeliveryState.QUEUED
                delivery.error = "process_restart_replay"
                self._counters["recovered_processing"] += 1
                self._recovery_dirty = True
            elif delivery.state is not DeliveryState.QUEUED:
                raise RealityEventFlowError("active queue contains a terminal delivery")
        self._queue = sorted(queue, key=lambda item: item.sequence)
        self._dead_letters = dead_letters
        self._receipts = {item.delivery_id: item for item in receipts}
        self._paused = {
            canonical_identifier(str(item), name="graph_id")
            for item in payload.get("paused") or ()
        }
        circuit_failures: dict[tuple[str, str], int] = {}
        for key, value in dict(payload.get("circuit_failures") or {}).items():
            graph_id, separator, node_id = str(key).partition("/")
            if not separator:
                raise RealityEventFlowError("event-flow circuit key is malformed")
            circuit_failures[
                (
                    canonical_identifier(graph_id, name="graph_id"),
                    canonical_identifier(node_id, name="node_id"),
                )
            ] = max(0, int(value))
        self._circuit_failures = circuit_failures
        self._circuit_open = {
            tuple(
                canonical_identifier(part, name="circuit_key")
                for part in str(item).split("/", 1)
            )
            for item in payload.get("circuit_open") or ()
            if "/" in str(item)
        }
        self._sequence = max(
            int(payload.get("sequence", 0)),
            *(item.sequence for item in (*self._queue, *self._dead_letters)),
            0,
        )
        for key, value in dict(payload.get("counters") or {}).items():
            if key in self._counters:
                self._counters[key] = max(self._counters[key], int(value))

    async def _persist(self) -> None:
        async with self._persist_lock:
            async with self._lock:
                receipts = list(self._receipts.values())[-self._max_receipts :]
                payload: dict[str, Any] = {
                    "saved_at_ns": int(self._wall_clock_ns()),
                    "sequence": self._sequence,
                    "graphs": [
                        item.to_dict()
                        for item in sorted(self._graphs.values(), key=lambda row: row.graph_id)
                    ],
                    "queue": [item.to_dict() for item in self._queue],
                    "dead_letters": [item.to_dict() for item in self._dead_letters],
                    "receipts": [item.to_dict() for item in receipts],
                    "paused": sorted(self._paused),
                    "circuit_failures": {
                        f"{graph_id}/{node_id}": count
                        for (graph_id, node_id), count in sorted(self._circuit_failures.items())
                    },
                    "circuit_open": [
                        f"{graph_id}/{node_id}"
                        for graph_id, node_id in sorted(self._circuit_open)
                    ],
                    "counters": dict(sorted(self._counters.items())),
                }
                payload["state_sha256"] = str(sha256_hex(canonical_json(payload)))
            await write_private_state_async(
                self._state_path, payload,
                scope_factory=lambda: local_internal_governed_scope(
                    "reality_reach.event_flow.persist", domain="state_mutation"),
                schema_version=_STATE_SCHEMA_VERSION, schema_name=_STATE_SCHEMA,
                source="reality_reach.event_flow.persist",
            )

    async def start(self) -> None:
        self._running = True
        if self._recovery_dirty:
            self._recovery_dirty = False
            await self._persist()
        if self._worker_enabled and (
            self._worker_task is None or self._worker_task.done()
        ):
            self._worker_task = get_task_tracker().create_task(
                self._worker_loop(),
                name="RealityEventFlowWorker",
            )
        if self._queue:
            self._wake.set()

    async def shutdown(self) -> None:
        self._running = False
        self._wake.set()
        worker = self._worker_task
        self._worker_task = None
        if worker is not None and not worker.done():
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        await self._persist()

    async def _worker_loop(self) -> None:
        while self._running:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=1.0)
            except TimeoutError:
                continue
            self._wake.clear()
            while self._running:
                try:
                    receipt = await self.run_once()
                    if receipt is not None:
                        self._worker_failures = 0
                        self._last_worker_error = ""
                except asyncio.CancelledError:
                    raise
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    self._worker_failures += 1
                    self._last_worker_error = f"{type(exc).__name__}:{exc}"[:1024]
                    logger.error(
                        "Reality event-flow worker failed (streak=%d): %s",
                        self._worker_failures,
                        self._last_worker_error,
                    )
                    backoff = min(5.0, 0.1 * (2 ** min(self._worker_failures, 6)))
                    await asyncio.sleep(backoff)
                    self._wake.set()
                    break
                if receipt is None:
                    break
                if receipt.state is DeliveryState.QUEUED:
                    retry_delay = (
                        0.5
                        if receipt.error.startswith("dependency_unavailable:")
                        else min(2.0, 0.05 * (2 ** min(receipt.attempt, 5)))
                    )
                    await asyncio.sleep(retry_delay)

    async def deploy(self, graph: FlowGraph) -> str:
        if not isinstance(graph, FlowGraph):
            raise TypeError("graph must be a FlowGraph")
        async with self._lock:
            prior = self._graphs.get(graph.graph_id)
            if prior is not None:
                if prior.sha256 == graph.sha256:
                    return graph.sha256
                if graph.revision <= prior.revision:
                    raise RealityEventFlowError("graph replacement requires a higher revision")
                if any(item.graph_id == graph.graph_id for item in self._queue):
                    raise RealityEventFlowError("a graph with queued work cannot be replaced")
            self._graphs[graph.graph_id] = graph
        await self._persist()
        return graph.sha256

    def register_handler(self, graph_id: str, node_id: str, handler: FlowHandler) -> None:
        graph = self._graph(graph_id)
        node = graph.node(node_id)
        if node.kind in {FlowNodeKind.SOURCE, FlowNodeKind.SERVICE, FlowNodeKind.ACTION}:
            raise RealityEventFlowError(f"{node.kind.value} nodes do not accept custom handlers")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handlers[(graph.graph_id, node.node_id)] = handler
        self._dependency_blocks.pop((graph.graph_id, node.node_id), None)
        self._wake.set()

    async def pause(self, graph_id: str) -> None:
        graph_id = self._graph(graph_id).graph_id
        async with self._lock:
            self._paused.add(graph_id)
        await self._persist()

    async def resume(self, graph_id: str) -> None:
        graph_id = self._graph(graph_id).graph_id
        async with self._lock:
            self._paused.discard(graph_id)
        await self._persist()
        self._wake.set()

    async def reset_circuit(self, graph_id: str, node_id: str) -> None:
        graph = self._graph(graph_id)
        node = graph.node(node_id)
        async with self._lock:
            self._circuit_failures.pop((graph.graph_id, node.node_id), None)
            self._circuit_open.discard((graph.graph_id, node.node_id))
        await self._persist()

    async def emit(
        self,
        graph_id: str,
        source_node: str,
        source_port: str,
        payload: Any,
        *,
        event_id: str | None = None,
        trace_id: str | None = None,
    ) -> tuple[str, ...]:
        if not self._running:
            raise RealityEventFlowError("event-flow runtime is not running")
        graph = self._graph(graph_id)
        source = graph.node(source_node)
        if source.kind is not FlowNodeKind.SOURCE:
            raise RealityEventFlowError("external events must enter through a source node")
        port = source.output(source_port)
        self._validate_value(port, payload)
        root_event_id = canonical_identifier(
            event_id or f"evt-{uuid.uuid4().hex}", name="event_id"
        )
        trace_id = canonical_identifier(
            trace_id or f"trace-{uuid.uuid4().hex}", name="trace_id"
        )
        async with self._lock:
            delivery_ids = self._enqueue_fanout_locked(
                graph,
                source.node_id,
                port.port_id,
                payload,
                event_id=root_event_id,
                trace_id=trace_id,
                parent_event_id=None,
            )
            self._counters["emitted"] += 1
        await self._persist()
        self._wake.set()
        return delivery_ids

    def _enqueue_fanout_locked(
        self,
        graph: FlowGraph,
        source_node: str,
        source_port: str,
        payload: Any,
        *,
        event_id: str,
        trace_id: str,
        parent_event_id: str | None,
        preflight: bool = True,
        capacity_credit: int = 0,
    ) -> tuple[str, ...]:
        edges = graph.outgoing(source_node, source_port)
        if preflight:
            self._preflight_edges_locked(
                graph,
                edges,
                capacity_credit=capacity_credit,
            )
        delivery_ids: list[str] = []
        for edge in edges:
            same_edge = [
                item
                for item in self._queue
                if item.edge_id == edge.edge_id and item.graph_id == graph.graph_id
            ]
            if len(same_edge) >= edge.queue_depth:
                queued = [item for item in same_edge if item.state is DeliveryState.QUEUED]
                if not queued:
                    raise RealityEventFlowError(f"edge queue has no replaceable event: {edge.edge_id}")
                victim = queued[-1] if edge.backpressure is BackpressurePolicy.COALESCE else queued[0]
                self._queue.remove(victim)
                victim.state = DeliveryState.CANCELLED
                victim.error = (
                    "coalesced_by_newer_event"
                    if edge.backpressure is BackpressurePolicy.COALESCE
                    else "dropped_oldest_under_backpressure"
                )
                self._record_terminal_locked(victim, (), victim.error)
                counter = (
                    "coalesced"
                    if edge.backpressure is BackpressurePolicy.COALESCE
                    else "dropped_oldest"
                )
                self._counters[counter] += 1
            self._sequence += 1
            delivery_id = f"delivery-{uuid.uuid4().hex}"
            delivery = FlowDelivery(
                delivery_id=delivery_id,
                event_id=event_id,
                trace_id=trace_id,
                parent_event_id=parent_event_id,
                graph_id=graph.graph_id,
                graph_sha256=graph.sha256,
                edge_id=edge.edge_id,
                target_node=edge.target_node,
                target_port=edge.target_port,
                payload=payload,
                sequence=self._sequence,
                emitted_at_ns=int(self._wall_clock_ns()),
            )
            self._queue.append(delivery)
            delivery_ids.append(delivery_id)
        return tuple(delivery_ids)

    def _preflight_edges_locked(
        self,
        graph: FlowGraph,
        edges: tuple[FlowEdge, ...],
        *,
        capacity_credit: int = 0,
    ) -> None:
        required_slots = 0
        for edge in edges:
            same_edge = [
                item
                for item in self._queue
                if item.edge_id == edge.edge_id and item.graph_id == graph.graph_id
            ]
            if len(same_edge) < edge.queue_depth:
                required_slots += 1
                continue
            queued = [item for item in same_edge if item.state is DeliveryState.QUEUED]
            if edge.backpressure is BackpressurePolicy.REJECT or not queued:
                self._counters["backpressure_rejections"] += 1
                raise RealityEventFlowError(f"edge queue is full: {edge.edge_id}")
        if len(self._queue) - max(0, capacity_credit) + required_slots > self._max_deliveries:
            self._counters["backpressure_rejections"] += 1
            raise RealityEventFlowError("event-flow global queue capacity exhausted")

    async def run_once(self, *, graph_id: str | None = None) -> FlowReceipt | None:
        if not self._running:
            raise RealityEventFlowError("event-flow runtime is not running")
        selected_graph = None if graph_id is None else self._graph(graph_id).graph_id
        async with self._lock:
            delivery = next(
                (
                    item
                    for item in self._queue
                    if item.state is DeliveryState.QUEUED
                    and item.graph_id not in self._paused
                    and (selected_graph is None or item.graph_id == selected_graph)
                ),
                None,
            )
            if delivery is None:
                return None
            delivery.state = DeliveryState.PROCESSING
            delivery.attempt += 1
            delivery.error = ""
        try:
            await self._persist()
        except (OSError, RuntimeError, TypeError, ValueError):
            async with self._lock:
                if delivery in self._queue and delivery.state is DeliveryState.PROCESSING:
                    delivery.state = DeliveryState.QUEUED
                    delivery.attempt = max(0, delivery.attempt - 1)
                    delivery.error = "claim_persistence_failed"
            raise
        return await self._execute(delivery)

    async def _execute(self, delivery: FlowDelivery) -> FlowReceipt:
        started_at_ns = int(self._wall_clock_ns())
        graph = self._graph(delivery.graph_id)
        if graph.sha256 != delivery.graph_sha256:
            return await self._dead_letter(
                delivery, started_at_ns, "graph_identity_changed_before_execution"
            )
        node = graph.node(delivery.target_node)
        self._validate_value(node.input(delivery.target_port), delivery.payload)
        circuit_key = (graph.graph_id, node.node_id)
        if circuit_key in self._circuit_open:
            return await self._dead_letter(delivery, started_at_ns, "node_circuit_open")
        try:
            outputs = await asyncio.wait_for(
                self._execute_node(graph, node, delivery),
                timeout=node.timeout_s,
            )
            normalized = self._validate_outputs(node, outputs)
        except RealityEventFlowDependencyUnavailableError as exc:
            error = f"dependency_unavailable:{exc}"[:1024]
            async with self._lock:
                delivery.state = DeliveryState.QUEUED
                delivery.attempt = max(0, delivery.attempt - 1)
                delivery.error = error
                self._dependency_blocks[circuit_key] = error
            await self._persist()
            return FlowReceipt(
                delivery_id=delivery.delivery_id,
                graph_id=delivery.graph_id,
                graph_sha256=delivery.graph_sha256,
                node_id=node.node_id,
                state=DeliveryState.QUEUED,
                attempt=delivery.attempt,
                output_event_ids=(),
                started_at_ns=started_at_ns,
                completed_at_ns=int(self._wall_clock_ns()),
                error=error,
            )
        except asyncio.CancelledError:
            async with self._lock:
                if delivery in self._queue:
                    delivery.state = DeliveryState.QUEUED
                    delivery.error = "execution_cancelled_for_shutdown"
            await self._persist()
            raise
        except Exception as exc:  # noqa: BLE001 - node boundary becomes durable state
            error = f"{type(exc).__name__}:{exc}"[:1024]
            return await self._retry_or_dead_letter(
                delivery,
                node,
                circuit_key,
                started_at_ns,
                error,
            )

        output_event_ids: list[str] = []
        try:
            async with self._lock:
                all_edges = tuple(
                    edge
                    for port_id in normalized
                    for edge in graph.outgoing(node.node_id, port_id)
                )
                self._preflight_edges_locked(graph, all_edges, capacity_credit=1)
                self._dependency_blocks.pop(circuit_key, None)
                self._circuit_failures.pop(circuit_key, None)
                self._circuit_open.discard(circuit_key)
                for port_id, payload in normalized.items():
                    output_event_id = f"evt-{uuid.uuid4().hex}"
                    queued = self._enqueue_fanout_locked(
                        graph,
                        node.node_id,
                        port_id,
                        payload,
                        event_id=output_event_id,
                        trace_id=delivery.trace_id,
                        parent_event_id=delivery.event_id,
                        preflight=False,
                    )
                    if queued:
                        output_event_ids.append(output_event_id)
                if delivery in self._queue:
                    self._queue.remove(delivery)
                delivery.state = DeliveryState.SUCCEEDED
                receipt = self._record_terminal_locked(delivery, tuple(output_event_ids), "")
                self._counters["completed"] += 1
        except (RuntimeError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}:{exc}"[:1024]
            return await self._retry_or_dead_letter(
                delivery,
                node,
                circuit_key,
                started_at_ns,
                error,
            )
        await self._persist()
        return receipt

    async def _retry_or_dead_letter(
        self,
        delivery: FlowDelivery,
        node: FlowNode,
        circuit_key: tuple[str, str],
        started_at_ns: int,
        error: str,
    ) -> FlowReceipt:
        async with self._lock:
            failures = self._circuit_failures.get(circuit_key, 0) + 1
            self._circuit_failures[circuit_key] = failures
            if failures >= node.circuit_breaker_failures:
                self._circuit_open.add(circuit_key)
            retry = delivery.attempt < node.max_attempts and circuit_key not in self._circuit_open
            if retry:
                delivery.state = DeliveryState.QUEUED
                delivery.error = error
                self._counters["retried"] += 1
        if retry:
            await self._persist()
            return FlowReceipt(
                delivery_id=delivery.delivery_id,
                graph_id=delivery.graph_id,
                graph_sha256=delivery.graph_sha256,
                node_id=node.node_id,
                state=DeliveryState.QUEUED,
                attempt=delivery.attempt,
                output_event_ids=(),
                started_at_ns=started_at_ns,
                completed_at_ns=int(self._wall_clock_ns()),
                error=error,
            )
        return await self._dead_letter(delivery, started_at_ns, error)

    async def _execute_node(
        self,
        graph: FlowGraph,
        node: FlowNode,
        delivery: FlowDelivery,
    ) -> FlowHandlerResult:
        if node.kind is FlowNodeKind.DELAY:
            delay_s = float(node.config.get("delay_s", 0.0))
            if delay_s < 0.0 or delay_s > node.timeout_s:
                raise FlowContractError("delay_s must lie inside the node timeout")
            if delay_s:
                await asyncio.sleep(delay_s)
            return {node.outputs[0].port_id: delivery.payload}
        if node.kind is FlowNodeKind.SERVICE:
            middleware = self._required_middleware()
            if not isinstance(delivery.payload, Mapping):
                raise FlowContractError("service node payload must be an object")
            request_id = f"flow-{delivery.idempotency_key.removeprefix('sha256:')[:48]}"
            receipt = await middleware.call_service(
                str(node.config["endpoint_id"]),
                delivery.payload,
                request_id=request_id,
                timeout_s=node.timeout_s,
            )
            if not receipt.ok:
                raise RealityEventFlowError(receipt.error or "physical service failed")
            return {node.outputs[0].port_id: receipt.response}
        if node.kind is FlowNodeKind.ACTION:
            middleware = self._required_middleware()
            if not isinstance(delivery.payload, Mapping):
                raise FlowContractError("action node payload must be an object")
            goal_id = f"flow-{delivery.idempotency_key.removeprefix('sha256:')[:48]}"
            await middleware.start_action(
                str(node.config["endpoint_id"]),
                delivery.payload,
                goal_id=goal_id,
                timeout_s=node.timeout_s,
                preempt=bool(node.config.get("preempt", False)),
            )
            result = await middleware.wait_action(goal_id, timeout_s=node.timeout_s)
            if str(result.get("state")) != "succeeded":
                raise RealityEventFlowError(
                    str(result.get("error") or f"physical action {result.get('state')}")
                )
            return {node.outputs[0].port_id: dict(result.get("result") or {})}
        handler = self._handlers.get((graph.graph_id, node.node_id))
        if handler is None:
            raise RealityEventFlowDependencyUnavailableError(
                f"node handler is not registered: {node.node_id}"
            )
        return await handler(delivery, node)

    def _required_middleware(self) -> RealityMiddlewareRuntime:
        if self._middleware is None:
            raise RealityEventFlowDependencyUnavailableError(
                "physical node requires Reality Middleware"
            )
        return self._middleware

    @staticmethod
    def _validate_value(port: Any, value: Any) -> None:
        try:
            canonical_json(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FlowContractError("flow values must be JSON serializable") from exc
        if not port.value_type.accepts(value):
            raise FlowContractError(
                f"value does not satisfy {port.port_id}:{port.value_type.value}"
            )

    def _validate_outputs(
        self,
        node: FlowNode,
        outputs: FlowHandlerResult,
    ) -> dict[str, Any]:
        if outputs is None:
            if node.kind not in {FlowNodeKind.FILTER, FlowNodeKind.SINK}:
                raise FlowContractError("only filter and sink nodes may emit no output")
            return {}
        if not isinstance(outputs, Mapping):
            raise FlowContractError("node output must map output port ids to values")
        normalized: dict[str, Any] = {}
        for port_id, value in outputs.items():
            port = node.output(str(port_id))
            self._validate_value(port, value)
            normalized[port.port_id] = value
        required = {item.port_id for item in node.outputs if item.required}
        missing = required - normalized.keys()
        if missing and node.kind not in {FlowNodeKind.FILTER, FlowNodeKind.ROUTER}:
            raise FlowContractError(f"node omitted required outputs: {sorted(missing)}")
        return normalized

    async def _dead_letter(
        self,
        delivery: FlowDelivery,
        started_at_ns: int,
        error: str,
    ) -> FlowReceipt:
        async with self._lock:
            if delivery in self._queue:
                self._queue.remove(delivery)
            delivery.state = DeliveryState.DEAD_LETTER
            delivery.error = error[:1024]
            self._dead_letters.append(delivery)
            if len(self._dead_letters) > self._max_dead_letters:
                del self._dead_letters[: len(self._dead_letters) - self._max_dead_letters]
            receipt = self._record_terminal_locked(delivery, (), delivery.error, started_at_ns)
            self._counters["dead_lettered"] += 1
        await self._persist()
        return receipt

    def _record_terminal_locked(
        self,
        delivery: FlowDelivery,
        output_event_ids: tuple[str, ...],
        error: str,
        started_at_ns: int | None = None,
    ) -> FlowReceipt:
        receipt = FlowReceipt(
            delivery_id=delivery.delivery_id,
            graph_id=delivery.graph_id,
            graph_sha256=delivery.graph_sha256,
            node_id=delivery.target_node,
            state=delivery.state,
            attempt=delivery.attempt,
            output_event_ids=output_event_ids,
            started_at_ns=started_at_ns or int(self._wall_clock_ns()),
            completed_at_ns=int(self._wall_clock_ns()),
            error=error,
        )
        self._receipts[delivery.delivery_id] = receipt
        while len(self._receipts) > self._max_receipts:
            self._receipts.pop(next(iter(self._receipts)))
        return receipt

    async def cancel_queued(self, graph_id: str, *, reason: str) -> int:
        graph_id = self._graph(graph_id).graph_id
        reason = str(reason or "flow_cancelled")[:1024]
        async with self._lock:
            targets = [
                item
                for item in self._queue
                if item.graph_id == graph_id and item.state is DeliveryState.QUEUED
            ]
            for delivery in targets:
                self._queue.remove(delivery)
                delivery.state = DeliveryState.CANCELLED
                delivery.error = reason
                self._record_terminal_locked(delivery, (), reason)
            self._counters["cancelled"] += len(targets)
        if targets:
            await self._persist()
        return len(targets)

    async def drain(self, graph_id: str, *, limit: int = 4096) -> tuple[FlowReceipt, ...]:
        graph_id = self._graph(graph_id).graph_id
        if isinstance(limit, bool) or not 1 <= int(limit) <= 65_536:
            raise ValueError("limit must lie inside [1, 65536]")
        receipts: list[FlowReceipt] = []
        for _ in range(int(limit)):
            receipt = await self.run_once(graph_id=graph_id)
            if receipt is None:
                break
            receipts.append(receipt)
        else:
            raise RealityEventFlowError("flow drain exceeded its bounded work limit")
        return tuple(receipts)

    def receipt(self, delivery_id: str) -> dict[str, Any]:
        delivery_id = canonical_identifier(delivery_id, name="delivery_id")
        receipt = self._receipts.get(delivery_id)
        if receipt is None:
            raise LookupError(delivery_id)
        return receipt.to_dict()

    def dead_letters(self, graph_id: str | None = None) -> tuple[dict[str, Any], ...]:
        selected = None if graph_id is None else self._graph(graph_id).graph_id
        return tuple(
            item.to_dict()
            for item in self._dead_letters
            if selected is None or item.graph_id == selected
        )

    def status(self) -> dict[str, Any]:
        queue_by_graph: dict[str, int] = {}
        for delivery in self._queue:
            queue_by_graph[delivery.graph_id] = queue_by_graph.get(delivery.graph_id, 0) + 1
        recent_receipts = list(self._receipts.values())[-256:]
        latencies_ms = sorted(
            max(0.0, (item.completed_at_ns - item.started_at_ns) / 1_000_000.0)
            for item in recent_receipts
        )
        now_ns = int(self._wall_clock_ns())
        oldest_queue_age_ms = max(
            (max(0, now_ns - item.emitted_at_ns) / 1_000_000.0 for item in self._queue),
            default=0.0,
        )
        return {
            "running": self._running,
            "ready": (
                self._running
                and not self._circuit_open
                and not self._dependency_blocks
                and (
                    not self._worker_enabled
                    or (self._worker_task is not None and not self._worker_task.done())
                )
            ),
            "worker": {
                "enabled": self._worker_enabled,
                "alive": bool(self._worker_task is not None and not self._worker_task.done()),
                "failure_streak": self._worker_failures,
                "last_error": self._last_worker_error,
            },
            "graph_count": len(self._graphs),
            "queue_depth": len(self._queue),
            "dead_letter_depth": len(self._dead_letters),
            "receipt_count": len(self._receipts),
            "paused_graphs": sorted(self._paused),
            "open_circuits": [
                {"graph_id": graph_id, "node_id": node_id}
                for graph_id, node_id in sorted(self._circuit_open)
            ],
            "blocked_dependencies": [
                {"graph_id": graph_id, "node_id": node_id, "error": error}
                for (graph_id, node_id), error in sorted(self._dependency_blocks.items())
            ],
            "queue_by_graph": dict(sorted(queue_by_graph.items())),
            "latency": {
                "sample_count": len(latencies_ms),
                "p50_ms": (
                    round(latencies_ms[(len(latencies_ms) - 1) // 2], 3)
                    if latencies_ms
                    else None
                ),
                "p95_ms": (
                    round(latencies_ms[max(0, int(len(latencies_ms) * 0.95) - 1)], 3)
                    if latencies_ms
                    else None
                ),
                "max_ms": round(latencies_ms[-1], 3) if latencies_ms else None,
                "oldest_queue_age_ms": round(oldest_queue_age_ms, 3),
            },
            "counters": dict(self._counters),
            "state_path": str(self._state_path),
        }

    def is_alive(self) -> bool:
        return self._running

    def is_ready(self) -> bool:
        return bool(self.status()["ready"])

    def _graph(self, graph_id: str) -> FlowGraph:
        normalized = canonical_identifier(graph_id, name="graph_id")
        graph = self._graphs.get(normalized)
        if graph is None:
            raise LookupError(normalized)
        return graph


__all__ = [
    "FlowHandler",
    "FlowHandlerResult",
    "RealityEventFlowError",
    "RealityEventFlowDependencyUnavailableError",
    "RealityEventFlowRuntime",
]
