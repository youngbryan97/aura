from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
import traceback
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.task_ownership import create_tracked_task

from .field import MorphogenField
from .governor import MorphBounds, MorphGovernor
from .graph import EdgeType, MorphEdge, MorphGraph
from .lineage import Lineage
from .metabolism import MetabolismManager
from .motifs import MotifLibrary
from .organs import OrganStabilizer
from .registry import MorphogenesisRegistry
from .substrate import LocalRuntimeSubstrate
from .types import MorphogenesisConfig, MorphogenSignal, SignalKind, stable_digest

logger = logging.getLogger("Aura.Morphogenesis.Runtime")

_MORPHOGENESIS_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _record_morphogenesis_runtime_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "morphogenesis.runtime",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation(
                "morphogenesis.runtime",
                error,
                severity=severity,
                action=action,
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.debug(
                "Morphogenesis runtime degradation could not be recorded: %s",
                signature_exc,
            )


class MorphogeneticRuntime:
    """Bounded self-organisation loop.

    Key hardening properties:
      - uses canonical task ownership instead of raw asyncio.create_task
      - never applies source-code patches directly
      - bridges high-danger events into AdaptiveImmuneSystem
      - persists registry through atomic writer when available
      - logs longitudinal episodes through EpisodicMemory when available
      - enforces caps on signals, cells, organs and actions per tick
      - holds the topology as state, and changes it only through the governor

    The topology is the part that was missing. The layer ran in production with
    a population and no bindings between them: every cell reached every other
    through one global signal queue, so two different shapes computed the same
    thing and the shape was decoration. The graph makes reachability real and
    the governor is the only thing allowed to change it.

    The live governor runs with governance armed and with the in-process
    substrate, which refuses ``migrate`` outright — a cell cannot leave this
    process, and reporting a move that did not happen would put the graph and
    the world out of agreement.
    """

    shutdown_timeout_s = 8.0

    def __init__(
        self,
        *,
        config: MorphogenesisConfig | None = None,
        registry: MorphogenesisRegistry | None = None,
        field: MorphogenField | None = None,
        metabolism: MetabolismManager | None = None,
        organ_stabilizer: OrganStabilizer | None = None,
    ):
        self.config = config or MorphogenesisConfig()
        self.registry = registry or MorphogenesisRegistry(config=self.config)
        self.field = field or MorphogenField(diffusion=self.config.field_diffusion, decay=self.config.field_decay)
        self.metabolism = metabolism or MetabolismManager(recovery_per_tick=self.config.energy_recovery_per_tick)
        self.organ_stabilizer = organ_stabilizer or OrganStabilizer(
            min_coactivations=self.config.organ_min_coactivations,
            min_members=self.config.organ_min_members,
            edge_threshold=self.config.organ_edge_threshold,
        )
        self.graph = MorphGraph(
            max_nodes=self.config.max_cells,
            max_edges=self.config.max_edges,
        )
        self.substrate = LocalRuntimeSubstrate(max_cells=self.config.max_cells)
        self.lineage = Lineage()
        self.motifs = MotifLibrary()
        self.governor = MorphGovernor(
            self.graph,
            self.substrate,
            bounds=MorphBounds(
                max_cells=self.config.max_cells,
                max_edges=self.config.max_edges,
            ),
            lineage=self.lineage,
            # No shadow evaluator on the live path yet: there is no offline
            # replica of the running system to measure a candidate shape
            # against, and an evaluator that cannot measure is a refusal. Until
            # one exists the live layer proposes bindings and refuses anything
            # above ROUTINE, which is Phase 2's work.
            shadow_evaluator=None,
            require_governance=self.config.require_governance_for_mutation,
            emit_receipts=True,
        )
        self.goal_demand: dict[str, float] = {}
        self._signals: deque[MorphogenSignal] = deque(maxlen=max(16, self.config.max_signals_per_tick * 4))
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._tick = 0
        self._events_since_episode = 0
        self._last_tick_error = ""
        self._last_tick_at = 0.0
        self._started_at = 0.0
        self._episode_buffer: deque[dict[str, Any]] = deque(maxlen=32)
        self._consecutive_tick_failures = 0
        self._last_degradation_at = 0.0
        self._hooks_wired = False
        self._last_hook_results: dict[str, Any] = {}
        self._stop_lock = asyncio.Lock()
        self._restart_lock = asyncio.Lock()
        self._persisted_on_stop = False
        queue_capacity = max(1, int(self.config.immunity_bridge_queue_capacity))
        self._immunity_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(
            maxsize=queue_capacity
        )
        self._immunity_pending_ids: set[str] = set()
        self._immunity_task: asyncio.Task | None = None
        self._immunity_inflight_id = ""
        self._immunity_inflight_started_at = 0.0
        self._immunity_enqueued = 0
        self._immunity_processed = 0
        self._immunity_failures = 0
        self._immunity_deduplicated = 0
        self._immunity_dropped = 0
        self._last_immunity_error = ""
        self._last_immunity_degradation_at = 0.0

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        if not self.config.enabled:
            logger.info("MorphogeneticRuntime disabled by config.")
            return
        self._consume_finished_task_failure(
            self._task,
            action="recovered a previously failed morphogenesis loop before restart",
        )
        self._consume_finished_task_failure(
            self._immunity_task,
            action="recovered a previously failed adaptive-immunity bridge worker before restart",
        )
        try:
            self.registry.load()
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
            self._last_degradation_at = time.time()
            _record_morphogenesis_runtime_degradation(
                exc,
                action="started morphogenesis runtime with fresh in-memory registry after registry load failed",
                severity="degraded",
            )
        self._stopping.clear()
        self._persisted_on_stop = False
        self._started_at = time.time()
        try:
            self._task = create_tracked_task(
                self._run_loop(),
                name="morphogenesis.runtime",
            )
            if self.config.adaptive_immunity_bridge:
                self._ensure_immunity_worker()
        except _MORPHOGENESIS_RECOVERABLE_ERRORS:
            task = self._task
            self._task = None
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise
        logger.info("MorphogeneticRuntime started.")

    async def stop(self) -> None:
        async with self._stop_lock:
            self._stopping.set()
            tasks = [
                task
                for task in (self._task, self._immunity_task)
                if task is not None and task is not asyncio.current_task()
            ]
            self._task = None
            self._immunity_task = None
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, asyncio.CancelledError) or result is None:
                        continue
                    if isinstance(result, BaseException):
                        self._last_degradation_at = time.time()
                        _record_morphogenesis_runtime_degradation(
                            result,
                            action="contained an owned morphogenesis task failure during lifecycle shutdown",
                            severity="degraded",
                        )
            if not self._persisted_on_stop:
                try:
                    await asyncio.to_thread(self.registry.save)
                    self._persisted_on_stop = True
                except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
                    self._last_degradation_at = time.time()
                    _record_morphogenesis_runtime_degradation(
                        exc,
                        action="completed morphogenesis shutdown while registry save failed; in-memory state was preserved until process exit",
                        severity="critical",
                    )
            logger.info("MorphogeneticRuntime stopped.")

    async def restart_async(self) -> None:
        """Restart the complete owned runtime without leaking prior task failures."""

        async with self._restart_lock:
            await self.stop()
            await self.start()
            if self.config.enabled and not self.status()["running"]:
                raise RuntimeError("morphogenesis runtime did not become ready after restart")

    async def on_stop_async(self) -> None:
        """ServiceContainer lifecycle hook — ensures clean shutdown."""
        await self.stop()

    def _consume_finished_task_failure(
        self,
        task: asyncio.Task | None,
        *,
        action: str,
    ) -> None:
        if task is None or not task.done():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        except _MORPHOGENESIS_RECOVERABLE_ERRORS as exc:
            error = exc
        if error is None:
            return
        self._last_degradation_at = time.time()
        _record_morphogenesis_runtime_degradation(
            error,
            action=action,
            severity="degraded",
        )

    def emit_signal(self, signal: MorphogenSignal) -> None:
        if signal.ttl_ticks <= 0:
            return
        self._signals.append(signal)
        try:
            self.field.ingest_signal(signal)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_degradation_at = time.time()
            _record_morphogenesis_runtime_degradation(
                exc,
                action="queued morphogen signal but skipped field ingestion after field update failed",
                severity="degraded",
                extra={"signal_kind": str(signal.kind), "subsystem": signal.subsystem},
            )

    def mark_hooks_wired(self, results: dict[str, Any] | None = None, *, ok: bool) -> None:
        self._hooks_wired = ok
        self._last_hook_results = dict(results or {})

    def observe_exception(
        self,
        *,
        subsystem: str,
        exc: BaseException,
        source: str = "runtime_exception",
        danger: float = 0.75,
        stack_trace: str | None = None,
    ) -> MorphogenSignal:
        stack = stack_trace or "".join(traceback.format_exception(type(exc), exc, getattr(exc, "__traceback__", None)))
        sig = MorphogenSignal(
            kind=SignalKind.EXCEPTION,
            source=source,
            subsystem=subsystem,
            intensity=danger,
            payload={
                "exception_type": type(exc).__name__,
                "message": str(exc)[:500],
                "stack_trace": stack[-4000:],
            },
            ttl_ticks=8,
        )
        self.emit_signal(sig)
        return sig

    async def tick(self) -> dict[str, Any]:
        started = time.monotonic()
        self._tick += 1
        self._last_tick_at = time.time()

        # Heartbeat SelfHealing watchdog — proves the runtime is alive.
        try:
            from core.morphogenesis.hooks import heartbeat_self_healing
            heartbeat_self_healing()
        except (ImportError, AttributeError, RuntimeError):
            pass  # no-op: intentional

        resource = self.metabolism.pulse()
        self._emit_system_signals(resource_pressure=resource.pressure)

        # Modulate MetabolicCoordinator energy refill based on field pressure.
        # This is how morphogenesis influences resource allocation: under high
        # danger/pressure, background autonomous tasks slow down; under high
        # growth/curiosity, they speed up.
        if self._tick % 5 == 0:
            try:
                from core.morphogenesis.hooks import modulate_metabolic_energy
                modulate_metabolic_energy()
            except (ImportError, AttributeError, RuntimeError):
                pass  # no-op: intentional

        active_signals = self._consume_signals()
        for sig in active_signals:
            self.field.ingest_signal(sig)
        self.field.diffuse_step()

        if self.config.adaptive_immunity_bridge:
            await self._bridge_signals_to_immunity(active_signals)

        active_results = []
        activated_ids: list[str] = []
        success = True

        cells = self.registry.active_cells()[: self.config.max_cells]
        for cell in cells:
            priority = max(0.05, float(cell.manifest.criticality))
            self.metabolism.ensure_budget(
                cell.cell_id,
                priority=priority,
                baseline=cell.manifest.baseline_energy,
                max_energy=cell.manifest.max_energy,
            )
            result = await cell.tick(
                signals=active_signals,
                field=self.field,
                global_energy=self.metabolism.global_energy,
                config=self.config,
            )
            if result.activated:
                activated_ids.append(cell.cell_id)
                active_results.append(result.to_dict())
                success = success and result.success
                for out in result.emitted_signals:
                    self.emit_signal(out)
                if len(active_results) >= self.config.max_cell_actions_per_tick:
                    break

        if len(activated_ids) >= 2:
            task_signature = self._task_signature(active_signals)
            subsystem = self._dominant_subsystem(active_signals) or "composite"
            self.organ_stabilizer.observe_activation(
                activated_ids,
                success=success,
                task_signature=task_signature,
                subsystem=subsystem,
            )
            for organ in self.organ_stabilizer.discover():
                cell = self.registry.register_organ(organ)
                if cell is not None:
                    self.emit_signal(
                        MorphogenSignal(
                            kind=SignalKind.GROWTH,
                            source="organ_stabilizer",
                            subsystem=organ.subsystem,
                            intensity=min(0.95, organ.confidence),
                            payload={"organ": organ.to_dict()},
                            ttl_ticks=10,
                        )
                    )
                    # Record organ formation in episodic memory — this is how
                    # morphogenesis drives long-term behavioural development.
                    try:
                        from core.morphogenesis.hooks import record_organ_formation_episode
                        from core.runtime.task_ownership import fire_and_forget
                        # Compatibility contract: get_task_tracker().create_task(record_organ_formation_episode...)
                        fire_and_forget(
                            record_organ_formation_episode(organ.to_dict()),
                            name="morphogenesis.organ_episode",
                            bounded=True,
                        )
                    except (ImportError, AttributeError, RuntimeError):
                        pass  # no-op: intentional

        if self.config.topology_enabled:
            self._sync_topology()
            self._strengthen_coactivation(activated_ids)

        if self._tick % max(1, self.config.prune_every_ticks) == 0:
            self._prune_population()

        if self._tick % max(1, self.config.telemetry_every_ticks) == 0:
            self._publish_telemetry()

        if self._tick % max(1, self.config.snapshot_every_ticks) == 0:
            await asyncio.to_thread(self.registry.save)

        await self._maybe_record_episode(active_results)

        return {
            "tick": self._tick,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
            "signals": [s.to_dict() for s in active_signals],
            "activated": activated_ids,
            "results": active_results,
            "resources": resource.to_dict(),
            "registry": self.registry.status(),
        }

    def _sync_topology(self) -> None:
        """Keep the graph's node set equal to the live population.

        Cells are registered and retired by paths outside the governor — boot
        registration, organ formation, quarantine — so the graph follows them
        rather than trying to own them. Bindings are the governor's alone.
        """
        try:
            live = {cell.cell_id for cell in self.registry.active_cells()}
            known = set(self.graph.nodes())
            if live == known:
                return
            for cell_id in live - known:
                self.substrate.place(cell_id)
                self.lineage.seed(cell_id, cause="registered")
                cell = self.registry.get(cell_id)
                if cell is not None:
                    self.governor.set_capabilities(cell_id, cell.manifest.capabilities)
            gone = known - live
            arrived = live - known

            def sync(scratch: Any) -> None:
                for cell_id in sorted(arrived):
                    scratch.add_node(cell_id)
                for cell_id in sorted(gone):
                    scratch.remove_node(cell_id)

            self.graph.transaction(sync, cause=f"population_sync@tick{self._tick}")
            if gone:
                # Whatever the graph recorded about these cells was decided
                # under a world that no longer holds.
                self.governor.invalidate_reversal_history(
                    cells=sorted(gone), reason="cells left the population"
                )
                for cell_id in gone:
                    self.substrate.retire(cell_id)
                    self.lineage.record_retirement(cell_id, cause="left the registry")
        except _MORPHOGENESIS_RECOVERABLE_ERRORS as exc:
            _record_morphogenesis_runtime_degradation(
                exc,
                action="kept the previous topology after a population sync failed",
                severity="warning",
                extra={"tick": self._tick},
            )

    def _strengthen_coactivation(self, activated: list[str]) -> None:
        """Record which cells fire together, on the cells themselves.

        ``MorphogenCell.strengthen`` and ``weaken`` were written and never
        called from anywhere, so ``neighbours`` was empty on every cell for the
        life of the process and every consumer of it read an empty dict.
        """
        if len(activated) < 2:
            return
        try:
            for cell_id in activated[:16]:
                cell = self.registry.get(cell_id)
                if cell is None:
                    continue
                for other in activated[:16]:
                    if other != cell_id:
                        cell.strengthen(other, amount=0.04)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_morphogenesis_runtime_degradation(
                exc,
                action="skipped one co-activation update",
                severity="warning",
            )

    def _prune_population(self) -> None:
        try:
            summary = self.registry.prune()
            if summary.get("removed"):
                logger.info("Morphogenesis pruned %s", summary)
        except _MORPHOGENESIS_RECOVERABLE_ERRORS as exc:
            _record_morphogenesis_runtime_degradation(
                exc, action="skipped one population prune", severity="warning",
            )

    def _publish_telemetry(self) -> None:
        try:
            from core.morphogenesis import telemetry

            status = self.governor.status()
            status["component_sizes"] = ",".join(
                str(len(component)) for component in self.graph.components()
            )
            telemetry.publish(status)
            telemetry.publish_motifs(self.motifs.status())
        except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            _record_morphogenesis_runtime_degradation(
                exc, action="skipped one telemetry publish", severity="warning",
            )

    def set_goal_demand(self, demand: Mapping[str, float]) -> None:
        """What the current work needs, by capability.

        The one global thing a local cell may read. A cell in a body does get
        told what the body is trying to do; it does not get told where the load
        is or who is struggling.
        """
        self.goal_demand = {str(k): float(v) for k, v in dict(demand or {}).items()}

    def propose(self, proposals: Sequence[Any]) -> list[Any]:
        """Submit topology proposals to the governor. The only way in."""
        if not self.config.topology_enabled:
            return []
        self.governor.set_port_contract(self._port_contract())
        return self.governor.submit(list(proposals))

    def _port_contract(self) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
        contract: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
        for cell in self.registry.active_cells():
            emits = frozenset(str(v) for v in cell.manifest.emits)
            consumes = frozenset(str(v) for v in cell.manifest.consumes)
            capabilities = frozenset(str(v) for v in cell.manifest.capabilities)
            contract[cell.cell_id] = (emits | capabilities, consumes | capabilities)
        return contract

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            sleep_s = max(0.05, self.config.tick_interval_s)
            try:
                if self._foreground_quiet_window_active():
                    self._last_tick_at = time.time()
                    with contextlib.suppress(Exception):
                        from core.morphogenesis.hooks import heartbeat_self_healing
                        heartbeat_self_healing()
                    await asyncio.sleep(max(0.5, self.config.tick_interval_s))
                    continue
                await self.tick()
                self._last_tick_error = ""
                self._consecutive_tick_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - supervised loop boundary
                self._consecutive_tick_failures += 1
                self._last_degradation_at = time.time()
                sleep_s = min(5.0, sleep_s * (1 + self._consecutive_tick_failures))
                _record_morphogenesis_runtime_degradation(
                    exc,
                    action="emitted morphogenesis error signal and backed off after tick failure",
                    severity="degraded",
                    extra={"consecutive_tick_failures": self._consecutive_tick_failures},
                )
                self._last_tick_error = f"{type(exc).__name__}: {exc}"
                logger.error("Morphogenesis tick failed: %s", self._last_tick_error, exc_info=True)
                self.emit_signal(
                    MorphogenSignal(
                        kind=SignalKind.ERROR,
                        source="morphogenesis.runtime",
                        subsystem="morphogenesis",
                        intensity=0.82,
                        payload={"error": self._last_tick_error},
                    )
                )
            await asyncio.sleep(sleep_s)

    @staticmethod
    def _foreground_quiet_window_active() -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return False
            quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
            return quiet_until > time.time()
        except (ImportError, AttributeError, RuntimeError):
            return False

    def _consume_signals(self) -> list[MorphogenSignal]:
        out: list[MorphogenSignal] = []
        while self._signals and len(out) < self.config.max_signals_per_tick:
            sig = self._signals.popleft()
            if sig.ttl_ticks <= 0:
                continue
            out.append(sig)
            # requeue if still alive
            new_ttl = sig.ttl_ticks - self.config.signal_decay_per_tick
            if new_ttl > 0 and sig.intensity > 0.02:
                self._signals.append(
                    MorphogenSignal(
                        kind=sig.kind,
                        source=sig.source,
                        subsystem=sig.subsystem,
                        intensity=sig.intensity * 0.92,
                        payload=sig.payload,
                        target_cell_id=sig.target_cell_id,
                        ttl_ticks=new_ttl,
                        timestamp=sig.timestamp,
                        signal_id=sig.signal_id,
                    )
                )
        return out

    def _emit_system_signals(self, *, resource_pressure: float) -> None:
        self.emit_signal(
            MorphogenSignal(
                kind=SignalKind.HEARTBEAT,
                source="morphogenesis.runtime",
                subsystem="global",
                intensity=0.18,
                ttl_ticks=2,
            )
        )
        if resource_pressure >= self.metabolism.high_pressure_threshold:
            self.emit_signal(
                MorphogenSignal(
                    kind=SignalKind.RESOURCE_PRESSURE,
                    source="metabolism",
                    subsystem="global",
                    intensity=resource_pressure,
                    ttl_ticks=4,
                )
            )

        # Read existing Aura state opportunistically. All failures are non-fatal.
        try:
            from core.container import ServiceContainer
            liquid = ServiceContainer.get("liquid_state", default=None)
            if liquid is not None and hasattr(liquid, "get_status"):
                status = liquid.get_status()
                if isinstance(status, dict):
                    curiosity = float(status.get("curiosity", 0.0)) / 100.0
                    energy = float(status.get("energy", 50.0)) / 100.0
                    if curiosity > 0.45:
                        self.emit_signal(MorphogenSignal(kind=SignalKind.CURIOSITY, source="liquid_state", subsystem="cognition", intensity=curiosity, ttl_ticks=3))
                    if energy < 0.25:
                        self.emit_signal(MorphogenSignal(kind=SignalKind.HOMEOSTASIS, source="liquid_state", subsystem="global", intensity=1.0 - energy, ttl_ticks=3))
        except (ImportError, AttributeError, RuntimeError):
            pass  # no-op: intentional

    async def _bridge_signals_to_immunity(self, signals: Sequence[MorphogenSignal]) -> None:
        """Queue high-danger signals without coupling immune latency to a tick."""

        if not self.config.adaptive_immunity_bridge or self._stopping.is_set():
            return
        self._ensure_immunity_worker()
        eligible = sorted(
            (
                signal
                for signal in signals
                if self._is_immunity_signal(signal)
            ),
            key=lambda signal: (-float(signal.intensity), float(signal.timestamp)),
        )
        limit = max(1, int(self.config.immunity_bridge_max_enqueue_per_tick))
        for sig in eligible[:limit]:
            signal_id = str(sig.signal_id)
            if signal_id in self._immunity_pending_ids:
                self._immunity_deduplicated += 1
                continue
            event = self._immunity_event(sig)
            try:
                self._immunity_queue.put_nowait((signal_id, event))
            except asyncio.QueueFull as exc:
                self._immunity_dropped += 1
                self._record_immunity_bridge_degradation(
                    exc,
                    action="kept morphogenesis responsive while the bounded adaptive-immunity bridge queue was full",
                    extra={
                        "queue_depth": self._immunity_queue.qsize(),
                        "queue_capacity": self._immunity_queue.maxsize,
                        "signal_kind": event["type"],
                    },
                )
                continue
            self._immunity_pending_ids.add(signal_id)
            self._immunity_enqueued += 1

    @staticmethod
    def _is_immunity_signal(signal: MorphogenSignal) -> bool:
        kind = signal.kind.value if hasattr(signal.kind, "value") else str(signal.kind)
        return kind in {
            SignalKind.ERROR.value,
            SignalKind.EXCEPTION.value,
            SignalKind.DANGER.value,
            SignalKind.RESOURCE_PRESSURE.value,
        } and float(signal.intensity) >= 0.55

    @staticmethod
    def _immunity_event(signal: MorphogenSignal) -> dict[str, Any]:
        kind = signal.kind.value if hasattr(signal.kind, "value") else str(signal.kind)
        resource_observation = kind == SignalKind.RESOURCE_PRESSURE.value
        return {
            "type": kind,
            "text": str(signal.payload.get("message") or signal.payload.get("error") or kind),
            "subsystem": signal.subsystem,
            "source": f"morphogenesis:{signal.source}",
            "source_domain": "environment" if resource_observation else "substrate",
            "observation_class": (
                "resource_telemetry" if resource_observation else "runtime_fault"
            ),
            "danger": float(signal.intensity),
            "resource_pressure": float(
                signal.intensity
                if kind == SignalKind.RESOURCE_PRESSURE.value
                else signal.payload.get("resource_pressure", 0.0)
            ),
            "stack_trace": str(signal.payload.get("stack_trace", ""))[-4000:],
            "exception_type": str(signal.payload.get("exception_type", "")),
            "timestamp": signal.timestamp,
            "error_signature": str(
                signal.payload.get("exception_type") or signal.payload.get("error") or kind
            )[:120],
        }

    def _ensure_immunity_worker(self) -> None:
        task = self._immunity_task
        if task is not None and not task.done():
            return
        self._consume_finished_task_failure(
            task,
            action="recovered the adaptive-immunity bridge after its worker stopped unexpectedly",
        )
        self._immunity_task = create_tracked_task(
            self._run_immunity_worker(),
            name="morphogenesis.immunity_bridge",
            owner="morphogenesis.runtime",
        )

    async def _run_immunity_worker(self) -> None:
        while not self._stopping.is_set():
            try:
                # Timed wait, not a forever-block: the worker wakes to
                # re-check _stopping even when the queue is quiet (the
                # bounded-await discipline — a bare .get() here is the
                # mind_tick wedge class).
                signal_id, event = await asyncio.wait_for(
                    self._immunity_queue.get(), timeout=5.0
                )
            except TimeoutError:
                continue
            self._immunity_inflight_id = signal_id
            self._immunity_inflight_started_at = time.time()
            try:
                from core.adaptation.adaptive_immunity import get_adaptive_immune_system

                immune = get_adaptive_immune_system()
                # observe_event is not always cheap and it is not always async.
                # Its synchronous path reaches Will.decide ->
                # _check_memory_relevance -> memory_facade.search_sync ->
                # rag._semantic_scores -> vector_memory_engine.embed, which is a
                # CPU-bound SentenceTransformer/BERT forward pass. Called
                # directly here it ran ON the event loop.
                #
                # Live 2026-07-26: 31 "HIGH EVENT LOOP LAG" events on one boot,
                # the worst 11.9s, all at context=idle — and the loop-stall dump
                # named this exact chain. A loop stalled for seconds cannot
                # drive the cortex warmup handshake either, so the runtime sat
                # in `warming` while nothing was actually wrong with the model.
                #
                # A background immunity observation has no business on the loop:
                # hand the synchronous path to a worker thread and await that.
                result = await asyncio.to_thread(immune.observe_event, event)
                if inspect.isawaitable(result):
                    await result
                self._immunity_processed += 1
                self._last_immunity_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - optional worker boundary
                self._immunity_failures += 1
                self._last_immunity_error = f"{type(exc).__name__}: {exc}"
                self._record_immunity_bridge_degradation(
                    exc,
                    action="contained one adaptive-immunity bridge failure while preserving the morphogenesis loop",
                    extra={
                        "signal_kind": event.get("type", "unknown"),
                        "subsystem": event.get("subsystem", "unknown"),
                    },
                )
            finally:
                self._immunity_pending_ids.discard(signal_id)
                self._immunity_inflight_id = ""
                self._immunity_inflight_started_at = 0.0
                self._immunity_queue.task_done()

    def _record_immunity_bridge_degradation(
        self,
        error: BaseException,
        *,
        action: str,
        extra: dict[str, object],
    ) -> None:
        self._last_degradation_at = time.time()
        interval = max(1.0, float(self.config.immunity_bridge_degradation_interval_s))
        if self._last_degradation_at - self._last_immunity_degradation_at < interval:
            return
        self._last_immunity_degradation_at = self._last_degradation_at
        _record_morphogenesis_runtime_degradation(
            error,
            action=action,
            severity="warning",
            extra=extra,
        )

    async def wait_for_immunity_idle(self, *, timeout_s: float = 10.0) -> None:
        """Wait for all accepted bridge work; intended for proof and shutdown gates."""

        await asyncio.wait_for(
            self._immunity_queue.join(),
            timeout=max(0.05, float(timeout_s)),
        )

    async def _maybe_record_episode(self, results: list[dict[str, Any]]) -> None:
        if not results:
            return
        self._events_since_episode += len(results)
        self._episode_buffer.extend(results[-8:])
        if self._events_since_episode < self.config.episode_every_events:
            return
        self._events_since_episode = 0

        try:
            from core.container import ServiceContainer
            mem = ServiceContainer.get("episodic_memory", default=None)
            if mem is None:
                try:
                    from core.memory.episodic_memory import get_episodic_memory
                    mem = get_episodic_memory()
                except (ImportError, AttributeError, RuntimeError):
                    mem = None
            if mem is None or not hasattr(mem, "record_episode_async"):
                return
            failures = [r for r in self._episode_buffer if not r.get("success", True)]
            await mem.record_episode_async(
                context="MorphogeneticRuntime self-organization cycle",
                action="cellular_tick_and_organ_stabilization",
                outcome=f"{len(self._episode_buffer)} cell activations, failures={len(failures)}",
                success=not failures,
                emotional_valence=-0.25 if failures else 0.18,
                tools_used=["morphogenesis_runtime"],
                lessons=[
                    "Stable co-activated cells can be formalized into organs",
                    "High-danger signals must route through adaptive immunity",
                ][: 1 + bool(failures)],
                importance=0.55 if not failures else 0.75,
                source="morphogenesis",
                metadata={"tick": self._tick, "failure_count": len(failures)},
            )
            self._episode_buffer.clear()
        except (ImportError, AttributeError, RuntimeError) as exc:
            self._last_degradation_at = time.time()
            _record_morphogenesis_runtime_degradation(
                exc,
                action="kept morphogenesis tick active while episodic episode write was skipped",
                severity="warning",
                extra={"buffered_events": len(self._episode_buffer)},
            )
            logger.debug("morphogenesis episode record skipped: %s", exc)

    @staticmethod
    def _task_signature(signals: Sequence[MorphogenSignal]) -> str:
        parts = []
        for s in signals[:5]:
            parts.append(str(s.kind.value if hasattr(s.kind, "value") else s.kind))
            if s.payload.get("task"):
                parts.append(str(s.payload.get("task"))[:80])
        return stable_digest(*parts, length=12) if parts else ""

    @staticmethod
    def _dominant_subsystem(signals: Sequence[MorphogenSignal]) -> str:
        if not signals:
            return ""
        counts: dict[str, float] = {}
        for s in signals:
            counts[s.subsystem] = counts.get(s.subsystem, 0.0) + s.intensity
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def status(self) -> dict[str, Any]:
        inflight_age_s = (
            max(0.0, time.time() - self._immunity_inflight_started_at)
            if self._immunity_inflight_started_at
            else 0.0
        )
        bridge_worker_running = bool(
            self._immunity_task and not self._immunity_task.done()
        )
        bridge_stalled = bool(
            self._immunity_inflight_started_at
            and inflight_age_s
            >= max(1.0, float(self.config.immunity_bridge_stall_warning_s))
        )
        bridge_required = self.config.enabled and self.config.adaptive_immunity_bridge
        return {
            "enabled": self.config.enabled,
            "running": bool(self._task and not self._task.done()),
            "tick": self._tick,
            "started_at": self._started_at,
            "last_tick_at": self._last_tick_at,
            "last_tick_error": self._last_tick_error,
            "consecutive_tick_failures": self._consecutive_tick_failures,
            "last_degradation_at": self._last_degradation_at,
            "hooks_wired": self._hooks_wired,
            "last_hook_results": self._last_hook_results,
            "queued_signals": len(self._signals),
            "immunity_bridge": {
                "enabled": self.config.adaptive_immunity_bridge,
                "healthy": bool(
                    not bridge_required or (bridge_worker_running and not bridge_stalled)
                ),
                "worker_running": bridge_worker_running,
                "stalled": bridge_stalled,
                "queue_depth": self._immunity_queue.qsize(),
                "queue_capacity": self._immunity_queue.maxsize,
                "inflight_signal_id": self._immunity_inflight_id,
                "inflight_age_s": round(inflight_age_s, 3),
                "enqueued": self._immunity_enqueued,
                "processed": self._immunity_processed,
                "failures": self._immunity_failures,
                "deduplicated": self._immunity_deduplicated,
                "dropped": self._immunity_dropped,
                "last_error": self._last_immunity_error,
            },
            "field": self.field.to_dict(),
            "metabolism": self.metabolism.status(),
            "registry": self.registry.status(),
            "organs": self.organ_stabilizer.to_dict().get("known_organs", {}),
            "topology": {
                "enabled": self.config.topology_enabled,
                "version": self.graph.version,
                "digest": self.graph.snapshot().digest(),
                "nodes": self.graph.node_count,
                "edges": self.graph.edge_count,
                "components": len(self.graph.components()),
                "substrate": self.substrate.health(),
            },
            "governor": self.governor.stats.to_dict(),
            "lineage": self.lineage.status(),
            "motifs": self.motifs.status(),
        }


_runtime_singleton: MorphogeneticRuntime | None = None


def get_morphogenetic_runtime() -> MorphogeneticRuntime:
    global _runtime_singleton
    if _runtime_singleton is None:
        _runtime_singleton = MorphogeneticRuntime()
    return _runtime_singleton
