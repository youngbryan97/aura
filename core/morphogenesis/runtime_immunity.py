"""The signal bus, and the bridge from it to the immune system.

Morphogenesis emits a signal whenever the shape of the runtime changes or
refuses to. Some of those signals are the immune system's business — a cell
that will not start, a port that keeps failing — and this is what carries them
across, on a worker of its own so that a slow immune response cannot hold the
morphogenetic tick.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("Aura.Morphogenesis.Runtime")


import asyncio
import inspect
import time
from collections.abc import Sequence
from typing import Any

from core.runtime.task_ownership import create_tracked_task

from .types import MorphogenSignal, SignalKind


class _BridgesSignalsToImmunity:
    """Lifted whole from MorphogeneticRuntime; see runtime.py."""

    def emit_signal(self, signal: MorphogenSignal) -> None:
        if signal.ttl_ticks <= 0:
            return
        self._signals.append(signal)
        try:
            self.field.ingest_signal(signal)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_degradation_at = time.time()
            # Imported here rather than at module level: runtime.py imports this
            # module to build the class, so the other direction has to be local.
            from .runtime import _record_morphogenesis_runtime_degradation

            _record_morphogenesis_runtime_degradation(
                exc,
                action="queued morphogen signal but skipped field ingestion after field update failed",
                severity="degraded",
                extra={"signal_kind": str(signal.kind), "subsystem": signal.subsystem},
            )

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
        # Imported here rather than at module level: runtime.py imports this
        # module to build the class, so the other direction has to be local.
        from .runtime import _record_morphogenesis_runtime_degradation

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
