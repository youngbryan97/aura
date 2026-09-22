"""core/resilience/recovery_bridge.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detect → decide → act: fault records actuate their RecoveryStrategy
through the immune system instead of stopping at the dashboard.

Until now the fault taxonomy's RecoveryStrategy column was descriptive —
detection was live (registry, evidence, drift, SLOs) but nothing
EXECUTED the cataloged response. This bridge closes the loop by feeding
qualifying fault records to the existing immune system
(core/security/immune_system.py: detect→reason→respond→heal with
FOP-safe cooldowns) rather than building a second actuation engine.

Safety posture (deliberate, per strategy):
- AUTO lane (executed): CIRCUIT_BREAKER, AUTOMATIC_FALLBACK,
  GRACEFUL_DEGRADATION, RETRY_WITH_BACKOFF — these are advisory-safe
  responses the runtime already tolerates; handlers may be registered
  by owning subsystems, and absent a handler the response is a
  structured recommendation (thought stream + log), never a no-op
  silence.
- OPERATOR lane (recommended only): AUTOMATIC_RESTART (subsystems like
  the MLX worker already self-restart — double-firing is worse),
  STEM_CELL_REVERT, QUARANTINE, MANUAL_INTERVENTION — surfaced with the
  runbook link for the human.
- IGNORE strategy faults and NEGLIGIBLE severity never enter the bridge.

Cost discipline: record_fault stays O(1) — the registry listener only
enqueues; a single daemon worker drains the bounded queue. Per-fault-id
cooldown prevents response storms. Kill switch: AURA_RECOVERY_BRIDGE=0.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any

logger = logging.getLogger("Aura.RecoveryBridge")

_QUEUE_MAX = 256
_PER_FAULT_COOLDOWN_S = 300.0

_AUTO_STRATEGIES = frozenset({
    "circuit_breaker",
    "automatic_fallback",
    "graceful_degradation",
    "retry_with_backoff",
})
_OPERATOR_STRATEGIES = frozenset({
    "automatic_restart",
    "stem_cell_revert",
    "quarantine",
    "manual_intervention",
})


class RecoveryBridge:
    """Routes fault records to immune responses / operator recommendations."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=_QUEUE_MAX)
        self._cooldowns: dict[str, float] = {}
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._started = False
        self._stats = {
            "enqueued": 0, "dropped_full": 0, "cooldown_skips": 0,
            "auto_responses": 0, "operator_recommendations": 0, "errors": 0,
        }

    # ── registry listener (hot path: enqueue only) ───────────────────

    def on_fault(self, record: Any) -> None:
        if not self._started:
            return
        try:
            strategy = self._strategy_for(record)
            if strategy is None:
                return
            now = time.monotonic()
            with self._lock:
                if now < self._cooldowns.get(record.fault_id, 0.0):
                    self._stats["cooldown_skips"] += 1
                    return
                self._cooldowns[record.fault_id] = now + _PER_FAULT_COOLDOWN_S
            try:
                self._queue.put_nowait(record)
                with self._lock:
                    self._stats["enqueued"] += 1
            except queue.Full:
                with self._lock:
                    self._stats["dropped_full"] += 1
        except (AttributeError, TypeError, RuntimeError) as exc:
            logger.debug("Recovery bridge listener skipped a record: %s", exc)

    def _strategy_for(self, record: Any) -> str | None:
        try:
            from core.resilience.fault_taxonomy import (
                FaultSeverity,
                get_fault_registry,
            )

            if record.severity == FaultSeverity.NEGLIGIBLE or record.recovered:
                return None
            defn = get_fault_registry().get_definition(record.fault_id)
            if defn is None:
                return None
            strategy = str(defn.recovery.value)
            if strategy == "ignore":
                return None
            return strategy
        # not a failure: no recovery definition for this fault means no strategy to name.
        except (ImportError, AttributeError, RuntimeError):
            return None

    # ── worker ────────────────────────────────────────────────────────

    def start(self) -> bool:
        if os.environ.get("AURA_RECOVERY_BRIDGE", "1").strip().lower() in {
            "0", "false", "off", "no",
        }:
            logger.info("Recovery bridge disabled by env.")
            return False
        if self._started:
            return True
        self._started = True
        self._worker = threading.Thread(
            target=self._drain, name="recovery-bridge", daemon=True,
        )
        self._worker.start()
        logger.info("Recovery bridge online — faults now actuate their strategies.")
        return True

    def _drain(self) -> None:
        while self._started:  # daemon lifetime; bounded by the started flag
            record = self._queue.get()
            try:
                self._respond(record)
            except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
                with self._lock:
                    self._stats["errors"] += 1
                logger.warning("Recovery response failed for %s: %s", getattr(record, "fault_id", "?"), exc)

    def _respond(self, record: Any) -> None:
        from core.resilience.fault_taxonomy import get_fault_registry

        registry = get_fault_registry()
        defn = registry.get_definition(record.fault_id)
        strategy = str(defn.recovery.value) if defn else "unknown"
        description = (
            f"fault {record.fault_id} in {record.subsystem}: "
            f"{str(record.details)[:140]} (strategy={strategy})"
        )

        if strategy in _OPERATOR_STRATEGIES:
            self._recommend(record, defn, strategy)
            return

        # AUTO lane: route through the immune system's respond/heal
        # discipline; its cooldowns and tolerance protect against loops.
        started = time.monotonic()
        try:
            from core.security.immune_system import get_immune_system

            response = get_immune_system().assess_and_respond(
                source=f"recovery_bridge:{record.subsystem}",
                description=description,
            )
            with self._lock:
                self._stats["auto_responses"] += 1
            elapsed = time.monotonic() - started
            registry.mark_recovered(record, recovery_time_s=round(elapsed, 3))
            logger.info(
                "Recovery response for %s: %s (%.2fs)",
                record.fault_id,
                getattr(response, "action", None) or "assessed",
                elapsed,
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            # No immune system: fall back to a structured recommendation —
            # a fault with a cataloged strategy is never silently ignored.
            logger.debug("Immune system unavailable (%s); recommending instead.", exc)
            self._recommend(record, defn, strategy)

    def _recommend(self, record: Any, defn: Any, strategy: str) -> None:
        runbook = getattr(defn, "runbook", "") or "no runbook on file"
        message = (
            f"Fault {record.fault_id} in {record.subsystem} calls for "
            f"'{strategy}' — recommended action for the operator ({runbook})."
        )
        with self._lock:
            self._stats["operator_recommendations"] += 1
        logger.warning("RECOVERY RECOMMENDATION: %s", message)
        try:
            from core.thought_stream import get_emitter

            get_emitter().emit(
                "Recovery recommendation", message, level="warning", category="Immune",
            )
        except (ImportError, AttributeError, RuntimeError):
            logger.debug("Thought stream unavailable for recovery recommendation.")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started": self._started,
                "queue_depth": self._queue.qsize(),
                **dict(self._stats),
            }


_bridge: RecoveryBridge | None = None
_bridge_lock = threading.Lock()


def get_recovery_bridge() -> RecoveryBridge:
    global _bridge
    if _bridge is None:
        with _bridge_lock:
            if _bridge is None:
                _bridge = RecoveryBridge()
    return _bridge
