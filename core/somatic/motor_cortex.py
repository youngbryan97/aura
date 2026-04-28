"""core/somatic/motor_cortex.py -- Fast Reflex Loop
====================================================
"Crossing the Rubicon" -- Motor Cortex Layer.

A 50ms-target async loop that runs INDEPENDENTLY from the main 1Hz cognitive
tick.  The motor cortex handles pre-approved reflex actions that don't need
LLM deliberation:

  - Screen capture on salience trigger
  - File system reactions (new file detected -> log it)
  - Health monitoring (thermal spike -> throttle)
  - Keyboard/mouse reflexes when HID control is available

The Will pre-authorizes **capability tokens** for these reflexes.  Each
completed action gets a lightweight receipt (not full Will deliberation) and
reports back to the cognitive loop for awareness.

Design invariants:
  1. NEVER blocks the main cognitive tick.
  2. NEVER calls the LLM.
  3. Runs on its own asyncio loop cadence (~20Hz / 50ms target).
  4. All actions require a valid CapabilityToken from the Will.
  5. Every completed action emits a MotorReceipt for audit + awareness.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation

from core.utils.task_tracker import get_task_tracker

import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Aura.MotorCortex")


# ---------------------------------------------------------------------------
# Enums & Data Structures
# ---------------------------------------------------------------------------

class ReflexClass(str, Enum):
    """Categories of reflex actions the motor cortex can handle."""
    SCREEN_CAPTURE = "screen_capture"          # Grab screen on salience
    FILE_REACTION = "file_reaction"            # New file detected -> log
    HEALTH_THROTTLE = "health_throttle"        # Thermal spike -> throttle
    HID_REFLEX = "hid_reflex"                  # Keyboard / mouse reflex
    PERCEPTION_REFRESH = "perception_refresh"  # Re-sample a sensor
    METRIC_SAMPLE = "metric_sample"            # Quick telemetry snapshot
    CUSTOM = "custom"                          # Extension point


class ReflexPriority(int, Enum):
    CRITICAL = 0   # Health / safety -- always first
    HIGH = 1       # User-triggered reflexes
    NORMAL = 2     # Perception refresh, file watch
    LOW = 3        # Background sampling


@dataclass
class CapabilityToken:
    """Pre-authorization from the Will for a specific reflex class.

    The Will issues these at boot or on-demand.  The motor cortex checks
    the token before executing any reflex.  Tokens expire after ``ttl``
    seconds and can be revoked.
    """
    token_id: str
    reflex_class: ReflexClass
    max_uses: int = -1            # -1 = unlimited
    ttl: float = 3600.0           # 1 hour default
    constraints: List[str] = field(default_factory=list)
    issued_at: float = field(default_factory=time.time)
    uses: int = 0
    revoked: bool = False

    @property
    def expired(self) -> bool:
        return (time.time() - self.issued_at) > self.ttl

    @property
    def valid(self) -> bool:
        if self.revoked or self.expired:
            return False
        if self.max_uses >= 0 and self.uses >= self.max_uses:
            return False
        return True

    def consume(self) -> bool:
        """Attempt to use this token.  Returns True if valid and consumed."""
        if not self.valid:
            return False
        self.uses += 1
        return True


@dataclass
class ReflexAction:
    """A pending reflex to be executed by the motor cortex."""
    reflex_class: ReflexClass
    handler_name: str
    priority: ReflexPriority = ReflexPriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    created_at: float = field(default_factory=time.time)


@dataclass
class MotorReceipt:
    """Lightweight proof of action -- reported back to the cognitive loop."""
    receipt_id: str
    reflex_class: ReflexClass
    handler_name: str
    success: bool
    latency_ms: float
    result_summary: str = ""
    error: str = ""
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reflex Handlers
# ---------------------------------------------------------------------------

class ReflexRegistry:
    """Registry of named reflex handler functions.

    Handlers are plain async callables: ``async def handler(payload) -> dict``
    They return ``{"success": bool, "summary": str, ...}``.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable] = {}

    def register(self, name: str, handler: Callable) -> None:
        self._handlers[name] = handler

    def get(self, name: str) -> Optional[Callable]:
        return self._handlers.get(name)

    @property
    def names(self) -> List[str]:
        return list(self._handlers.keys())


# ---------------------------------------------------------------------------
# Built-in Reflex Handlers
# ---------------------------------------------------------------------------

async def _reflex_screen_capture(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Capture a screenshot when a salience trigger fires."""
    try:
        vision = ServiceContainer.get("continuous_vision", default=None)
        if vision is None:
            vision = ServiceContainer.get("screen_vision", default=None)
        if vision and hasattr(vision, "capture_frame"):
            frame = await asyncio.wait_for(
                asyncio.coroutine(vision.capture_frame)()
                if not asyncio.iscoroutinefunction(vision.capture_frame)
                else vision.capture_frame(),
                timeout=0.5,
            )
            return {"success": True, "summary": "screen_captured", "frame_size": len(frame) if frame else 0}
        return {"success": False, "summary": "no_vision_service"}
    except Exception as exc:
        record_degradation('motor_cortex', exc)
        return {"success": False, "summary": f"capture_failed: {exc}"}


async def _reflex_health_check(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Quick health telemetry sample and throttle if needed."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        thermal_pressure = 0.0

        # macOS thermal state
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                first_key = next(iter(temps))
                temp_c = temps[first_key][0].current
                thermal_pressure = max(0.0, min(1.0, (temp_c - 60) / 40))
        except Exception:
            pass  # no-op: intentional

        throttle_action = None
        if thermal_pressure > 0.8 or cpu > 95:
            # Request throttle via world state
            try:
                from core.world_state import get_world_state
                ws = get_world_state()
                ws.thermal_pressure = thermal_pressure
                ws.push_event(
                    "thermal_spike",
                    source="motor_cortex",
                    salience=0.8,
                    metadata={"cpu": cpu, "thermal": thermal_pressure},
                )
                throttle_action = "thermal_throttle_requested"
            except Exception:
                pass  # no-op: intentional

        return {
            "success": True,
            "summary": throttle_action or "health_ok",
            "cpu": cpu,
            "mem_pct": mem.percent,
            "thermal": round(thermal_pressure, 3),
        }
    except ImportError:
        return {"success": False, "summary": "psutil_not_available"}
    except Exception as exc:
        record_degradation('motor_cortex', exc)
        return {"success": False, "summary": f"health_check_failed: {exc}"}


async def _reflex_file_reaction(payload: Dict[str, Any]) -> Dict[str, Any]:
    """React to a detected file system change."""
    path = payload.get("path", "")
    event_type = payload.get("event_type", "unknown")
    try:
        from core.world_state import get_world_state
        ws = get_world_state()
        ws.push_event(
            f"file_{event_type}: {path}",
            source="motor_cortex",
            salience=0.4,
            metadata={"path": path, "event_type": event_type},
        )
        return {"success": True, "summary": f"file_{event_type}_logged", "path": path}
    except Exception as exc:
        record_degradation('motor_cortex', exc)
        return {"success": False, "summary": f"file_reaction_failed: {exc}"}


async def _reflex_metric_sample(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Quick telemetry snapshot for the proprioceptive loop."""
    try:
        import psutil
        return {
            "success": True,
            "summary": "metric_sampled",
            "cpu": psutil.cpu_percent(interval=0),
            "mem_pct": psutil.virtual_memory().percent,
            "timestamp": time.time(),
        }
    except Exception as exc:
        record_degradation('motor_cortex', exc)
        return {"success": False, "summary": f"metric_sample_failed: {exc}"}


# ---------------------------------------------------------------------------
# The Motor Cortex
# ---------------------------------------------------------------------------

class MotorCortex:
    """Fast reflex loop -- the somatic action engine.

    Runs independently of the 1Hz cognitive tick at ~20Hz (50ms target).
    Executes pre-approved reflex actions using capability tokens from the Will.

    Usage:
        cortex = MotorCortex()
        await cortex.start()
        cortex.submit_reflex(ReflexAction(...))
        # ... later ...
        await cortex.stop()
    """

    _MAX_QUEUE_SIZE = 256
    _MAX_RECEIPT_TRAIL = 500
    _CYCLE_TARGET_S = 0.050   # 50ms target cycle time
    _HEALTH_INTERVAL_S = 5.0  # Health check every 5s

    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self._MAX_QUEUE_SIZE
        )
        self._seq = 0  # Sequence counter for stable priority ordering

        # Capability tokens (issued by the Will)
        self._tokens: Dict[ReflexClass, CapabilityToken] = {}

        # Action receipts (reported back to cognitive loop)
        self._receipts: Deque[MotorReceipt] = deque(maxlen=self._MAX_RECEIPT_TRAIL)
        self._pending_cognitive_reports: Deque[MotorReceipt] = deque(maxlen=64)

        # Reflex handler registry
        self._registry = ReflexRegistry()
        self._register_builtins()

        # Telemetry
        self._cycle_count = 0
        self._total_actions = 0
        self._total_latency_ms = 0.0
        self._last_health_check = 0.0
        self._boot_time = time.time()

        logger.info("MotorCortex created -- awaiting start()")

    def _register_builtins(self) -> None:
        """Register the built-in reflex handlers."""
        self._registry.register("screen_capture", _reflex_screen_capture)
        self._registry.register("health_check", _reflex_health_check)
        self._registry.register("file_reaction", _reflex_file_reaction)
        self._registry.register("metric_sample", _reflex_metric_sample)

    # ------------------------------------------------------------------
    # Capability Token Management
    # ------------------------------------------------------------------

    def issue_token(
        self,
        reflex_class: ReflexClass,
        *,
        max_uses: int = -1,
        ttl: float = 3600.0,
        constraints: Optional[List[str]] = None,
    ) -> CapabilityToken:
        """Issue a new capability token for a reflex class.

        Typically called by the Will or during boot to pre-authorize reflexes.
        """
        token_id = "mct_" + hashlib.sha256(
            f"{time.time():.6f}:{reflex_class.value}".encode()
        ).hexdigest()[:12]

        token = CapabilityToken(
            token_id=token_id,
            reflex_class=reflex_class,
            max_uses=max_uses,
            ttl=ttl,
            constraints=constraints or [],
        )
        self._tokens[reflex_class] = token
        logger.debug("MotorCortex: issued token %s for %s", token_id, reflex_class.value)
        return token

    def revoke_token(self, reflex_class: ReflexClass) -> None:
        """Revoke a capability token."""
        token = self._tokens.get(reflex_class)
        if token:
            token.revoked = True
            logger.info("MotorCortex: revoked token for %s", reflex_class.value)

    def _check_authorization(self, reflex_class: ReflexClass) -> bool:
        """Check if a reflex class has a valid capability token."""
        token = self._tokens.get(reflex_class)
        if token is None:
            return False
        return token.valid

    def _consume_authorization(self, reflex_class: ReflexClass) -> bool:
        """Consume a use of the capability token."""
        token = self._tokens.get(reflex_class)
        if token is None:
            return False
        return token.consume()

    # ------------------------------------------------------------------
    # Reflex Submission
    # ------------------------------------------------------------------

    def submit_reflex(self, action: ReflexAction) -> bool:
        """Submit a reflex action to the motor cortex queue.

        Returns True if queued, False if rejected (queue full, no token, etc).
        """
        if not self._running:
            logger.debug("MotorCortex: not running, rejecting reflex %s", action.handler_name)
            return False

        if not self._check_authorization(action.reflex_class):
            logger.debug(
                "MotorCortex: no valid token for %s, rejecting %s",
                action.reflex_class.value,
                action.handler_name,
            )
            return False

        try:
            self._seq += 1
            # Priority queue: (priority, sequence, action)
            self._queue.put_nowait((action.priority.value, self._seq, action))
            return True
        except asyncio.QueueFull:
            logger.warning("MotorCortex: queue full, dropping reflex %s", action.handler_name)
            return False

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register a custom reflex handler."""
        self._registry.register(name, handler)

    # ------------------------------------------------------------------
    # Cognitive Loop Interface
    # ------------------------------------------------------------------

    def drain_pending_reports(self) -> List[MotorReceipt]:
        """Drain pending receipts for the cognitive loop to consume.

        Called by the proprioceptive loop or affect phase to become
        aware of motor actions that happened since last tick.
        """
        reports = list(self._pending_cognitive_reports)
        self._pending_cognitive_reports.clear()
        return reports

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the motor cortex reflex loop."""
        if self._running:
            return

        # Register in ServiceContainer
        ServiceContainer.register_instance("motor_cortex", self, required=False)

        # Issue default capability tokens for standard reflexes
        self._issue_default_tokens()

        self._running = True
        self._task = get_task_tracker().create_task(self._run_loop(), name="motor_cortex_loop")
        logger.info("MotorCortex ONLINE -- 50ms reflex loop active")

    async def stop(self) -> None:
        """Stop the motor cortex."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # no-op: intentional
            self._task = None
        logger.info("MotorCortex stopped -- %d total actions executed", self._total_actions)

    def _issue_default_tokens(self) -> None:
        """Issue default tokens for standard reflexes at boot."""
        defaults = [
            (ReflexClass.HEALTH_THROTTLE, -1, 86400.0),   # Unlimited, 24h
            (ReflexClass.METRIC_SAMPLE, -1, 86400.0),     # Unlimited, 24h
            (ReflexClass.SCREEN_CAPTURE, 1000, 3600.0),   # Max 1000/hour
            (ReflexClass.FILE_REACTION, 500, 3600.0),     # Max 500/hour
            (ReflexClass.PERCEPTION_REFRESH, -1, 86400.0),
            (ReflexClass.CUSTOM, 100, 3600.0),
        ]
        for cls, max_uses, ttl in defaults:
            self.issue_token(cls, max_uses=max_uses, ttl=ttl)

    # ------------------------------------------------------------------
    # The Fast Loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main reflex loop -- targets 50ms cycle time."""
        logger.debug("MotorCortex: reflex loop started")
        while self._running:
            cycle_start = time.monotonic()
            self._cycle_count += 1

            try:
                # Process queued reflexes (batch up to 4 per cycle)
                processed = 0
                while processed < 4 and not self._queue.empty():
                    try:
                        _priority, _seq, action = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await self._execute_reflex(action)
                    processed += 1

                # Periodic health check (every 5s)
                now = time.time()
                if now - self._last_health_check > self._HEALTH_INTERVAL_S:
                    self._last_health_check = now
                    self.submit_reflex(ReflexAction(
                        reflex_class=ReflexClass.HEALTH_THROTTLE,
                        handler_name="health_check",
                        priority=ReflexPriority.LOW,
                        source="motor_cortex_periodic",
                    ))

            except Exception as exc:
                record_degradation('motor_cortex', exc)
                logger.error("MotorCortex: loop error: %s", exc, exc_info=True)

            # Sleep to maintain target cycle time
            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0.0, self._CYCLE_TARGET_S - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _execute_reflex(self, action: ReflexAction) -> None:
        """Execute a single reflex action and record the receipt."""
        t0 = time.monotonic()

        # Authorization check (consume a use)
        if not self._consume_authorization(action.reflex_class):
            receipt = MotorReceipt(
                receipt_id=self._make_receipt_id(t0, action),
                reflex_class=action.reflex_class,
                handler_name=action.handler_name,
                success=False,
                latency_ms=0.0,
                error="authorization_denied",
            )
            self._record_receipt(receipt)
            return

        handler = self._registry.get(action.handler_name)
        if handler is None:
            receipt = MotorReceipt(
                receipt_id=self._make_receipt_id(t0, action),
                reflex_class=action.reflex_class,
                handler_name=action.handler_name,
                success=False,
                latency_ms=0.0,
                error=f"handler_not_found: {action.handler_name}",
            )
            self._record_receipt(receipt)
            return

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(handler(action.payload), timeout=2.0)
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, handler, action.payload),
                    timeout=2.0,
                )
        except asyncio.TimeoutError:
            result = {"success": False, "summary": "handler_timeout"}
        except Exception as exc:
            record_degradation('motor_cortex', exc)
            result = {"success": False, "summary": f"handler_error: {exc}"}

        latency_ms = (time.monotonic() - t0) * 1000

        receipt = MotorReceipt(
            receipt_id=self._make_receipt_id(t0, action),
            reflex_class=action.reflex_class,
            handler_name=action.handler_name,
            success=bool(result.get("success", False)),
            latency_ms=round(latency_ms, 2),
            result_summary=str(result.get("summary", ""))[:200],
            error=str(result.get("error", ""))[:200] if not result.get("success") else "",
            payload={k: v for k, v in result.items() if k not in ("success", "summary", "error")},
        )

        self._record_receipt(receipt)
        self._total_actions += 1
        self._total_latency_ms += latency_ms

        # Feed back into affect system on failure
        if not receipt.success:
            self._emit_affect_feedback(receipt)

    def _record_receipt(self, receipt: MotorReceipt) -> None:
        """Record a receipt and queue it for cognitive loop awareness."""
        self._receipts.append(receipt)
        self._pending_cognitive_reports.append(receipt)

        # Publish to event bus for system-wide observability
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe("motor_cortex.action", {
                "receipt_id": receipt.receipt_id,
                "reflex_class": receipt.reflex_class.value,
                "handler": receipt.handler_name,
                "success": receipt.success,
                "latency_ms": receipt.latency_ms,
                "summary": receipt.result_summary,
                "timestamp": receipt.timestamp,
            })
        except Exception:
            pass  # no-op: intentional

    def _emit_affect_feedback(self, receipt: MotorReceipt) -> None:
        """Push failure feedback into the affect system (cortisol bump)."""
        try:
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect is None:
                affect = ServiceContainer.get("affect_facade", default=None)
            if affect and hasattr(affect, "inject_percept"):
                affect.inject_percept({
                    "type": "motor_failure",
                    "source": "motor_cortex",
                    "handler": receipt.handler_name,
                    "error": receipt.error,
                    "salience": 0.4,
                })
        except Exception:
            pass  # no-op: intentional

    @staticmethod
    def _make_receipt_id(ts: float, action: ReflexAction) -> str:
        raw = f"{ts:.6f}:{action.handler_name}:{action.reflex_class.value}"
        return "mr_" + hashlib.sha256(raw.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Status & Health
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return current motor cortex status."""
        avg_latency = (
            round(self._total_latency_ms / max(1, self._total_actions), 2)
            if self._total_actions > 0
            else 0.0
        )
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "total_actions": self._total_actions,
            "avg_latency_ms": avg_latency,
            "queue_size": self._queue.qsize(),
            "pending_reports": len(self._pending_cognitive_reports),
            "active_tokens": {
                cls.value: token.valid
                for cls, token in self._tokens.items()
            },
            "registered_handlers": self._registry.names,
            "uptime_s": round(time.time() - self._boot_time, 1),
        }

    def get_recent_receipts(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return recent motor receipts for audit."""
        recent = list(self._receipts)[-n:]
        return [
            {
                "receipt_id": r.receipt_id,
                "reflex_class": r.reflex_class.value,
                "handler": r.handler_name,
                "success": r.success,
                "latency_ms": r.latency_ms,
                "summary": r.result_summary,
                "timestamp": r.timestamp,
            }
            for r in recent
        ]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_motor_cortex_instance: Optional[MotorCortex] = None


def get_motor_cortex() -> MotorCortex:
    """Get the singleton MotorCortex instance."""
    global _motor_cortex_instance
    if _motor_cortex_instance is None:
        _motor_cortex_instance = MotorCortex()
    return _motor_cortex_instance
