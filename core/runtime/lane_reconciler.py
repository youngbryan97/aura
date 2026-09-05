"""core/runtime/lane_reconciler.py — K1 declarative lane reconciler + K4 crash-loop backoff.

The model-serving lane's recovery today is imperative and scattered:
watchdogs force-kill wedged workers, ensure-paths respawn them, callers
block on warmups. Each piece is individually correct and the composition
still produced the recorded doom loop: stall → force-kill → cold reload →
stall, a successful spawn every cycle so the existing *spawn-failure*
backoff never engages.

Two pieces, Kubernetes-shaped:

CrashLoopBreaker (K4) — CrashLoopBackOff proper. Counts *short-lived
worker runs* (spawn succeeded, worker died young), not failed spawns.
Deliberate administrative kills (yields, promotions, idle scavenge) never
count. Three young deaths inside the window trip the breaker: the lane
refuses respawns for an exponentially growing backoff and the escalation
ladder (cortex → cloud → reflex) answers instead of the host thrashing
through 20 GB cold reloads. After the backoff one probe spawn is allowed
(half-open); another young death re-trips at double the backoff; a
long-lived run closes the breaker.

LaneReconciler (K1) — a control loop that converges observed state onto
desired state: {the primary cortex is warm, joint declared footprints fit
the host budget}. It heals the primary when nothing else is (respecting
the breaker and never fighting a foreground turn), and evicts
lowest-QoS lanes when the budget is breached. Observation and action are
injected callables so the loop is hermetically testable; production
defaults bind to the live mlx_client registry.

Kill switches: AURA_LANE_RECONCILER=0 (loop never acts),
AURA_CRASHLOOP_BREAKER=0 (breaker records but never blocks).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.shutdown_coordinator import is_shutdown_requested

logger = logging.getLogger("Aura.LaneReconciler")

SERVICE_NAME = "lane_reconciler"

_ACTION_RING_SIZE = 64

_OWNER = "core.runtime.lane_reconciler"
_YOUNG_S_FLAG = declare(
    "AURA_CRASHLOOP_YOUNG_S",
    kind=FlagKind.FLOAT,
    default=180.0,
    description="Worker lifetime below which a death counts toward the crash loop",
    owner=_OWNER,
)
_THRESHOLD_FLAG = declare(
    "AURA_CRASHLOOP_THRESHOLD",
    kind=FlagKind.INT,
    default=3,
    description="Young deaths inside the window that trip the breaker",
    owner=_OWNER,
)
_WINDOW_FLAG = declare(
    "AURA_CRASHLOOP_WINDOW_S",
    kind=FlagKind.FLOAT,
    default=900.0,
    description="Sliding window for counting young deaths",
    owner=_OWNER,
)
_BASE_BACKOFF_FLAG = declare(
    "AURA_CRASHLOOP_BASE_BACKOFF_S",
    kind=FlagKind.FLOAT,
    default=30.0,
    description="First-trip respawn backoff; doubles per consecutive trip",
    owner=_OWNER,
)
_MAX_BACKOFF_FLAG = declare(
    "AURA_CRASHLOOP_MAX_BACKOFF_S",
    kind=FlagKind.FLOAT,
    default=600.0,
    description="Respawn backoff ceiling",
    owner=_OWNER,
)
_BREAKER_ENABLED_FLAG = declare(
    "AURA_CRASHLOOP_BREAKER",
    kind=FlagKind.BOOL,
    default=True,
    description="Kill switch: when false the breaker records but never blocks",
    owner=_OWNER,
)
_RECONCILER_ENABLED_FLAG = declare(
    "AURA_LANE_RECONCILER",
    kind=FlagKind.BOOL,
    default=True,
    description="Kill switch for the lane reconciler control loop",
    owner=_OWNER,
)
_RECONCILE_INTERVAL_FLAG = declare(
    "AURA_LANE_RECONCILE_INTERVAL_S",
    kind=FlagKind.FLOAT,
    default=20.0,
    description="Seconds between reconcile passes (floor 5)",
    owner=_OWNER,
)

# Administrative kill reasons: the runtime chose to stop this worker; the
# death says nothing about the lane's health.
DELIBERATE_DEATH_PREFIXES: tuple[str, ...] = (
    "yield_to_",
    "promoted_artifact_swap",
    "idle_vram_scavenge",
    "manual_reboot",
    "memory_pressure_guard",
    "reconcile_evict",
    "shutdown",
    "expert_adapter_",
    # A generation force-abort (inference-gate timeout / first-token
    # watchdog) is a POLICY recycle, not a spontaneous crash: the gate chose
    # to kill a slow generation. Counting it toward crash-loop backoff backed
    # off the FAST FALLBACK (reflex/brainstem) whenever it timed out queued
    # behind a busy foreground 32B under single-slot GPU serialization —
    # removing the fast answer path and cascading into deeper contention
    # (2026-07-15 soak). The breaker exists for workers that die UNEXPECTEDLY
    # (crash/OOM-kill/process_died_unexpectedly/init_timeout); a deliberate
    # recycle just respawns clean. (hard_generation_deadline stays tripping —
    # it is the escalated hard ceiling, distinct from a routine timeout.)
    "inference_gate_generation_timeout",
    "first_token",
)


def death_is_deliberate(reason: str) -> bool:
    lowered = str(reason or "").strip().lower()
    return any(lowered.startswith(prefix) for prefix in DELIBERATE_DEATH_PREFIXES)


def disruption_budget_blocks(candidate_path: str, lanes: list[Any]) -> str | None:
    """K5 disruption budget: never VOLUNTARILY kill the last warm lane.

    Applies to administrative disruptions only (budget evictions, yields for
    background warmups) — involuntary recovery kills are exempt, because
    force-killing a wedged worker IS the recovery (the 0-token-stall
    lesson). A host with one warm model must keep it: a cold gap with
    nothing warm is strictly worse than a briefly-over-budget host.
    """
    alive = [l for l in lanes if float(getattr(l, "footprint_gb", 0.0)) > 0.0]
    if len(alive) == 1 and str(getattr(alive[0], "model_path", "")) == str(candidate_path):
        return "disruption_budget:last_warm_lane"
    return None


@dataclass
class _LaneCrashState:
    young_deaths: list[float] = field(default_factory=list)
    trips: int = 0
    blocked_until: float = 0.0
    half_open: bool = False
    last_reason: str = ""


class CrashLoopBreaker:
    """Per-lane circuit breaker over short-lived worker runs. Thread-safe:
    deaths are reported from watchdog threads and async paths alike."""

    def __init__(self) -> None:
        self._lanes: dict[str, _LaneCrashState] = {}
        self._lock = threading.Lock()

    # ── policy knobs ───────────────────────────────────────────────

    @staticmethod
    def young_s() -> float:
        return float(_YOUNG_S_FLAG.value())

    @staticmethod
    def _threshold() -> int:
        return max(1, int(_THRESHOLD_FLAG.value()))

    @staticmethod
    def _window_s() -> float:
        return float(_WINDOW_FLAG.value())

    @staticmethod
    def _base_backoff_s() -> float:
        return float(_BASE_BACKOFF_FLAG.value())

    @staticmethod
    def _max_backoff_s() -> float:
        return float(_MAX_BACKOFF_FLAG.value())

    @staticmethod
    def _enforcing() -> bool:
        return bool(_BREAKER_ENABLED_FLAG.value())

    # ── event intake ───────────────────────────────────────────────

    def note_death(self, lane_key: str, *, lifetime_s: float, reason: str) -> None:
        """Record a worker death. Only young, non-deliberate deaths count."""
        if death_is_deliberate(reason):
            return
        now = time.time()
        with self._lock:
            state = self._lanes.setdefault(str(lane_key), _LaneCrashState())
            state.last_reason = str(reason or "")
            if lifetime_s >= self.young_s():
                # A worker that lived long enough was genuinely serving:
                # whatever killed it, this is not a crash loop. Close fully.
                self._close(state)
                return
            if state.half_open:
                # The probe spawn died young too: re-trip at double backoff.
                self._trip(state, now)
                return
            state.young_deaths = [
                t for t in state.young_deaths if (now - t) <= self._window_s()
            ]
            state.young_deaths.append(now)
            if len(state.young_deaths) >= self._threshold():
                self._trip(state, now)

    def note_healthy(self, lane_key: str) -> None:
        """A worker has been observed alive past the young threshold."""
        with self._lock:
            state = self._lanes.get(str(lane_key))
            if state is not None:
                self._close(state)

    # ── verdicts ───────────────────────────────────────────────────

    def blocked(self, lane_key: str) -> str | None:
        with self._lock:
            state = self._lanes.get(str(lane_key))
            if state is None:
                return None
            now = time.time()
            if state.blocked_until > now:
                if not self._enforcing():
                    return None
                remaining = state.blocked_until - now
                return (
                    f"crash_loop_backoff:trip={state.trips}"
                    f":retry_in={remaining:.0f}s:last={state.last_reason or 'unknown'}"
                )
            if state.blocked_until > 0.0 and not state.half_open:
                # Backoff expired: allow exactly one probe spawn.
                state.half_open = True
                state.blocked_until = 0.0
            return None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            return {
                "enforcing": self._enforcing(),
                "lanes": {
                    key: {
                        "trips": state.trips,
                        "half_open": state.half_open,
                        "blocked_for_s": max(0.0, round(state.blocked_until - now, 1)),
                        "young_deaths_in_window": len(
                            [
                                t
                                for t in state.young_deaths
                                if (now - t) <= self._window_s()
                            ]
                        ),
                        "last_reason": state.last_reason,
                    }
                    for key, state in self._lanes.items()
                },
            }

    def reset_for_test(self) -> None:
        with self._lock:
            self._lanes.clear()

    # ── internals (caller holds the lock) ──────────────────────────

    def _trip(self, state: _LaneCrashState, now: float) -> None:
        state.trips += 1
        backoff = min(
            self._max_backoff_s(),
            self._base_backoff_s() * (2 ** (state.trips - 1)),
        )
        state.blocked_until = now + backoff
        state.half_open = False
        state.young_deaths.clear()
        logger.warning(
            "⛔ Crash-loop breaker tripped (trip %d): lane backing off %.0fs (last death: %s)",
            state.trips,
            backoff,
            state.last_reason or "unknown",
        )

    @staticmethod
    def _close(state: _LaneCrashState) -> None:
        state.young_deaths.clear()
        state.trips = 0
        state.blocked_until = 0.0
        state.half_open = False


_BREAKER: CrashLoopBreaker | None = None
_BREAKER_LOCK = threading.Lock()


def get_crash_loop_breaker() -> CrashLoopBreaker:
    global _BREAKER
    if _BREAKER is None:
        with _BREAKER_LOCK:
            if _BREAKER is None:
                _BREAKER = CrashLoopBreaker()
    return _BREAKER


# ═══════════════════════════════════════════════════════════════════════
# K1 — the reconciler
# ═══════════════════════════════════════════════════════════════════════


def _default_observe_lanes() -> list[Any]:
    from core.brain.llm import mlx_client

    observed: Any = mlx_client._observed_active_lanes()
    return list(observed or [])


def _default_primary_alive() -> bool | None:
    """True/False for the primary lane's health; None when unobservable."""
    try:
        from core.brain.llm import mlx_client
        from core.brain.llm.model_registry import ACTIVE_MODEL, get_model_path

        primary_path = mlx_client._real_model_path(get_model_path(ACTIVE_MODEL))
        client = mlx_client._CLIENTS.get(primary_path)
        if client is None:
            return False
        return bool(client.is_alive() and getattr(client, "_init_done", False))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _default_primary_key() -> str:
    try:
        from core.brain.llm import mlx_client
        from core.brain.llm.model_registry import ACTIVE_MODEL, get_model_path

        return str(mlx_client._real_model_path(get_model_path(ACTIVE_MODEL)))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return "primary"


def _default_primary_age_s() -> float:
    try:
        from core.brain.llm import mlx_client

        client = mlx_client._CLIENTS.get(_default_primary_key())
        started = float(getattr(client, "_process_started_at", 0.0) or 0.0)
        return (time.time() - started) if started > 0.0 else 0.0
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return 0.0


async def _default_spawn_primary() -> bool:
    from core.brain.llm.mlx_client import get_mlx_client

    client = get_mlx_client(_default_primary_key())
    # Background semantics: every existing deferral (foreground owner,
    # memory pressure, background policy) applies. The reconciler heals in
    # the gaps; it never competes with a user turn.
    return bool(await client.warmup(foreground_request=False))


async def _default_evict_lane(model_path: str) -> bool:
    from core.brain.llm import mlx_client

    client = mlx_client._CLIENTS.get(model_path)
    if client is None or not client.is_alive():
        return False
    await client.reboot_worker(reason="reconcile_evict:budget", mark_failed=False)
    return True


#: How long a warm lane may sit unproven before the runtime proves it itself.
#: Long enough that an ordinary turn arrives first and does the proving for
#: free; short enough that an idle machine is not left unable to act.
UNPROVEN_TOO_LONG_S = 120.0

async def _default_prove_lane() -> str:
    """Make the runtime prove its own conversation lane, and say what happened.

    A lane that is loaded is not a lane that is serving, and the runtime knew
    the difference: optional background work is gated on the lane having
    produced at least one visible reply, and the executive raises the threat
    level while it has not.

    Nothing produced that reply. The gate blocked on the proof and never made
    one, so the proof could only ever arrive from outside — a person typing
    something. LIVE 2026-08-29: a transient memory blip deferred one recovery
    warmup, the lane stayed unproven, the threat level went critical, and every
    desktop action was refused for four hours. The lane was fine the whole
    time. One chat message healed it instantly, which is the evidence that
    nothing inside was ever going to.

    Runs only when no turn is in flight, so it never competes with a person,
    and it is bounded: a proof that hangs is worth less than no proof.
    """
    try:
        from core.container import ServiceContainer

        gate = ServiceContainer.peek("inference_gate", default=None)
        if gate is None or not hasattr(gate, "get_conversation_status"):
            return ""
        lane = dict(gate.get_conversation_status() or {})
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return ""
    if not lane or bool(lane.get("conversation_ready", False)):
        return ""
    if bool(lane.get("foreground_owned")) or int(lane.get("active_generations", 0) or 0) > 0:
        return ""
    if bool(lane.get("warmup_in_flight", False)):
        return ""
    since = float(lane.get("last_visible_readiness_at", 0.0) or 0.0)
    unproven_for = (time.time() - since) if since > 0.0 else float("inf")
    if unproven_for < UNPROVEN_TOO_LONG_S:
        return ""
    try:
        from core.brain.llm.mlx_client import get_mlx_client

        client = get_mlx_client(_default_primary_key())
        # One place means "the lane has been seen to answer", and it does the
        # recording as well as the asking. Running a health probe here and
        # reading the text would succeed, report success, and leave readiness
        # exactly as unproven as it found it — health_probe suppresses the
        # user-facing mark by design, so the proof would prove nothing.
        return await client.prove_visible_readiness(budget_s=UNPROVEN_TOO_LONG_S)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError):
        return "proof_failed"


def _default_foreground_active() -> bool:
    try:
        from core.brain.llm import mlx_client

        return bool(mlx_client._foreground_owner_active())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False


class LaneReconciler:
    """Converges the model-serving lane onto its declared desired state."""

    def __init__(
        self,
        *,
        observe_lanes: Callable[[], list[Any]] | None = None,
        primary_alive: Callable[[], bool | None] | None = None,
        primary_key: Callable[[], str] | None = None,
        primary_age_s: Callable[[], float] | None = None,
        spawn_primary: Callable[[], Awaitable[bool]] | None = None,
        evict_lane: Callable[[str], Awaitable[bool]] | None = None,
        foreground_active: Callable[[], bool] | None = None,
        prove_lane: Callable[[], Awaitable[str]] | None = None,
        breaker: CrashLoopBreaker | None = None,
    ) -> None:
        self._observe_lanes = observe_lanes or _default_observe_lanes
        self._primary_alive = primary_alive or _default_primary_alive
        self._primary_key = primary_key or _default_primary_key
        self._primary_age_s = primary_age_s or _default_primary_age_s
        self._spawn_primary = spawn_primary or _default_spawn_primary
        self._evict_lane = evict_lane or _default_evict_lane
        self._foreground_active = foreground_active or _default_foreground_active
        self._prove_lane = prove_lane or _default_prove_lane
        self._breaker = breaker or get_crash_loop_breaker()
        self._actions: deque[dict[str, Any]] = deque(maxlen=_ACTION_RING_SIZE)
        self._loop_task: asyncio.Task[Any] | None = None
        self._reconcile_inflight = False
        self._running = False

    # ── the control loop ───────────────────────────────────────────

    @staticmethod
    def enabled() -> bool:
        return bool(_RECONCILER_ENABLED_FLAG.value())

    @staticmethod
    def interval_s() -> float:
        return max(5.0, float(_RECONCILE_INTERVAL_FLAG.value()))

    async def start(self) -> None:
        if self._running:
            return
        if is_shutdown_requested():
            logger.info("LaneReconciler start skipped: runtime shutdown requested.")
            return
        self._running = True
        from core.utils.task_tracker import get_task_tracker

        self._loop_task = get_task_tracker().create_task(
            self._run_loop(), name="LaneReconciler"
        )

    async def stop(self) -> None:
        self._running = False
        task, self._loop_task = self._loop_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # no-op: intentional

    async def _run_loop(self) -> None:
        while self._running:
            if is_shutdown_requested():
                self._running = False
                return
            try:
                if self.enabled():
                    await self.reconcile_once()
            except asyncio.CancelledError:
                raise
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                record_degradation(
                    "lane_reconciler",
                    exc,
                    severity="warning",
                    action="skipped one reconcile pass; loop continues",
                )
            await asyncio.sleep(self.interval_s())

    # ── one convergence step ───────────────────────────────────────

    async def reconcile_once(self) -> list[dict[str, Any]]:
        """Observe → diff against desired state → act. Returns the actions.

        Single-flight: overlapping calls (loop + manual) collapse to one.
        """
        if is_shutdown_requested():
            return [self._note("skipped", detail="runtime_shutdown")]
        if self._reconcile_inflight:
            return [self._note("skipped", detail="reconcile_already_inflight")]
        self._reconcile_inflight = True
        try:
            return await self._reconcile_inner()
        finally:
            self._reconcile_inflight = False

    async def _reconcile_inner(self) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        primary_key = self._primary_key()
        primary_alive = self._primary_alive()

        # Rule 0 — observation feedback: a primary that has been up past the
        # young threshold closes its breaker (the crash loop is over).
        if primary_alive and self._primary_age_s() >= CrashLoopBreaker.young_s():
            self._breaker.note_healthy(primary_key)

        # Rule 1 — converge the primary cortex toward warm.
        #
        # A DEAD primary converges regardless of foreground ownership. The
        # old foreground deferral here was a priority inversion, caught live
        # (2026-07-10, 75 min of 216s turns): each waiting turn held the
        # foreground lane, which deferred the very convergence it was
        # waiting on — cortex warming forever, every turn answered by
        # fallback at the wall. A dead lane cannot be disrupted; foreground
        # ownership protects a SERVING primary (Rule 0 side), never blocks
        # reviving one.
        if primary_alive is False:
            blocked = self._breaker.blocked(primary_key)
            if blocked:
                actions.append(self._note("held", lane=primary_key, detail=blocked))
            else:
                ok = bool(await self._spawn_primary())
                actions.append(
                    self._note(
                        "warm_requested" if ok else "warm_deferred",
                        lane=primary_key,
                        detail=(
                            "reconciler_prewarm_foreground_waiting"
                            if self._foreground_active()
                            else "reconciler_prewarm"
                        ),
                    )
                )

        # Rule 1b — proof: a warm lane that has never answered is not serving.
        #
        # Everything downstream distinguishes a loaded lane from a serving one,
        # and nothing produced the evidence. Convergence onto the desired state
        # includes proving the state is real.
        if not self._foreground_active():
            proved = await self._prove_lane()
            if proved:
                actions.append(self._note("proof", lane=primary_key, detail=proved))

        # Rule 2 — budget: joint declared footprints must fit the envelope.
        try:
            from core.brain.lane_admission import (
                QoSClass,
                _eviction_shield_s,
                lane_budget_gb,
            )

            lanes = list(self._observe_lanes())
            committed = sum(float(getattr(l, "footprint_gb", 0.0)) for l in lanes)
            budget = lane_budget_gb()
            if committed > budget:
                # Largest, lowest-QoS, non-recently-user-facing lanes first.
                shield_s = _eviction_shield_s()
                rank = {QoSClass.BEST_EFFORT: 0, QoSClass.BURSTABLE: 1, QoSClass.GUARANTEED: 2}
                candidates = [
                    l
                    for l in lanes
                    if getattr(l, "qos", None) is not QoSClass.GUARANTEED
                    and (
                        getattr(l, "last_user_facing_age_s", None) is None
                        or l.last_user_facing_age_s >= shield_s
                    )
                ]
                candidates.sort(
                    key=lambda l: (rank.get(getattr(l, "qos", QoSClass.BURSTABLE), 1), -l.footprint_gb)
                )
                for lane in candidates:
                    if committed <= budget:
                        break
                    blocked = disruption_budget_blocks(lane.model_path, lanes)
                    if blocked:
                        actions.append(
                            self._note("held", lane=lane.model_path or lane.lane, detail=blocked)
                        )
                        continue
                    evicted = bool(await self._evict_lane(lane.model_path or lane.lane))
                    if evicted:
                        committed -= lane.footprint_gb
                        actions.append(
                            self._note(
                                "evicted",
                                lane=lane.model_path or lane.lane,
                                detail=f"over_budget:committed>{budget:.1f}GB",
                            )
                        )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "lane_reconciler",
                exc,
                severity="warning",
                action="skipped budget rule for this pass",
            )

        self._publish_conditions(primary_alive, actions)
        return actions

    def _publish_conditions(
        self, primary_alive: bool | None, actions: list[dict[str, Any]]
    ) -> None:
        """K6: expose the lane's state as typed conditions with reasons.

        Best-effort — condition publication must never break convergence.
        """
        try:
            from core.runtime.conditions import ConditionType, get_component_conditions

            conditions = get_component_conditions("cortex_lane")
            taken = {a["action"] for a in actions}

            if primary_alive is not None:
                if primary_alive:
                    conditions.set(
                        ConditionType.READY, True, reason="PrimaryWarm",
                        message="primary cortex worker alive and initialized",
                    )
                elif "held" in taken:
                    detail = next(a["detail"] for a in actions if a["action"] == "held")
                    conditions.set(
                        ConditionType.READY, False, reason="CrashLoopBackOff",
                        message=detail,
                    )
                else:
                    conditions.set(
                        ConditionType.READY, False, reason="PrimaryDown",
                        message="primary cortex not warm; reconciler converging",
                    )

            conditions.set(
                ConditionType.PROGRESSING,
                "warm_requested" in taken,
                reason="WarmupRequested" if "warm_requested" in taken else "Idle",
                message="background warmup scheduled" if "warm_requested" in taken else "",
            )

            degraded = bool({"held", "evicted"} & taken)
            reason = "CrashLoopBackOff" if "held" in taken else (
                "BudgetEviction" if "evicted" in taken else "None"
            )
            conditions.set(
                ConditionType.DEGRADED, degraded, reason=reason,
                message="; ".join(
                    f"{a['action']}:{a.get('lane', '')}" for a in actions
                ) if degraded else "",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, StopIteration) as exc:
            logger.debug("Condition publication skipped: %s", exc)

    # ── observability ──────────────────────────────────────────────

    def _note(self, action: str, **fields: Any) -> dict[str, Any]:
        entry = {"action": action, "at": time.time(), **fields}
        self._actions.append(entry)
        if action in {"evicted", "held"}:
            logger.warning("Lane reconciler: %s", entry)
        else:
            logger.info("Lane reconciler: %s", entry)
        try:
            from core.runtime.flight_recorder import record_event

            record_event(
                kind=f"reconcile_{action}",
                source="lane_reconciler",
                summary=str(fields.get("detail", "") or action),
                lane=str(fields.get("lane", "")),
            )
        except (ImportError, AttributeError, RuntimeError):
            pass  # no-op: black-box feed is best-effort by design
        return entry

    def snapshot(self) -> dict[str, Any]:
        try:
            from core.runtime.conditions import get_component_conditions

            conditions = get_component_conditions("cortex_lane").snapshot()
        except (ImportError, AttributeError, RuntimeError):
            conditions = {}
        return {
            "alive": self.is_alive(),
            "ready": self.is_ready(),
            "running": self._running,
            "enabled": self.enabled(),
            "interval_s": self.interval_s(),
            "recent_actions": list(self._actions)[-10:],
            "breaker": self._breaker.snapshot(),
            "conditions": conditions,
        }

    def is_alive(self) -> bool:
        return bool(
            self._running
            and self._loop_task is not None
            and not self._loop_task.done()
        )

    def is_ready(self) -> bool:
        return self.is_alive()

    def get_status(self) -> dict[str, Any]:
        return self.snapshot()


_RECONCILER: LaneReconciler | None = None
_RECONCILER_LOCK = threading.Lock()


def get_lane_reconciler() -> LaneReconciler:
    global _RECONCILER
    if _RECONCILER is None:
        with _RECONCILER_LOCK:
            if _RECONCILER is None:
                _RECONCILER = LaneReconciler()
    return _RECONCILER
