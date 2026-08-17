from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from core.health.degraded_events import get_unified_failure_state
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.resource_observation import get_resource_observer

logger = logging.getLogger(__name__)
_PROCESS_STARTED_AT = time.time()
_FOREGROUND_ONLY_FLAG = declare(
    "AURA_FOREGROUND_ONLY",
    kind=FlagKind.BOOL,
    default=False,
    description="Run only user-facing foreground work",
    owner="core.runtime.background_policy",
)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.debug("Invalid %s=%r; using %.1f", name, raw, default)
        return float(default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def foreground_only_runtime() -> bool:
    """Return True when Aura should boot only foreground/user-facing loops."""

    return bool(_FOREGROUND_ONLY_FLAG.value())


def background_cognition_disabled_reason(*, allow_desktop_safe_boot: bool = False) -> str:
    """Return why optional background cognition must stay offline — almost never.

    Background cognition is part of the full, live Aura, not an optional extra: a normal
    launch AND a protected/safe boot both run the complete background runtime under resource
    admission. A "safe boot" is a resource-protection posture (memory guards, process
    ownership, tighter admission), NOT a lesser mind — so it no longer suppresses cognition.

    The ONLY thing that takes background cognition offline is an explicit operator override,
    ``AURA_ENABLE_BACKGROUND_COGNITION=0``, kept as a deliberate last-resort kill-switch for
    genuine crash-loop recovery. Everything else keeps the whole mind live.
    """
    del allow_desktop_safe_boot  # safe/protected boot no longer reduces cognition

    configured = str(os.getenv("AURA_ENABLE_BACKGROUND_COGNITION", "") or "").strip().lower()
    if configured in {"0", "false", "no", "off"}:
        return "background_cognition_disabled"
    return ""


def background_loop_start_reason(
    origin: Any = None,
    *,
    allow_desktop_safe_boot: bool = False,
) -> str:
    """Explain why a persistent background loop must not start.

    ``background_activity_reason`` gates individual work items. This helper
    gates loop creation itself for modes where idle autonomy would contaminate
    proof artifacts or compete with the live user lane.
    """

    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        if is_shutdown_requested():
            return "shutdown_requested"
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background loop start because shutdown probe failed",
        )
        logger.warning("Background loop shutdown probe failed: %s", _exc)
        return "shutdown_probe_unavailable"

    try:
        from core.runtime.proof_policy import proof_run_active

        if proof_run_active(origin=origin):
            return "proof_run_active"
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background loop start because proof-run signal failed",
        )
        logger.warning("Background loop proof-run probe failed: %s", _exc)
        return "proof_signal_unavailable"

    if foreground_only_runtime():
        return "foreground_only_runtime"

    disabled_reason = background_cognition_disabled_reason(
        allow_desktop_safe_boot=allow_desktop_safe_boot,
    )
    if disabled_reason:
        return disabled_reason

    return ""


def background_loop_start_allowed(
    origin: Any = None,
    *,
    allow_desktop_safe_boot: bool = False,
) -> bool:
    return not background_loop_start_reason(
        origin,
        allow_desktop_safe_boot=allow_desktop_safe_boot,
    )


_USER_FACING_ORIGIN_TOKENS = frozenset({
    "user",
    "voice",
    "admin",
    "api",
    "gui",
    "ws",
    "websocket",
    "desktop",
    "ui",
    "external",
    "direct",
    "embodied",
    "reflex",
    "motor",
    "test",
})

_BACKGROUND_ORIGIN_HINTS = frozenset({
    "affect",
    "autonomous",
    "background",
    "constitutive",
    "consolidation",
    "context",
    "dream",
    "growth",
    "impulse",
    "internal",
    "memory",
    "metabolic",
    "mist",
    "monitor",
    "motivation",
    "parallel",
    "perception",
    "phenomenological",
    "proactive",
    "pruner",
    "scanner",
    "sensory",
    "spontaneous",
    "stream",
    "structured",
    "subconscious",
    # "system" intentionally omitted: it is too broad and would misclassify
    # user-adjacent routing paths that still use the historical default.
    "terminal",
    "volition",
    "witness",
})


@dataclass(frozen=True)
class BackgroundPolicyProfile:
    min_idle_seconds: float = 10.0
    max_memory_percent: float = 90.0
    max_failure_pressure: float = 0.60
    require_conversation_ready: bool = False


@dataclass(frozen=True)
class ConstitutiveComputeBudget:
    """Runtime budget for always-on embodied/cognitive loops.

    These loops should stay alive, but they must yield hard priority to the live
    foreground conversation and to system memory pressure. The budget is a
    throttle, not a shutdown signal.
    """

    component: str
    base_hz: float
    effective_hz: float
    interval_s: float
    reason: str
    foreground_active: bool
    memory_percent: float | None = None


@dataclass(frozen=True)
class _MemoryPressureSnapshot:
    pressure_pct: float
    reason: str
    refuse_heavy_local_generation: bool = False
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""


def _read_compute_pressure_reason() -> str:
    """Return a host heat/load reason for deferring optional background work.

    Aura's background cognition should stay alive in normal launches, but it
    must not cook the host machine. This guard is intentionally narrow: it
    defers optional background actions and throttles constitutive loops under
    clear CPU/load/thermal pressure while leaving foreground replies and core
    runtime health checks available.
    """

    if not _env_flag("AURA_BACKGROUND_HEAT_GUARD", True):
        return ""
    observer = get_resource_observer()
    provenance = observer.provenance

    try:
        compute = observer.compute()
        if not compute.available:
            return f"compute_observation_unavailable_source_{provenance.source.value}"
        cpu_pct = float(compute.cpu_percent)
        max_cpu_pct = _env_float("AURA_BACKGROUND_MAX_CPU_PERCENT", 88.0)
        if cpu_pct >= max_cpu_pct:
            return f"cpu_pressure_{cpu_pct:.1f}_source_{provenance.source.value}"
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return f"compute_observation_unavailable_source_{provenance.source.value}"

    try:
        load_per_core = float(compute.load_1m) / max(1, int(compute.cpu_count))
        max_load_per_core = _env_float("AURA_BACKGROUND_MAX_LOAD_PER_CORE", 0.90)
        if load_per_core >= max_load_per_core:
            return f"load_pressure_{load_per_core:.2f}_source_{provenance.source.value}"
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return f"compute_observation_unavailable_source_{provenance.source.value}"

    try:
        reading = observer.thermal()
        if not reading.available:
            return f"thermal_observation_unavailable_source_{provenance.source.value}"
        max_level = int(_env_float("AURA_BACKGROUND_MAX_THERMAL_LEVEL", 2.0))
        if reading.level >= max_level:
            return (
                f"thermal_pressure_level_{reading.level}_{reading.provider}"
                f"_source_{provenance.source.value}"
            )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return f"thermal_observation_unavailable_source_{provenance.source.value}"

    return ""


def _read_memory_pressure_snapshot() -> _MemoryPressureSnapshot:
    """Read memory through the canonical attributable runtime guard."""

    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        runtime = get_memory_pressure_snapshot()
        runtime_percent = float(getattr(runtime, "pressure_pct", 0.0) or 0.0)
        runtime_reason = str(getattr(runtime, "reason", "") or "")
        runtime_refuse = bool(getattr(runtime, "refuse_heavy_local_generation", False))
        reason = runtime_reason or f"memory_pressure_{runtime_percent:.1f}"
        return _MemoryPressureSnapshot(
            pressure_pct=runtime_percent,
            reason=reason,
            refuse_heavy_local_generation=runtime_refuse,
            observation_source=str(
                getattr(runtime, "observation_source", "unavailable")
            ),
            observation_scenario_id=str(
                getattr(runtime, "observation_scenario_id", "")
            ),
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        provenance = get_resource_observer().provenance
        return _MemoryPressureSnapshot(
            pressure_pct=100.0,
            reason="memory_observation_unavailable",
            refuse_heavy_local_generation=True,
            observation_source=provenance.source.value,
            observation_scenario_id=provenance.scenario_id,
        )


# ── Named background-yield profiles: THE shared vocabulary ────────────────
#
# One organism, one discipline for when background work yields to the user.
# Rather than 25+ call sites each spelling out magic-number thresholds, every
# background loop should name the class of work it is and let the profile
# carry the thresholds. Tiers (idle gate increases with cost; all yield to
# the live conversation and to memory pressure):
#
#   IDLE_COGNITION — light always-on cognitive loops that must yield to the
#                    conversation but may run during warmup (mind-tick
#                    reflection, phenomenology, metacognition, momentum). This
#                    names the emergent inline convention (180s / 78%).
#   THOUGHT        — deliberate background thinking that should wait for a
#                    ready conversation lane.
#   RESEARCH       — multi-step background research: long idle gate.
#   MAINTENANCE    — expensive upkeep (defrag, evolution, training): longest.
#
# The magic-number inline callers should migrate onto IDLE_COGNITION; that
# migration is behavior-preserving and tracked separately.

IDLE_COGNITION_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=180.0,
    max_memory_percent=78.0,
    # 0.60 matches the function default the bare inline callers already get,
    # so `profile=IDLE_COGNITION` is a true drop-in for `min_idle_seconds=180,
    # max_memory_percent=78`.
    max_failure_pressure=0.60,
    require_conversation_ready=False,
)

THOUGHT_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=30.0,
    max_memory_percent=85.0,
    max_failure_pressure=0.50,
    require_conversation_ready=True,
)

RESEARCH_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=900.0,
    max_memory_percent=85.0,
    max_failure_pressure=0.50,
    require_conversation_ready=True,
)

MAINTENANCE_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=1800.0,
    max_memory_percent=92.0,
    max_failure_pressure=0.75,
    require_conversation_ready=True,
)

# Skill-preflight profiles (capability engine, Jul 7-8 live hardening).
# Deliberately stricter on memory than MAINTENANCE: these gate SKILL
# EXECUTION beside the resident 32B, where 92% memory would admit work
# straight into the contention band that starves first tokens.
HEAVY_SKILL_PREFLIGHT_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=600.0,
    max_memory_percent=72.0,
    max_failure_pressure=0.20,
    require_conversation_ready=False,
)

RECON_SCAN_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=1800.0,
    max_memory_percent=72.0,
    max_failure_pressure=0.20,
    require_conversation_ready=False,
)


def _component_env_name(component: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(component or "loop"))
    return f"AURA_CONSTITUTIVE_{normalized.upper()}_HZ"


def _bounded_hz(value: float, *, lower: float = 0.1, upper: float = 1000.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = lower
    if not value or value != value or value < lower:
        return lower
    return min(float(upper), value)


def constitutive_compute_budget(
    component: str,
    base_hz: float,
    *,
    min_hz: float = 0.5,
    foreground_hz: float = 2.0,
    memory_high_hz: float = 2.0,
    memory_critical_hz: float = 0.5,
    memory_high_percent: float = 85.0,
    memory_critical_percent: float = 92.0,
    compute_pressure_hz: float = 1.0,
    failure_pressure_hz: float = 1.0,
    max_failure_pressure: float = 0.75,
    serves_foreground: bool = False,
) -> ConstitutiveComputeBudget:
    """Return a safe update budget for continuous constitutive loops.

    Unlike ``background_activity_reason``, this does not block the loop. It
    caps its frequency during foreground inference, memory pressure, proof runs,
    or failure pressure so substrate/field/HOT machinery cannot compete with a
    user-facing model turn or drive host RAM spikes.
    """

    base = _bounded_hz(base_hz, lower=0.1, upper=1000.0)
    floor = _bounded_hz(min_hz, lower=0.1, upper=base)
    effective = base
    reason = "nominal"
    foreground_active = False
    memory_percent: float | None = None

    component_override = os.getenv(_component_env_name(component))
    if component_override is not None:
        effective = min(effective, _bounded_hz(component_override, lower=floor, upper=base))
        reason = "component_override"

    global_override = os.getenv("AURA_CONSTITUTIVE_MAX_HZ")
    if global_override is not None:
        effective = min(effective, _bounded_hz(global_override, lower=floor, upper=base))
        reason = "global_override" if reason == "nominal" else f"{reason}+global_override"

    try:
        from core.runtime.proof_policy import proof_run_active

        if proof_run_active():
            effective = min(effective, floor)
            reason = "proof_run_active"
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="throttled constitutive loop because proof-run signal failed",
        )
        effective = min(effective, floor)
        reason = "proof_signal_unavailable"

    if foreground_only_runtime():
        effective = min(effective, floor)
        reason = "foreground_only_runtime"

    disabled_reason = background_cognition_disabled_reason()
    if disabled_reason:
        effective = min(effective, floor)
        reason = disabled_reason

    foreground_reason = _foreground_activity_reason()
    if foreground_reason:
        foreground_active = True
        if serves_foreground:
            # This loop is not COMPETING with the foreground turn, it is what
            # the turn is made of.
            #
            # The clamp below exists so constitutive machinery cannot steal
            # cycles from a user-facing generation. Applied to perception that
            # the generation DEPENDS ON, it does the opposite of its purpose:
            # continuous_vision passes foreground_hz=0.1, so the instant she
            # began working on a request her sight dropped to one frame every
            # ten seconds — she went nearly blind exactly while acting. Any
            # task needing look-act-look — dragging something, watching a
            # progress bar, reacting to a game, verifying a click landed — is
            # unreachable at that cadence, and the cause is invisible because
            # nothing failed and no error was raised.
            #
            # Only THIS clamp is skipped. Memory pressure, compute pressure,
            # proof runs and failure pressure still apply below, because those
            # protect the host rather than arbitrating between two kinds of
            # work.
            reason = f"{foreground_reason}+serving_foreground"
        else:
            effective = min(effective, _bounded_hz(foreground_hz, lower=floor, upper=base))
            reason = foreground_reason

    compute_reason = _read_compute_pressure_reason()
    if compute_reason:
        effective = min(effective, _bounded_hz(compute_pressure_hz, lower=floor, upper=base))
        reason = compute_reason

    try:
        memory = _read_memory_pressure_snapshot()
        memory_percent = float(memory.pressure_pct)
        memory_reason = str(memory.reason or f"memory_pressure_{memory_percent:.1f}")
        if memory.refuse_heavy_local_generation:
            effective = min(
                effective,
                _bounded_hz(memory_critical_hz, lower=floor, upper=base),
            )
            reason = memory_reason
        elif memory_percent >= float(memory_critical_percent):
            effective = min(
                effective,
                _bounded_hz(memory_critical_hz, lower=floor, upper=base),
            )
            reason = f"memory_critical_{memory_percent:.1f}"
        elif memory_percent >= float(memory_high_percent):
            effective = min(effective, _bounded_hz(memory_high_hz, lower=floor, upper=base))
            reason = f"memory_pressure_{memory_percent:.1f}"
    except (ImportError, OSError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="throttled constitutive loop because memory-pressure probe failed",
        )
        effective = min(effective, floor)
        reason = "memory_probe_unavailable"

    try:
        failure = get_unified_failure_state()
        pressure = float(failure.get("pressure", 0.0) or 0.0)
        if pressure >= float(max_failure_pressure):
            effective = min(effective, _bounded_hz(failure_pressure_hz, lower=floor, upper=base))
            reason = f"failure_pressure_{pressure:.2f}"
    except (OSError, ConnectionError, TimeoutError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="throttled constitutive loop because failure-state probe failed",
        )
        effective = min(effective, floor)
        reason = "failure_state_unavailable"

    effective = max(floor, min(base, float(effective)))
    return ConstitutiveComputeBudget(
        component=str(component or "loop"),
        base_hz=base,
        effective_hz=effective,
        interval_s=1.0 / max(floor, effective),
        reason=reason,
        foreground_active=foreground_active,
        memory_percent=memory_percent,
    )


def normalize_origin(origin: Any) -> str:
    normalized = str(origin or "").strip().lower().replace("-", "_")
    while normalized.startswith("routing_"):
        normalized = normalized[len("routing_"):]
    return normalized


def origin_tokens(origin: Any) -> set[str]:
    normalized = normalize_origin(origin)
    return {token for token in normalized.split("_") if token}


def is_user_facing_origin(origin: Any) -> bool:
    normalized = normalize_origin(origin)
    if not normalized:
        return False
    if normalized in _USER_FACING_ORIGIN_TOKENS:
        return True
    return bool(origin_tokens(normalized) & _USER_FACING_ORIGIN_TOKENS)


def is_background_origin(origin: Any, *, explicit_background: bool = False) -> bool:
    if explicit_background:
        return True
    tokens = origin_tokens(origin)
    if not tokens:
        return False
    if tokens & _USER_FACING_ORIGIN_TOKENS:
        return False
    return bool(tokens & _BACKGROUND_ORIGIN_HINTS)


def _last_user_interaction_time(orchestrator: Any = None) -> float:
    orch = orchestrator
    if orch is None:
        return 0.0

    value = float(getattr(orch, "_last_user_interaction_time", 0.0) or 0.0)
    if value > 0.0:
        return value

    status = getattr(orch, "status", None)
    if status is not None:
        value = float(getattr(status, "last_user_interaction_time", 0.0) or 0.0)
        if value > 0.0:
            return value

    return 0.0


def _runtime_uptime_seconds(orchestrator: Any = None) -> float:
    if orchestrator is None:
        return 0.0

    candidates = [
        _PROCESS_STARTED_AT,
        getattr(orchestrator, "start_time", None),
        getattr(getattr(orchestrator, "status", None), "start_time", None),
    ]
    valid_starts: list[float] = []
    for candidate in candidates:
        try:
            start = float(candidate or 0.0)
        except (TypeError, ValueError):
            continue
        if start > 0.0:
            valid_starts.append(start)
    if not valid_starts:
        return 0.0
    # Snapshots may restore historical organism uptime. Optional background
    # work must still honor the current process incarnation's boot grace.
    return max(0.0, time.time() - max(valid_starts))


def _foreground_activity_reason() -> str:
    guard_reason = ""
    try:
        from core.runtime.foreground_guard import foreground_activity_reason

        guard_reason = foreground_activity_reason()
        if guard_reason == "foreground_chat_active":
            return guard_reason
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background work because foreground guard probe failed",
        )
        logger.warning("Background policy foreground guard probe failed: %s", _exc)
        return "foreground_guard_unavailable"

    try:
        from core.container import ServiceContainer

        gate = ServiceContainer.peek("inference_gate", default=None)
        if gate and hasattr(gate, "get_conversation_status"):
            lane = dict(gate.get_conversation_status() or {})
            if bool(lane.get("foreground_owned")) or int(lane.get("active_generations", 0) or 0) > 0:
                return "foreground_generation_active"
            if bool(lane.get("kernel_lock_held")):
                return "foreground_kernel_lock"
            request_age = float(lane.get("request_age_s", 0.0) or 0.0)
            if request_age > 0.0 and str(lane.get("foreground_owner") or "").strip():
                return "foreground_request_active"
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background work because inference foreground probe failed",
        )
        logger.warning("Background policy inference foreground probe failed: %s", _exc)
        return "foreground_generation_status_unavailable"
    if guard_reason:
        return guard_reason
    return ""


def _first_visible_conversation_probe_reason() -> str:
    """Return a startup-only gate for optional background work.

    A loaded Cortex worker is not enough for daily-use readiness. The live
    desktop lane must first prove it can produce at least one visible reply.
    Optional autonomy can run after that proof, but letting it fire before the
    first verified reply competes with the warmup/probe path and can wedge the
    UI in ``visible_conversation_probe_missing`` while tools are already acting.
    """

    try:
        from core.container import ServiceContainer

        gate = ServiceContainer.peek("inference_gate", default=None)
        if not gate or not hasattr(gate, "get_conversation_status"):
            return ""
        lane = dict(gate.get_conversation_status() or {})
        if not lane or bool(lane.get("conversation_ready", False)):
            return ""
        if bool(lane.get("foreground_owned")) or int(lane.get("active_generations", 0) or 0) > 0:
            return ""
        if bool(lane.get("warmup_in_flight", False)):
            return "conversation_warmup_in_flight"
        last_visible = float(lane.get("last_visible_readiness_at", 0.0) or 0.0)
        blockers = {
            str(item or "").strip()
            for item in (lane.get("readiness_blockers") or [])
            if str(item or "").strip()
        }
        reason = str(lane.get("last_failure_reason") or "").strip()
        if last_visible <= 0.0 or "visible_conversation_probe_missing" in blockers or reason == "visible_conversation_probe_missing":
            return "first_visible_conversation_probe_pending"
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked optional background work because conversation probe state was unavailable",
        )
        logger.warning("Background policy first-visible conversation probe failed: %s", _exc)
        return "conversation_probe_state_unavailable"
    return ""


def background_activity_reason(
    orchestrator: Any = None,
    *,
    profile: BackgroundPolicyProfile | None = None,
    min_idle_seconds: float | None = None,
    max_memory_percent: float | None = None,
    max_failure_pressure: float | None = None,
    require_conversation_ready: bool | None = None,
    allow_no_user_anchor: bool = False,
    allow_desktop_safe_boot: bool = False,
) -> str:
    if profile is not None:
        if min_idle_seconds is None:
            min_idle_seconds = profile.min_idle_seconds
        if max_memory_percent is None:
            max_memory_percent = profile.max_memory_percent
        if max_failure_pressure is None:
            max_failure_pressure = profile.max_failure_pressure
        if require_conversation_ready is None:
            require_conversation_ready = profile.require_conversation_ready

    min_idle_seconds = float(min_idle_seconds if min_idle_seconds is not None else 10.0)
    max_memory_percent = float(max_memory_percent if max_memory_percent is not None else 90.0)
    max_failure_pressure = float(max_failure_pressure if max_failure_pressure is not None else 0.60)
    require_conversation_ready = bool(
        False if require_conversation_ready is None else require_conversation_ready
    )

    now = time.time()

    try:
        from core.runtime.proof_policy import proof_run_active

        if proof_run_active():
            return "proof_run_active"
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="continued background policy evaluation without proof-run signal",
        )
        logger.debug("Proof-run background policy check unavailable: %s", _exc)

    if foreground_only_runtime():
        return "foreground_only_runtime"

    disabled_reason = background_cognition_disabled_reason(
        allow_desktop_safe_boot=allow_desktop_safe_boot,
    )
    if disabled_reason:
        return disabled_reason

    first_probe_reason = _first_visible_conversation_probe_reason()
    if first_probe_reason:
        return first_probe_reason

    orch = orchestrator
    if orch is not None:
        if bool(getattr(orch, "is_busy", False)):
            return "orchestrator_busy"

        if float(getattr(orch, "_suppress_unsolicited_proactivity_until", 0.0) or 0.0) > now:
            return "suppressed"

        last_user = _last_user_interaction_time(orch)
        if last_user <= 0.0 and not allow_no_user_anchor:
            return "no_user_anchor"

    foreground_reason = _foreground_activity_reason()
    if foreground_reason:
        return foreground_reason

    if orch is not None:
        quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
        if quiet_until > now:
            return "foreground_quiet_window"

        if (now - last_user) < min_idle_seconds:
            return f"recent_user_{int(now - last_user)}"

    compute_reason = _read_compute_pressure_reason()
    if compute_reason:
        return compute_reason

    try:
        memory = _read_memory_pressure_snapshot()
        memory_pct = float(memory.pressure_pct)
        if bool(memory.refuse_heavy_local_generation):
            return str(memory.reason or f"memory_pressure_guard_{memory_pct:.1f}")
        if memory_pct >= max_memory_percent:
            return f"memory_pressure_{memory_pct:.1f}"
    except (ImportError, OSError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background work because memory-pressure probe failed",
        )
        logger.warning("Background policy memory-pressure probe failed: %s", _exc)
        return "memory_probe_unavailable"

    try:
        failure = get_unified_failure_state()
        pressure = float(failure.get("pressure", 0.0) or 0.0)
        if pressure >= max_failure_pressure:
            return f"failure_lockdown_{pressure:.2f}"
    except (OSError, ConnectionError, TimeoutError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background work because failure-state probe failed",
        )
        logger.warning("Background policy failure-state probe failed: %s", _exc)
        return "failure_state_unavailable"

    try:
        from core.organism.welfare import welfare_block_reason

        welfare_reason = welfare_block_reason()
        if welfare_reason:
            # The organism's vital interests (memory integrity, repair
            # capacity) gate optional work: welfare is causal machinery
            # here, not narrative.
            return welfare_reason
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="continued background gating without welfare model",
        )

    if require_conversation_ready:
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.peek("inference_gate", default=None)
            if gate and hasattr(gate, "get_conversation_status"):
                lane = gate.get_conversation_status() or {}
                if not bool(lane.get("conversation_ready", False)):
                    return f"conversation_lane_{str(lane.get('state', 'unready') or 'unready').lower()}"
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation(
                "background_policy",
                _exc,
                action="blocked background work because conversation readiness probe failed",
            )
            logger.warning("Background policy conversation readiness probe failed: %s", _exc)
            return "conversation_lane_probe_unavailable"

    if orch is not None:
        # Boot grace is a coarse resource policy, so report it only after every
        # specific foreground, pressure, welfare, and readiness blocker above.
        # That keeps telemetry causally useful while preserving the same
        # admission behavior.
        boot_grace_s = _env_float("AURA_BACKGROUND_BOOT_GRACE_S", 300.0)
        uptime_s = _runtime_uptime_seconds(orch)
        if boot_grace_s > 0.0 and 0.0 < uptime_s < boot_grace_s:
            return f"boot_grace_{int(uptime_s)}s"

    return ""


def background_activity_allowed(
    orchestrator: Any = None,
    *,
    profile: BackgroundPolicyProfile | None = None,
    min_idle_seconds: float | None = None,
    max_memory_percent: float | None = None,
    max_failure_pressure: float | None = None,
    require_conversation_ready: bool | None = None,
    allow_no_user_anchor: bool = False,
    allow_desktop_safe_boot: bool = False,
) -> bool:
    return not background_activity_reason(
        orchestrator,
        profile=profile,
        min_idle_seconds=min_idle_seconds,
        max_memory_percent=max_memory_percent,
        max_failure_pressure=max_failure_pressure,
        require_conversation_ready=require_conversation_ready,
        allow_no_user_anchor=allow_no_user_anchor,
        allow_desktop_safe_boot=allow_desktop_safe_boot,
    )
