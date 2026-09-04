"""core/morphogenesis/hooks.py

Bidirectional integration hooks that give morphogenesis *real influence*
over Aura's runtime behaviour.

This module is the difference between a decorative dashboard and a genuine
developmental layer. Each hook:

  1. Reads morphogenetic state (field gradients, metabolism, cell health)
  2. Feeds that state into an existing Aura subsystem's decision path
  3. Or feeds subsystem events back into morphogenesis

The hooks are designed to be individually non-fatal: if the morphogenetic
runtime is offline, every hook degrades to a no-op.

Influence map:
  - StabilityGuardian  → cell health check registered via add_check()
  - SelfHealing        → morphogenesis runtime registered as a watched service
  - MetabolicCoordinator → field pressure modulates energy refill rate
  - InferenceGate      → metabolism high-pressure flag downgrades foreground tier
  - Orchestrator       → exceptions/tick events feed signals back into morphogenesis
  - EpisodicMemory     → organ stabilisation events trigger memory consolidation
"""
from __future__ import annotations

import inspect
import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Morphogenesis.Hooks")


# ---------------------------------------------------------------------------
# 1. StabilityGuardian ← morphogenesis cell health check
# ---------------------------------------------------------------------------

def register_stability_guardian_check() -> bool:
    """Register a health check with StabilityGuardian that reports
    aggregate cell/field health from the morphogenetic layer.

    Returns True if the check was registered, False otherwise.
    """
    try:
        from core.container import ServiceContainer
        guardian = ServiceContainer.get("stability_guardian", default=None)
        if guardian is None or not hasattr(guardian, "add_check"):
            return False
        guardian.add_check(_morphogenesis_health_check)
        logger.info("🧬 Morphogenesis health check registered with StabilityGuardian.")
        return True
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('hooks', exc)
        logger.debug("StabilityGuardian hook skipped: %s", exc)
        return False


def _morphogenesis_health_check():
    """Called every 10s by StabilityGuardian to check morphogenesis health."""
    try:
        from core.resilience.stability_guardian import HealthCheckResult
    except ImportError:
        return None

    try:
        from core.container import ServiceContainer
        rt = ServiceContainer.get("morphogenetic_runtime", default=None)
        if rt is None:
            return HealthCheckResult(
                name="morphogenesis",
                healthy=True,
                message="Morphogenesis runtime not registered",
                severity="info",
            )

        status = rt.status()
        running = status.get("running", False)
        tick = status.get("tick", 0)
        last_error = status.get("last_tick_error", "")
        # Check cell ecology health
        registry_status = status.get("registry", {})
        cell_count = registry_status.get("cells", 0)
        by_state = registry_status.get("by_state", {}) or {}
        quarantined = registry_status.get("quarantined", by_state.get("quarantined", 0))
        dead = registry_status.get("dead", by_state.get("dead", 0))

        # Check metabolism health
        metabolism = status.get("metabolism", {})
        global_energy = metabolism.get("global_energy", 1.0)
        high_pressure = metabolism.get("high_pressure", False)
        immunity_bridge = status.get("immunity_bridge", {}) or {}

        # Determine health
        issues = []
        severity = "info"

        if not running and status.get("enabled", False):
            issues.append("runtime stopped but enabled")
            severity = "warning"

        if last_error:
            issues.append(f"last_error: {last_error[:80]}")
            severity = "warning"

        if quarantined > 0:
            issues.append(f"{quarantined} quarantined cell(s)")
            severity = "warning"

        if dead > 0:
            issues.append(f"{dead} dead cell(s)")

        if global_energy < 0.15:
            issues.append(f"global_energy critically low ({global_energy:.2f})")
            severity = "warning"

        if high_pressure:
            issues.append("high resource pressure")

        if immunity_bridge.get("enabled") and not immunity_bridge.get("worker_running"):
            issues.append("adaptive-immunity bridge worker stopped")
            severity = "warning"

        if immunity_bridge.get("stalled"):
            issues.append(
                "adaptive-immunity bridge stalled "
                f"({float(immunity_bridge.get('inflight_age_s', 0.0)):.1f}s)"
            )
            severity = "warning"

        healthy = severity != "warning" and severity != "error"

        if issues:
            message = f"Morphogenesis: {'; '.join(issues)} (tick={tick}, cells={cell_count})"
        else:
            message = f"Morphogenesis OK: tick={tick}, cells={cell_count}, energy={global_energy:.2f}"

        return HealthCheckResult(
            name="morphogenesis",
            healthy=healthy,
            message=message,
            severity=severity,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation('hooks', exc)
        from core.resilience.stability_guardian import HealthCheckResult
        return HealthCheckResult(
            name="morphogenesis",
            healthy=False,
            message=f"Morphogenesis health probe failed: {type(exc).__name__}: {exc}",
            severity="warning",
        )


# ---------------------------------------------------------------------------
# 2. SelfHealing ← morphogenesis runtime as a watched service
# ---------------------------------------------------------------------------

def register_self_healing_watch() -> bool:
    """Register the morphogenetic runtime with SelfHealing so that
    if the runtime loop stalls, it gets auto-restarted.
    """
    try:
        from core.container import ServiceContainer
        from core.runtime.self_healing import get_healer

        async def _restart_morphogenesis():
            rt = ServiceContainer.get("morphogenetic_runtime", default=None)
            if rt is None:
                raise RuntimeError("morphogenesis runtime is not registered")
            logger.warning("🧬 SelfHealing restarting morphogenesis runtime...")
            restart = getattr(rt, "restart_async", None)
            if callable(restart):
                result = restart()
                if inspect.isawaitable(result):
                    await result
            else:
                await rt.stop()
                await rt.start()
            status = rt.status()
            if bool(getattr(rt.config, "enabled", True)) and not status.get("running", False):
                raise RuntimeError("morphogenesis restart completed without a live runtime task")
            logger.info("🧬 Morphogenesis runtime restarted by SelfHealing.")

        runtime = ServiceContainer.get("morphogenetic_runtime", default=None)
        tick_interval_s = float(
            getattr(getattr(runtime, "config", None), "tick_interval_s", 1.0) or 1.0
        )
        expected_interval_s = min(60.0, max(5.0, tick_interval_s * 4.0))

        healer = get_healer()
        healer.watch(
            "morphogenesis_runtime",
            expected_interval_s=expected_interval_s,
            restart_async=_restart_morphogenesis,
            container_key="morphogenetic_runtime",
        )
        logger.info("🧬 Morphogenesis registered with SelfHealing watchdog.")
        return True
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('hooks', exc)
        logger.debug("SelfHealing hook skipped: %s", exc)
        return False


def heartbeat_self_healing() -> None:
    """Called from the morphogenesis runtime tick to keep SelfHealing happy."""
    try:
        from core.runtime.self_healing import get_healer
        get_healer().heartbeat("morphogenesis_runtime")
    except (ImportError, AttributeError, RuntimeError):
        pass  # no-op: intentional


# ---------------------------------------------------------------------------
# 3. MetabolicCoordinator ← field pressure modulates energy refill
# ---------------------------------------------------------------------------

def modulate_metabolic_energy() -> float | None:
    """Read morphogenesis field pressure and modulate the MetabolicCoordinator's
    energy refill rate. Returns the applied modifier or None.

    When the morphogenetic field shows high danger/resource pressure, the
    MetabolicCoordinator slows down energy recovery so that background tasks
    (autonomous thought, RL training, persona evolution) are naturally
    throttled. When field shows growth/curiosity, recovery is boosted.
    """
    try:
        from core.container import ServiceContainer
        rt = ServiceContainer.get("morphogenetic_runtime", default=None)
        if rt is None:
            return None

        coord = ServiceContainer.get("metabolic_coordinator", default=None)
        if coord is None:
            return None

        # Sample the global field state
        field_state = rt.field.sample("global")
        danger = field_state.get("danger", 0.0)
        resource_pressure = field_state.get("resource_pressure", 0.0)
        growth = field_state.get("growth", 0.0)
        curiosity = field_state.get("curiosity", 0.0)

        # Compute a modifier: 0.5 = halve recovery, 1.5 = 50% boost
        #   High danger/pressure → suppress → modifier < 1
        #   High growth/curiosity → encourage → modifier > 1
        suppress = max(danger, resource_pressure) * 0.5
        encourage = max(growth, curiosity) * 0.3
        modifier = max(0.3, min(1.5, 1.0 - suppress + encourage))

        # Apply the modifier to the coordinator's OWN base rate, remembered
        # once. The previous version multiplied a hardcoded 0.05 and wrote the
        # result back every five ticks, so whatever the coordinator had been
        # configured with was overwritten with a constant this module invented,
        # and the modulation compounded against its own last output.
        attribute = "_energy_refill_rate"
        if not hasattr(coord, attribute):
            return None
        base = getattr(coord, "_morphogenesis_base_refill_rate", None)
        if base is None:
            base = float(getattr(coord, attribute, 0.05) or 0.05)
            try:
                coord._morphogenesis_base_refill_rate = base
            except (AttributeError, TypeError):
                return None
        try:
            setattr(coord, attribute, base * modifier)
        except (AttributeError, TypeError):
            return None
        return modifier
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('hooks', exc)
        logger.debug("Metabolic modulation skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 4. InferenceGate ← metabolism pressure influences tier selection
# ---------------------------------------------------------------------------

def get_morphogenesis_routing_advice() -> dict[str, Any]:
    """Return routing advice based on morphogenetic state.

    InferenceGate can call this to decide whether to attempt the heavy
    32B cortex (primary) or downgrade to the 7B brainstem (tertiary)
    when the system is under developmental stress.

    Returns:
        {
            "pressure": float,       # 0.0-1.0 overall pressure
            "recommend_downgrade": bool,
            "reason": str,
            "field_snapshot": dict,
        }
    """
    default = {
        "pressure": 0.0,
        "recommend_downgrade": False,
        "reason": "morphogenesis_offline",
        "field_snapshot": {},
    }
    try:
        from core.container import ServiceContainer
        rt = ServiceContainer.get("morphogenetic_runtime", default=None)
        if rt is None:
            return default

        metabolism = rt.metabolism
        field_state = rt.field.sample("global")

        pressure = metabolism._last_snapshot.pressure
        global_energy = metabolism.global_energy
        danger = field_state.get("danger", 0.0)
        resource_pressure = field_state.get("resource_pressure", 0.0)

        # Recommend downgrade when:
        #   - System resource pressure > 0.82 (high CPU/RAM), OR
        #   - Morphogenetic global energy < 0.12, OR
        #   - Combined danger + resource_pressure > 1.2
        recommend = (
            pressure > 0.82
            or global_energy < 0.12
            or (danger + resource_pressure) > 1.2
        )

        if recommend:
            reason = (
                f"pressure={pressure:.2f} energy={global_energy:.2f} "
                f"danger={danger:.2f} resource_pressure={resource_pressure:.2f}"
            )
        else:
            reason = "nominal"

        return {
            "pressure": pressure,
            "recommend_downgrade": recommend,
            "reason": reason,
            "field_snapshot": field_state,
        }
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('hooks', exc)
        logger.debug("Routing advice skipped: %s", exc)
        return default


# ---------------------------------------------------------------------------
# 5. Orchestrator → morphogenesis: feed exceptions and tick events
# ---------------------------------------------------------------------------

def observe_orchestrator_exception(
    *,
    subsystem: str,
    exc: BaseException,
    source: str = "orchestrator",
) -> None:
    """Feed an orchestrator-level exception into the morphogenetic field.

    This allows the cell ecology to react to failures across all subsystems,
    not just the ones morphogenesis cells explicitly watch.
    """
    try:
        from core.container import ServiceContainer
        rt = ServiceContainer.get("morphogenetic_runtime", default=None)
        if rt is not None:
            rt.observe_exception(
                subsystem=subsystem,
                exc=exc,
                source=source,
                danger=0.65,
            )
    except (ImportError, AttributeError, RuntimeError):
        pass  # no-op: intentional


def emit_task_signal(
    *,
    subsystem: str,
    task_description: str = "",
    intensity: float = 0.4,
) -> None:
    """Emit a task signal into the morphogenetic field when a turn starts.

    Had no callers for the life of the module — a writer with no writer, which
    is the same defect as a channel with no reader and just as invisible. The
    live seam is now :func:`core.morphogenesis.bridge.announce_demand`, called
    from the chat turn; this stays for callers that already know their
    subsystem and do not need the capability router to work it out.
    """
    try:
        from core.container import ServiceContainer
        rt = ServiceContainer.get("morphogenetic_runtime", default=None)
        if rt is None:
            return
        from core.morphogenesis.types import MorphogenSignal, SignalKind
        rt.emit_signal(MorphogenSignal(
            kind=SignalKind.TASK,
            source="orchestrator",
            subsystem=subsystem,
            intensity=min(1.0, max(0.05, intensity)),
            payload={"task": task_description[:200]},
            ttl_ticks=6,
        ))
    except (ImportError, AttributeError, RuntimeError):
        pass  # no-op: intentional


# ---------------------------------------------------------------------------
# 6. EpisodicMemory ← organ stabilisation triggers consolidation
# ---------------------------------------------------------------------------

async def record_organ_formation_episode(organ_data: dict[str, Any]) -> None:
    """When a new organ is discovered, record it as a significant episode
    in episodic memory. This ensures long-term behavioural development:
    the system remembers which cell coalitions proved useful.
    """
    try:
        from core.container import ServiceContainer
        mem = ServiceContainer.get("episodic_memory", default=None)
        if mem is None:
            try:
                from core.memory.episodic_memory import get_episodic_memory
                mem = get_episodic_memory()
            except (ImportError, AttributeError, RuntimeError):
                return

        if not hasattr(mem, "record_episode_async"):
            return

        members = organ_data.get("members", [])
        organ_name = organ_data.get("name", "unknown")
        subsystem = organ_data.get("subsystem", "composite")
        confidence = organ_data.get("confidence", 0.0)

        await mem.record_episode_async(
            context=f"Morphogenetic organ formation: {organ_name}",
            action="organ_stabilization",
            outcome=(
                f"New organ '{organ_name}' emerged from {len(members)} cells "
                f"in subsystem '{subsystem}' with confidence {confidence:.2f}"
            ),
            success=True,
            emotional_valence=0.35,
            tools_used=["morphogenesis_organ_stabilizer"],
            lessons=[
                f"Cells {', '.join(members[:4])} form a stable coalition for {subsystem}",
                "Repeated co-activation above threshold crystallises into an organ",
            ],
            importance=0.7,
            source="morphogenesis",
            metadata={
                "organ": organ_data,
                "event_type": "organ_formation",
            },
        )
        logger.debug("🧬 Organ formation episode recorded: %s", organ_name)
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('hooks', exc)
        logger.debug("Organ formation episode recording failed: %s", exc)


# ---------------------------------------------------------------------------
# 7. Autonomous initiative modulation
# ---------------------------------------------------------------------------

def should_suppress_autonomous_initiative() -> bool:
    """Returns True if the morphogenetic field indicates that autonomous
    initiative (boredom/reflection impulses, proactive messages) should be
    suppressed due to high system stress.

    The MetabolicCoordinator's impulse triggers can check this before
    spawning expensive background tasks.
    """
    try:
        from core.container import ServiceContainer
        rt = ServiceContainer.get("morphogenetic_runtime", default=None)
        if rt is None:
            return False  # No morphogenesis = no opinion

        field_state = rt.field.sample("global")
        danger = field_state.get("danger", 0.0)
        resource_pressure = field_state.get("resource_pressure", 0.0)
        inhibition = field_state.get("inhibition", 0.0)

        # Suppress if danger or resource pressure is elevated
        return (danger > 0.6 or resource_pressure > 0.7 or inhibition > 0.5)
    except (ImportError, AttributeError, RuntimeError):
        return False


# ---------------------------------------------------------------------------
# 8. Tool selection influence
# ---------------------------------------------------------------------------

def get_cell_capability_boost(tool_name: str) -> float:
    """Return a priority boost (0.0 to 0.5) for a tool/skill based on
    whether morphogenesis cells with matching capabilities are active
    and healthy.

    This allows the system to prefer tools that have an active, healthy
    cell ecology behind them.
    """
    try:
        from core.container import ServiceContainer
        rt = ServiceContainer.get("morphogenetic_runtime", default=None)
        if rt is None:
            return 0.0

        boost = 0.0
        for cell in rt.registry.active_cells():
            caps = {str(c) for c in cell.manifest.capabilities}
            if tool_name in caps or any(tool_name.lower() in c.lower() for c in caps):
                # Active, healthy cell with this capability → boost
                health = cell.state.health
                confidence = cell.state.confidence
                boost = max(boost, min(0.5, health * 0.3 + confidence * 0.2))
        return boost
    except (ImportError, AttributeError, RuntimeError):
        return 0.0


# ---------------------------------------------------------------------------
# 9. Master wiring — called once during boot
# ---------------------------------------------------------------------------

async def wire_all_hooks() -> dict[str, bool]:
    """Wire all morphogenesis hooks into the existing Aura subsystems.

    Called by integration.start_morphogenesis_runtime() after the runtime
    is started. Returns a dict of hook_name → success.
    """
    results: dict[str, bool] = {}

    results["stability_guardian"] = register_stability_guardian_check()
    results["self_healing"] = register_self_healing_watch()

    # Metabolic modulation runs on each morphogenesis tick, so we just
    # verify the coordinator exists.
    try:
        from core.container import ServiceContainer
        coord = ServiceContainer.get("metabolic_coordinator", default=None)
        results["metabolic_modulation"] = coord is not None
    except (ImportError, AttributeError, RuntimeError):
        results["metabolic_modulation"] = False

    # Routing advice is pulled on-demand by InferenceGate, no registration needed.
    results["routing_advice"] = True

    # Exception observer is called manually from exception handlers, no registration.
    results["exception_observer"] = True

    # Initiative suppression is polled on-demand.
    results["initiative_suppression"] = True

    # Tool capability boost is polled on-demand.
    results["tool_capability_boost"] = True

    # Organ formation episodes are triggered by runtime.tick() — we wire that
    # into the runtime's organ callback.
    results["organ_episodes"] = True

    logger.info(
        "🧬 Morphogenesis hooks wired: %s",
        ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in results.items()),
    )
    return results
