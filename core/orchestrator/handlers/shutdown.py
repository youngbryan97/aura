"""core/orchestrator/handlers/shutdown.py
Extracted shutdown orchestration from RobustOrchestrator.stop().
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING

from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log

if TYPE_CHECKING:
    from core.orchestrator.main import RobustOrchestrator

logger = logging.getLogger("Aura.Core.Orchestrator.Shutdown")


def _record_shutdown_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("shutdown", exc, severity=severity, action=action)


async def _gracefully_stop_actor_via_bus(
    orch: RobustOrchestrator,
    actor_name: str,
    *,
    stop_budget_s: float = 2.0,
) -> None:
    """Request actor shutdown over the runtime bus before the supervisor kills it."""
    bus = getattr(orch, "_actor_bus", None) or getattr(orch, "actor_bus", None)
    if bus is None:
        return

    has_actor = getattr(bus, "has_actor", None)
    if callable(has_actor) and not has_actor(actor_name):
        return

    try:
        await asyncio.wait_for(
            bus.request(
                actor_name,
                "stop",
                {"source": "orchestrator_shutdown", "reason": "graceful_shutdown"},
                timeout=stop_budget_s,
            ),
            timeout=stop_budget_s,
        )
    except (RuntimeError, asyncio.CancelledError, AttributeError) as exc:
        _record_shutdown_degradation(
            exc,
            action=f"continued shutdown after actor bus stop request failed for {actor_name}",
        )
        logger.debug("Graceful stop request failed for %s: %s", actor_name, exc)
        return

    supervisor = getattr(orch, "_supervisor_tree", None) or getattr(orch, "supervisor", None)
    is_actor_running = getattr(supervisor, "is_actor_running", None)
    if not callable(is_actor_running):
        return

    loop = asyncio.get_running_loop()
    deadline = loop.time() + stop_budget_s
    while loop.time() < deadline:
        try:
            if not is_actor_running(actor_name):
                return
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_shutdown_degradation(
                exc,
                action=f"continued shutdown after supervisor liveness probe failed for {actor_name}",
            )
            logger.debug("Supervisor liveness probe failed for %s: %s", actor_name, exc)
            return
        await asyncio.sleep(0.05)


async def orchestrator_shutdown(orch: RobustOrchestrator) -> None:
    """Gracefully shut down all orchestrator subsystems in priority order."""
    if hasattr(orch, "status") and not orch.status.running:
        return

    stack = inspect.stack()
    stack_str = "\n".join([f"  {s.filename}:{s.lineno} in {s.function}" for s in stack[:8]])
    logger.info("Initiating secure shutdown sequence... Called by:\n%s", stack_str)
    orch.status.running = False
    orch.status.is_processing = False

    # 1. Stop high-priority substrate loops
    if hasattr(orch, "substrate"):
        try:
            await asyncio.wait_for(orch.substrate.stop(), timeout=5.0)
        except TimeoutError:
            _record_shutdown_degradation(
                TimeoutError("substrate stop timed out"),
                action="continued shutdown after substrate stop timed out",
                severity="degraded",
            )
            logger.error("Substrate failed to stop within timeout")
        except (RuntimeError, asyncio.CancelledError, AttributeError) as exc:
            _record_shutdown_degradation(
                exc,
                action="continued shutdown after substrate stop failed",
                severity="degraded",
            )
            logger.debug("Substrate stop error: %s", exc)

    if hasattr(orch, "mind_tick") and orch.mind_tick:
        try:
            await asyncio.wait_for(orch.mind_tick.stop(), timeout=5.0)
            logger.info("💓 MindTick: Stopped.")
        except TimeoutError:
            _record_shutdown_degradation(
                TimeoutError("MindTick stop timed out"),
                action="continued shutdown after MindTick stop timed out",
                severity="degraded",
            )
            logger.error("MindTick: Failed to stop within timeout")
        except (RuntimeError, asyncio.CancelledError, AttributeError) as exc:
            _record_shutdown_degradation(
                exc,
                action="continued shutdown after MindTick stop failed",
                severity="degraded",
            )
            logger.error("MindTick: Failed to stop: %s", exc)

    # 2. Flush memory buffers / Snapshot management
    try:
        from core.resilience.snapshot_manager import SnapshotManager

        snapshot_mgr = SnapshotManager(orch)
        snapshot_mgr.freeze()
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_shutdown_degradation(
            exc,
            action="continued shutdown after cognitive snapshot freeze failed",
            severity="degraded",
        )
        logger.error("Failed to freeze cognitive snapshot: %s", exc)

    try:
        orch._save_state("shutdown")
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_shutdown_degradation(
            exc,
            action="continued shutdown after final state save failed",
            severity="degraded",
        )
        logger.debug("Final state save failed: %s", exc)

    try:
        state = await orch.state_repo.get_current()
        transport_ready = True
        transport_probe = getattr(orch.state_repo, "_transport_has_vault", None)
        if callable(transport_probe) and not getattr(orch.state_repo, "is_vault_owner", False):
            transport_ready = transport_probe()
        if state is not None and transport_ready:
            await orch.state_repo.commit(state.derive("shutdown"), "shutdown")
            logger.info("💾 UPSO: Shutdown state committed.")
        else:
            logger.info("💾 UPSO: Skipping shutdown state commit; state transport unavailable.")
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_shutdown_degradation(
            exc,
            action="continued shutdown after UPSO shutdown state commit failed",
            severity="degraded",
        )
        logger.error("UPSO: Failed to commit shutdown state: %s", exc)

    # 3. Release service locks / Graceful shutdown of subsystems
    orch._publish_status({"event": "stopping", "message": "Graceful shutdown initiated"})

    if hasattr(orch, "_stop_event") and orch._stop_event:
        orch._stop_event.set()

    consciousness = getattr(orch, "consciousness", None)
    if consciousness and hasattr(consciousness, "stop"):
        try:
            res = consciousness.stop()
            if inspect.isawaitable(res):
                await res
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_shutdown_degradation(
                exc,
                action="continued shutdown after consciousness system stop failed",
                severity="degraded",
            )
            capture_and_log(exc, {"module": __name__})

    if hasattr(orch, "conversation_loop") and orch.conversation_loop:
        try:
            await asyncio.wait_for(orch.conversation_loop.stop(), timeout=5.0)
        except TimeoutError as _exc:
            _record_shutdown_degradation(
                _exc,
                action="continued shutdown after conversation loop stop timed out",
            )
            logger.debug("Suppressed asyncio.TimeoutError: %s", _exc)
        except (RuntimeError, asyncio.CancelledError, AttributeError) as exc:
            _record_shutdown_degradation(
                exc,
                action="continued shutdown after conversation loop stop failed",
            )
            logger.debug("Conversation loop stop error: %s", exc)

    if orch.kernel_interface and hasattr(orch.kernel_interface, "shutdown"):
        try:
            await asyncio.wait_for(orch.kernel_interface.shutdown(), timeout=5.0)
        except TimeoutError:
            _record_shutdown_degradation(
                TimeoutError("KernelInterface shutdown timed out"),
                action="continued shutdown after KernelInterface shutdown timed out",
                severity="degraded",
            )
            logger.error("KernelInterface shutdown timed out")
        except (RuntimeError, asyncio.CancelledError, AttributeError) as exc:
            _record_shutdown_degradation(
                exc,
                action="continued shutdown after KernelInterface shutdown failed",
                severity="degraded",
            )
            logger.error("KernelInterface shutdown failed: %s", exc)

    if hasattr(orch, "_actor_bus") and orch._actor_bus:
        await _gracefully_stop_actor_via_bus(orch, "state_vault", stop_budget_s=2.0)
        try:
            await asyncio.wait_for(orch._actor_bus.stop(), timeout=5.0)
        except TimeoutError as _exc:
            _record_shutdown_degradation(
                _exc,
                action="continued shutdown after actor bus stop timed out",
            )
            logger.debug("Suppressed asyncio.TimeoutError: %s", _exc)
        except (RuntimeError, asyncio.CancelledError, AttributeError) as exc:
            _record_shutdown_degradation(
                exc,
                action="continued shutdown after actor bus stop failed",
            )
            logger.debug("ActorBus stop error: %s", exc)

    if hasattr(orch, "_supervisor_tree") and orch._supervisor_tree:
        try:
            await asyncio.wait_for(orch._supervisor_tree.stop(), timeout=5.0)
        except TimeoutError:
            _record_shutdown_degradation(
                TimeoutError("supervisor tree stop timed out"),
                action="continued shutdown after supervisor tree stop timed out",
                severity="degraded",
            )
            logger.error("Supervisor tree failed to stop within timeout")
        except (RuntimeError, asyncio.CancelledError, AttributeError) as exc:
            _record_shutdown_degradation(
                exc,
                action="continued shutdown after supervisor tree stop failed",
                severity="degraded",
            )
            logger.error("Supervisor tree shutdown failed: %s", exc)

    if hasattr(orch, "state_repo") and orch.state_repo:
        try:
            await asyncio.wait_for(orch.state_repo.close(), timeout=5.0)
        except TimeoutError:
            _record_shutdown_degradation(
                TimeoutError("StateRepository close timed out"),
                action="continued shutdown after StateRepository close timed out",
                severity="degraded",
            )
            logger.error("StateRepository close timed out")
        except (RuntimeError, asyncio.CancelledError, AttributeError) as exc:
            _record_shutdown_degradation(
                exc,
                action="continued shutdown after StateRepository close failed",
                severity="degraded",
            )
            logger.error("StateRepository close failed: %s", exc)

    sensory = getattr(orch, "_sensory_actor", None)
    if sensory and hasattr(sensory, "is_alive") and sensory.is_alive():
        logger.info("🛑 Terminating SensoryGate Actor...")
        sensory.terminate()
        await asyncio.to_thread(sensory.join, 2.0)
        if sensory.is_alive():
            sensory.kill()

    if hasattr(orch, "swarm") and orch.swarm:
        try:
            await asyncio.wait_for(orch.swarm.stop(), timeout=5.0)
        except TimeoutError as _exc:
            _record_shutdown_degradation(
                _exc,
                action="continued shutdown after swarm stop timed out",
            )
            logger.debug("Suppressed asyncio.TimeoutError: %s", _exc)
        except (RuntimeError, asyncio.CancelledError, AttributeError) as exc:
            _record_shutdown_degradation(
                exc,
                action="continued shutdown after swarm stop failed",
            )
            logger.debug("Swarm stop error: %s", exc)

    try:
        from core.container import ServiceContainer
        delegator = ServiceContainer.get("agent_delegator", default=None)
        if delegator and hasattr(delegator, "stop"):
            await asyncio.wait_for(delegator.stop(), timeout=5.0)
    except (ImportError, AttributeError, RuntimeError, TimeoutError) as exc:
        _record_shutdown_degradation(
            exc,
            action="continued shutdown after agent delegator stop failed",
        )
        logger.debug("Agent delegator stop error: %s", exc)

    try:
        from core.container import ServiceContainer

        await asyncio.wait_for(ServiceContainer.shutdown(), timeout=5.0)
    except TimeoutError:
        _record_shutdown_degradation(
            TimeoutError("ServiceContainer shutdown timed out"),
            action="continued shutdown after ServiceContainer shutdown timed out",
            severity="degraded",
        )
        logger.error("ServiceContainer shutdown timed out")
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_shutdown_degradation(
            exc,
            action="continued shutdown after ServiceContainer shutdown failed",
            severity="degraded",
        )
        logger.error("Error during ServiceContainer shutdown: %s", exc)

    if hasattr(orch, "_event_loop_monitor") and orch._event_loop_monitor:
        try:
            await asyncio.wait_for(orch._event_loop_monitor.stop(), timeout=5.0)
        except TimeoutError as _exc:
            _record_shutdown_degradation(
                _exc,
                action="continued shutdown after event loop monitor stop timed out",
            )
            logger.debug("Suppressed asyncio.TimeoutError: %s", _exc)
        except (RuntimeError, asyncio.CancelledError, AttributeError) as exc:
            _record_shutdown_degradation(
                exc,
                action="continued shutdown after event loop monitor stop failed",
            )
            logger.debug("Event loop monitor stop error: %s", exc)

    try:
        from core.event_bus import get_event_bus

        await get_event_bus().shutdown()
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_shutdown_degradation(
            exc,
            action="continued shutdown after event bus shutdown failed",
        )
        logger.warning("Event bus shutdown failed: %s", exc, exc_info=True)

    try:
        from core.utils.task_tracker import get_task_tracker

        tracker_shutdown = get_task_tracker().shutdown(timeout=3.0)
        if asyncio.iscoroutine(tracker_shutdown):
            await tracker_shutdown
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_shutdown_degradation(
            exc,
            action="completed shutdown after task tracker shutdown failed",
            severity="degraded",
        )
        logger.warning("Task tracker shutdown failed: %s", exc)

    logger.info("✅ Orchestrator stopped.")
