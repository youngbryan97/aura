from __future__ import annotations

import inspect
import logging
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import Severity, record_degradation
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.Dream")

_DREAM_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    OSError,
    ConnectionError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _error_summary(error: BaseException) -> str:
    return f"{type(error).__qualname__}: {error}"[:240]


def _record_dream_degradation(
    error: BaseException,
    *,
    subsystem: str,
    action: str,
    severity: Severity = "warning",
) -> None:
    record_degradation(
        "dream_skill",
        error,
        severity=severity,
        action=action,
        extra={"subsystem": subsystem},
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _set_status(results: dict[str, dict[str, Any]], subsystem: str, status: str, **fields: Any) -> None:
    payload = {"status": status}
    payload.update({key: value for key, value in fields.items() if value is not None})
    results[subsystem] = payload


def _close_coro(coro: Any) -> None:
    if inspect.iscoroutine(coro):
        coro.close()


def _schedule_background_task(coro: Any, *, name: str, subsystem: str) -> dict[str, Any]:
    try:
        from core.utils.task_tracker import get_task_tracker

        get_task_tracker().create_task(coro, name=name)
        return {"status": "queued", "task": name}
    except _DREAM_RECOVERABLE_ERRORS as exc:
        _close_coro(coro)
        action = f"Closed unscheduled coroutine for {name}; subsystem remains available for later retry"
        _record_dream_degradation(exc, subsystem=subsystem, action=action, severity="warning")
        logger.debug("Dream background task %s could not be scheduled: %s", name, exc)
        return {"status": "failed", "task": name, "error": _error_summary(exc), "action": action}


def _get_service(results: dict[str, dict[str, Any]], name: str, subsystem: str) -> Any | None:
    try:
        return ServiceContainer.get(name, default=None)
    except _DREAM_RECOVERABLE_ERRORS as exc:
        action = f"Service lookup for {name} failed; skipped {subsystem}"
        _record_dream_degradation(exc, subsystem=subsystem, action=action, severity="warning")
        _set_status(results, subsystem, "failed", error=_error_summary(exc), action=action)
        return None


class DreamSkill(BaseSkill):
    """
    Triggers an immediate 'Dream Cycle' across the cognitive and semantic layers.
    This consolidates old, fragmented memories into denser concepts, and
    reprocesses the dead-letter queue for missed thoughts.
    """

    name = "force_dream_cycle"
    description = "Initiates immediate memory consolidation and dead-letter queue (DLQ) re-processing."
    inputs = {
        "focus": "(Optional) A specific concept or timeframe to focus consolidation on."
    }

    async def execute(
        self,
        goal: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Triggers the semantic defragger, dream journal synthesis, and DLQ re-ingestion."""
        logger.info("💤 Manual Dream Cycle Initiated via Skill.")
        results: dict[str, dict[str, Any]] = {}

        # 1. DreamJournal — qualia-driven creative synthesis from salient memories
        dream_journal = _get_service(results, "dream_journal", "dream_journal")
        if dream_journal and hasattr(dream_journal, "synthesize_dream"):
            try:
                dream_result = await _maybe_await(dream_journal.synthesize_dream())
                if dream_result:
                    _set_status(
                        results,
                        "dream_journal",
                        "completed",
                        content=str(dream_result.get("dream_content", ""))[:500],
                        seed_count=dream_result.get("seed_count", 0),
                    )
                else:
                    _set_status(results, "dream_journal", "skipped", reason="insufficient_salient_material")
            except _DREAM_RECOVERABLE_ERRORS as exc:
                action = "Dream journal synthesis failed; continued with defrag and restoration lanes"
                _record_dream_degradation(exc, subsystem="dream_journal", action=action, severity="warning")
                _set_status(results, "dream_journal", "failed", error=_error_summary(exc), action=action)
        elif "dream_journal" not in results:
            _set_status(results, "dream_journal", "unavailable")

        orchestrator = _get_service(results, "orchestrator", "orchestrator")

        # 2. Semantic Defrag (ChromaDB Vector Consolidation)
        if orchestrator and hasattr(orchestrator, "semantic_defrag") and getattr(orchestrator.semantic_defrag, "run_defrag_cycle", None):
            try:
                schedule = _schedule_background_task(
                    orchestrator.semantic_defrag.run_defrag_cycle(),
                    name="dream_skill.semantic_defrag",
                    subsystem="semantic_defrag",
                )
                results["semantic_defrag"] = schedule
            except _DREAM_RECOVERABLE_ERRORS as exc:
                action = "Semantic defrag scheduling failed before task handoff; continued dream cycle"
                _record_dream_degradation(exc, subsystem="semantic_defrag", action=action, severity="warning")
                _set_status(results, "semantic_defrag", "failed", error=_error_summary(exc), action=action)
        else:
            _set_status(results, "semantic_defrag", "unavailable")

        # 3. Dream Cycle (DLQ Re-ingestion)
        if orchestrator and hasattr(orchestrator, "dream_cycle") and getattr(orchestrator.dream_cycle, "process_dreams", None):
            try:
                schedule = _schedule_background_task(
                    orchestrator.dream_cycle.process_dreams(),
                    name="dream_skill.process_dreams",
                    subsystem="dlq_cycle",
                )
                results["dlq_cycle"] = schedule
            except _DREAM_RECOVERABLE_ERRORS as exc:
                action = "DLQ dream-cycle scheduling failed before task handoff; continued dream cycle"
                _record_dream_degradation(exc, subsystem="dlq_cycle", action=action, severity="warning")
                _set_status(results, "dlq_cycle", "failed", error=_error_summary(exc), action=action)
        else:
            _set_status(results, "dlq_cycle", "unavailable")

        # 4. Heuristic Synthesis — extract learned instincts from recent telemetry
        heuristic_synthesizer = _get_service(results, "heuristic_synthesizer", "heuristic_synthesis")
        if heuristic_synthesizer and hasattr(heuristic_synthesizer, "synthesize_from_telemetry"):
            try:
                hs_result = await _maybe_await(heuristic_synthesizer.synthesize_from_telemetry())
                if isinstance(hs_result, dict):
                    results["heuristic_synthesis"] = {"status": "completed", **hs_result}
                else:
                    _set_status(results, "heuristic_synthesis", "completed", result=hs_result)
            except _DREAM_RECOVERABLE_ERRORS as exc:
                action = "Heuristic synthesis failed; retained other dream-cycle outputs"
                _record_dream_degradation(exc, subsystem="heuristic_synthesis", action=action, severity="warning")
                _set_status(results, "heuristic_synthesis", "failed", error=_error_summary(exc), action=action)
        elif "heuristic_synthesis" not in results:
            _set_status(results, "heuristic_synthesis", "unavailable")

        # 5. Drive restoration (dreaming restores energy)
        drive = _get_service(results, "drive_engine", "drive_restoration")
        if drive and hasattr(drive, "satisfy"):
            try:
                await _maybe_await(drive.satisfy("energy", 20.0))
                _set_status(results, "drive_restoration", "completed", restored={"energy": 20.0})
            except _DREAM_RECOVERABLE_ERRORS as exc:
                action = "Drive restoration failed; returned dream outputs without claiming rest recovery"
                _record_dream_degradation(exc, subsystem="drive_restoration", action=action, severity="warning")
                _set_status(results, "drive_restoration", "failed", error=_error_summary(exc), action=action)
        elif "drive_restoration" not in results:
            _set_status(results, "drive_restoration", "unavailable")

        # 6. WorldState event
        try:
            from core.world_state import get_world_state

            get_world_state().record_event(
                "Dream cycle completed",
                source="dream_skill",
                salience=0.3,
                ttl=14400,
            )
            _set_status(results, "world_state", "completed")
        except _DREAM_RECOVERABLE_ERRORS as exc:
            action = "World-state dream event write failed; returned subsystem receipts instead"
            _record_dream_degradation(exc, subsystem="world_state", action=action, severity="debug")
            _set_status(results, "world_state", "failed", error=_error_summary(exc), action=action)

        completed = [
            key
            for key, value in results.items()
            if value.get("status") in {"completed", "queued"}
        ]
        degraded = [
            key
            for key, value in results.items()
            if value.get("status") == "failed"
        ]
        summary = f"Dream cycle: {len(completed)}/{len(results)} subsystems engaged."
        if degraded:
            summary += f" Degraded: {', '.join(degraded[:4])}."

        return {
            "ok": True,
            "summary": summary,
            "message": summary,
            "subsystems": results,
            "completed_subsystems": completed,
            "degraded_subsystems": degraded,
        }
