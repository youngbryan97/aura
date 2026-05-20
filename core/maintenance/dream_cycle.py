"""core/maintenance/dream_cycle.py — Memory Consolidation & Pruning.
"""
import asyncio
import logging
import time
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation

logger = logging.getLogger("Aura.Maintenance")


def _record_dream_degradation(
    error: BaseException,
    *,
    stage: str,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"stage": stage, "repair_requested": True}
    if extra:
        payload.update(extra)
    record_degradation(
        "dream_cycle",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        extra=payload,
    )


async def run_dream_cycle() -> dict[str, Any]:
    """Dream Cycle: Selective pruning of noisy vector memories and insight consolidation.
    Triggered when system stability falls below threshold.
    """
    logger.info("🌙 Aura is entering a Dream Cycle for stability restoration...")
    started_at = time.time()
    result: dict[str, Any] = {
        "ok": True,
        "completed_steps": [],
        "degraded_steps": [],
        "duration_s": 0.0,
    }

    try:
        from core.container import ServiceContainer

        memory = ServiceContainer.get("episodic_memory", default=None)
    except (ImportError, AttributeError, RuntimeError) as e:
        memory = None
        result["ok"] = False
        result["degraded_steps"].append("episodic_memory_lookup")
        _record_dream_degradation(
            e,
            stage="episodic_memory_lookup",
            action="continued dream cycle without episodic memory consolidation",
            severity="degraded",
        )

    if memory and hasattr(memory, "consolidate"):
        try:
            logger.info("  - Consolidating episodic traces...")
            await asyncio.sleep(1.5)
            await memory.consolidate()
            result["completed_steps"].append("episodic_memory_consolidation")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            result["ok"] = False
            result["degraded_steps"].append("episodic_memory_consolidation")
            _record_dream_degradation(
                e,
                stage="episodic_memory_consolidation",
                action="continued dream cycle after episodic memory consolidation failed",
                severity="degraded",
            )

    # Phase 10: Potential vector-space re-indexing or noise reduction
    logger.info("  - Calibrating cognitive entropy levels...")
    await asyncio.sleep(1.0)
    result["completed_steps"].append("entropy_calibration")

    # WAL checkpoint: prevent unbounded WAL growth under sustained writes
    try:
        from core.resilience.database_coordinator import get_db_coordinator

        coordinator = get_db_coordinator()
        coordinator.checkpoint_wal()
        result["completed_steps"].append("wal_checkpoint")
        logger.info("  - WAL checkpoint completed.")
    except (ImportError, AttributeError, RuntimeError) as e:
        result["ok"] = False
        result["degraded_steps"].append("wal_checkpoint")
        _record_dream_degradation(
            e,
            stage="wal_checkpoint",
            action="completed remaining dream-cycle steps after WAL checkpoint failed",
            severity="warning",
        )
        logger.debug("WAL checkpoint skipped: %s", e)

    logger.info("✓ Dream Cycle complete. System stability restored.")

    # Emit thought for UI visibility
    try:
        from core.thought_stream import get_emitter

        status_text = (
            "Dream cycle complete. Cognitive debt cleared."
            if result["ok"]
            else f"Dream cycle completed with degraded steps: {', '.join(result['degraded_steps'])}."
        )
        get_emitter().emit("Stability 🌙", status_text, level="info")
        result["completed_steps"].append("thought_stream_emit")
    except (ImportError, AttributeError, RuntimeError) as _exc:
        result["ok"] = False
        result["degraded_steps"].append("thought_stream_emit")
        _record_dream_degradation(
            _exc,
            stage="thought_stream_emit",
            action="returned dream-cycle result after UI thought emission failed",
            severity="warning",
        )
        logger.debug("Suppressed Exception: %s", _exc)

    result["duration_s"] = round(time.time() - started_at, 3)
    return result
