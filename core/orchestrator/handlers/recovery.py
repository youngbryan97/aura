"""core/orchestrator/handlers/recovery.py
Extracted cognitive recovery and circuit-breaker reset logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service, register_runtime_service

if TYPE_CHECKING:
    from core.orchestrator.main import RobustOrchestrator

logger = logging.getLogger("Aura.Core.Orchestrator.Recovery")
_RECOVERY_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError, Exception)


def _record_recovery_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("recovery", error, severity=severity, action=action)


async def retry_cognitive_connection(orch: RobustOrchestrator) -> bool:
    """Manually retry connecting to the cognitive brain (LLM).
    Forces a full re-wire of the Cognitive Engine AND resets the circuit breaker.
    """
    logger.info("Manual Cognitive Retry Initiated...")

    # Reset the Cognitive Circuit Breakers
    try:
        # 1. Reset LLM Router rate limits & health (includes local & remote)
        router = get_runtime_service("llm_router", default=None)
        if router and hasattr(router, "clear_rate_limits"):
            router.clear_rate_limits()
            logger.info("LLM Router health and rate limits RESET")

        # 2. Reset specific Local MLX Client if it exists
        mlx = get_runtime_service("mlx_client", default=None)
        if mlx and hasattr(mlx, "_circuit_open"):
            try:
                mlx._circuit_open = False
                mlx._consecutive_failures = 0
                logger.info("Local MLX Client circuit breaker RESET")
            except AttributeError as _exc:
                logger.debug("Suppressed AttributeError: %s", _exc)

        # Also try through the cognitive engine
        ce = getattr(orch, "cognitive_engine", None)
        if ce and hasattr(ce, "client") and hasattr(ce.client, "_circuit_open"):
            ce.client._circuit_open = False
            ce.client._circuit_open_until = 0.0
            ce.client._consecutive_failures = 0
            logger.info("Circuit breaker FORCE RESET on CognitiveEngine.client")

    except _RECOVERY_ERRORS as cb_err:
        _record_recovery_degradation(
            cb_err,
            action="continued cognitive retry after circuit-breaker reset failed",
        )
        logger.warning("Circuit breaker/RateLimit reset skipped: %s", cb_err)

    try:
        from core.brain.cognitive_engine import CognitiveEngine

        ce = getattr(orch, "cognitive_engine", None)
        if ce is None:
            ce = CognitiveEngine()

        try:
            ce.setup()
        except _RECOVERY_ERRORS as exc:
            _record_recovery_degradation(
                exc,
                action="continued cognitive retry after CognitiveEngine setup failed",
                severity="error",
            )
            logger.error("Setup failed: %s", exc)

        if not ce.lobotomized:
            register_runtime_service("cognitive_engine", ce)
            logger.info("Cognitive Engine ONLINE — Safe Mode deactivated")

            try:
                from core.thought_stream import get_emitter

                get_emitter().emit(
                    "System",
                    "Cognitive Connection Re-established",
                    level="success",
                    category="Brain",
                )
            except _RECOVERY_ERRORS as exc:
                _record_recovery_degradation(
                    exc,
                    action="returned successful cognitive retry after thought-stream emission failed",
                )
                logger.debug("ThoughtStream emit failed during cognitive retry: %s", exc)

            return True
        else:
            logger.error("Cognitive Retry Failed: Engine still lobotomized after re-wire")
            return False

    except _RECOVERY_ERRORS as exc:
        _record_recovery_degradation(
            exc,
            action="returned failed cognitive retry after CognitiveEngine reconstruction failed",
            severity="error",
        )
        logger.error("Cognitive Retry Exception: %s", exc)
        return False
