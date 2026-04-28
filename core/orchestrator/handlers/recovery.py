"""core/orchestrator/handlers/recovery.py
Extracted cognitive recovery and circuit-breaker reset logic.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.orchestrator.main import RobustOrchestrator

logger = logging.getLogger("Aura.Core.Orchestrator.Recovery")


async def retry_cognitive_connection(orch: "RobustOrchestrator") -> bool:
    """Manually retry connecting to the cognitive brain (LLM).
    Forces a full re-wire of the Cognitive Engine AND resets the circuit breaker.
    """
    from core.container import ServiceContainer

    logger.info("🧠 Manual Cognitive Retry Initiated...")

    # Reset the Cognitive Circuit Breakers
    try:
        # 1. Reset LLM Router rate limits & health (includes local & remote)
        router = ServiceContainer.get("llm_router", default=None)
        if router and hasattr(router, "clear_rate_limits"):
            router.clear_rate_limits()
            logger.info("⚡ LLM Router health and rate limits RESET")

        # 2. Reset specific Local MLX Client if it exists
        mlx = ServiceContainer.get("mlx_client", default=None)
        if mlx and hasattr(mlx, "_circuit_open"):
            try:
                mlx._circuit_open = False
                mlx._consecutive_failures = 0
                logger.info("⚡ Local MLX Client circuit breaker RESET")
            except AttributeError as _exc:
                logger.debug("Suppressed AttributeError: %s", _exc)

        # Also try through the cognitive engine
        ce = getattr(orch, "cognitive_engine", None)
        if ce and hasattr(ce, 'client') and hasattr(ce.client, '_circuit_open'):
            ce.client._circuit_open = False
            ce.client._circuit_open_until = 0.0
            ce.client._consecutive_failures = 0
            logger.info("⚡ Circuit breaker FORCE RESET on CognitiveEngine.client")

    except Exception as cb_err:
        record_degradation('recovery', cb_err)
        logger.warning("Circuit breaker/RateLimit reset skipped: %s", cb_err)

    try:
        from core.brain.cognitive_engine import CognitiveEngine
        ce = getattr(orch, "cognitive_engine", None)
        if ce is None:
            ce = CognitiveEngine()

        try:
            ce.setup()
        except Exception as exc:
            record_degradation('recovery', exc)
            logger.error("Setup failed: %s", exc)

        if not ce.lobotomized:
            ServiceContainer.register_instance("cognitive_engine", ce)
            logger.info("✅ Cognitive Engine ONLINE — Safe Mode deactivated")

            try:
                from core.thought_stream import get_emitter
                get_emitter().emit(
                    "System",
                    "Cognitive Connection Re-established",
                    level="success",
                    category="Brain",
                )
            except Exception as exc:
                record_degradation('recovery', exc)
                logger.debug("ThoughtStream emit failed during cognitive retry: %s", exc)

            return True
        else:
            logger.error("❌ Cognitive Retry Failed: Engine still lobotomized after re-wire")
            return False

    except Exception as exc:
        record_degradation('recovery', exc)
        logger.error("Cognitive Retry Exception: %s", exc)
        return False
