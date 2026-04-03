"""core/utils/safe_llm.py — Timeout-Guarded LLM Wrapper

All LLM calls should go through this wrapper to prevent hung local inference
from blocking the entire system indefinitely.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("Aura.SafeLLM")


@dataclass
class SafeThought:
    """Fallback thought object returned on timeout or error."""
    content: str = ""
    timed_out: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0


async def safe_think(brain, prompt: str, timeout: float = 60.0,
                     fallback: str = "", **kwargs) -> Any:
    """Timeout-guarded wrapper around brain.think().
    
    Args:
        brain: CognitiveEngine or any object with an async think() method
        prompt: The prompt to send
        timeout: Maximum seconds to wait (default 60)
        fallback: Text to return as content if timeout/error occurs
        **kwargs: Passed through to brain.think()
    
    Returns:
        The thought result from brain.think(), or a SafeThought on failure
    """
    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            brain.think(prompt, **kwargs),
            timeout=timeout
        )
        latency = (time.monotonic() - start) * 1000
        if latency > 30000:  # Log slow calls (>30s)
            logger.warning("Slow LLM call: %.1fms for prompt: %.60s...", latency, prompt)
        return result
    except asyncio.TimeoutError:
        latency = (time.monotonic() - start) * 1000
        logger.error("⏰ LLM TIMEOUT after %.1fs — prompt: %.80s...", timeout, prompt)
        return SafeThought(
            content=fallback,
            timed_out=True,
            error=f"LLM call timed out after {timeout}s",
            latency_ms=latency,
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        logger.error("LLM call failed after %.1fms: %s", latency, e)
        return SafeThought(
            content=fallback,
            timed_out=False,
            error=str(e),
            latency_ms=latency,
        )