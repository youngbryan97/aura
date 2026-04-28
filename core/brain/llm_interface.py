# core/brain/llm_interface.py
from core.runtime.errors import record_degradation
from typing import Any, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

class LLMInterface:
    """
    Minimal adapter interface for an LLM client.
    Implement generate_sync or generate_async depending on your client.
    """

    def __init__(self):
        # optional configuration
        self.default_opts: Dict[str, Any] = {}
        # SAFE-01: Bound thread pool to prevent system-wide exhaustion
        self._executor = ThreadPoolExecutor(max_workers=10)

    def generate_sync(self, prompt: str, **opts) -> str:
        """
        Blocking variant. Override if your client is sync.
        """
        raise NotImplementedError()

    async def generate(self, prompt: str, **opts) -> str:
        """
        Async variant. Default wrapper calls generate_sync in executor.
        Override with an async API if available.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, 
            lambda: self.generate_sync(prompt, **{**self.default_opts, **opts})
        )

async def call_llm_with_timeout(llm_call_coro, timeout: float = 10.0, fallback_fn=None):
    """
    Stabilization Helper: Wraps LLM calls with a hard timeout and optional fallback.
    Prevents UI 'Thinking' freeze when endpoints are slow or stalled.
    """
    import logging
    logger = logging.getLogger("Aura.LLMTimeout")
    
    try:
        return await asyncio.wait_for(llm_call_coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("LLM call timed out after %.1fs", timeout)
    except Exception as e:
        record_degradation('llm_interface', e)
        logger.exception("LLM call exception: %s", e)

    # try fallback
    if fallback_fn:
        try:
            if asyncio.iscoroutinefunction(fallback_fn):
                return await fallback_fn()
            else:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, fallback_fn)
        except Exception:
            logger.exception("Fallback LLM also failed")
    return ""  # safe empty string
