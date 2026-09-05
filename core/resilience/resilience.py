"""Aura Zenith Resilience Framework: Circuit Breakers and Retries.
"""
import asyncio
import logging
import random
from collections.abc import Callable

from core.exceptions import LLMError, NetworkError

logger = logging.getLogger(__name__)


async def retry_with_backoff(func: Callable, max_attempts=5, base_delay=0.1):
    """Exponential backoff retry decorator logic."""
    for attempt in range(max_attempts):
        try:
            return await func()
        except (LLMError, NetworkError) as e:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
            logger.info("⏳ Retry %s/%s after %ss due to %s", attempt+1, max_attempts, f"{delay:.2f}", type(e).__name__)
            await asyncio.sleep(delay)