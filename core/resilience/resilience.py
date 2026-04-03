"""Aura Zenith Resilience Framework: Circuit Breakers and Retries.
"""
import asyncio
import time
import random
import logging
from typing import Callable, Any
from core.exceptions import CircuitOpenError, LLMError, NetworkError

logger = logging.getLogger(__name__)

from .circuit_breaker import (
    CircuitBreaker as SmartCircuitBreaker, 
    PROMETHEUS_AVAILABLE, 
    CIRCUIT_STATE, 
    CIRCUIT_FAILURES
)

async def retry_with_backoff(func: Callable, max_attempts=5, base_delay=0.1):
    """Exponential backoff retry decorator logic."""
    for attempt in range(max_attempts):
        try:
            return await func()
        except (LLMError, NetworkError) as e:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
            logger.info(f"⏳ Retry {attempt+1}/{max_attempts} after {delay:.2f}s due to {type(e).__name__}")
            await asyncio.sleep(delay)