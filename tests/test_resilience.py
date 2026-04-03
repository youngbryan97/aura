################################################################################

"""
tests/test_resilience.py
────────────────────────
Verify async resilience primitives.
"""

import pytest
import asyncio
import time
from infrastructure.resilience import (
    retry_async, AsyncCircuitBreaker, CircuitBreaker, CircuitState,
    RetryExhausted, resilient
)

@pytest.mark.asyncio
async def test_retry_async_success():
    call_count = 0
    async def success_fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    res = await retry_async(success_fn)
    assert res == "ok"
    assert call_count == 1

@pytest.mark.asyncio
async def test_retry_async_failure_capture():
    call_count = 0
    async def fail_fn():
        nonlocal call_count
        call_count += 1
        raise ValueError("Boom")

    with pytest.raises(RetryExhausted):
        await retry_async(fail_fn, attempts=3, base_delay=0.01)
    
    assert call_count == 3

@pytest.mark.asyncio
async def test_circuit_breaker_logic():
    cb = CircuitBreaker("test-breaker", failure_threshold=2, recovery_timeout=0.1)
    
    # Needs 2 failures to trip
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    
    # Wait for recovery
    await asyncio.sleep(0.15)
    
    # State check should trigger transition to half-open
    assert cb.state == CircuitState.HALF_OPEN
    
    # Success should close it
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb._failures == 0

@pytest.mark.asyncio
async def test_resilient_decorator():
    """Verify decorator works on async functions."""
    
    @resilient("test-component", retry_attempts=2)
    async def unstable_api(succeed):
        if not succeed:
            raise ValueError("Fail")
        return "Success"
        
    # Should succeed
    assert await unstable_api(True) == "Success"
    
    # Should fail after retries (raises RetryExhausted)
    with pytest.raises(RetryExhausted):
        await unstable_api(False)


##
