import asyncio
import time
from typing import Dict, Optional

class TokenBucketRateLimiter:
    """Token Bucket rate limiter for managing API throughput.
    
    Ensures that calls to external services (OpenAI, Google) do not
    exceed provider tier limits.
    """
    def __init__(self, rpm: int = 60, burst: int = 5):
        self.rpm = rpm
        self.capacity = burst
        self.tokens = float(burst)
        self.last_fill = time.time()
        self.refill_rate = rpm / 60.0 # tokens per second
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens. Blocks until available (Audit-39 fix)."""
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                # Calculate wait time based on missing tokens
                wait_time = max(0.01, (tokens - self.tokens) / self.refill_rate)
            
            # Sleep OUTSIDE the lock to allow other tasks to progress
            await asyncio.sleep(wait_time)

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_fill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_fill = now

class RateLimitManager:
    """Registry for tiered rate limiters."""
    _limiters: Dict[str, TokenBucketRateLimiter] = {}

    @classmethod
    def get_limiter(cls, key: str, rpm: int = 60, burst: int = 5) -> TokenBucketRateLimiter:
        if key not in cls._limiters:
            cls._limiters[key] = TokenBucketRateLimiter(rpm, burst)
        return cls._limiters[key]

# Singleton helper
async def limit_rate(key: str, rpm: int = 60, burst: int = 5):
    limiter = RateLimitManager.get_limiter(key, rpm, burst)
    await limiter.acquire()
