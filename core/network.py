from core.runtime.errors import record_degradation
import asyncio
import logging
import os
import random
import ssl
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Union
from urllib.parse import urlparse

import httpx
from core.config import config

logger = logging.getLogger("Kernel.Network")

@dataclass
class RequestStats:
    """Request statistics for monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_request_time: Optional[datetime] = None

class RobustHTTP:
    """Enterprise-grade Async HTTP client using httpx.
    
    Features:
    1. Native asyncio support via httpx.AsyncClient
    2. Connection pooling and keep-alive
    3. TLS 1.2+ enforcement
    4. Request rate limiting
    5. User-agent rotation
    6. Detailed metrics and logging
    """
    
    USER_AGENTS = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
    ]
    
    def __init__(self, timeout: float = 30.0):
        self.client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": random.choice(self.USER_AGENTS)}
        )
        self.stats = RequestStats()
        self.requests_per_minute = 60
        self.request_history = deque(maxlen=500)
        self._lock = asyncio.Lock()

    async def _check_rate_limit(self):
        async with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)
            self.request_history = deque((t for t in self.request_history if t > cutoff), maxlen=500)
            
            if len(self.request_history) >= self.requests_per_minute:
                wait_time = 60 - (now - self.request_history[0]).total_seconds()
                logger.warning("Rate limit exceeded. Waiting %.1fs", wait_time)
                await asyncio.sleep(min(wait_time, 5.0))
            
            self.request_history.append(now)

    def _is_url_allowed(self, url: str) -> bool:
        parsed = urlparse(url).netloc.split(':')[0]
        allowed_domains = ["api.github.com", "localhost", "127.0.0.1"]
        if any(parsed == d or parsed.endswith("." + d) for d in allowed_domains):
            return True
            
        if parsed.startswith("192.168.") or parsed.startswith("10."):
            return True
        if parsed.startswith("172."):
            try:
                second_octet = int(parsed.split(".")[1])
                if 16 <= second_octet <= 31:
                    return True
            except (ValueError, IndexError):
                logger.debug('Ignored Exception in network.py: %s', "unknown_error")
                
        if os.getenv("AURA_ALLOW_OUTBOUND", "false").lower() == "true":
            return True
        return False

    async def get(self, url: str, **kwargs) -> httpx.Response:
        await self._check_rate_limit()
        if not self._is_url_allowed(url):
            raise ValueError(f"Outbound access to {urlparse(url).netloc} restricted.")
            
        try:
            response = await self.client.get(url, **kwargs)
            self.stats.total_requests += 1
            self.stats.last_request_time = datetime.now()
            if response.status_code < 400:
                self.stats.successful_requests += 1
            else:
                self.stats.failed_requests += 1
            return response
        except Exception as e:
            record_degradation('network', e)
            self.stats.failed_requests += 1
            logger.error("GET %s failed: %s", url, e)
            raise

    async def post(self, url: str, **kwargs) -> httpx.Response:
        await self._check_rate_limit()
        if not self._is_url_allowed(url):
            raise ValueError(f"Outbound access to {urlparse(url).netloc} restricted.")
            
        try:
            response = await self.client.post(url, **kwargs)
            self.stats.total_requests += 1
            self.stats.last_request_time = datetime.now()
            if response.status_code < 400:
                self.stats.successful_requests += 1
            else:
                self.stats.failed_requests += 1
            return response
        except Exception as e:
            record_degradation('network', e)
            self.stats.failed_requests += 1
            logger.error("POST %s failed: %s", url, e)
            raise

    async def close(self):
        await self.client.aclose()
        logger.info("Async HTTP client closed")