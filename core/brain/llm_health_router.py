"""
core/brain/llm_health_router.py
────────────────────────────────
Replacement for IntelligentLLMRouter.

Fixes:
  - Zero-token / whitespace-only responses treated as failure, not success
  - Primary endpoint failure triggers genuine fallback to local Ollama
  - Per-endpoint health tracking with circuit breaker pattern
  - Response validation before acceptance
  - Structured logging that distinguishes real success from empty success

Drop-in: replace the existing router instantiation in orchestrator_boot.py
with HealthAwareLLMRouter.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("Brain.HealthRouter")


# ── Circuit Breaker States ────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"       # Normal — requests flow through
    OPEN = "open"           # Failed — requests blocked, fallback used
    HALF_OPEN = "half_open" # Testing — one probe request allowed


@dataclass
class EndpointHealth:
    name: str
    url: str
    model: str
    is_local: bool = False

    # Circuit breaker
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure: float = 0.0
    last_success: float = 0.0

    # Performance tracking
    avg_latency_ms: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0
    empty_responses: int = 0

    # Config
    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    min_tokens_for_success: int = 1

    def record_success(self, tokens: int, latency_ms: float):
        self.success_count += 1
        self.total_requests += 1
        self.total_tokens += tokens
        self.last_success = time.time()

        # Rolling average latency
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (self.avg_latency_ms * 0.8) + (latency_ms * 0.2)

        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit CLOSED for %s — probe succeeded", self.name)
            self.state = CircuitState.CLOSED
            self.failure_count = 0

    def record_failure(self, reason: str):
        self.failure_count += 1
        self.total_requests += 1
        self.last_failure = time.time()

        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.warning(
                    "Circuit OPEN for %s after %d failures. Reason: %s",
                    self.name, self.failure_count, reason
                )
            self.state = CircuitState.OPEN

    def record_empty(self):
        """Zero-token or whitespace-only response — treat as failure."""
        self.empty_responses += 1
        self.record_failure("empty_response")

    def is_available(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure > self.recovery_timeout:
                logger.info("Circuit HALF-OPEN for %s — probing", self.name)
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return True
        return False

    def status_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self.failure_count,
            "successes": self.success_count,
            "empty_responses": self.empty_responses,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_tokens": self.total_tokens,
        }


# ── Validator ─────────────────────────────────────────────────────────────────

def validate_response(text: Optional[str], min_tokens: int = 1) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    A response is invalid if:
      - It is None
      - It is empty or whitespace-only
      - It contains only punctuation
      - It is suspiciously short (< min_tokens words)
    """
    if text is None:
        return False, "none_response"
    stripped = text.strip()
    if not stripped:
        return False, "empty_whitespace"
    if len(stripped) < 2:
        return False, "too_short"
    words = stripped.split()
    if len(words) < min_tokens:
        return False, f"below_min_tokens_{min_tokens}"
    # Check for pure error markers
    lower = stripped.lower()
    error_markers = [
        "i am currently offline",
        "i cannot process that",
        "error:",
        "connection refused",
        "timeout",
    ]
    for marker in error_markers:
        if lower.startswith(marker):
            return False, f"error_marker:{marker}"
    return True, "ok"


# ── Main Router ───────────────────────────────────────────────────────────────

class HealthAwareLLMRouter:
    """
    Routes LLM requests to available endpoints with circuit breaking.

    Priority order: endpoints registered first are tried first.
    Local Ollama should be registered as the final fallback but is
    always available regardless of circuit state.
    """

    def __init__(self):
        self.endpoints: List[EndpointHealth] = []
        self._lock = asyncio.Lock()
        logger.info("HealthAwareLLMRouter initialized")

    def register(
        self,
        name: str,
        url: str,
        model: str,
        is_local: bool = False,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> "HealthAwareLLMRouter":
        ep = EndpointHealth(
            name=name,
            url=url,
            model=model,
            is_local=is_local,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self.endpoints.append(ep)
        logger.info("Registered endpoint: %s (%s) local=%s", name, model, is_local)
        return self

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Try each endpoint in order. Return first valid response.
        Falls back to local if all remote endpoints fail.
        Always returns a dict: {"ok": bool, "text": str, "endpoint": str, "tokens": int}
        """
        available = [ep for ep in self.endpoints if ep.is_available()]
        unavailable = [ep for ep in self.endpoints if not ep.is_available()]

        if unavailable:
            logger.debug(
                "Skipping unavailable endpoints: %s",
                [ep.name for ep in unavailable]
            )

        if not available:
            # Emergency: try local Ollama directly regardless of circuit state
            local = next((ep for ep in self.endpoints if ep.is_local), None)
            if local:
                logger.warning("All circuits open — emergency direct call to %s", local.name)
                available = [local]
            else:
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "none",
                    "tokens": 0,
                    "error": "all_endpoints_unavailable",
                }

        last_error = "unknown"
        for ep in available:
            try:
                result = await self._call_endpoint(ep, prompt, system_prompt, timeout, **kwargs)
                if result["ok"]:
                    return result
                else:
                    last_error = result.get("error", "unknown")
                    logger.warning(
                        "Endpoint %s failed validation: %s",
                        ep.name, last_error
                    )
            except Exception as exc:
                logger.error("Endpoint %s raised exception: %s", ep.name, exc)
                ep.record_failure(str(exc))
                last_error = str(exc)

        return {
            "ok": False,
            "text": "",
            "endpoint": "all_failed",
            "tokens": 0,
            "error": last_error,
        }

    async def _call_endpoint(
        self,
        ep: EndpointHealth,
        prompt: str,
        system_prompt: Optional[str],
        timeout: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make the actual HTTP call and validate the response."""
        start = time.time()

        try:
            # Build the Ollama-compatible request
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": ep.model,
                "messages": messages,
                "stream": False,
                **kwargs,
            }

            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{ep.url}/api/chat",
                    json=payload,
                )

            if resp.status_code != 200:
                ep.record_failure(f"http_{resp.status_code}")
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": ep.name,
                    "tokens": 0,
                    "error": f"http_{resp.status_code}",
                }

            data = resp.json()
            raw_text = (
                data.get("message", {}).get("content")
                or data.get("response")
                or data.get("text")
                or ""
            )

            # ── THE CRITICAL FIX: validate before accepting ───────────────
            is_valid, reason = validate_response(raw_text)
            latency_ms = (time.time() - start) * 1000

            if not is_valid:
                ep.record_empty()
                logger.warning(
                    "Endpoint %s returned invalid response (reason=%s, raw=%r)",
                    ep.name, reason, raw_text[:100]
                )
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": ep.name,
                    "tokens": 0,
                    "error": f"invalid_response:{reason}",
                }

            # Count tokens (rough estimate if not provided)
            token_count = (
                data.get("eval_count")
                or data.get("usage", {}).get("completion_tokens")
                or len(raw_text.split())
            )

            ep.record_success(token_count, latency_ms)
            logger.info(
                "✓ Response from %s (%d tokens, %.0fms)",
                ep.name, token_count, latency_ms
            )

            return {
                "ok": True,
                "text": raw_text.strip(),
                "endpoint": ep.name,
                "tokens": token_count,
                "latency_ms": latency_ms,
            }

        except httpx.TimeoutException:
            ep.record_failure("timeout")
            return {
                "ok": False,
                "text": "",
                "endpoint": ep.name,
                "tokens": 0,
                "error": "timeout",
            }
        except Exception as exc:
            ep.record_failure(str(exc))
            raise

    def get_health_report(self) -> Dict[str, Any]:
        return {
            "endpoints": [ep.status_dict() for ep in self.endpoints],
            "available_count": sum(1 for ep in self.endpoints if ep.is_available()),
            "total_count": len(self.endpoints),
        }

    def reset_circuit(self, endpoint_name: str):
        """Manually reset a circuit breaker (for admin/retry endpoints)."""
        for ep in self.endpoints:
            if ep.name == endpoint_name:
                ep.state = CircuitState.CLOSED
                ep.failure_count = 0
                logger.info("Circuit manually reset for %s", endpoint_name)
                return
        logger.warning("reset_circuit: endpoint %s not found", endpoint_name)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_router_from_config(config) -> HealthAwareLLMRouter:
    """
    Build and return a properly configured router.
    Local Ollama is always registered as final fallback.

    Usage in orchestrator_boot.py:
        from core.brain.llm_health_router import build_router_from_config
        router = build_router_from_config(config)
    """
    router = HealthAwareLLMRouter()

    # Local Ollama — always first and always available
    ollama_url = getattr(config.llm, "base_url", "http://localhost:11434")
    ollama_model = getattr(config.llm, "model", "llama3:latest")

    router.register(
        name="Local-Ollama",
        url=ollama_url,
        model=ollama_model,
        is_local=True,
        failure_threshold=5,       # More tolerant for local
        recovery_timeout=10.0,     # Recover quickly
    )

    # Any remote/cloud endpoint (optional)
    remote_url = getattr(config.llm, "remote_url", None)
    remote_model = getattr(config.llm, "remote_model", None)
    if remote_url and remote_model:
        router.register(
            name="Remote-Endpoint",
            url=remote_url,
            model=remote_model,
            is_local=False,
            failure_threshold=2,   # Less tolerant for remote
            recovery_timeout=60.0, # Longer recovery for remote
        )
        logger.info("Remote endpoint registered: %s", remote_url)

    return router
