"""core/brain/llm/gemini_adapter.py
Frontier LLM Adapter for Google Gemini API.

Provides PRIMARY-tier reasoning via Gemini 1.5 Pro/Flash while respecting
free tier rate limits to prevent charges and 429 errors.

Free tier limits (as of March 2026):
    - Gemini Pro:        50 RPD,  2 RPM
    - Gemini Flash:      1500 RPD, 15 RPM

Strategy: Use Flash for streaming chat (250 RPD budget), Pro for deep
reasoning only when explicitly requested (100 RPD budget). Automatic
fallback to local models when daily quota is exhausted.
"""
from core.utils.exceptions import capture_and_log
import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx

logger = logging.getLogger("Brain.Gemini")


class DailyRateLimiter:
    """Tracks per-model daily usage and enforces free-tier limits.
    
    Resets at midnight Pacific Time (Google's reset schedule).
    Saves state to disk so restarts don't lose the count.
    """
    
    DEFAULT_LIMITS = {
        "gemini-pro": int(os.environ.get("AURA_GEMINI_RPD_PRO", 2000)),
        "gemini-2.5-flash": int(os.environ.get("AURA_GEMINI_RPD_DEEP", 2000)),
        "gemini-flash-latest": int(os.environ.get("AURA_GEMINI_RPD_FLASH", 10000)),
        "gemini-2.0-flash": int(os.environ.get("AURA_GEMINI_RPD_FLASH", 10000)),
        "gemini-2.5-pro": int(os.environ.get("AURA_GEMINI_RPD_THINKING", 2000)),
    }
    
    # Per-minute limits (High-performance baseline for paid tiers)
    RPM_LIMITS = {
        "gemini-pro": int(os.environ.get("AURA_GEMINI_RPM_PRO", 50)),
        "gemini-2.5-flash": int(os.environ.get("AURA_GEMINI_RPM_DEEP", 50)),
        "gemini-flash-latest": int(os.environ.get("AURA_GEMINI_RPM_FLASH", 500)),
        "gemini-2.0-flash": int(os.environ.get("AURA_GEMINI_RPM_FLASH", 500)),
        "gemini-2.5-pro": int(os.environ.get("AURA_GEMINI_RPM_THINKING", 50)),
    }
    
    def __init__(self, state_path: Optional[str] = None):
        self._counts: Dict[str, int] = defaultdict(int)
        self._reset_date: str = self._today()
        self._state_path = state_path
        self._minute_timestamps: Dict[str, deque] = defaultdict(deque)
        self._backoff_until: Dict[str, float] = {}  # model -> timestamp
        self._cluster_backoff_until: float = 0.0     # All Gemini models
        self._boot_time: float = time.monotonic()
        self.COLD_START_GRACE_S: float = 90.0  # 90s boot grace window
        self._load_state()
    
    def _today(self) -> str:
        """Current date in Pacific time (Google's billing day boundary)."""
        import datetime as dt
        # Approximate Pacific time as UTC-8
        pt = datetime.now(timezone.utc).astimezone(
            dt.timezone(dt.timedelta(hours=-8))
        )
        return pt.strftime("%Y-%m-%d")
    
    def _load_state(self):
        """Load daily counts from disk if available."""
        if self._state_path:
            try:
                import json
                from pathlib import Path
                p = Path(self._state_path)
                if p.exists():
                    data = json.loads(p.read_text())
                    if data.get("date") == self._today():
                        self._counts = defaultdict(int, data.get("counts", {}))
                        self._reset_date = data["date"]
                        logger.info("📊 Loaded Gemini usage: %s", dict(self._counts))
                    else:
                        logger.info("📊 New day — resetting Gemini usage counters")
            except Exception as e:
                logger.debug("Failed to load rate limiter state: %s", e)
    
    def _save_state(self):
        """Persist daily counts to disk."""
        if self._state_path:
            try:
                from pathlib import Path
                Path(self._state_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self._state_path).write_text(json.dumps({
                    "date": self._reset_date,
                    "counts": dict(self._counts),
                }))
            except Exception as e:
                capture_and_log(e, {'module': __name__})
    
    def _maybe_reset(self):
        """Reset counters if it's a new day."""
        today = self._today()
        if today != self._reset_date:
            logger.info("📊 Daily reset: Gemini quotas refreshed (%s → %s)", 
                       self._reset_date, today)
            self._counts.clear()
            self._reset_date = today
            self._save_state()
    
    def _check_rpm(self, model: str) -> bool:
        """Check if we're within per-minute rate limit."""
        now = time.monotonic()
        rpm_limit = self.RPM_LIMITS.get(model, 3)
        
        # Clean old timestamps with O(1) amortized popleft
        dq = self._minute_timestamps[model]
        while dq and now - dq[0] >= 60:
            dq.popleft()
        
        return len(dq) < rpm_limit
    
    def _record_rpm(self, model: str):
        """Record a call for RPM tracking."""
        self._minute_timestamps[model].append(time.monotonic())
    
    def mark_429(self, model: str, retry_after: float = 60.0):
        """Mark a 429 backoff for this model."""
        self._backoff_until[model] = time.monotonic() + retry_after
        logger.warning("🚫 Gemini %s: 429 backoff for %.0fs", model, retry_after)
    
    def is_backed_off(self, model: str) -> bool:
        """Check if this model is in a 429 backoff period."""
        now = time.monotonic()
        
        # Check cluster-level backoff first
        if now < self._cluster_backoff_until:
            return True
            
        until = self._backoff_until.get(model, 0)
        if now < until:
            return True
        return False

    def mark_cluster_429(self, retry_after: float = 5.0):
        """Mark a brief cooldown for ALL Gemini models to prevent project quota thumping."""
        self._cluster_backoff_until = time.monotonic() + retry_after
        logger.info("🛡️ Gemini Cluster: Entering %.0fs project-level backoff", retry_after)
    
    def is_cold_start(self) -> bool:
        """True during the first 90s after boot — protect against startup RPM storms."""
        return (time.monotonic() - self._boot_time) < self.COLD_START_GRACE_S

    def can_call(self, model: str, is_background: bool = False, priority: float = 0.5) -> bool:
        """Check if we have remaining quota for this model.
        Background calls are prioritized lower to save credits for chat.
        """
        self._maybe_reset()
        
        # Block non-urgent background calls during boot grace window
        if is_background and self.is_cold_start():
            logger.debug("⏳ Cold-start grace: deferring background Gemini call for %s", model)
            return False
            
        # Check 429 backoff first
        if self.is_backed_off(model):
            return False
        
        # Check RPM
        if not self._check_rpm(model):
            logger.debug("⏳ Gemini %s: RPM limit reached, waiting...", model)
            return False
        
        # Check daily limit
        limit = self.DEFAULT_LIMITS.get(model, 80)
        
        # [Pipeline Hardening] Conservative background limit: Stop using Gemini for background tasks
        # once we hit the preservation threshold (default 30%), preserving 70% for the User.
        preservation_threshold = float(os.environ.get("AURA_GEMINI_BACKGROUND_THRESHOLD", 0.3))
        if is_background and self._counts[model] > (limit * preservation_threshold):
            logger.debug("📉 Preserving Gemini %s quota: background call diverted (threshold: %.1f).", model, preservation_threshold)
            return False

        return self._counts[model] < limit

    def reset_manual(self):
        """Force a reset of all daily counters and backoffs."""
        self._counts.clear()
        self._backoff_until.clear()
        self._reset_date = self._today()
        self._save_state()
        logger.info("📊 Gemini rate limits manually RESET.")
    
    def record_call(self, model: str):
        """Record a successful API call."""
        self._maybe_reset()
        self._counts[model] += 1
        self._record_rpm(model)
        remaining = self.DEFAULT_LIMITS.get(model, 80) - self._counts[model]
        if remaining <= 10:
            logger.warning("⚠️ Gemini %s: %d calls remaining today", model, remaining)
        self._save_state()
    
    def get_usage(self) -> Dict:
        """Return current usage stats."""
        self._maybe_reset()
        return {
            model: {
                "used": self._counts.get(model, 0),
                "limit": limit,
                "remaining": limit - self._counts.get(model, 0),
            }
            for model, limit in self.DEFAULT_LIMITS.items()
        }


from core.resilience.factory import circuit_breaker

class GeminiAdapter:
    """Adapter for Google Gemini API — slots into IntelligentLLMRouter as PRIMARY tier.
    
    Provides both streaming (generate_text_stream_async) and non-streaming (call)
    interfaces so the router's race_think_stream and think_stream both work.
    """
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    # Gemini 2.x/2.5: Flash for speed, 2.5-Flash for deep, 2.5-Pro for thinking
    CHAT_MODEL = "gemini-2.0-flash"
    DEEP_MODEL = "gemini-2.5-flash"  # Stable deep fallback (GA June 2025)
    THINKING_MODEL = "gemini-2.5-pro"  # Best reasoning model (GA March 2025)
    
    def __init__(self, api_key: str, model: str = None, 
                 rate_limiter: Optional[DailyRateLimiter] = None,
                 timeout: float = 120.0):
        self.api_key = api_key
        self.model = model or self.CHAT_MODEL
        self.timeout = timeout
        self.rate_limiter = rate_limiter or DailyRateLimiter()
        self._client: Optional[httpx.AsyncClient] = None
        logger.info("✨ GeminiAdapter initialized: model=%s", self.model)
    
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                headers={"Content-Type": "application/json"},
            )
        return self._client
    
    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def _handle_error(self, response: httpx.Response):
        """Standardized error handling for Gemini API."""
        error_body = await response.aread()
        text = error_body.decode('utf-8', errors='replace')
        
        if response.status_code == 429:
            retry_after = self._parse_retry_after(error_body)
            # Permanent Quota exhaustion detection
            if "quota" in text.lower():
                logger.error("🚫 Gemini %s: DAILY QUOTA EXHAUSTED", self.model)
                self.rate_limiter._counts[self.model] = self.rate_limiter.DEFAULT_LIMITS.get(self.model, 80)
                self.rate_limiter._save_state()
            else:
                self.rate_limiter.mark_429(self.model, retry_after)
                # Phase 39: Protect project quota by pulsing a cluster-wide backoff
                self.rate_limiter.mark_cluster_429(min(retry_after, 5.0))
            
            msg = f"🚫 Gemini {self.model}: 429 rate limited, backoff {retry_after:.0f}s"
            logger.warning(msg)
            raise Exception(msg)
        else:
            msg = f"Gemini API error {response.status_code}: {text[:500]}"
            logger.error(msg)
            raise Exception(msg)

    def _parse_retry_after(self, error_body: bytes) -> float:
        """Extract retry-after duration from a 429 error response."""
        try:
            text = error_body.decode('utf-8', errors='replace')
            # Look for "Please retry in Xs" pattern
            import re
            match = re.search(r'retry in (\d+\.?\d*)', text)
            if match:
                return float(match.group(1))
        except Exception as e:
            capture_and_log(e, {'module': __name__})
        return 60.0  # Default 60s backoff
    
    @circuit_breaker(service_name="gemini-api")
    async def generate_text_stream_async(
        self, prompt: str, 
        system_prompt: Optional[str] = None,
        cancel_event=None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream tokens from Gemini — compatible with LLMRouter's race_think_stream."""
        is_background = kwargs.get("is_background", False)
        if not self.rate_limiter.can_call(self.model, is_background=is_background):
            msg = f"🚫 Gemini {self.model} local rate limited"
            logger.warning(msg)
            raise Exception(msg)
        
        contents = []
        system_instruction = None
        
        if system_prompt:
            system_instruction = {"parts": [{"text": system_prompt}]}
        
        parts = kwargs.get("parts")
        if not parts:
            # Guard: prompt can be empty when user input is in system_prompt
            if prompt and prompt.strip():
                parts = [{"text": prompt}]
            elif system_prompt:
                # Move system_prompt to user content so Gemini has valid data
                parts = [{"text": system_prompt}]
                system_instruction = None  # Don't double-send
            else:
                logger.warning("⚠️ Gemini stream: No prompt or system_prompt provided")
                return
            
        # Final guard: filter out any parts with empty/None text
        parts = [p for p in parts if p.get("text")]
        if not parts:
            logger.warning("⚠️ Gemini stream: All parts were empty after filtering")
            return
            
        contents.append({
            "role": "user",
            "parts": parts
        })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.8),
                "maxOutputTokens": kwargs.get("max_tokens", 2048),
                "topP": 0.95,
            },
        }
        
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        
        url = f"{self.BASE_URL}/models/{self.model}:streamGenerateContent?key={self.api_key}&alt=sse"
        
        client = self._get_client()
        tokens_yielded = 0
        t0_stream = time.monotonic()
        
        try:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    await self._handle_error(response)
                
                self.rate_limiter.record_call(self.model)
                
                async for line in response.aiter_lines():
                    if cancel_event and cancel_event.is_set():
                        return
                    
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    
                    json_str = line[6:]  # Strip "data: " prefix
                    if json_str == "[DONE]":
                        return
                    
                    try:
                        chunk = json.loads(json_str)
                        # Extract token usage from any chunk (usually final)
                        usage = chunk.get("usageMetadata")
                        if usage:
                            # Yield metadata as a special non-string chunk
                            yield {
                                "type": "metadata",
                                "tokens_used": usage.get("totalTokenCount", 0),
                                "prompt_tokens": usage.get("promptTokenCount", 0),
                                "completion_tokens": usage.get("candidatesTokenCount", 0)
                            }
                            
                        candidates = chunk.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                text = part.get("text", "")
                                if text:
                                    yield text
                                    tokens_yielded += 1
                    except json.JSONDecodeError:
                        continue
            
            if tokens_yielded == 0:
                logger.warning("⚠️ Gemini stream yielded 0 tokens")
            else:
                # [Phase 18] Record Metabolic Cost for streaming
                try:
                    from core.ops.metabolic_monitor import get_cost_tracker
                    get_cost_tracker().record_operation(
                        op_type="gemini_stream",
                        tokens=tokens_yielded,
                        duration_s=time.monotonic() - t0_stream, # Need to define t0_stream
                        model_tier="PRIMARY" if self.model == self.DEEP_MODEL else "SECONDARY"
                    )
                except Exception as _e:
                    logger.debug('Ignored Exception in gemini_adapter.py: %s', _e)
                
        except httpx.TimeoutException:
            logger.warning("Gemini stream timed out after %.0fs", self.timeout)
        except Exception as e:
            logger.error("Gemini stream error: %s", e)

    @circuit_breaker(service_name="gemini-api")
    async def call(
        self, prompt: str, **kwargs
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Non-streaming call — compatible with LLMRouter's _think_internal fallback."""
        is_background = kwargs.get("is_background", False)
        if not self.rate_limiter.can_call(self.model, is_background=is_background):
            logger.warning("🚫 Gemini %s rate limited", self.model)
            return False, "", {"error": "Rate limit reached"}
        
        contents = []
        system_instruction = None
        
        sys_prompt = kwargs.get("system_prompt") or kwargs.get("system", "")
        if sys_prompt:
            system_instruction = {"parts": [{"text": sys_prompt}]}
        
        parts = kwargs.get("parts")
        if not parts:
            # Handle standard 'messages' format if provided
            messages = kwargs.get("messages")
            if messages and isinstance(messages, list):
                # Map to Gemini-style contents
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        # Convert system message to instruction or just part of the flow
                        system_instruction = {"parts": [{"text": content}]}
                    else:
                        contents.append({
                            "role": "user" if role == "user" else "model",
                            "parts": [{"text": content}]
                        })
                
                # If we have contents, we skip the prompt-based construction below
                if contents:
                    parts = True # sentinel to skip next block
            
        if not parts:
            # Guard: prompt can be empty when user input is in system_prompt
            if prompt and prompt.strip():
                parts = [{"text": prompt}]
            elif sys_prompt:
                # Move system_prompt to user content so Gemini has valid data
                parts = [{"text": sys_prompt}]
                system_instruction = None  # Don't double-send
            else:
                return False, "", {"error": "No content to send to Gemini"}
        
        if contents:
            # Already built via messages
            pass
        else:
            # Final guard: filter out any parts with empty/None text
            parts = [p for p in parts] if isinstance(parts, list) else []
            parts = [p for p in parts if p.get("text")]
            if not parts:
                return False, "", {"error": "All parts were empty after filtering"}
                
            contents.append({
                "role": "user",
                "parts": parts
            })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.8),
                "maxOutputTokens": kwargs.get("max_tokens", 2048),
                "topP": 0.95,
            },
        }
        
        # [v11.0 HARDENING] Native JSON Mode & Schema Enforcement
        schema = kwargs.get("schema")
        if schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            if isinstance(schema, dict):
                payload["generationConfig"]["responseSchema"] = schema
            logger.info("🎯 Gemini: JSON Mode ACTIVE with schema enforcement.")

        if system_instruction:
            payload["systemInstruction"] = system_instruction
        
        url = f"{self.BASE_URL}/models/{self.model}:generateContent?key={self.api_key}"
        
        metadata = {
            "model": self.model,
            "endpoint": "gemini_frontier",
            "latency_ms": 0,
        }
        
        t0 = time.monotonic()
        client = self._get_client()
        
        try:
            response = await client.post(url, json=payload)
            metadata["latency_ms"] = int((time.monotonic() - t0) * 1000)
            
            if response.status_code != 200:
                try:
                    await self._handle_error(response)
                except Exception as e:
                    return False, "", {"error": str(e)}
            
            self.rate_limiter.record_call(self.model)
            
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return False, "", {"error": "No candidates in response"}
            
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            
            # Extract token usage
            usage = data.get("usageMetadata", {})
            metadata["tokens_used"] = usage.get("totalTokenCount", 0)
            metadata["prompt_tokens"] = usage.get("promptTokenCount", 0)
            metadata["completion_tokens"] = usage.get("candidatesTokenCount", 0)
            
            if not text:
                return False, "", {"error": "Empty response from Gemini"}
            
            # [Phase 18] Record Metabolic Cost
            try:
                from core.ops.metabolic_monitor import get_cost_tracker
                get_cost_tracker().record_operation(
                    op_type="gemini_call",
                    tokens=metadata.get("tokens_used", 0),
                    duration_s=(time.monotonic() - t0),
                    model_tier="PRIMARY" if self.model == self.DEEP_MODEL else "SECONDARY"
                )
            except Exception as _e:
                logger.debug('Ignored Exception in gemini_adapter.py: %s', _e)
            
            return True, text, metadata
            
        except httpx.TimeoutException:
            metadata["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return False, "", {"error": f"Timeout after {self.timeout}s"}
        except Exception as e:
            metadata["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return False, "", {"error": str(e)}

    async def generate_text_async(
        self, prompt: str, 
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Compatibility method for code paths that call generate_text_async."""
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        success, text, meta = await self.call(prompt, **kwargs)
        return text if success else ""

    async def generate(
        self, prompt: str,
        system_prompt: str = "",
        **kwargs
    ) -> str:
        """Compatibility method for LLM router's generate()."""
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        success, text, meta = await self.call(prompt, **kwargs)
        return text if success else ""

    async def think(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Unified interface for non-chat callers.
        Returns the simplified string result from Gemini.
        """
        # Distinguish between prompt string and message list
        if not prompt and "messages" in kwargs:
            msgs = kwargs.get("messages", [])
            if msgs and isinstance(msgs, list):
                prompt = msgs[-1].get("content", "")
                if not system_prompt:
                    system_prompt = next((m["content"] for m in msgs if m["role"] == "system"), None)

        if not prompt:
            return None

        # Call generate with the prompt
        text = await self.generate(prompt, system_prompt=system_prompt, **kwargs)
        return text if text and text.strip() else None

    async def unload_models(self):
        """No-op for API models — nothing to unload."""
        pass
    
    async def generate_stream(self, prompt: str, system_prompt: str = None, **kwargs):
        """Alias for generate_text_stream_async — matches the interface expected by LLMRouter.think_stream."""
        async for chunk in self.generate_text_stream_async(prompt, system_prompt=system_prompt, **kwargs):
            yield chunk

    def get_usage_stats(self) -> Dict:
        """Return human-readable usage stats for the UI."""
        return self.rate_limiter.get_usage()
