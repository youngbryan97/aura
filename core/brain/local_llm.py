import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from core.config import config

logger = logging.getLogger("Aura.LocalLLM")

# Configurable timeouts (Increased for Llama 3.1 8b)
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 300.0  # 5 minutes to allow slow model loading
_DEFAULT_TIMEOUT = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=30.0, pool=10.0)

# ── Model Tier Defaults (v27: Upgraded for M5/64GB) ─────────────────────────
# Old values (v26): num_ctx=4096, num_predict=512 — cripplingly low
_MODEL_TIER_DEFAULTS = {
    "coding":  {"num_ctx": 16384, "num_predict": 4096, "temperature": 0.4},
    "chat":    {"num_ctx": 16384, "num_predict": 2048, "temperature": 0.7},
    "summary": {"num_ctx": 8192,  "num_predict": 1024, "temperature": 0.3},
    "default": {"num_ctx": 16384, "num_predict": 2048, "temperature": 0.7},
}

# ── Thought Circulation (from gemini-skills) ────────────────────────────────
_CODING_KEYWORDS = {
    "code", "function", "class", "implement", "debug", "fix", "refactor",
    "script", "api", "endpoint", "database", "sql", "query", "test",
    "error", "bug", "traceback", "exception", "compile", "build",
    "deploy", "docker", "git", "commit", "merge", "python", "javascript",
    "typescript", "rust", "java", "create a", "write a", "modify",
    "html", "css", "react", "fastapi", "flask", "django",
}

_THOUGHT_CIRCULATION_DIRECTIVE = """Before responding, reason step-by-step in <think> tags about:
1. What the user is asking for and any implicit requirements
2. What files, APIs, or systems are involved
3. Edge cases, error handling, and potential pitfalls
4. Your exact implementation approach
Then provide your implementation outside the think tags."""


def detect_task_tier(prompt: str, system_prompt: str = "") -> str:
    """Detect the task tier from the prompt content."""
    combined = (prompt + " " + (system_prompt or "")).lower()
    for keyword in _CODING_KEYWORDS:
        if keyword in combined:
            return "coding"
    # Check for summary/compression tasks
    if any(kw in combined for kw in ["summarize", "compress", "distill", "snapshot"]):
        return "summary"
    return "chat"


class LocalBrain:
    """Sovereign Intelligence Bridge connecting to local Ollama instance.
    v6.1: Retry logic, proper client lifecycle, configurable timeouts.
    """
    
    def __init__(self, model_name: str = None):
        self.base_url = config.llm.base_url
        self.model = model_name or config.llm.model
        self.timeout = getattr(config.llm, "timeout", 300)
        self._client = None
        self._warmed = False
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until = 0.0
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=_DEFAULT_TIMEOUT
            )
        return self._client

    # --- Lifecycle ---
    async def close(self):
        """Close the underlying httpx client to release connections."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # --- Health Checks ---
    def check_health(self) -> bool:
        """Sovereign health check (Synchronous — uses httpx sync client)."""
        try:
            with httpx.Client(base_url=self.base_url, timeout=3) as client:
                response = client.get("/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def check_health_async(self) -> bool:
        """Sovereign health check (Async)."""
        try:
            response = await self.client.get("/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False
    
    async def warmup(self) -> bool:
        """v6.0: Pre-warm the model by sending a minimal generate request.
        This loads the model into GPU memory for faster first-response.
        """
        if self._warmed:
            return True
        try:
            logger.info("🔥 Warming up model: %s", self.model)
            response = await self.client.post("/api/generate", json={
                "model": self.model,
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 1}
            }, timeout=120)
            response.raise_for_status()
            self._warmed = True
            logger.info("🔥 Model %s warmed up successfully", self.model)
            return True
        except Exception as e:
            logger.warning("Model warmup failed: %s", e)
            return False
    
    # --- Circuit Breaker ---
    def _check_circuit(self) -> bool:
        """Check if circuit breaker allows the call."""
        if self._circuit_open:
            if time.time() < self._circuit_open_until:
                logger.warning("Circuit breaker OPEN — skipping LLM call")
                return False
            # Half-open: allow one attempt
            self._circuit_open = False
        return True
    
    def _record_success(self):
        """Record a successful call."""
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until = 0.0
    
    def _record_failure(self):
        """Record a failed call. Opens circuit after 5 consecutive failures."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            self._circuit_open = True
            self._circuit_open_until = time.time() + 30.0  # 30s recovery
            logger.error("Circuit breaker OPENED after %s failures. Cooldown: 30s", self._consecutive_failures)
    
    # --- Generation ---
    # --- Helper: DeepSeek Think Extraction ---
    THOUGHT_STREAM_PREFIX = "__THOUGHT__:"
    THOUGHT_STREAM_SUFFIX = ":__ENDTHOUGHT__"

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> blocks from static text (legacy compat)."""
        return self._extract_think_segments(text)[0]

    @staticmethod
    def _extract_think_segments(text: str) -> tuple[str, str]:
        """Extract thinking content and cleaned response from <think> tagged text.

        Returns:
            (cleaned_response, thought_content)
        """
        import re
        thoughts: list[str] = []
        for m in re.finditer(r'<think>(.*?)</think>', text, flags=re.DOTALL):
            thought_text = m.group(1).strip()
            if thought_text:
                thoughts.append(thought_text)
        # Remove balanced tags
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove stray closing tags if any
        cleaned = cleaned.replace('</think>', '')
        # Remove stray opening tags if at the end (incomplete thought)
        cleaned = cleaned.replace('<think>', '')
        return cleaned.strip(), "\n\n".join(thoughts)

    # --- Generation ---
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, options: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, str]:
        """Send a generation request to the local Ollama instance (Async).
        v15: Task-tier-aware defaults, thought circulation, 3 retries w/ exp backoff.
        """
        if not self._check_circuit():
            return {"response": "Error: Local brain temporarily unavailable (circuit breaker open).", "thought": ""}
        
        # v27: Task-tier-aware defaults (replaces hard 4K/512 ceiling)
        task_tier = detect_task_tier(prompt, system_prompt or "")
        tier_defaults = _MODEL_TIER_DEFAULTS.get(task_tier, _MODEL_TIER_DEFAULTS["default"])

        final_options = {
            "temperature": tier_defaults["temperature"],
            "num_predict": tier_defaults["num_predict"],
            "num_ctx": tier_defaults["num_ctx"],
        }
        if options:
            final_options.update(options)

        # v27: Thought circulation injection for coding tasks
        effective_system = system_prompt or ""
        if task_tier == "coding" and _THOUGHT_CIRCULATION_DIRECTIVE not in effective_system:
            effective_system = f"{effective_system}\n\n{_THOUGHT_CIRCULATION_DIRECTIVE}" if effective_system else _THOUGHT_CIRCULATION_DIRECTIVE

        url = "/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": final_options
        }
        if effective_system:
            payload["system"] = effective_system
        
        last_error = None
        max_retries = 4  # v27: Increased from 2 to 4 (3 retries)
        for attempt in range(max_retries):
            try:
                logger.info("Ollama Request: %s | Tier: %s | Prompt len: %d | Attempt: %d/%d",
                           self.model, task_tier, len(prompt), attempt + 1, max_retries)
                response = await self.client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                self._record_success()
                
                # Extract thinking segments
                raw_response = data.get("response", "").strip()
                cleaned, thought = self._extract_think_segments(raw_response)
                return {"response": cleaned, "thought": thought}
                
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_error = e
                if attempt < max_retries - 1:
                    backoff = min(2 ** attempt, 8)  # 1s, 2s, 4s, 8s
                    logger.warning("Transient LLM error (retry %d in %ds): %s", attempt + 1, backoff, e)
                    await asyncio.sleep(backoff)
                    continue
            except Exception as e:
                last_error = e
                break
        
        self._record_failure()
        logger.error("Local LLM Error: %s", last_error)
        return {"response": f"Error: Local brain (Ollama) is unreachable. {str(last_error)}", "thought": ""}

    async def generate_text_stream_async(self, prompt: str, system_prompt: Optional[str] = None, cancel_event=None, options: Optional[Dict[str, Any]] = None, **kwargs):
        """Stream tokens from the local Ollama instance.
        v13B: Filters <think> blocks in real-time.
        """
        if not self._check_circuit():
            yield "Error: Local brain temporarily unavailable."
            return
        
        # v27: Task-tier-aware defaults for text streaming
        task_tier = detect_task_tier(prompt, system_prompt or "")
        tier_defaults = _MODEL_TIER_DEFAULTS.get(task_tier, _MODEL_TIER_DEFAULTS["default"])

        final_options = {
            "temperature": tier_defaults["temperature"],
            "num_predict": tier_defaults["num_predict"],
            "num_ctx": tier_defaults["num_ctx"],
        }
        if options:
            final_options.update(options)

        # v27: Thought circulation for coding
        effective_system = system_prompt or ""
        if task_tier == "coding" and _THOUGHT_CIRCULATION_DIRECTIVE not in effective_system:
            effective_system = f"{effective_system}\n\n{_THOUGHT_CIRCULATION_DIRECTIVE}" if effective_system else _THOUGHT_CIRCULATION_DIRECTIVE

        url = "/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "options": final_options
        }
        if effective_system:
            payload["system"] = effective_system

        try:
            # Thinking State Machine
            in_think_block = False
            thought_buffer = []
            
            async with self.client.stream("POST", url, json=payload, timeout=_DEFAULT_TIMEOUT) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if cancel_event and cancel_event.is_set():
                        logger.info("Stream cancelled by caller")
                        break
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        
                        if data.get("done"):
                            # Emit collected thought at the end as a protocol token
                            if thought_buffer:
                                yield f"{self.THOUGHT_STREAM_PREFIX}{''.join(thought_buffer)}{self.THOUGHT_STREAM_SUFFIX}"
                            break
                            
                        # Stream Filtering Logic for <think>
                        if in_think_block:
                            if "</think>" in token:
                                in_think_block = False
                                parts = token.split("</think>")
                                # Capture the thinking content
                                thought_buffer.append(parts[0])
                                # Emit whatever follows the closing tag
                                if len(parts) > 1 and parts[1]:
                                    yield parts[1]
                            else:
                                thought_buffer.append(token)
                            continue
                            
                        if "<think>" in token:
                            in_think_block = True
                            parts = token.split("<think>")
                            if parts[0]:
                                yield parts[0]
                            # Capture any text after the opening tag
                            if len(parts) > 1 and parts[1]:
                                thought_buffer.append(parts[1])
                            continue
                            
                        if token:
                            # UX Tuning: 50% slower stream for readability
                            await asyncio.sleep(0.03) 
                            yield token
                            
                    except json.JSONDecodeError:
                        continue
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error("Streaming Error: %s", e)
            yield f" [Error: Sovereign stream interrupted: {str(e)}]"

    # --- Chat ---
    async def chat(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, str]:
        """Async chat interface for the orchestrator.
        v15: Task-tier-aware defaults, thought circulation, 3 retries w/ exp backoff.
        """
        if not self._check_circuit():
            return {"response": "Error: Local brain temporarily unavailable (circuit breaker open).", "thought": ""}
        
        # v27: Detect task tier from message content
        all_content = " ".join(m.get("content", "") for m in messages[-3:])  # Last 3 messages
        task_tier = detect_task_tier(all_content)
        tier_defaults = _MODEL_TIER_DEFAULTS.get(task_tier, _MODEL_TIER_DEFAULTS["default"])

        final_options = {
            "temperature": tier_defaults["temperature"],
            "num_predict": tier_defaults["num_predict"],
            "num_ctx": tier_defaults["num_ctx"],
        }
        if options:
             final_options.update(options)

        # v27: Thought circulation — inject into system message for coding tasks
        effective_messages = list(messages)
        if task_tier == "coding" and effective_messages:
            if effective_messages[0].get("role") == "system":
                sys_content = effective_messages[0]["content"]
                if _THOUGHT_CIRCULATION_DIRECTIVE not in sys_content:
                    effective_messages[0] = {
                        **effective_messages[0],
                        "content": f"{sys_content}\n\n{_THOUGHT_CIRCULATION_DIRECTIVE}"
                    }
            else:
                effective_messages.insert(0, {"role": "system", "content": _THOUGHT_CIRCULATION_DIRECTIVE})
        
        url = "/api/chat"
        payload = {
            "model": self.model,
            "messages": effective_messages,
            "stream": False,
            "keep_alive": "30m",
            "options": final_options
        }
        
        last_error = None
        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = await self.client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                self._record_success()
                
                # Extract thinking segments
                raw_response = data.get("message", {}).get("content", "").strip()
                cleaned, thought = self._extract_think_segments(raw_response)
                return {"response": cleaned, "thought": thought}
                
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_error = e
                if attempt < max_retries - 1:
                    backoff = min(2 ** attempt, 8)
                    logger.warning("Transient chat error (retry %d in %ds): %s", attempt + 1, backoff, e)
                    await asyncio.sleep(backoff)
                    continue
            except Exception as e:
                last_error = e
                break
        
        self._record_failure()
        logger.error("Local Chat Error: %s", last_error)
        return {"response": "Internal Error: Sovereign cognitive stream interrupted.", "thought": ""}

    async def chat_stream_async(self, messages: List[Dict[str, str]], cancel_event=None, options: Optional[Dict[str, Any]] = None, **kwargs):
        """Stream chat tokens from the local Ollama instance.
        v13B: Filters <think> blocks in real-time.
        """
        if not self._check_circuit():
            yield "Error: Local brain temporarily unavailable."
            return
        
        # v27: Task-tier-aware defaults for streaming chat
        all_content = " ".join(m.get("content", "") for m in messages[-3:])
        task_tier = detect_task_tier(all_content)
        tier_defaults = _MODEL_TIER_DEFAULTS.get(task_tier, _MODEL_TIER_DEFAULTS["default"])

        final_options = {
            "temperature": tier_defaults["temperature"],
            "num_predict": tier_defaults["num_predict"],
            "num_ctx": tier_defaults["num_ctx"],
        }
        if options:
            final_options.update(options)

        # v27: Thought circulation for coding
        effective_messages = list(messages)
        if task_tier == "coding" and effective_messages:
            if effective_messages[0].get("role") == "system":
                sys_content = effective_messages[0]["content"]
                if _THOUGHT_CIRCULATION_DIRECTIVE not in sys_content:
                    effective_messages[0] = {
                        **effective_messages[0],
                        "content": f"{sys_content}\n\n{_THOUGHT_CIRCULATION_DIRECTIVE}"
                    }

        url = "/api/chat"
        payload = {
            "model": self.model,
            "messages": effective_messages,
            "stream": True,
            "keep_alive": "30m",
            "options": final_options
        }

        try:
            # Thinking State Machine
            in_think_block = False
            thought_buffer = []
            
            async with self.client.stream("POST", url, json=payload, timeout=_DEFAULT_TIMEOUT) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if cancel_event and cancel_event.is_set():
                        logger.info("Chat stream cancelled by caller")
                        break
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        
                        if data.get("done"):
                            # Emit collected thought at the end as a protocol token
                            if thought_buffer:
                                yield f"{self.THOUGHT_STREAM_PREFIX}{''.join(thought_buffer)}{self.THOUGHT_STREAM_SUFFIX}"
                            break
                            
                        # Stream Filtering Logic for <think>
                        if in_think_block:
                            if "</think>" in token:
                                in_think_block = False
                                parts = token.split("</think>")
                                thought_buffer.append(parts[0])
                                if len(parts) > 1 and parts[1]:
                                    yield parts[1]
                            else:
                                thought_buffer.append(token)
                            continue
                            
                        if "<think>" in token:
                            in_think_block = True
                            parts = token.split("<think>")
                            if parts[0]:
                                yield parts[0]
                            if len(parts) > 1 and parts[1]:
                                thought_buffer.append(parts[1])
                            continue
                            
                        if token:
                            # UX Tuning: 50% slower stream for readability
                            await asyncio.sleep(0.03)
                            yield token
                            
                    except json.JSONDecodeError:
                        continue
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error("Chat Streaming Error: %s (%s)", e, type(e).__name__)
            yield f" [Error: Sovereign chat stream interrupted: {type(e).__name__}: {str(e) or repr(e)}]"
