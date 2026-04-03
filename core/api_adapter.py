"""core/api_adapter.py — Aura APIAdapter v1.0
=============================================
Unified multi-model API client.

Provides a single interface for all LLM backends:
  - Google Gemini  (api_deep/api_fast)
  - local          → Aura's managed on-device runtime

Config (reads from Aura's existing config / env):
  GEMINI_API_KEY      → enables Gemini

Usage:
    adapter = APIAdapter()
    await adapter.start()
    response = await adapter.generate(prompt, {"model_tier": "api_fast"})
"""

import asyncio
import json
import logging
import os
import time
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as _e:
    logger.debug('Ignored ImportError in api_adapter.py: %s', _e)

try:
    from core.schemas import ChatStreamEvent
except ImportError:
    ChatStreamEvent = Any
logger = logging.getLogger("Aura.APIAdapter")


# ─── Model definitions ───────────────────────────────────────────────────────

GEMINI_MODELS = {
    "api_deep":  "gemini-2.0-flash",
    "api_fast":  "gemini-2.0-flash",
}

try:
    from core.brain.llm.mlx_client import get_mlx_client
    _HAS_LOCAL_RUNTIME = True
except ImportError:
    _HAS_LOCAL_RUNTIME = False


# ─── APIAdapter ──────────────────────────────────────────────────────────────

class APIAdapter:
    """
    Unified LLM client with automatic fallback.
    Integrates with Aura's existing config and ServiceContainer.
    """
    name = "api_adapter"

    def __init__(self):
        self._gemini_client     = None
        self._local_client      = None
        self._http_session      = None

        # Capability flags (set after start())
        self.has_gemini  = False
        self.has_local   = False

        # Usage tracking
        self._call_count: Dict[str, int] = {"gemini": 0, "local": 0}
        self._error_count: Dict[str, int] = {"gemini": 0, "local": 0}
        self._total_tokens: int = 0
        self._gemini_backoff_until: float = 0.0

        logger.info("APIAdapter constructed.")

    async def start(self):
        """Initialize clients from environment / Aura config."""
        # ISSUE #29 - Create shared HTTP session to prevent connection pooling exhaustion
        import aiohttp
        self._http_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=100, keepalive_timeout=60)
        )

        # Load config from Aura's config system
        gemini_key    = None

        try:
            from core.config import config
            # ISSUE #28 - gemini_api_key config precedence
            if hasattr(config, "llm") and hasattr(config.llm, "gemini_api_key") and config.llm.gemini_api_key:
                gemini_key = config.llm.gemini_api_key
            else:
                gemini_key = os.getenv("GEMINI_API_KEY")
        except Exception:
            gemini_key = os.getenv("GEMINI_API_KEY")

        # Initialize Gemini
        if gemini_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=gemini_key)
                self.has_gemini = True
                logger.info("✅ APIAdapter: Gemini enabled (%s)", GEMINI_MODELS["api_fast"])
            except ImportError:
                logger.warning("APIAdapter: 'google-genai' package not installed.")
            except Exception as e:
                logger.error("APIAdapter: Gemini init failed: %s", e)

        # Initialize Aura's local runtime
        if _HAS_LOCAL_RUNTIME:
            try:
                self._local_client = get_mlx_client()
                self.has_local = True
                logger.info("✅ APIAdapter: Local runtime enabled.")
            except Exception as e:
                logger.error("APIAdapter: local runtime init failed: %s", e)

        if not self.has_gemini and not self.has_local:
            logger.error("APIAdapter: NO LLM AVAILABLE. Set GEMINI_API_KEY or verify the local runtime.")

    async def setup_memory_facade(self):
        """Standard integration for MemoryFacade and AgencyFacade."""
        try:
            from core.container import ServiceContainer
            from core.agency.agency_facade import AgencyFacade
            if ServiceContainer.get("agency_facade", default=None) is None:
                fa = AgencyFacade()
                ServiceContainer.register("agency_facade", fa)
                logger.info("✅ AgencyFacade registered for MemoryFacade")
        except ImportError:
            logger.warning("⚠️ [BOOT] Early Facade registration deferred: AgencyFacade missing.")
        except Exception as e:
            logger.error("❌ [BOOT] AgencyFacade registration error: %s", e)

    async def stop(self):
        if self._http_session:
            await self._http_session.close()
        logger.info("APIAdapter stopped. Calls: %s | Tokens: %d",
                    self._call_count, self._total_tokens)

    # ─── Main API ────────────────────────────────────────────────────────────

    async def generate(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response. Tier is specified in config["model_tier"].
        """
        config = config or {}
        tier        = config.get("model_tier", "local")
        temperature = config.get("temperature", 0.7)
        max_tokens  = config.get("max_tokens", 800)
        purpose     = config.get("purpose", "general")

        start = time.monotonic()

        # Tier routing with fallback
        result = await self._route_generate(prompt, tier, temperature, max_tokens)

        elapsed = (time.monotonic() - start) * 1000
        logger.debug("APIAdapter.generate: tier=%s purpose=%s %.1fms len=%d",
                     tier, purpose, elapsed, len(result))
        return result

    async def generate_stream(
        self, prompt: str, config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """Streaming generation."""
        config = config or {}
        tier        = config.get("model_tier", "local")
        temperature = config.get("temperature", 0.7)
        max_tokens  = config.get("max_tokens", 800)

        async for chunk in self._route_stream(prompt, tier, temperature, max_tokens):
            yield chunk

    # ─── Routing ─────────────────────────────────────────────────────────────

    async def _route_generate(
        self, prompt: str, tier: str, temperature: float, max_tokens: int
    ) -> str:
        """Route with automatic fallback chain."""

        # Cloud chain (Gemini only)
        if tier in ("api_deep", "api_fast"):
            if self.has_gemini and time.monotonic() >= self._gemini_backoff_until:
                result = await self._gemini_generate(prompt, tier, temperature, max_tokens)
                if result:
                    return result

        # Local fallback chain
        if self.has_local:
            result = await self._local_generate(prompt, temperature, max_tokens)
            if result:
                return result

        logger.error("APIAdapter: all backends failed for tier=%s", tier)
        return ""

    async def _route_stream(
        self, prompt: str, tier: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        # ISSUE #10 - generate_stream tier="local" missing yielding + error fallback
        if tier in ("api_fast", "api_deep") and self.has_gemini:
            async for chunk in self._gemini_stream(prompt, tier, temperature, max_tokens):
                yield chunk
            return
            
        # Local fallback
        if self.has_local:
            async for chunk in self._local_stream(prompt, temperature, max_tokens):
                yield chunk
            return
            
        # Cloud fallback for local request
        if tier == "local" and self.has_gemini:
            logger.warning("Local runtime missing, falling back to Gemini API")
            async for chunk in self._gemini_stream(prompt, "api_fast", temperature, max_tokens):
                yield chunk
            return
            
        logger.error("APIAdapter: all streams failed for tier=%s", tier)
        yield ChatStreamEvent(type="error", content="No LLM backend available for streaming")

    # ─── Gemini ──────────────────────────────────────────────────────────────

    async def _gemini_generate(
        self, prompt: str, tier: str, temperature: float, max_tokens: int, system_instruction: Optional[str] = None
    ) -> Optional[str]:
        if self._gemini_client and self.has_gemini:
            model_name = GEMINI_MODELS.get(tier, GEMINI_MODELS["api_fast"])
            try:
                from google import genai
                config = genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_instruction if system_instruction else None,
                )
                response = await self._gemini_client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                self._call_count["gemini"] += 1
                return response.text or ""
            except Exception as e:
                err_text = str(e)
                if "429" in err_text or "quota" in err_text.lower():
                    self._gemini_backoff_until = time.monotonic() + 60.0
                logger.warning("Gemini %s failed: %s", model_name, e)
                self._error_count["gemini"] += 1
        return None

    async def _gemini_stream(
        self, prompt: str, tier: str, temperature: float, max_tokens: int, system_instruction: Optional[str] = None
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        if not self._gemini_client:
            return
        model_name = GEMINI_MODELS.get(tier, GEMINI_MODELS["api_fast"])
        try:
            from google import genai
            system_text, user_text = self._split_prompt(prompt)
            config = genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction or system_text or None,
            )
            async for chunk in self._gemini_client.aio.models.generate_content_stream(
                model=model_name,
                contents=user_text,
                config=config,
            ):
                if chunk.text:
                    yield ChatStreamEvent(type="token", content=chunk.text)
            yield ChatStreamEvent(type="end")
            self._call_count["gemini"] += 1
        except Exception as e:
            logger.warning("Gemini streaming failed: %s", e)

    # ─── Local Runtime ───────────────────────────────────────────────────────

    async def _local_generate(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> Optional[str]:
        if not self._local_client:
            return None
        try:
            system_text, user_text = self._split_prompt(prompt)
            result = await self._local_client.generate(
                user_text, 
                system_prompt=system_text,
                temp=temperature, 
                max_tokens=max_tokens
            )
            
            # Prevent hallucinated human turns from local models
            if result:
                stop_marker = "\nHuman:"
                idx = result.find(stop_marker)
                if idx != -1:
                    result = result[:idx].strip()
                    
            self._call_count["local"] += 1
            return result
        except Exception as e:
            logger.warning("Local runtime generate failed: %s", e)
            self._error_count["local"] += 1
        return None

    async def _local_stream(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        if not self._local_client:
            return
        try:
            system_text, user_text = self._split_prompt(prompt)
            buffer = ""
            async for chunk in self._local_client.generate_stream(
                user_text,
                system_prompt=system_text,
                temp=temperature,
                max_tokens=max_tokens
            ):
                content = chunk if isinstance(chunk, str) else chunk.content if hasattr(chunk, 'content') else str(chunk)
                buffer += content
                # ISSUE #11 - local_stream prefix match buffer holding on newlines
                stop_marker = "Human:"
                if stop_marker in buffer:
                    idx = buffer.find(stop_marker)
                    valid_part = buffer[:idx].rstrip()
                    if valid_part:
                        yield ChatStreamEvent(type="token", content=valid_part)
                    break
                else:
                    if any(buffer.endswith(stop_marker[:i]) for i in range(1, len(stop_marker) + 1)):
                        pass # keep in buffer
                    else:
                        yield ChatStreamEvent(type="token", content=buffer)
                        buffer = ""
                        
            if buffer and "Human:" not in buffer:
                yield ChatStreamEvent(type="token", content=buffer)
                
            yield ChatStreamEvent(type="end")
            self._call_count["local"] += 1
        except Exception as e:
            logger.warning("Local runtime stream failed: %s", e)
            self._error_count["local"] += 1

    # ─── Embeddings ──────────────────────────────────────────────────────────

    async def embed_async(self, text: str) -> List[float]:
        """Generate embeddings for text. Uses Gemini as primary, then a local shim."""
        if self.has_gemini:
            try:
                res = self._gemini_client.models.embed_content(
                    model="text-embedding-004",
                    contents=text,
                )
                return res.embeddings[0].values
            except Exception as e:
                logger.debug("Gemini embedding failed: %s", e)

        # Deterministic local embedding fallback using bag-of-words hashing.
        # Unlike random seeded vectors, this preserves semantic similarity:
        # texts sharing words will have non-zero cosine similarity.
        # Each word hashes to a fixed position in the 768-dim vector and adds
        # a contribution, so keyword overlap → vector overlap → search works.
        return self._local_bow_embed(text)

    def embed_sync(self, text: str) -> List[float]:
        """Synchronous wrapper for embeddings."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # ISSUE #7 - embed_sync blocking loop indefinitely
                # We use a thread pool to safely execute the async function
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    future = pool.submit(lambda: asyncio.run(self.embed_async(text)))
                    return future.result()

            return asyncio.run(self.embed_async(text))
        except Exception:
            # Fallback to local bag-of-words embedding
            return self._local_bow_embed(text)

    @staticmethod
    def _local_bow_embed(text: str, dim: int = 768) -> List[float]:
        """Bag-of-words hashing embedding that preserves semantic similarity.

        Each word is hashed to 3 positions in the vector and contributes a
        signed value. Texts sharing words will have proportional cosine
        similarity. IDF-like weighting is approximated by word length
        (longer words are rarer and contribute more). The result is
        L2-normalized to unit length.

        This is NOT as good as a real embedding model, but it makes semantic
        memory retrieval, consolidation, and deduplication actually work
        when cloud embeddings are unavailable — unlike random vectors which
        produce near-zero similarity for all pairs.
        """
        import hashlib
        import numpy as np

        vec = np.zeros(dim, dtype=np.float64)
        words = text.lower().split()
        if not words:
            # Empty text gets a zero vector
            return vec.tolist()

        for word in words:
            # Strip punctuation
            clean = ''.join(c for c in word if c.isalnum())
            if not clean:
                continue
            # IDF-like weight: longer words are rarer and matter more
            weight = 1.0 + min(len(clean), 12) * 0.15
            # Hash to 3 positions for better coverage and collision resistance
            for salt in (b"a", b"b", b"c"):
                h = hashlib.md5(salt + clean.encode()).digest()
                idx = int.from_bytes(h[:2], "big") % dim
                sign = 1.0 if h[2] & 1 else -1.0
                vec[idx] += sign * weight

        # L2 normalize to unit length
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec.tolist()

    # ─── Utilities ───────────────────────────────────────────────────────────

    def _split_prompt(self, prompt: str) -> tuple:
        # ISSUE #8 & #9 - _split_prompt logic & trailing strip
        marker = "\nHuman:"
        idx = prompt.rfind(marker)
        if idx == -1:
            system_part = ""
            user_part = prompt
        else:
            system_part = prompt[:idx].strip()
            user_part   = prompt[idx + len(marker):].strip()

        # Remove trailing Aura: marker accurately
        user_part = re.sub(r"\s*Aura:\s*$", "", user_part).strip()

        return system_part, user_part

    def get_status(self) -> Dict[str, Any]:
        return {
            "gemini":       self.has_gemini,
            "local":        self.has_local,
            "calls":        self._call_count,
            "errors":       self._error_count,
            "total_tokens": self._total_tokens,
        }

    def get_available_tiers(self) -> List[str]:
        tiers = ["local"] if self.has_local else []
        if self.has_gemini:
            tiers = ["api_fast", "api_deep"] + tiers
        return tiers


# ─── Singleton ───────────────────────────────────────────────────────────────

import threading
_adapter_instance: Optional[APIAdapter] = None
_adapter_lock = threading.Lock()

def get_api_adapter() -> APIAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        with _adapter_lock:
            if _adapter_instance is None:
                _adapter_instance = APIAdapter()
    return _adapter_instance
