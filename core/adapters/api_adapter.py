"""Aura's local inference compatibility adapter.

This service remains the stable facade used by memory and epistemic systems,
but every generation and embedding stays on the host.  Historical callers may
still send the old ``api_fast``/``api_deep`` tier labels; those labels now mean
local latency/capability preferences and cannot select a remote provider.

Usage:
    adapter = APIAdapter()
    await adapter.start()
    response = await adapter.generate(prompt, {"model_tier": "api_fast"})
"""

import asyncio
import contextvars
import inspect
import logging
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any

from core.adapters.prompt_boundary import split_prompt
from core.adapters.provider_receipt import (
    digest,
    provider_receipt,
)
from core.adapters.provider_tools import MAX_TOOLS_PER_REQUEST
from core.runtime.errors import Severity, record_degradation

logger = logging.getLogger("Aura.APIAdapter")

#: Per-task generation provenance. See APIAdapter.__init__ for why this
#: is not an instance field.
_LAST_GENERATION: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "api_adapter_last_generation", default={}
)


def _record_api_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "api_adapter",
        error,
        severity=severity,
        action=action,
        extra=extra,
    )


try:
    from core.schemas import ChatStreamEvent
except ImportError:
    ChatStreamEvent = Any


# ─── Model definitions ───────────────────────────────────────────────────────

# Typed admission bounds for caller-supplied generation controls. These
# values flow into paid provider requests and local decode budgets, so they
# are validated here rather than trusted from config.
_VALID_TIERS = {"local", "api_fast", "api_deep"}
_MAX_OUTPUT_TOKENS = 32768
_MAX_PROMPT_CHARS = 500_000


def _bounded_float(value: Any, *, default: float, low: float, high: float) -> float:
    """Finite float in [low, high]; NaN/inf/garbage fall back to default."""
    try:
        candidate = float(default if value is None else value)
    except (TypeError, ValueError):
        return default
    if candidate != candidate or candidate in (float("inf"), float("-inf")):
        return default
    return max(low, min(high, candidate))


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        candidate = int(default if value is None else value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(low, min(high, candidate))


try:
    from core.brain.llm.mlx_client import get_mlx_client
    _HAS_LOCAL_RUNTIME = True
except ImportError:
    _HAS_LOCAL_RUNTIME = False


class _StreamFailed(RuntimeError):
    """A provider stream ended without completing.

    Raised by the provider legs so the ROUTER decides what the caller sees.
    The legs used to swallow their own failures and simply stop yielding,
    which is indistinguishable from a completed stream (CP126 ``88bb1083``).
    """


# ─── APIAdapter ──────────────────────────────────────────────────────────────

class APIAdapter:
    """
    Unified LLM client with automatic fallback.
    Integrates with Aura's existing config and ServiceContainer.
    """
    name = "api_adapter"

    #: Total deadline for one non-stream provider call. Every provider call
    #: was awaited directly, so a backend that accepted the request and then
    #: stopped answering held a conversation lane open with no bound at all
    #: (CP126 ``cf57b7f2``).
    GENERATE_TIMEOUT_S = 120.0
    #: Longest gap between two stream chunks before the stream is declared
    #: dead. A total deadline would cut off a long healthy answer; silence is
    #: the signal that matters.
    STREAM_INACTIVITY_TIMEOUT_S = 45.0

    def __init__(self):
        self._local_client      = None
        self._last_embed_space  = ""

        # Capability flags (set after start())
        self.has_local   = False

        # Usage tracking
        self._call_count: dict[str, int] = {"local": 0}
        self._error_count: dict[str, int] = {"local": 0}
        self._total_tokens: int = 0
        self._exact_token_reports: int = 0
        self._estimated_token_reports: int = 0
        self._last_boundary_provenance: str = ""
        # Provenance of the LAST generation, per execution context. One
        # shared dict meant a concurrent request overwrote it between a
        # caller's generate() and its get_last_generation_metadata(), so the
        # caller read another request's provider and fallback chain (CP126
        # ``63f2b817``). A contextvar follows the task that made the call.
        self._last_generation_metadata: dict[str, Any] = {}

        logger.info("APIAdapter constructed.")

    async def start(self):
        """Initialize the managed local runtime client."""
        # There used to be a shared aiohttp.ClientSession here, opened on
        # every start() with a 100-connection TCPConnector, "to prevent
        # connection pooling exhaustion". Nothing in this class ever made a
        # request with it — it was created, tracked, and closed, and that was
        # the whole of its life. Generation goes through the local backend.
        # Removed rather than routed, because
        # routing a session nobody uses would have preserved the confusion.

        # Initialize Aura's local runtime
        if _HAS_LOCAL_RUNTIME:
            try:
                self._local_client = get_mlx_client()
                self.has_local = True
                logger.info("✅ APIAdapter: Local runtime enabled.")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_api_degradation(
                    e,
                    action="marked the local inference facade unavailable until runtime recovery",
                    extra={"backend": "local", "phase": "start"},
                )
                logger.error("APIAdapter: local runtime init failed: %s", e)

        if not self.has_local:
            logger.error("APIAdapter: managed local runtime unavailable.")

    async def setup_memory_facade(self):
        """Standard integration for MemoryFacade and AgencyFacade."""
        try:
            from core.agency.agency_facade import AgencyFacade
            from core.container import ServiceContainer
            if ServiceContainer.get("agency_facade", default=None) is None:
                fa = AgencyFacade()
                ServiceContainer.register("agency_facade", fa)
                logger.info("✅ AgencyFacade registered for MemoryFacade")
        except ImportError:
            logger.warning("⚠️ [BOOT] Early Facade registration deferred: AgencyFacade missing.")
        except (AttributeError, RuntimeError) as e:
            _record_api_degradation(
                e,
                action="deferred AgencyFacade registration; memory facade setup can retry after container boot",
                extra={"phase": "setup_memory_facade"},
            )
            logger.error("❌ [BOOT] AgencyFacade registration error: %s", e)

    async def stop(self):
        # A stopped adapter must not keep advertising generation capability.
        self.has_local = False
        self._local_client = None
        await self._close_http_session()
        logger.info("APIAdapter stopped. Calls: %s | Tokens: %d",
                    self._call_count, self._total_tokens)

    async def _close_http_session(self) -> None:
        """Close a shared HTTP session, if anything ever sets one.

        ``start()`` no longer opens one — see the note there; the old
        aiohttp session was created, tracked, closed, and never used to make
        a request, so it was removed rather than routed. What survived the
        removal was ``on_stop_async``'s docstring, still describing itself as
        "the shutdown hook for the shared HTTP session" while closing
        nothing.

        Kept as a real close rather than deleting the promise, because the
        next person to add a session will add it to ``start()`` and will not
        think to add a matching close — an aiohttp session that outlives
        shutdown holds its connector and its sockets. Now the hook does what
        it says, whether or not there is anything to do.
        """
        session = getattr(self, "_http_session", None)
        if session is None:
            return
        self._http_session = None
        close = getattr(session, "close", None)
        if close is None:
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — shutdown must complete
            record_degradation(
                "api_adapter",
                exc,
                severity="warning",
                action="dropped the HTTP session reference after close failed",
            )

    async def on_stop_async(self) -> None:
        """ServiceContainer shutdown hook for the shared HTTP session."""
        await self.stop()

    # ─── Main API ────────────────────────────────────────────────────────────

    async def generate(self, prompt: str, config: dict[str, Any] | None = None) -> str:
        """
        Generate a response. Tier is specified in config["model_tier"].
        """
        config = config or {}
        tier        = config.get("model_tier", "local")
        purpose     = config.get("purpose", "general")

        start = time.monotonic()

        result_metadata = await self.generate_with_metadata(prompt, config)
        result = str(result_metadata.get("text") or "")

        # An all-backend failure must NOT come back as an empty string. This
        # is the exact error-versus-empty ambiguity the adapter layer exists
        # to prevent: callers could not distinguish "the model produced
        # nothing" from "every backend failed", and downstream code went on
        # to parse, store, or serve the emptiness as a real answer.
        if not result_metadata.get("ok", True) and not result:
            raise RuntimeError(
                "api_adapter_generation_failed:"
                f"{result_metadata.get('error') or 'unknown'}"
            )

        elapsed = (time.monotonic() - start) * 1000
        logger.debug("APIAdapter.generate: tier=%s purpose=%s %.1fms len=%d",
                     tier, purpose, elapsed, len(result))
        return result

    def get_last_generation_metadata(self) -> dict[str, Any]:
        """Provenance of the last generation made BY THIS TASK.

        Read from a contextvar, not a shared field: two concurrent requests
        used to race here and a caller could be handed the other one's
        provider and fallback chain (CP126 ``63f2b817``). The instance field
        is kept in step for readers that still touch it directly, but the
        contextvar is the answer.
        """
        scoped = _LAST_GENERATION.get()
        if scoped:
            return dict(scoped)
        return dict(self._last_generation_metadata)

    def _publish_generation_metadata(self, result: dict[str, Any]) -> None:
        payload = dict(result)
        _LAST_GENERATION.set(payload)
        self._last_generation_metadata = payload

    async def generate_with_metadata(
        self,
        prompt: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate text with truthful provider, model, and fallback provenance."""

        config = config or {}
        # Typed, finite, bounded admission. These values came straight from
        # caller config into provider requests: a NaN temperature, a string
        # where a number was expected, an unbounded max_tokens, or an
        # unknown tier could raise before routing or request pathological
        # amounts of work from a paid backend.
        tier = str(config.get("model_tier", "local") or "local").strip().lower()
        if tier not in _VALID_TIERS:
            logger.warning("APIAdapter: unknown model_tier %r; defaulting to local.", tier)
            tier = "local"
        temperature = _bounded_float(config.get("temperature"), default=0.7, low=0.0, high=2.0)
        max_tokens = _bounded_int(
            config.get("max_tokens"), default=800, low=1, high=_MAX_OUTPUT_TOKENS
        )
        prompt = str(prompt or "")
        if len(prompt) > _MAX_PROMPT_CHARS:
            logger.warning(
                "APIAdapter: prompt of %d chars exceeds the %d-char ceiling; refusing.",
                len(prompt),
                _MAX_PROMPT_CHARS,
            )
            return {
                "ok": False,
                "text": "",
                "provider": "none",
                "model": "",
                "error": f"prompt_too_large:{len(prompt)}",
            }
        result = await self._route_generate_with_metadata(
            prompt,
            tier,
            temperature,
            max_tokens,
            config=config,
        )
        self._publish_generation_metadata(result)
        return dict(result)

    async def generate_stream(
        self, prompt: str, config: dict[str, Any] | None = None
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
        self, prompt: str, tier: str, temperature: float, max_tokens: int, config: dict[str, Any] | None = None
    ) -> str:
        """Route with automatic fallback chain."""
        result = await self._route_generate_with_metadata(
            prompt,
            tier,
            temperature,
            max_tokens,
            config=config,
        )
        self._publish_generation_metadata(result)
        return str(result.get("text") or "")

    async def _route_generate_with_metadata(
        self,
        prompt: str,
        tier: str,
        temperature: float,
        max_tokens: int,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Route through the managed local runtime with explicit provenance."""

        config = config or {}
        fallback_chain: list[dict[str, str]] = []
        requested_tier = tier
        if tier in ("api_deep", "api_fast"):
            tier = "local"
        if bool(config.get("cloud_only", False)):
            return {
                "ok": False,
                "text": "",
                "endpoint": "APIAdapter-remote-provider-removed",
                "provider": "none",
                "model": "",
                "is_local": True,
                "tier_requested": requested_tier,
                "fallback_chain": [],
                "error": "remote_model_provider_removed",
            }

        if self.has_local:
            result = await self._local_generate(prompt, temperature, max_tokens)
            if result:
                model_name = str(
                    getattr(self._local_client, "model_name", None)
                    or getattr(self._local_client, "model_path", None)
                    or "managed-local-runtime"
                )
                fallback_chain.append(
                    {"provider": "local", "model": model_name, "status": "success"}
                )
                receipt = provider_receipt(
                    provider="local",
                    model=model_name,
                    prompt=prompt,
                    response=str(result),
                    system_instruction=None,
                    transport="mlx_client.generate",
                )
                return {
                    "ok": True,
                    "text": str(result),
                    "endpoint": f"Local-APIAdapter:{model_name}",
                    "provider": "local",
                    "model": model_name,
                    "is_local": True,
                    # Every result carries a receipt so downstream systems do
                    # not need provider-specific provenance rules.
                    "provider_receipt": receipt,
                    "provider_attribution": "provider_receipt",
                    "provider_verified": receipt["response_sha256"]
                    == digest(str(result)),
                    "tier_requested": requested_tier,
                    "tier_resolved": tier,
                    "fallback_chain": fallback_chain,
                    "error": "",
                }
            fallback_chain.append(
                {"provider": "local", "model": "managed-local-runtime", "status": "no_text"}
            )

        logger.error("APIAdapter: local runtime failed for tier=%s", requested_tier)
        return {
            "ok": False,
            "text": "",
            "endpoint": "APIAdapter-all-failed",
            "provider": "none",
            "model": "",
            "is_local": False,
            "fallback_chain": fallback_chain,
            "error": "local_runtime_failed",
        }

    async def _route_stream(
        self, prompt: str, tier: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """Stream from the managed local runtime and terminate exactly once."""
        produced_any = False
        errors: list[str] = []
        if self.has_local:
            try:
                source = self._local_stream(prompt, temperature, max_tokens)
                async for chunk in self._with_inactivity_deadline(source, "local"):
                    produced_any = True
                    yield chunk
                if produced_any:
                    yield ChatStreamEvent(type="end")
                    return
                errors.append("local: produced no tokens")
            except _StreamFailed as exc:
                errors.append(f"local: {exc}")
                if produced_any:
                    yield ChatStreamEvent(
                        type="error",
                        content=f"stream ended early after a local runtime failure: {exc}",
                    )
                    return
        else:
            errors.append("local: runtime unavailable")

        logger.error("APIAdapter: all streams failed for tier=%s (%s)", tier, errors)
        yield ChatStreamEvent(
            type="error",
            content="No LLM backend produced a stream: " + "; ".join(errors[:4]),
        )

    async def _with_inactivity_deadline(
        self, source: AsyncGenerator[ChatStreamEvent, None], backend: str
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """Fail a stream that has gone quiet rather than waiting forever.

        Both stream iterators were awaited with no deadline of any kind, so
        a backend that accepted the request and then stopped sending held
        the conversation lane open indefinitely (CP126 ``cf57b7f2``).
        """
        iterator = source.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(
                    iterator.__anext__(), timeout=self.STREAM_INACTIVITY_TIMEOUT_S
                )
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                await self._aclose_quietly(source)
                raise _StreamFailed(
                    f"no token for {self.STREAM_INACTIVITY_TIMEOUT_S:.0f}s"
                ) from exc
            yield chunk

    @staticmethod
    async def _aclose_quietly(source: Any) -> None:
        aclose = getattr(source, "aclose", None)
        if aclose is None:
            return
        try:
            await aclose()
        except (RuntimeError, GeneratorExit, asyncio.CancelledError) as exc:
            logger.debug("APIAdapter: stream close raised: %s", exc)

    # ─── Local Runtime ───────────────────────────────────────────────────────

    async def _local_generate(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str | None:
        if not self._local_client:
            return None
        try:
            system_text, user_text = split_prompt(prompt)
            result = await asyncio.wait_for(
                self._local_client.generate(
                    user_text,
                    system_prompt=system_text,
                    temp=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=self.GENERATE_TIMEOUT_S,
            )
            
            # Prevent hallucinated human turns from local models
            if result:
                stop_marker = "\nHuman:"
                idx = result.find(stop_marker)
                if idx != -1:
                    result = result[:idx].strip()
                    
            self._call_count["local"] += 1
            self._count_tokens(result or "")
            return result
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            # The MLX client raises ordinary RuntimeError for model admission,
            # decode, worker-state and lane failures, and TypeError/ValueError
            # /AttributeError for malformed client state. Catching only the
            # first three let those escape the local facade entirely.
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as e:
            _record_api_degradation(
                e,
                action="incremented local error count and returned None so routing can fail over",
                extra={"backend": "local", "phase": "generate"},
            )
            logger.warning("Local runtime generate failed: %s", e)
            self._error_count["local"] += 1
        return None

    async def _local_stream(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """Yield tokens. Raise on failure. Never emit a terminal event."""
        if not self._local_client:
            raise _StreamFailed("no local client")
        try:
            system_text, user_text = split_prompt(prompt)
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
                        self._count_tokens(valid_part)
                        yield ChatStreamEvent(type="token", content=valid_part)
                    break
                else:
                    if any(buffer.endswith(stop_marker[:i]) for i in range(1, len(stop_marker) + 1)):
                        pass # keep in buffer
                    else:
                        self._count_tokens(buffer)
                        yield ChatStreamEvent(type="token", content=buffer)
                        buffer = ""
                        
            if buffer and "Human:" not in buffer:
                self._count_tokens(buffer)
                yield ChatStreamEvent(type="token", content=buffer)

            self._call_count["local"] += 1
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as e:
            self._error_count["local"] += 1
            _record_api_degradation(
                e,
                action="raised a typed stream failure so the router can fail over and terminate the stream",
                extra={"backend": "local", "phase": "stream"},
            )
            logger.warning("Local runtime stream failed: %s", e)
            raise _StreamFailed(str(e) or type(e).__name__) from e

    # ─── Embeddings ──────────────────────────────────────────────────────────

    # Persisted consumers use this fixed lexical space. The stronger semantic
    # memory engine owns its separate Qwen3 embedding space.
    LOCAL_EMBED_SPACE = "local:bow-hash-768"

    def last_embedding_space(self) -> str:
        """Vector space of the most recent embedding (see *_EMBED_SPACE)."""
        return getattr(self, "_last_embed_space", "")

    async def embed_async(self, text: str) -> list[float]:
        """Generate a deterministic host-local compatibility embedding."""
        # This lexical space intentionally remains 768-wide for persisted
        # legacy collections. New semantic memory uses the shared Qwen3
        # embedding runtime rather than silently mixing vector spaces.
        # token overlap only — it cannot represent synonymy, relations, or
        # context, so it is not a semantic embedding and must not be
        # described as one. Texts sharing literal words get non-zero cosine
        # similarity; paraphrases with no shared tokens get zero.
        self._last_embed_space = self.LOCAL_EMBED_SPACE
        return self._local_bow_embed(text)

    def embed_sync(self, text: str) -> list[float]:
        """Synchronous wrapper for embeddings."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # This operation is already local and CPU-bounded, so return
                # directly rather than manufacturing a thread/loop bridge.
                self._last_embed_space = self.LOCAL_EMBED_SPACE
                _record_api_degradation(
                    RuntimeError("embed_sync called from a running event loop"),
                    severity="info",
                    action="returned the local compatibility embedding",
                )
                return self._local_bow_embed(text)

            return asyncio.run(self.embed_async(text))
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_api_degradation(
                exc,
                severity="warning",
                action="returned deterministic local bag-of-words embedding from synchronous wrapper",
                extra={"phase": "embed_sync"},
            )
            logger.debug("Synchronous embedding wrapper failed; falling back to local bag-of-words embedding: %s", exc)
            # Fallback to local bag-of-words embedding
            return self._local_bow_embed(text)

    @staticmethod
    def _local_bow_embed(text: str, dim: int = 768) -> list[float]:
        """Bag-of-words hashing embedding that preserves semantic similarity.

        Each word is hashed to 3 positions in the vector and contributes a
        signed value. Texts sharing words will have proportional cosine
        similarity. IDF-like weighting is approximated by word length
        (longer words are rarer and contribute more). The result is
        L2-normalized to unit length.

        This is NOT as good as a real embedding model, but it makes semantic
        memory retrieval, consolidation, and deduplication actually work
        for compatibility collections, unlike random vectors which
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

    #: The literal that separates instructions from the person's turn in a
    #: flat prompt. It is ordinary text, so anyone who can put text in the
    #: prompt can write it.
    ROLE_MARKER = "\nHuman:"
    #: Re-exported so a reader of this class can find the boundary rule
    #: without hunting for the module it moved to.
    ROLE_MARKER = "\nHuman:"
    MAX_TOOLS_PER_REQUEST = MAX_TOOLS_PER_REQUEST

    @staticmethod
    def _split_prompt(prompt: str) -> tuple[str, str]:
        """See :func:`core.adapters.prompt_boundary.split_prompt`."""
        return split_prompt(prompt)

    #: Characters per token, for the paths where a provider reports no usage
    #: count. An estimate that says it is one, rather than a zero that reads
    #: as a measurement.
    CHARS_PER_TOKEN_ESTIMATE = 4.0

    def _count_tokens(self, text: str, *, exact: int | None = None) -> None:
        """Advance the token counter that `get_status` reports.

        `total_tokens` was initialized, reported, and never incremented by
        any generation or stream path, so status published a permanent,
        technically valid zero (CP126 ``82ec3ab8``). An exact count is used
        when the provider gives one; otherwise this estimates and the status
        says which it is.
        """
        if exact is not None:
            self._total_tokens += max(0, int(exact))
            self._exact_token_reports += 1
            return
        chars = len(str(text or ""))
        if chars:
            self._total_tokens += max(1, int(chars / self.CHARS_PER_TOKEN_ESTIMATE))
            self._estimated_token_reports += 1

    def get_status(self) -> dict[str, Any]:
        # Copies, not references: the live counter dicts were handed out by
        # reference, so any consumer could mutate adapter telemetry without
        # going through the adapter.
        return {
            "local":        self.has_local,
            "calls":        dict(self._call_count),
            "errors":       dict(self._error_count),
            "total_tokens": self._total_tokens,
            # How the number was arrived at, so nobody reads an estimate as
            # a billing figure. Zero of both means nothing has generated yet
            # — which is a different fact from "the counter is broken", and
            # that is what this used to be unable to say.
            "token_accounting": {
                "exact_reports": self._exact_token_reports,
                "estimated_reports": self._estimated_token_reports,
                "chars_per_token_estimate": self.CHARS_PER_TOKEN_ESTIMATE,
            },
            "remote_model_providers": (),
            "embedding_space": self._last_embed_space,
        }

    def get_available_tiers(self) -> list[str]:
        tiers = ["local"] if self.has_local else []
        return tiers


# ─── Singleton ───────────────────────────────────────────────────────────────

_adapter_instance: APIAdapter | None = None
_adapter_lock = threading.Lock()

def get_api_adapter() -> APIAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        with _adapter_lock:
            if _adapter_instance is None:
                _adapter_instance = APIAdapter()
    return _adapter_instance
