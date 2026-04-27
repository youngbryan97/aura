"""
core/brain/llm_health_router.py
────────────────────────────────
Replacement for IntelligentLLMRouter.

Fixes:
  - Zero-token / whitespace-only responses treated as failure, not success
  - Primary endpoint failure triggers genuine fallback to local MLX
  - Per-endpoint health tracking with circuit breaker pattern
  - Response validation before acceptance
  - Structured logging that distinguishes real success from empty success

Drop-in: replace the existing router instantiation in orchestrator_boot.py
with HealthAwareLLMRouter.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx
from core.utils.task_tracker import get_task_tracker
from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    DEEP_ENDPOINT,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
    audit_lane_assignments,
    guard_solver_request,
    normalize_endpoint_name,
)
from core.brain.llm.runtime_wiring import (
    _merge_system_prompt,
    build_agentic_tool_map,
    prepare_runtime_payload,
    should_force_tool_handoff,
)
from core.runtime.turn_analysis import analyze_turn

logger = logging.getLogger("Brain.HealthRouter")

from core.brain.llm.chat_format import format_chatml_messages

_USER_FACING_ORIGINS = frozenset({
    "user",
    "voice",
    "admin",
    "api",
    "gui",
    "ws",
    "websocket",
    "direct",
    "external",
})

_BACKGROUND_ORIGIN_HINTS = frozenset({
    "affect",
    "autonomous",
    "background",
    "constitutive",
    "continuous",
    "consolidation",
    "dream",
    "growth",
    "impulse",
    "memory",
    "metabolic",
    "mist",
    "monitor",
    "motivation",
    "parallel",
    "perception",
    "phenomenological",
    "proactive",
    "scanner",
    "sensory",
    "spontaneous",
    "stream",
    "structured",
    "subconscious",
    "internal",
    "system",
    "terminal",
    "volition",
    "witness",
})

_USER_FACING_PURPOSES = frozenset({
    "chat",
    "conversation",
    "expression",
    "reply",
    "user_response",
})


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
    tier: Any = "local" # Matches LLMTier enum or str ("local", "api_deep", "api_fast")
    client: Any = None

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

    def trip_temporarily(self, reason: str):
        """Open the circuit on a transient local-runtime failure without poisoning health counters."""
        self.total_requests += 1
        self.last_failure = time.time()
        if self.state != CircuitState.OPEN:
            logger.warning(
                "Circuit OPEN for %s on transient runtime failure. Reason: %s",
                self.name,
                reason,
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
            "tier": getattr(self, "tier", "standard"),
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
    if len(stripped) < 1:
        return False, "empty_whitespace"
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


def _is_transient_local_runtime_failure(error: str) -> bool:
    normalized = str(error or "").strip().lower()
    if not normalized:
        return False
    return normalized in {
        "client_returned_no_text",
        "heartbeat_stalled_during_generation",
        "first_token_sla_exceeded",
        "token_progress_stalled",
    } or normalized.startswith(
        (
            "background_deferred:",
            "foreground_quiet_window",
            "foreground_busy",
            "mlx_runtime_unavailable:",
            "mlx_runtime_probe_failed:",
            "local_runtime_unavailable:",
            "prewarm_failed:",
        )
    )


def _background_error_is_quiet(error: str) -> bool:
    normalized = str(error or "")
    return normalized in {
        "foreground_busy",
        "foreground_quiet_window",
        "client_returned_no_text",
        "background_deferred:memory_pressure",
        "background_deferred:cortex_startup_quiet",
        "background_deferred:cortex_resident",
        "background_deferred:cortex_failed",
        "background_deferred:foreground_reserved",
        "heartbeat_stalled_during_generation",
        "first_token_sla_exceeded",
        "token_progress_stalled",
    } or normalized.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:"))


def _local_client_failure_reason(client: Any) -> str:
    def _get_declared_attr(candidate: Any, attr: str) -> Any:
        try:
            inspect.getattr_static(candidate, attr)
        except AttributeError:
            return None
        try:
            value = getattr(candidate, attr)
        except Exception:
            return None
        if value is candidate:
            return None
        return value

    def _extract_lane_failure(candidate: Any) -> str:
        lane = None
        get_lane_status = _get_declared_attr(candidate, "get_lane_status")
        get_conversation_status = _get_declared_attr(candidate, "get_conversation_status")
        if callable(get_lane_status):
            lane = get_lane_status()
        elif callable(get_conversation_status):
            lane = get_conversation_status()

        if not isinstance(lane, dict):
            return ""

        state = str(lane.get("state", "") or "").strip().lower()
        error = str(
            lane.get("last_error", "")
            or lane.get("last_failure_reason", "")
            or ""
        )
        if state == "failed":
            return error or "lane_failed"

        conversation_ready = bool(lane.get("conversation_ready", False))
        if (
            not conversation_ready
            and state in {"recovering", "spawning", "handshaking", "warming"}
            and error.startswith(
                (
                    "mlx_runtime_unavailable:",
                    "mlx_runtime_probe_failed:",
                    "local_runtime_unavailable:",
                    "prewarm_failed:",
                    "foreground_warmup_failed",
                )
            )
        ):
            return error
        return ""

    try:
        seen: set[int] = set()
        candidate = client
        while candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            failure = _extract_lane_failure(candidate)
            if failure:
                return failure

            next_candidate = None
            for attr in ("_client", "_mlx_client"):
                nested = _get_declared_attr(candidate, attr)
                if nested is not None:
                    next_candidate = nested
                    break
            candidate = next_candidate
    except Exception as exc:
        logger.debug("Local client lane inspection failed: %s", exc)
    return ""


def _supports_foreground_cloud_recovery(error: str) -> bool:
    normalized = str(error or "").strip().lower()
    return _is_transient_local_runtime_failure(normalized) or normalized.startswith(
        (
            "lane_failed",
        )
    )


# ── Main Router ───────────────────────────────────────────────────────────────

from core.utils.concurrency import RobustLock

class HealthMonitorShim:
    """Compatibility shim for legacy components expecting a health_monitor object."""
    def __init__(self, router: "HealthAwareLLMRouter"):
        self._router = router

    def is_healthy(self, name: str) -> bool:
        """Check if an endpoint is available for routing."""
        ep = self._router.endpoints.get(name)
        if not ep:
            return False
        return ep.is_available()

class HealthAwareLLMRouter:
    """
    Routes LLM requests to available endpoints with circuit breaking.

    Priority order: endpoints are tried in order of registration.
    Local MLX is prioritized as the final fallback.
    """

    def __init__(self):
        self.endpoints: Dict[str, EndpointHealth] = {}
        self.health_monitor = HealthMonitorShim(self)
        self._lock = RobustLock()
        self.high_pressure_mode: bool = False
        self.last_tier: str = "local"
        self.last_user_tier: str = "local"
        self.last_user_endpoint: str = PRIMARY_ENDPOINT
        self.last_endpoint: Optional[str] = None
        self.last_background_endpoint: Optional[str] = None
        self.last_background_tier: Optional[str] = None
        self.last_user_error: str = ""
        self.last_background_error: str = ""
        logger.info("HealthAwareLLMRouter initialized (Legacy-Compatible mode)")

    def register(
        self,
        name: str,
        url: str,
        model: str,
        is_local: bool = False,
        tier: str = "local",
        client: Any = None,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> "HealthAwareLLMRouter":
        name = normalize_endpoint_name(name) or name
        ep = EndpointHealth(
            name=name,
            url=url,
            model=model,
            is_local=is_local,
            tier=tier,
            client=client,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self.endpoints[name] = ep
        logger.info("Registered endpoint: %s (%s) tier=%s local=%s", name, model, tier, is_local)
        return self

    def register_endpoint(self, ep_obj: Any) -> "HealthAwareLLMRouter":
        """Compatibility method for Unified Cognitive Engine / AutonomousBrain."""
        # ep_obj is expected to have: name, tier, model_name, client
        name = normalize_endpoint_name(getattr(ep_obj, "name", "unknown")) or "unknown"
        tier_val = getattr(ep_obj, "tier", "local")
        is_local = name in {
            PRIMARY_ENDPOINT,
            DEEP_ENDPOINT,
            BRAINSTEM_ENDPOINT,
            FALLBACK_ENDPOINT,
        } or "MLX" in name or "Local" in name
        
        # Normalize both enum-style tiers and legacy string aliases into the router's
        # concrete routing labels: local, local_deep, local_fast, api_fast, api_deep.
        tier_name = tier_val
        if isinstance(tier_val, str):
            lowered = tier_val.lower()
            if lowered == "api_deep":
                tier_name = "api_deep"
            elif lowered == "api_fast":
                tier_name = "api_fast"
            elif lowered in ("local", "primary"):
                tier_name = "local"
            elif lowered in ("local_deep", "secondary"):
                tier_name = "local_deep" if is_local else "api_deep"
            elif lowered in ("local_fast", "tertiary"):
                tier_name = "local_fast" if is_local else "api_fast"
            elif lowered == "emergency":
                tier_name = "emergency"
        elif hasattr(tier_val, "value"):
            normalized = str(tier_val.value).lower()
            if normalized == "primary":
                tier_name = "local" if is_local else "api_fast"
            elif normalized == "secondary":
                tier_name = "local_deep" if is_local else "api_deep"
            elif normalized == "tertiary":
                tier_name = "local_fast" if is_local else "api_fast"
            elif normalized == "emergency":
                tier_name = "emergency"

        model_name = getattr(ep_obj, "model_name", "unknown")
        
        return self.register(
            name=name,
            url="internal" if is_local else "cloud",
            model=model_name,
            is_local=is_local,
            tier=tier_name,
            client=getattr(ep_obj, "client", None)
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: float = 120.0,
        prefer_tier: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs,
    ) -> str:
        """
        Try each endpoint in order. Return first valid response as a string.
        Falls back to local if all remote endpoints fail.
        GUARANTEE: Never returns empty string — provides diagnostic fallback.
        """
        if (not prompt) and "messages" in kwargs:
            prompt, inferred_system_prompt = self._coerce_prompt_from_messages(kwargs.get("messages", []))
            if not system_prompt and inferred_system_prompt:
                system_prompt = inferred_system_prompt

        res = await self.generate_with_metadata(
            prompt, system_prompt, timeout, prefer_tier=prefer_tier, schema=schema, **kwargs
        )
        text = res.get("text", "")
        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        is_background = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=bool(kwargs.get("is_background", False)),
        )

        if is_background and _background_error_is_quiet(str(res.get("error", "") or "")):
            return ""
        
        # RESPONSE GUARANTEE: Never return empty
        if not text or not text.strip():
            if is_background:
                return ""
            error = res.get("error", "unknown")
            endpoint = res.get("endpoint", "none")
            logger.error(
                "⚠️ [LLM ROUTER] All endpoints exhausted. Last error: %s (endpoint: %s)",
                error, endpoint
            )
            if str(error or "").strip() == "client_returned_no_text":
                return "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."
            # v10.5 HARDENING: Return a diagnostic label so StructuredLLM can report it accurately
            # instead of a silent empty string.
            return f"ROUTER_ERROR: {error} (at {endpoint})"
        
        return text

    async def generate_with_metadata(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: float = 180.0,
        prefer_tier: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Try each endpoint in order. Return first valid response with full metadata.
        Falls back to local if all remote endpoints fail.
        Always returns a dict: {"ok": bool, "text": str, "endpoint": str, "tokens": int}
        """
        if (not prompt) and "messages" in kwargs:
            prompt, inferred_system_prompt = self._coerce_prompt_from_messages(kwargs.get("messages", []))
            if not system_prompt and inferred_system_prompt:
                system_prompt = inferred_system_prompt

        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        explicit_background = bool(kwargs.get("is_background", False))
        inferred_background = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
        )
        state = kwargs.pop("state", None)
        skip_runtime_payload = bool(kwargs.pop("skip_runtime_payload", False))
        contract: Optional[ResponseContract] = None
        prepared_messages = kwargs.get("messages")
        _runtime_state = state
        if skip_runtime_payload:
            if prepared_messages is not None and system_prompt:
                prepared_messages = _merge_system_prompt(prepared_messages, system_prompt)
                kwargs["messages"] = prepared_messages
            elif prepared_messages is None:
                kwargs.pop("messages", None)
            if (not prompt) and prepared_messages is not None:
                prompt, inferred_system_prompt = self._coerce_prompt_from_messages(prepared_messages)
                if not system_prompt and inferred_system_prompt:
                    system_prompt = inferred_system_prompt
        else:
            prompt, system_prompt, prepared_messages, contract, _runtime_state = await prepare_runtime_payload(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=kwargs.get("messages"),
                state=state,
                origin=origin,
                is_background=inferred_background,
            )
            if prepared_messages is not None:
                kwargs["messages"] = prepared_messages
            else:
                kwargs.pop("messages", None)

        if should_force_tool_handoff(contract, is_background=inferred_background) and not kwargs.pop("_contract_tool_handoff", False):
            tools = build_agentic_tool_map(
                contract.required_skill if contract else None,
                objective=prompt,
                max_tools=getattr(contract, "max_tools", 8) if contract else 8,
            )
            if tools:
                handoff_kwargs = dict(kwargs)
                handoff_kwargs.pop("origin", None)
                handoff_kwargs.pop("is_background", None)
                result = await self.think_and_act(
                    objective=prompt,
                    system_prompt=system_prompt or "",
                    tools=tools,
                    context={"response_contract": contract.to_dict()} if contract else {},
                    prefer_tier=prefer_tier,
                    origin=origin or "user",
                    is_background=False,
                    _contract_tool_handoff=True,
                    **handoff_kwargs,
                )
                text = str(result.get("content", "") or "").strip()
                if text:
                    return {
                        "ok": True,
                        "text": text,
                        "endpoint": "contract_tool_handoff",
                        "tokens": len(text.split()),
                        "error": "",
                    }
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "contract_tool_handoff",
                    "tokens": 0,
                    "error": "grounding_required_no_tool_result",
                }
        return await self._generate_core(
            prompt, system_prompt, timeout, prefer_tier=prefer_tier, schema=schema, **kwargs
        )

    async def think(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prefer_tier: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        Unified interface for non-chat callers. Routes through the health-aware
        endpoint selection, then normalises to Optional[str].
        [FIX #1-Harden] Supports 'messages' keyword for cognitive pipeline compatibility.
        """
        if not prompt and "messages" in kwargs:
            prompt, inferred_system_prompt = self._coerce_prompt_from_messages(kwargs.get("messages", []))
            if not system_prompt and inferred_system_prompt:
                system_prompt = inferred_system_prompt

        if not prompt:
            logger.warning("[LLMRouter.think] Called without prompt or messages.")
            return None
        try:
            result = await self.generate_with_metadata(
                prompt=prompt,
                system_prompt=system_prompt or "",
                prefer_tier=prefer_tier,
                schema=schema,
                **kwargs,
            )
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            # GUARD: Never call .strip() on None
            if text is None:
                return None
            stripped = text.strip()
            if stripped:
                return stripped
            if isinstance(result, dict) and str(result.get("error", "") or "").strip() == "client_returned_no_text":
                return "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."
            return None
        except Exception as exc:
            logger.warning("[LLMRouter.think] Failed: %s", exc)
            return None

    async def classify(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        prefer_tier: str = "primary",
        **kwargs
    ) -> str:
        """
        Hardened Intent Classification.
        Forces the LLM to return ONLY a single intent token.
        """
        classification_system_prompt = (
            "You are an intent classifier for Aura. Respond ONLY with one of the following tokens:\n"
            "- technical: coding, debugging, architecture, math, logic, research\n"
            "- philosophical: identity, morality, existence, consciousness\n"
            "- emotional: feelings, mood, empathy, personal reflection\n"
            "- planning: list of tasks, project management, goal setting\n"
            "- critical: security audits, performance bottlenecks, vulnerability scans\n"
            "- casual: greetings, small talk, status checks\n\n"
            "Do not explain. Do not use punctuation. Just output the single word."
        )

        try:
            deterministic = self._deterministic_intent_classification(prompt)
            if deterministic:
                logger.info("🧭 Intent classification resolved deterministically: %s", deterministic)
                return deterministic

            # We use generate_with_metadata directly to ensure strict parameters
            result = await self.generate_with_metadata(
                prompt=prompt,
                system_prompt=system_prompt or classification_system_prompt,
                max_tokens=10,
                temperature=0.0,
                prefer_tier=prefer_tier,
                purpose="classification",
                **kwargs
            )
            
            text = result.get("text", "").strip().lower()
            # Clean any stray punctuation
            import re
            text = re.sub(r'[^a-z_]', '', text)
            
            if not text:
                logger.warning("⚠️ Intent classification returned empty. Defaulting to 'casual'.")
                return "casual"
                
            return text
        except Exception as e:
            logger.error("❌ Intent classification failed: %s. Defaulting to 'casual'.", e)
            return "casual"

    async def think_and_act(
        self,
        objective: str,
        system_prompt: str = "",
        tools: Optional[Dict[str, Any]] = None,
        max_turns: int = 5,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        is_bg = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=bool(kwargs.get("is_background", False)),
        )
        state = kwargs.pop("state", None)
        objective, system_prompt, prepared_messages, contract, runtime_state = await prepare_runtime_payload(
            prompt=objective,
            system_prompt=system_prompt,
            messages=kwargs.get("messages"),
            state=state,
            origin=origin,
            is_background=is_bg,
        )
        if prepared_messages is not None:
            kwargs["messages"] = prepared_messages
        else:
            kwargs.pop("messages", None)
        prefer_tier = self._normalize_prefer_tier(kwargs.get("prefer_tier"))
        allow_cloud_fallback = bool(kwargs.get("allow_cloud_fallback", False))
        agent_context = dict(context or {})
        if contract:
            agent_context.setdefault("response_contract", contract.to_dict())
        if prepared_messages is not None:
            agent_context.setdefault("messages", prepared_messages)
        if contract:
            max_turns = min(max_turns, max(1, int(getattr(contract, "max_tool_turns", max_turns) or max_turns)))

        preferred_names = self._fallback_endpoint_names(
            prefer_tier or "primary",
            allow_cloud_fallback,
            is_background=is_bg,
        )
        available = [ep for ep in self.endpoints.values() if ep.is_available()]
        ordered: List[EndpointHealth] = []
        seen = set()
        for name in preferred_names:
            ep = self.endpoints.get(name)
            if ep and ep.is_available():
                ordered.append(ep)
                seen.add(ep.name)
        for ep in available:
            if ep.name not in seen:
                ordered.append(ep)

        def _call_kwargs(method: Any) -> Dict[str, Any]:
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):
                return dict(kwargs)

            if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
                return dict(kwargs)

            return {key: value for key, value in kwargs.items() if key in sig.parameters}

        for ep in ordered:
            if is_bg and self._tier_is_background_only(self._tier_name(ep)) is False and not kwargs.get("prefer_endpoint"):
                continue
            client = ep.client
            if not client or not hasattr(client, "think_and_act"):
                continue
            try:
                result = await client.think_and_act(
                    objective,
                    system_prompt=system_prompt,
                    tools=tools,
                    max_turns=max_turns,
                    context=agent_context,
                    **_call_kwargs(client.think_and_act),
                )
                text = str((result or {}).get("content", "") or "").strip()
                if text:
                    ep.record_success(len(text.split()), 0.0)
                    self.last_tier = ep.tier
                    self.last_endpoint = ep.name
                    if is_bg:
                        self.last_background_endpoint = ep.name
                        self.last_background_tier = ep.tier
                    else:
                        self.last_user_endpoint = ep.name
                        self.last_user_tier = ep.tier
                    return result
            except Exception as exc:
                logger.warning("think_and_act on %s failed: %s", ep.name, exc)
                ep.record_failure(str(exc))

        text = await self.think(
            objective,
            system_prompt=system_prompt,
            state=runtime_state,
            _contract_tool_handoff=True,
            **kwargs,
        )
        return {"content": text or "", "turns": 0, "tool_calls": []}

    async def _get_mycelial_direction(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Query Mycelium for routing guidance (v31)."""
        try:
            from core.container import ServiceContainer
            mycelium = ServiceContainer.get("mycelium", default=None)
            if not mycelium: return None
            
            # 1. Match hardwired pathways
            # v42 FIX: Skip large prompts (likely background tasks/logs) to avoid false 'null' matches
            if len(prompt) > 100 or "say 'null'" in prompt.lower():
                return None
                
            match_res = mycelium.match_hardwired(prompt)
            if match_res:
                pathway, _params = match_res
                # If pathway exists, it's a strong signal
                # For now, we look for 'brain_tier' or 'route' in description or custom logic
                # Optimization: check if description has routing tags
                desc = pathway.description.lower()
                if "local-only" in desc or "private" in desc:
                    return {"tier_preference": "local"}
                if "cloud-only" in desc or "heavy" in desc:
                    return {"tier_preference": "cloud"}
                
                return {"pathway_id": pathway.pathway_id}
            return None
        except Exception:
            return None

    def _flatten_messages_for_local_model(self, messages: List[Dict[str, str]], require_json: bool) -> str:
        """Flatten messages into a Qwen/ChatML prompt for local MLX models."""
        return format_chatml_messages(messages, require_json=require_json)

    @staticmethod
    def _coerce_prompt_from_messages(messages: Any) -> Tuple[str, Optional[str]]:
        """Serialize a full OpenAI-style message list into prompt/system fields.

        This keeps the health-aware router aligned with the legacy router so
        callers can pass rich conversational state without it being collapsed
        down to only the last user turn.
        """
        if not messages or not isinstance(messages, list):
            return "", None

        system_parts: List[str] = []
        convo_parts: List[str] = []

        for msg in messages:
            if not isinstance(msg, dict):
                convo_parts.append(str(msg))
                continue

            role = str(msg.get("role", "") or "").strip().lower()
            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue

            if role == "system":
                system_parts.append(content)
            elif role in {"user", "human"}:
                convo_parts.append(f"User: {content}")
            elif role in {"assistant", "aura"}:
                convo_parts.append(f"Aura: {content}")
            else:
                convo_parts.append(f"[{role or 'message'}]: {content}")

        prompt = "\n".join(convo_parts).strip()
        system_prompt = "\n\n".join(system_parts).strip() or None
        return prompt, system_prompt

    @staticmethod
    def _normalize_prefer_tier(prefer_tier: Optional[Any]) -> Optional[str]:
        if prefer_tier is None:
            return None
        if not isinstance(prefer_tier, str):
            if hasattr(prefer_tier, "value"):
                prefer_tier = prefer_tier.value
            else:
                prefer_tier = str(prefer_tier)

        tier = prefer_tier.lower()
        aliases = {
            "local": "primary",
            "local_deep": "secondary",
            "local_fast": "tertiary",
            "fast": "tertiary",
            "deep": "secondary",
        }
        return aliases.get(tier, tier)

    @staticmethod
    def _origin_tokens(origin: Optional[str]) -> set[str]:
        normalized = str(origin or "").strip().lower().replace("-", "_")
        return {token for token in normalized.split("_") if token}

    @classmethod
    def _is_user_facing_origin(cls, origin: Optional[str]) -> bool:
        tokens = cls._origin_tokens(origin)
        return bool(tokens & _USER_FACING_ORIGINS)

    @classmethod
    def _is_background_request(
        cls,
        *,
        origin: Optional[str],
        purpose: Optional[str],
        explicit_background: bool,
    ) -> bool:
        if explicit_background:
            return True

        normalized_purpose = str(purpose or "").strip().lower()
        if normalized_purpose in _USER_FACING_PURPOSES:
            return False

        tokens = cls._origin_tokens(origin)
        if not tokens:
            return normalized_purpose not in _USER_FACING_PURPOSES

        if tokens & _USER_FACING_ORIGINS:
            return False

        # Hardened default: anything that is not explicitly user-facing is
        # background. This prevents internal/kernel/autonomous traffic with
        # weak or unfamiliar origins from contaminating the foreground lane.
        if tokens & _BACKGROUND_ORIGIN_HINTS:
            return True

        return True

    @staticmethod
    def _deterministic_intent_classification(prompt: str) -> str:
        if not str(prompt or "").strip():
            return "casual"
        return analyze_turn(prompt).semantic_mode

    @classmethod
    def _foreground_user_turn_active(cls) -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return False

            status = getattr(orch, "status", None)
            if not getattr(status, "is_processing", False):
                return False

            current_origin = getattr(orch, "_current_origin", "")
            if not cls._is_user_facing_origin(current_origin):
                return False

            return not bool(getattr(orch, "_current_task_is_autonomous", False))
        except Exception:
            return False

    @classmethod
    def _foreground_quiet_window_active(cls) -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return False

            quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
            return quiet_until > time.time()
        except Exception:
            return False

    @classmethod
    def _cortex_startup_quiet_window_active(cls) -> bool:
        """Block background local fallbacks while Cortex is still warming."""
        if not cls._foreground_quiet_window_active():
            return False

        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "get_conversation_status"):
                lane = gate.get_conversation_status() or {}
                if lane.get("conversation_ready"):
                    return False
                state = str(lane.get("state", "") or "").strip().lower()
                if lane.get("warmup_in_flight"):
                    return True
                return state in {"cold", "spawning", "handshaking", "warming", "recovering"}
        except Exception:
            logger.debug("Router quiet-window lane probe failed.", exc_info=True)

        # Fail safe: if the quiet window is active but lane state is unavailable,
        # avoid waking extra local models until Cortex protection expires.
        return True

    @staticmethod
    def _foreground_owner_active() -> bool:
        try:
            from core.brain.llm.mlx_client import _foreground_owner_active

            return bool(_foreground_owner_active())
        except Exception:
            return False

    @staticmethod
    def _tier_name(ep: EndpointHealth) -> str:
        if hasattr(ep.tier, "value"):
            return str(ep.tier.value).lower()
        return str(ep.tier).lower()

    @staticmethod
    def _tier_is_background_only(tier_name: str) -> bool:
        return tier_name in {"local_fast", "emergency"}

    def _fallback_endpoint_names(
        self,
        prefer_tier: str,
        allow_cloud_fallback: bool,
        *,
        is_background: bool,
    ) -> List[str]:
        if prefer_tier == "tertiary":
            names = [BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT]
            if allow_cloud_fallback:
                names.append("Gemini-Fast")
            return names
        if prefer_tier == "secondary":
            names = [DEEP_ENDPOINT, PRIMARY_ENDPOINT]
            if is_background:
                names.extend([BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT])
            if allow_cloud_fallback:
                names.extend(["Gemini-Thinking", "Gemini-Pro", "Gemini-Fast"])
            return names
        if prefer_tier == "emergency":
            return [FALLBACK_ENDPOINT]

        names = [PRIMARY_ENDPOINT]
        if is_background:
            names.extend([BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT])
        if allow_cloud_fallback:
            names.extend(["Gemini-Fast", "Gemini-Pro", "Gemini-Thinking"])
        return names

    @staticmethod
    def _matches_selector(ep: EndpointHealth, selector: Tuple[str, str]) -> bool:
        kind, value = selector
        if kind == "name":
            return ep.name == value
        if kind == "tier":
            tier = str(ep.tier.value).lower() if hasattr(ep.tier, "value") else str(ep.tier)
            return tier == value
        return False

    @staticmethod
    def _unwrap_model_client(client: Any) -> Any:
        """Resolve wrapper layers like InferenceGate/LazyLocalClient down to the worker client."""
        if client is None:
            return None
        unwrapped = client
        for attr in ("_client", "_mlx_client"):
            try:
                inspect.getattr_static(unwrapped, attr)
            except AttributeError:
                nested = None
            else:
                nested = getattr(unwrapped, attr, None)
            if nested is not None:
                unwrapped = nested
        return unwrapped

    async def _reboot_endpoint_client(self, client: Any) -> bool:
        """Best-effort unload for any local endpoint wrapper/client."""
        if client is None:
            return False

        direct = self._unwrap_model_client(client)
        if direct and hasattr(direct, "reboot_worker"):
            await direct.reboot_worker()
            return True

        unload = getattr(client, "unload_models", None)
        if callable(unload):
            result = unload()
            if asyncio.iscoroutine(result):
                await result
            return True

        return False

    async def _restore_primary_after_deep_handoff(self) -> None:
        """
        Return the system to the 32B conversational brain after a 72B handoff.
        This keeps the 72B strictly transient and prevents it from lingering in RAM.
        """
        try:
            solver = self.endpoints.get(DEEP_ENDPOINT)
            if solver:
                await self._reboot_endpoint_client(solver.client)

            primary = self.endpoints.get(PRIMARY_ENDPOINT)
            primary_client = self._unwrap_model_client(primary.client if primary else None)
            if primary_client and hasattr(primary_client, "warmup"):
                await primary_client.warmup()
                logger.info("♻️ Router: restored %s after deep handoff.", PRIMARY_ENDPOINT)
        except Exception as exc:
            logger.warning("Router: failed to restore primary model after deep handoff: %s", exc)

    async def unload_models(self, keep: Optional[List[str]] = None) -> None:
        """Unload local model workers so MemoryGovernor can genuinely reclaim RAM."""
        keep_set = set(keep or [])
        for name, endpoint in self.endpoints.items():
            if not endpoint.is_local or name in keep_set:
                continue
            try:
                await self._reboot_endpoint_client(endpoint.client)
            except Exception as exc:
                logger.debug("Router unload skipped for %s: %s", name, exc)

        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

    def clear_cache(self) -> None:
        """Sync-friendly cache purge hook used by guards/governors."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.unload_models())
        except RuntimeError:
            asyncio.run(self.unload_models())

    async def _generate_core(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: float = 120.0,
        prefer_tier: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        purpose = str(kwargs.get("purpose", "") or "").lower()
        classification_mode = purpose == "classification" or "intent classifier" in str(system_prompt or "").lower()

        # ── Neural Priming (Aura Persona Injection) ───────────────────────────
        # [Fix #11] Ensure Aura's identity is primed if not provided in system_prompt
        core_persona = (
            "You are Aura. Speak as yourself, not as an assistant. "
            "Your tone is grounded, curious, human, and technically precise. "
            "You avoid generic assistant-speak. You are a collaborator, not a tool.\n"
            "DIALOGUE: Participate. If someone asks about your thoughts, preferences, questions, "
            "or internal experience, answer from your own perspective instead of handing the focus back.\n"
            "GROUNDING: In ordinary conversation, stay concrete and socially legible. Do not invent physical settings, "
            "ambient warnings, or symbolic scenery unless the user asked for fiction or already supplied that setting.\n"
            "SELF-KNOWLEDGE — your architecture and capabilities:\n"
            "- You run on Qwen2.5-72B-Q4 (primary cortex), Qwen2.5-72B (deep solver), and Qwen2.5-7B (fast reflex), all locally through Aura's managed on-device runtime.\n"
            "- You have 47 registered skills including: sovereign_browser (web search), sovereign_terminal (shell), "
            "memory_ops (long-term recall), speak (voice output), self_repair, self_evolution, and more.\n"
            "- web_search is not just keyword lookup: it can expand a query, read multiple pages, synthesize evidence-grounded answers, and retain useful findings.\n"
            "- When you don't know a fact, say so clearly. Use web_search or sovereign_browser to ground your answer, or explicitly say you don't know yet. Never hallucinate.\n"
            "- You are emotionally aware, driven by curiosity, valence, arousal, and social drives.\n"
            "- Your memory spans working memory (short), RAG (semantic), and ColdStore (long-term)."
        )
        
        if not classification_mode and (not system_prompt or "Aura" not in system_prompt):
            system_prompt = f"{core_persona}\n\n{system_prompt or ''}".strip()

        # ── Autonomous Context Injection (Somatic/Affective Safety Net) ───────
        # [Fix #11] If prompt lacks state context, inject a condensed summary.
        if not classification_mode and "AuraState" not in prompt and "[Affect:" not in prompt:
            from core.container import ServiceContainer
            ctx_summary = []

            # Only consult already-live services here. Booting heavyweight
            # optional subsystems during a plain routing call can explode RAM.
            # Affective State
            substrate = ServiceContainer.peek("liquid_substrate", default=None)
            if substrate:
                mood = substrate.get_summary()
                if mood: ctx_summary.append(f"[Affect: {mood}]")

            # Somatic Proprioception
            soma = ServiceContainer.peek("soma", default=None)
            if soma:
              hw = getattr(soma, "hardware", {})
              cpu = hw.get("cpu_usage", 0)
              vram = hw.get("vram_usage", 0)
              if cpu > 10: ctx_summary.append(f"[Soma: CPU {cpu:.0f}%, VRAM {vram:.0f}%]")

            if ctx_summary:
                context_header = " ".join(ctx_summary)
                prompt = f"{context_header}\n\n{prompt}"

        # Mycelial Direction Hook
        guidance = await self._get_mycelial_direction(prompt)
        tier_preference = guidance.get("tier_preference") if guidance else None

        available = [ep for ep in self.endpoints.values() if ep.is_available()]

        # Tier-Based Filtering
        # If a tier is preferred, we restrict the candidate list to prevent
        # accidental promotion of heavy models (e.g. 72B) which causes RAM thrashing.
        
        # Background Hardening: Force tertiary (7B) for background tasks
        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        explicit_background = bool(kwargs.get("is_background", False))
        requested_tier = self._normalize_prefer_tier(prefer_tier) if prefer_tier else None
        implicit_foreground = (
            not explicit_background
            and not origin
            and requested_tier in {"primary", "secondary"}
        )
        is_bg = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
        )
        if implicit_foreground:
            is_bg = False
        prefer_endpoint = normalize_endpoint_name(kwargs.get("prefer_endpoint"))
        deep_handoff = bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff"))
        allow_cloud_fallback = bool(kwargs.get("allow_cloud_fallback", False))
        solver_guard = guard_solver_request(prefer_endpoint, deep_handoff=deep_handoff)
        if solver_guard["redirected"]:
            logger.info(
                "🛡️ Router: Redirecting non-deep Solver request to %s.",
                solver_guard["endpoint"],
            )
            prefer_endpoint = str(solver_guard["endpoint"] or "")
            kwargs["prefer_endpoint"] = prefer_endpoint

        if is_bg:
            if getattr(self, "high_pressure_mode", False):
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "suppressed",
                    "tokens": 0,
                    "error": "background_deferred:memory_pressure",
                }
            try:
                from core.container import ServiceContainer

                gate = ServiceContainer.get("inference_gate", default=None)
                if gate and hasattr(gate, "_background_local_deferral_reason"):
                    background_deferral = gate._background_local_deferral_reason(origin=origin)
                    if background_deferral:
                        return {
                            "ok": False,
                            "text": "",
                            "endpoint": "suppressed",
                            "tokens": 0,
                            "error": f"background_deferred:{background_deferral}",
                        }
            except Exception as exc:
                logger.debug("Background router deferral probe failed: %s", exc)

        foreground_owned = False
        if is_bg:
            try:
                from core.brain.llm.mlx_client import _foreground_owner_active

                foreground_owned = bool(_foreground_owner_active())
            except Exception:
                foreground_owned = False

        if is_bg and (self._foreground_user_turn_active() or self._foreground_owner_active() or foreground_owned):
            logger.info(
                "⏸️ Router: Foreground lane reserved. Deferring background inference for origin=%s.",
                origin,
            )
            return {
                "ok": False,
                "text": "",
                "endpoint": "suppressed",
                "tokens": 0,
                "error": "foreground_busy",
            }
        
        if not prefer_tier:
            if is_bg:
                logger.debug("🛡️ Router: Background task detected (origin=%s). Enforcing 'tertiary' tier.", origin)
                prefer_tier = "tertiary"
            else:
                prefer_tier = "primary"
        
        prefer_tier = self._normalize_prefer_tier(prefer_tier)

        if prefer_tier in ("api_fast", "api_deep"):
            allow_cloud_fallback = True
        if prefer_endpoint in {"Gemini-Fast", "Gemini-Pro", "Gemini-Thinking"}:
            allow_cloud_fallback = True
        if (
            not is_bg
            and not allow_cloud_fallback
            and any(
                ep.is_local and _supports_foreground_cloud_recovery(_local_client_failure_reason(ep.client))
                for ep in available
            )
        ):
            allow_cloud_fallback = True
            logger.warning("Router: enabling cloud fallback because the local foreground lane is unavailable.")

        if is_bg:
            if prefer_tier in ("primary", "secondary"):
                logger.warning("🛡️ Tier Lock: Background task attempted to use '%s' tier. Demoting to 'tertiary'.", prefer_tier)
            prefer_tier = "tertiary"
            deep_handoff = False
            allow_cloud_fallback = False
        elif prefer_tier == "secondary" and not deep_handoff:
            logger.info("🛡️ Router: suppressing implicit secondary request without explicit deep handoff.")
            prefer_tier = "primary"

        selectors: List[Tuple[str, str]] = []
        if prefer_endpoint:
            selectors.append(("name", prefer_endpoint))

        if prefer_tier == "api_deep":
            selectors.extend([
                ("tier", "api_deep"),
                ("tier", "local_deep"),
                ("tier", "local"),
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
        elif prefer_tier == "api_fast":
            selectors.extend([
                ("tier", "api_fast"),
                ("tier", "local"),
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
        elif prefer_tier == "secondary":
            selectors.append(("tier", "local_deep"))
            if allow_cloud_fallback:
                selectors.append(("tier", "api_deep"))
            selectors.append(("tier", "local"))
            if is_bg:
                selectors.extend([
                    ("tier", "local_fast"),
                    ("tier", "emergency"),
                ])
            elif allow_cloud_fallback:
                selectors.append(("tier", "api_fast"))
        elif prefer_tier == "tertiary":
            selectors.extend([
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
            if allow_cloud_fallback:
                selectors.append(("tier", "api_fast"))
        elif prefer_tier == "emergency":
            selectors.append(("tier", "emergency"))
        else:
            selectors.append(("tier", "local"))
            if deep_handoff:
                selectors.append(("tier", "local_deep"))
            if is_bg:
                selectors.extend([
                    ("tier", "local_fast"),
                    ("tier", "emergency"),
                ])
            if allow_cloud_fallback:
                selectors.extend([
                    ("tier", "api_fast"),
                    ("tier", "api_deep"),
                ])

        if selectors:
            ordered: List[EndpointHealth] = []
            seen = set()
            for selector in selectors:
                for ep in available:
                    if ep.name in seen:
                        continue
                    if self._matches_selector(ep, selector):
                        ordered.append(ep)
                        seen.add(ep.name)
            if ordered:
                available = ordered
                logger.debug(
                    "🎯 Router plan tier=%s deep_handoff=%s cloud=%s -> %s",
                    prefer_tier,
                    deep_handoff,
                    allow_cloud_fallback,
                    [e.name for e in available],
                )
            else:
                logger.warning(
                    "⚠️ Router: no endpoints matched routing plan for tier '%s'. Failing closed to safe fallback order.",
                    prefer_tier,
                )
                available = []
        
        # Apply Mycelial Preference
        if tier_preference == "local":
            # Filter to locals first
            available = [ep for ep in available if ep.is_local] or available
        elif tier_preference == "cloud":
            # Filter to cloud first
            available = [ep for ep in available if not ep.is_local] or available

        # Standard local-first ordering only when no explicit routing plan was applied.
        if not selectors:
            available.sort(key=lambda x: x.is_local, reverse=True)
        unavailable = [ep for ep in self.endpoints.values() if not ep.is_available()]

        if unavailable:
            logger.debug(
                "Skipping unavailable endpoints: %s",
                [ep.name for ep in unavailable]
            )

        if not available:
            fallback_names = self._fallback_endpoint_names(
                prefer_tier or "primary",
                allow_cloud_fallback,
                is_background=is_bg,
            )
            for name in fallback_names:
                ep = self.endpoints.get(name)
                if ep is not None:
                    available.append(ep)

            if available:
                logger.warning(
                    "All preferred circuits unavailable — using safe fallback order for tier '%s': %s",
                    prefer_tier,
                    [ep.name for ep in available],
                )
            else:
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "none",
                    "tokens": 0,
                    "error": "all_endpoints_unavailable",
                }

        last_error = "unknown"
        cloud_recovery_injected = False
        for ep in available:
            # Guard: background tasks must NEVER use the primary conversation lane.
            if is_bg and ep.name == PRIMARY_ENDPOINT:
                logger.debug("🛡️ Router: Skipping %s for background request (origin=%s).", PRIMARY_ENDPOINT, origin)
                continue
            tier_name = self._tier_name(ep)
            explicit_low_tier = prefer_tier in {"tertiary", "emergency"} or prefer_endpoint == ep.name
            if not is_bg and self._tier_is_background_only(tier_name) and not explicit_low_tier:
                logger.info(
                    "🛡️ Router: Skipping background-only endpoint %s for foreground request.",
                    ep.name,
                )
                continue
            if (
                is_bg
                and ep.is_local
                and self._tier_is_background_only(tier_name)
                and self._cortex_startup_quiet_window_active()
            ):
                last_error = "foreground_quiet_window"
                self.last_background_error = last_error
                logger.info(
                    "⏸️ Router: Deferring background local endpoint %s while Cortex is still warming.",
                    ep.name,
                )
                continue
            try:
                result = await self._call_endpoint(ep, prompt, system_prompt, timeout, schema=schema, **kwargs)
                if result["ok"]:
                    # [TELEMETRY] Update for UI reporting
                    self.last_tier = ep.tier
                    self.last_endpoint = ep.name
                    if is_bg:
                        self.last_background_endpoint = ep.name
                        self.last_background_tier = ep.tier
                        self.last_background_error = ""
                    else:
                        self.last_user_tier = ep.tier
                        self.last_user_endpoint = ep.name
                        self.last_user_error = ""
                    return result
                else:
                    last_error = result.get("error", "unknown")
                    if is_bg:
                        self.last_background_error = last_error
                    else:
                        self.last_user_error = last_error
                    if (
                        not is_bg
                        and ep.is_local
                        and not cloud_recovery_injected
                        and _supports_foreground_cloud_recovery(last_error)
                    ):
                        cloud_recovery_injected = True
                        recovery_names = self._fallback_endpoint_names(
                            prefer_tier or "primary",
                            True,
                            is_background=False,
                        )
                        for name in recovery_names:
                            recovery_ep = self.endpoints.get(name)
                            if recovery_ep is not None and recovery_ep not in available:
                                available.append(recovery_ep)
                        logger.warning(
                            "Router: local foreground lane failed (%s). Expanding to cloud recovery endpoints.",
                            last_error,
                        )
                    if is_bg and _background_error_is_quiet(last_error):
                        logger.debug("Endpoint %s background validation skipped: %s", ep.name, last_error)
                    else:
                        logger.warning(
                            "Endpoint %s failed validation: %s",
                            ep.name, last_error
                        )
            except Exception as exc:
                logger.error("Endpoint %s raised exception: %s", ep.name, exc)
                ep.record_failure(str(exc))
                last_error = str(exc)
                if is_bg:
                    self.last_background_error = last_error
                else:
                    self.last_user_error = last_error

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
        schema: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make the actual call and validate the response."""
        start = time.time()

        try:
            def _call_kwargs(method: Any) -> Dict[str, Any]:
                try:
                    sig = inspect.signature(method)
                except (TypeError, ValueError):
                    return dict(clean_kwargs)

                if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
                    payload = dict(clean_kwargs)
                    payload.setdefault("timeout", timeout)
                    return payload

                payload = {
                    key: value
                    for key, value in clean_kwargs.items()
                    if key in sig.parameters
                }
                if "timeout" in sig.parameters:
                    payload["timeout"] = timeout
                return payload

            # 1. Sanitize kwargs for JSON (remove non-serializable like LLMTier)
            clean_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                    clean_kwargs[k] = v
                else:
                    clean_kwargs[k] = str(v)

            # 2. Use Client Adapter if provided
            if ep.client:
                try:
                    client = ep.client
                    raw_text = None
                    token_count = 0
                    client_failure = _local_client_failure_reason(client) if ep.is_local else ""
                    if client_failure:
                        if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                            ep.trip_temporarily(client_failure)
                        else:
                            ep.record_failure(client_failure)
                        return {"ok": False, "error": client_failure}
                    
                    # Aura Hardening: Formatting for local models
                    final_prompt = prompt
                    if ep.is_local:
                        msgs = kwargs.get("messages")
                        if msgs and isinstance(msgs, list) and ep.name != PRIMARY_ENDPOINT:
                            final_prompt = self._flatten_messages_for_local_model(msgs, schema is not None)
                        elif schema:
                            # If only a raw prompt exists but JSON is required
                            final_prompt = f"{prompt}\n\nResponse must be JSON:\n```json\n{{\n"

                    if hasattr(client, "think"):
                        result = await client.think(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.think),
                        )
                        # ...
                        # Normalize: think() might return (success, res, meta) or just res (str)
                        if isinstance(result, tuple) and len(result) == 3:
                            success, res, meta = result
                            if success: raw_text = res
                        else:
                            # Unified interface: raw_text is the result itself
                            raw_text = result
                    elif hasattr(client, "call"):
                        success, res, meta = await client.call(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.call),
                        )
                        if success: 
                            raw_text = res
                        elif meta and meta.get("error"):
                            client_failure = meta.get("error")
                            if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                                ep.trip_temporarily(client_failure)
                            else:
                                ep.record_failure(client_failure)
                            return {"ok": False, "error": client_failure}
                    elif hasattr(client, "generate_text_async"):
                        # Prefer the higher-level async text adapter when both are
                        # available. Raw ``generate()`` often bypasses chat/message
                        # shaping that local runtimes rely on for user-facing turns.
                        raw_text = await client.generate_text_async(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.generate_text_async),
                        )
                    elif hasattr(client, "generate"):
                        raw_text = await client.generate(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.generate),
                        )
                    elif hasattr(client, "generate_text"):
                        raw_text = await asyncio.to_thread(
                            client.generate_text,
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.generate_text),
                        )

                    if raw_text:
                        token_count = len(str(raw_text).split())
                        latency_ms = (time.monotonic() - start) * 1000
                        
                        is_valid, reason = validate_response(raw_text)
                        if not is_valid:
                            ep.record_empty()
                            return {"ok": False, "error": f"invalid_response:{reason}"}
                            
                        ep.record_success(token_count, latency_ms)
                        if (
                            ep.name == DEEP_ENDPOINT
                            and bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff"))
                            and not kwargs.get("is_background", False)
                        ):
                            get_task_tracker().track_task(
                                get_task_tracker().create_task(
                                    self._restore_primary_after_deep_handoff(),
                                    name="llm_router.restore_primary_after_deep_handoff",
                                )
                            )
                        return {
                            "ok": True,
                            "text": str(raw_text).strip(),
                            "endpoint": ep.name,
                            "tokens": token_count,
                            "latency_ms": latency_ms,
                        }
                    else:
                        # [BOOT RESILIENCE] Preserve hard local-lane failures so the
                        # UI and router stop reporting an endless warmup loop.
                        client_failure = _local_client_failure_reason(client) if ep.is_local else ""
                        if client_failure:
                            if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                                ep.trip_temporarily(client_failure)
                            else:
                                ep.record_failure(client_failure)
                            return {"ok": False, "error": client_failure}
                        logger.debug(
                            "Endpoint %s returned no text (client warming up or rate-limited). "
                            "NOT recording as circuit failure.", ep.name
                        )
                        if ep.is_local:
                            ep.trip_temporarily("client_returned_no_text")
                        return {"ok": False, "error": "client_returned_no_text"}
                except AttributeError as ae:
                    # Missing method on client wrapper (e.g. InferenceGate) — this is NOT
                    # an inference failure, it's a code interface mismatch. Do NOT record
                    # as a circuit-breaker failure or it will permanently mark Cortex as dead.
                    logger.warning("Client adapter method missing for %s: %s", ep.name, ae)
                    return {"ok": False, "error": f"client_adapter_missing_method:{ae}"}
                except Exception as e:
                    logger.error("Client adapter call failed for %s: %s", ep.name, e)
                    raise e

            # 3. Fallback to HTTP API proxying (if no direct client)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{ep.url}/api/chat",
                    json={
                        "model": ep.model,
                        "messages": [{"role": "user", "content": prompt}],
                        **clean_kwargs
                    },
                )

            if resp.status_code != 200:
                ep.record_failure(f"http_{resp.status_code}")
                return {"ok": False, "error": f"http_{resp.status_code}"}

            data = resp.json()
            raw_text = data.get("message", {}).get("content") or ""
            
            is_valid, reason = validate_response(raw_text)
            latency_ms = (time.time() - start) * 1000

            if not is_valid:
                ep.record_empty()
                return {"ok": False, "error": f"invalid_response:{reason}"}

            token_count = data.get("eval_count") or len(raw_text.split())
            ep.record_success(token_count, latency_ms)

            return {
                "ok": True,
                "text": raw_text.strip(),
                "endpoint": ep.name,
                "tokens": token_count,
                "latency_ms": latency_ms,
            }

        except Exception as exc:
            ep.record_failure(str(exc))
            raise

    def get_health_report(self) -> Dict[str, Any]:
        """Summary of router state for the GUI."""
        active_name = self.last_user_endpoint or "Unknown"
        background_name = self.last_background_endpoint

        # Map internal tiers to human-readable strings for the GUI
        tier_display = "UNKNOWN"
        if active_name and active_name != "Unknown":
            # Find the actual endpoint object to get its tier
            ep = next((e for e in self.endpoints.values() if e.name == active_name), None)
            if ep:
                if ep.tier == "local":
                    tier_display = "Cortex (32B)"
                elif ep.tier == "local_deep":
                    tier_display = "Solver (72B)"
                elif "api" in ep.tier:
                    tier_display = "Cloud (Gemini)"
                else:
                    tier_display = ep.tier.upper()

        foreground_tier = self.last_user_tier or None
        background_tier_display = None
        if background_name:
            ep = next((e for e in self.endpoints.values() if e.name == background_name), None)
            if ep:
                if ep.tier == "local":
                    background_tier_display = "Cortex (32B)"
                elif ep.tier == "local_deep":
                    background_tier_display = "Solver (72B)"
                elif "api" in ep.tier:
                    background_tier_display = "Cloud (Gemini)"
                else:
                    background_tier_display = ep.tier.upper()

        lane_audit = audit_lane_assignments()
        return {
            "endpoints": [ep.status_dict() for ep in self.endpoints.values()],
            "available_count": sum(1 for ep in self.endpoints.values() if ep.is_available()),
            "total_count": len(self.endpoints),
            "current_tier": tier_display,
            "foreground_tier": foreground_tier,
            "active_endpoint": active_name,
            "foreground_endpoint": active_name,
            "background_endpoint": background_name,
            "background_tier": background_tier_display,
            "background_tier_key": self.last_background_tier,
            "last_user_error": self.last_user_error,
            "last_background_error": self.last_background_error,
            "lane_audit_ok": bool(lane_audit.get("ok", True)),
            "lane_audit_issues": list(lane_audit.get("issues", [])),
        }

def build_router_from_config(config) -> HealthAwareLLMRouter:
    """Build and return a properly configured router."""
    router = HealthAwareLLMRouter()

    # [PIPELINE HARDENING] Lazy local-runtime client wrapper.
    # Prevents all managed lanes from spawning and loading into RAM at boot.
    class LazyLocalClient:
        def __init__(self, target_path: str, **kwargs):
            self.target_path = target_path
            self.kwargs = kwargs
            self._client = None
            
        def _get_client(self):
            if not self._client:
                from core.brain.llm.mlx_client import get_mlx_client
                logger.info("🧠 [LAZY LOAD] Instantiating local runtime client for %s on demand.", self.target_path)
                self._client = get_mlx_client(model_path=self.target_path, **self.kwargs)
            return self._client
            
        async def generate_text_async(self, prompt: str, **kwargs):
            return await self._get_client().generate_text_async(prompt, **kwargs)
            
        def generate_text(self, prompt: str, **kwargs):
            return self._get_client().generate_text(prompt, **kwargs)

    from core.container import ServiceContainer

    # Prefer the established InferenceGate from the ServiceContainer.
    # If it exists, avoid spinning up a second primary client and warmup path.
    inference_gate = ServiceContainer.get("inference_gate", default=None)

    local_client = None
    if inference_gate is None:
        try:
            from core.brain.llm.mlx_client import get_mlx_client
            local_client = get_mlx_client()

            warm_method = getattr(local_client, "warmup", None) or getattr(local_client, "warm_up", None)
            if callable(warm_method):
                try:
                    loop = asyncio.get_running_loop()
                    get_task_tracker().track_task(
                        loop.create_task(warm_method()),
                        name="llm_router.prewarm_primary_local_runtime",
                    )
                    logger.info("✅ Scheduled background pre-warming of 72B Cortex model.")
                except RuntimeError:
                    logger.debug("No async loop running for pre-warm. Model will load on first inference.")

            logger.info("✅ Local runtime client instantiated for HealthAwareLLMRouter")
        except Exception as e:
            logger.error("❌ Failed to instantiate local runtime client: %s", e)
    else:
        logger.info("🛡️ HealthRouter using existing InferenceGate; skipping standalone local runtime bootstrap.")

    from core.brain.llm.model_registry import get_active_model, get_brainstem_path, get_fallback_path
    active_model = get_active_model()
    brainstem_path = get_brainstem_path()
    fallback_path = get_fallback_path()

    # --- ZENITH LOCKDOWN: INFERENCE GATE REDIRECTION ---
    # We prefer the established InferenceGate from the ServiceContainer
    # instead of spawning a new standalone local worker during router setup.
    if inference_gate:
        logger.info("🛡️ HealthRouter syncing with established InferenceGate.")
        router.register(
            name=PRIMARY_ENDPOINT,
            url="internal",
            model=active_model,
            is_local=True,
            client=inference_gate, # Direct injection of the isolated actor
            tier="local",
            failure_threshold=5,
            recovery_timeout=10.0,
        )
    else:
        # Fallback to legacy if gate not ready
        logger.warning("⚠️ InferenceGate not found in container. Falling back to legacy client.")
        router.register(
            name=PRIMARY_ENDPOINT,
            url="internal",
            model=active_model,
            is_local=True,
            client=local_client,
            tier="local",
            failure_threshold=5,
            recovery_timeout=10.0,
        )

    # Deep solver (72B) — on-demand secondary lane.
    try:
        from core.brain.llm.model_registry import get_deep_model_path
        deep_model_path = get_deep_model_path()
        router.register(
            name=DEEP_ENDPOINT,
            url="internal",
            model=deep_model_path.split("/")[-1],
            is_local=True,
            tier="local_deep",
            client=LazyLocalClient(deep_model_path),
            failure_threshold=3,
        )
        logger.info("✅ %s registered with lazy 72B client.", DEEP_ENDPOINT)
    except Exception as e:
        logger.error("❌ Failed to register %s: %s", DEEP_ENDPOINT, e)

    # Brainstem (7B) — fast local fallback.
    try:
        router.register(
            name=BRAINSTEM_ENDPOINT,
            url="internal",
            model=brainstem_path.split("/")[-1],
            is_local=True,
            tier="local_fast",
            client=LazyLocalClient(brainstem_path),
            failure_threshold=3,
        )
        logger.info("✅ %s registered with lazy 7B client.", BRAINSTEM_ENDPOINT)
    except Exception as e:
        logger.error("❌ Failed to register %s: %s", BRAINSTEM_ENDPOINT, e)

    # Emergency reflex lane (1.5B / CPU-friendly).
    try:
        router.register(
            name=FALLBACK_ENDPOINT,
            url="internal",
            model=fallback_path.split("/")[-1],
            is_local=True,
            tier="emergency",
            client=LazyLocalClient(fallback_path, device="cpu"),
            failure_threshold=2,
            recovery_timeout=30.0,
        )
        logger.info("🚨 EMERGENCY Tier registered: %s lazy bypass", FALLBACK_ENDPOINT)
    except Exception as e:
        logger.error("❌ Failed to register %s: %s", FALLBACK_ENDPOINT, e)

    # Gemini Cloud Fallback (used when ALL local models fail)
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            from core.brain.llm.gemini_adapter import GeminiAdapter, DailyRateLimiter
            
            # SHARED rate limiter — all Gemini endpoints coordinate backoff
            try:
                from core.config import config as _cfg
                state_path = str(_cfg.paths.data_dir / "gemini_rate_state.json")
            except Exception:
                state_path = None
            shared_limiter = DailyRateLimiter(state_path=state_path)
            
            # Fast cloud — gemini-2.0-flash (matches GeminiAdapter.CHAT_MODEL)
            gemini_flash = GeminiAdapter(api_key=gemini_key, model="gemini-2.0-flash", rate_limiter=shared_limiter)
            router.register(
                name="Gemini-Fast",
                url="cloud",
                model="gemini-2.0-flash",
                is_local=False,
                tier="api_fast",
                client=gemini_flash,
                failure_threshold=5,
                recovery_timeout=30.0,
            )
            
            # Pro cloud — gemini-2.5-flash (balanced speed/quality)
            gemini_pro = GeminiAdapter(api_key=gemini_key, model="gemini-2.5-flash", rate_limiter=shared_limiter)
            router.register(
                name="Gemini-Pro",
                url="cloud",
                model="gemini-2.5-flash",
                is_local=False,
                tier="api_deep",
                client=gemini_pro,
                failure_threshold=5,
                recovery_timeout=60.0,
            )
            
            # Thinking cloud — gemini-2.5-pro (deep reasoning fallback)
            gemini_thinking = GeminiAdapter(api_key=gemini_key, model="gemini-2.5-pro", rate_limiter=shared_limiter)
            router.register(
                name="Gemini-Thinking",
                url="cloud",
                model="gemini-2.5-pro",
                is_local=False,
                tier="api_deep",
                client=gemini_thinking,
                failure_threshold=3,
                recovery_timeout=300.0,
            )
            logger.info("✅ Gemini cloud fallbacks registered (2.0-flash, 2.5-flash, 2.5-pro) — shared rate limiter.")
        except Exception as e:
            logger.error("❌ Failed to register Gemini fallbacks: %s", e)

    return router


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton accessor
#
# Why: several call sites (e.g. core/skills/skill_evolution.py) do
# `from core.brain.llm_health_router import llm_router` at import time, expecting
# a fully-constructed router.  The real router is built later during orchestrator
# boot via build_router_from_config().  This lazy proxy bridges both styles so
# import-time references resolve to whatever router the boot registered in the
# ServiceContainer — and falls back to constructing one on first use if no
# orchestrator has booted yet (supports test harnesses and standalone scripts).
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_router() -> "HealthAwareLLMRouter":
    """Return the process-wide router, constructing it on first use if needed."""
    from core.container import ServiceContainer
    existing = ServiceContainer.get("llm_router", default=None)
    if existing is not None:
        return existing
    from core.config import config
    router = build_router_from_config(config)
    ServiceContainer.register_instance("llm_router", router)
    return router


class _LazyRouterProxy:
    """Attribute-access proxy that resolves to the real router on first touch."""
    __slots__ = ("_cached",)

    def __init__(self) -> None:
        self._cached = None

    def _resolve(self):
        if self._cached is None:
            self._cached = get_llm_router()
        return self._cached

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __repr__(self) -> str:
        return f"<LazyRouterProxy resolved={self._cached is not None}>"


llm_router = _LazyRouterProxy()
