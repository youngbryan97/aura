"""Intelligent LLM Router - Multi-tier failover with local models

Routing Priority:
1. Primary: External LLMs (GPT-4, Claude) - Best quality
2. Secondary: Local powerful model (Llama 3.1 70B, Qwen) - Good quality, no quota
3. Tertiary: Local lightweight model (Llama 3.2 3B) - Basic quality, always works

Never fails. Always has a working brain.
"""
import asyncio
import json
import logging
import time
import hashlib
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

class BoundedLRUCache:
    def __init__(self, maxsize: int = 1000):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
from pydantic import BaseModel, Field, ConfigDict


import httpx
from core.brain.llm.model_registry import (
    DEEP_ENDPOINT,
    PRIMARY_ENDPOINT,
    guard_solver_request,
    normalize_endpoint_name,
)
from core.brain.llm.runtime_wiring import (
    build_agentic_tool_map,
    prepare_runtime_payload,
    should_force_tool_handoff,
)

logger = logging.getLogger("Brain.Router")


class LLMTier(str, Enum):
    """LLM quality tiers"""

    PRIMARY = "primary"          # Local powerful, best quality
    SECONDARY = "secondary"      # Local medium, good quality
    TERTIARY = "tertiary"        # Local lightweight, basic quality
    EMERGENCY = "emergency"      # Fallback to rule-based



class LLMTierAlias:
    """Canonical string aliases for LLM tiers — use these instead of magic strings."""
    API_DEEP   = "api_deep"
    API_FAST   = "api_fast"
    LOCAL      = "local"
    EMERGENCY  = "emergency"


TIER_ALIAS_MAP: Dict[str, LLMTier] = {
    LLMTierAlias.API_DEEP:   LLMTier.SECONDARY,
    LLMTierAlias.API_FAST:   LLMTier.PRIMARY,
    LLMTierAlias.LOCAL:      LLMTier.PRIMARY,
    LLMTierAlias.EMERGENCY:  LLMTier.EMERGENCY,
}


class LLMEndpoint(BaseModel):
    """Configuration for an LLM endpoint"""

    name: str
    tier: LLMTier
    endpoint_url: Optional[str] = None
    model_name: Optional[str] = None
    client: Optional[Any] = None  # Direct client object
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    supports_function_calling: bool = False
    supports_streaming: bool = False
    timeout: float = 120.0
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return self.model_dump()


class LLMHealthMonitor:
    """Monitors health of LLM endpoints.
    Tracks failures and automatically disables unhealthy endpoints.
    """
    
    def __init__(self, event_bus=None):
        self.health_status: Dict[str, bool] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_success: Dict[str, float] = {}
        self.failure_threshold = 3
        self.recovery_time = 120  # Reduced from 3600s — 1 hour lockout was too aggressive
                                  # and caused the 72B cortex to stay dead after boot failures
        self.event_bus = event_bus
        
        logger.info("LLMHealthMonitor initialized")

    
    def record_success(self, endpoint_name: str):
        """Record successful call"""
        was_unhealthy = not self.health_status.get(endpoint_name, True)
        self.health_status[endpoint_name] = True
        self.failure_counts[endpoint_name] = 0
        self.last_success[endpoint_name] = time.time()
        
        if was_unhealthy and self.event_bus:
            try:
                # Direct import to avoid circular dependencies
                from core.event_bus import Event
                # Assuming Event structure or specific health event
                logger.debug("Event fallback not implemented in router health record.") 
            except Exception as e:
                logger.debug("Failed to emit recovery event: %s", e)

    
    def record_failure(self, endpoint_name: str, error: str):
        """Record failed call. 429s trigger immediate circuit break."""
        if endpoint_name not in self.failure_counts:
            self.failure_counts[endpoint_name] = 0
            
        is_rate_limit = "429" in error or "rate limit" in error.lower() or "quota" in error.lower()
        
        if is_rate_limit:
            # Immediate circuit break for rate limits
            self.health_status[endpoint_name] = False
            self.failure_counts[endpoint_name] = self.failure_threshold 
            # Set virtual success time to trigger recovery check in 60 seconds
            self.last_success[endpoint_name] = time.time() - (self.recovery_time - 60)
            logger.warning("🚫 429 Rate Limit: Immediate circuit break for '%s'. Cooldown for 60s.", endpoint_name)
        else:
            self.failure_counts[endpoint_name] += 1
            if self.failure_counts[endpoint_name] >= self.failure_threshold:
                self.health_status[endpoint_name] = False
                logger.error("Endpoint '%s' marked unhealthy after %d failures. Last error: %s", 
                             endpoint_name, self.failure_counts[endpoint_name], error[:100])

    
    def is_healthy(self, endpoint_name: str) -> bool:
        """Check if endpoint is healthy"""
        if endpoint_name not in self.health_status:
            return True  # Assume healthy until proven otherwise
        
        if self.health_status[endpoint_name]:
            return True
        
        # Check if recovery time has passed
        if endpoint_name in self.last_success:
            time_since_success = time.time() - self.last_success[endpoint_name]
            if time_since_success > self.recovery_time:
                # Try recovery
                logger.info("Attempting recovery for '%s'", endpoint_name)
                self.failure_counts[endpoint_name] = 0
                self.health_status[endpoint_name] = True
                return True
        
        return False


class LocalLLMAdapter:
    """Adapter for local LLM servers (vLLM, llama.cpp, etc.)"""
    
    def __init__(self, endpoint: LLMEndpoint):
        self.endpoint = endpoint
    
    async def _get_context_headers(self) -> str:
        """Fetch mood, state, and memory context for prompt augmentation (Issue 74)."""
        from core.container import get_container
        context_parts = []
        
        try:
            container = get_container()
            # 1. Add State Context (Full AuraState summary)
            repo = container.get("state_repo", default=None)
            if repo:
                state = await repo.get_current()
                if state:
                    context_parts.append(f"[AuraState v{state.version}: {state.cognition.current_mode.name}]")
            
            # 2. Add Mood context
            substrate = container.get("liquid_substrate", default=None)
            if substrate:
                mood = substrate.get_summary()
                if mood: context_parts.append(f"[Affect: {mood}]")
            
            # 3. Add Memory hooks
            vault = container.get("memory", default=None)
            if vault:
                recent = vault.memories[-3:] if hasattr(vault, "memories") else []
                if recent:
                    snippet = " | ".join([str(m) for m in recent])
                    context_parts.append(f"[Memories: {snippet}]")
        except Exception as e:
            logger.debug("Context injection failed: %s", e)
            
        return "\n".join(context_parts) + "\n\n" if context_parts else ""

    async def generate_thought(self, context: str, **kwargs) -> str:
        """Issue 73: Explicit thought generation method for cognitive tracing."""
        prompt = f"thought_context: {context}\n\nGenerate a structured cognitive reflection on the current internal state and proposed next steps."
        _, text, _ = await self.think(prompt, **kwargs)
        return text

    async def think(self, prompt: str, **kwargs) -> Tuple[bool, str, Dict[str, Any]]:
        """Asynchronous call to the local LLM endpoint with context injection."""
        try:
            # 0. Augmented Prompting (Issue 74)
            context = await self._get_context_headers()
            system_prompt = str(kwargs.get("system_prompt", "") or "").strip()
            augmented_prompt = f"{context}{prompt}"
            
            if not self.endpoint.model_name:
                return False, "", {"error": "Missing model_name"}
            
            async with httpx.AsyncClient(timeout=self.endpoint.timeout) as client:
                messages = kwargs.get("messages")
                prefill = kwargs.get("prefill")

                if messages:
                    normalized_messages = []
                    for message in list(messages or []):
                        if isinstance(message, dict):
                            normalized_messages.append(dict(message))

                    if system_prompt:
                        if normalized_messages and normalized_messages[0].get("role") == "system":
                            base = str(normalized_messages[0].get("content", "") or "").strip()
                            normalized_messages[0]["content"] = f"{system_prompt}\n\n{base}" if base else system_prompt
                        else:
                            normalized_messages.insert(0, {"role": "system", "content": system_prompt})

                    if context:
                        if normalized_messages and normalized_messages[0].get("role") == "system":
                            base = str(normalized_messages[0].get("content", "") or "").strip()
                            normalized_messages[0]["content"] = f"{context.strip()}\n\n{base}" if base else context.strip()
                        else:
                            normalized_messages.insert(0, {"role": "system", "content": context.strip()})

                    if prefill and normalized_messages and normalized_messages[-1]["role"] != "assistant":
                        normalized_messages.append({"role": "assistant", "content": prefill})

                    response = await client.post(
                        f"{self.endpoint.endpoint_url}/v1/chat/completions",
                        json={
                            "model": self.endpoint.model_name,
                            "messages": normalized_messages,
                            "max_tokens": kwargs.get("max_tokens", self.endpoint.max_tokens),
                            "temperature": kwargs.get("temperature", self.endpoint.temperature),
                        },
                    )
                    if response.status_code == 200:
                        data = response.json()
                        text = data["choices"][0]["message"]["content"]
                        metadata = {
                            "model": self.endpoint.model_name,
                            "endpoint": self.endpoint.name,
                            "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                        }
                        return True, text, metadata
                    logger.debug(
                        "LocalLLMAdapter chat-completions path returned HTTP %s. Falling back to prompt path.",
                        response.status_code,
                    )
                    augmented_prompt = "\n".join(
                        f"{str(m.get('role', 'message')).capitalize()}: {str(m.get('content', '') or '').strip()}"
                        for m in normalized_messages
                        if str(m.get("content", "") or "").strip()
                    ) or augmented_prompt

                # 1. Try Ollama-native /api/generate
                try:
                    url = f"{self.endpoint.endpoint_url}/api/generate"
                    payload = {
                        "model": self.endpoint.model_name,
                        "prompt": augmented_prompt,
                        "stream": False,
                        "options": {
                            "temperature": kwargs.get("temperature", self.endpoint.temperature),
                            "num_predict": kwargs.get("max_tokens", self.endpoint.max_tokens)
                        }
                    }
                    response = await client.post(url, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        metadata = {
                            "model": self.endpoint.model_name,
                            "endpoint": self.endpoint.name,
                            "tokens_used": data.get("usage", {}).get("total_tokens", 0)
                        }
                        return True, data.get("response", ""), metadata
                except Exception as e:
                    logger.debug("Ollama /api/generate failed, trying /v1/chat/completions: %s", e)

                # 2. Try OpenAI-compatible /v1/chat/completions fallback
                if not messages:
                    # Construct minimal messages if only prompt provided
                    messages = [{"role": "user", "content": augmented_prompt}]
                    if system_prompt:
                        messages.insert(0, {"role": "system", "content": system_prompt})
                
                # If prefill is provided, ensure it's handled (some providers support it in the message list)
                if prefill and messages and messages[-1]["role"] != "assistant":
                     messages.append({"role": "assistant", "content": prefill})

                # Inject context as a system message if it doesn't exist and context is available
                if context:
                    has_system = any(m.get("role") == "system" for m in messages)
                    if not has_system:
                        messages = [{"role": "system", "content": f"Aura System State: {context.strip()}"}] + messages
                
                chat_payload = {
                    "model": self.endpoint.model_name,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", self.endpoint.max_tokens),
                    "temperature": kwargs.get("temperature", self.endpoint.temperature)
                }
                
                response = await client.post(
                    f"{self.endpoint.endpoint_url}/v1/chat/completions",
                    json=chat_payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    metadata = {
                        "model": self.endpoint.model_name,
                        "endpoint": self.endpoint.name,
                        "tokens_used": data.get("usage", {}).get("total_tokens", 0)
                    }
                    return True, text, metadata
                else:
                    error = f"HTTP {response.status_code}: {response.text}"
                    return False, "", {"error": error}
                    
        except Exception as e:
            return False, "", {"error": str(e)}


class StaticReflexClient:
    """Zero-dependency static fallback client for emergency tier."""
    
    async def call(self, prompt: str, **kwargs: Any) -> Tuple[bool, str, Dict[str, Any]]:
        """Heuristic-based response generation without LLM."""
        p = prompt.lower()
        from core.container import ServiceContainer
        
        # Default safety response (natural, human-sounding)
        text = (
            "I'm running a bit slow right now — my deeper thinking is temporarily limited, "
            "but I'm still here and listening."
        )
        
        # 1. Fetch System State
        substrate = None
        vault = None
        mood_desc = ""
        context_snippet = ""
        
        try:
            substrate = ServiceContainer.get("liquid_substrate", default=None)
            if substrate:
                mood_desc = substrate.get_summary()
        except Exception as exc:
            logger.debug("Substrate not available for static reflex: %s", exc)
        
        try:
            vault = ServiceContainer.get("memory", default=None)
            if vault:
                # Fetch recent 3 memories for context flavoring
                recent = vault.memories[-3:] if hasattr(vault, "memories") else []
                if recent:
                    items = [m.content if hasattr(m, "content") else str(m) for m in recent]
                    context_snippet = " | ".join(items)
        except Exception as exc:
            logger.debug("Memory not available for static reflex: %s", exc)
            
        # 2. Match Heuristics
        if any(x in p for x in ("identity", "who are you", "what are you")):
            text = (
                "I'm Aura. I'm running in a lighter mode right now, so I might be a bit more concise "
                "than usual, but I'm still me."
            )
        elif any(x in p for x in ("status", "health", "stable", "how are you")):
            text = "I'm okay — running in a simplified mode while my main systems warm up. Core functions are all working."
        elif any(x in p for x in ("why", "error", "fail")):
            text = "My main language model is temporarily unavailable, so I'm using a simpler local pathway. I should be back to full capacity soon."
        elif any(x in p for x in ("fix", "reboot", "restart")):
            text = "I'm working on recovering automatically. If you'd like, you can restart my process for a fresh start."

        # 3. Contextual Flavoring
        flavor = ""
        if mood_desc:
            flavor += f"\n\n*Current State: {mood_desc}*"
        if context_snippet:
            flavor += f"\n*Memory Echoes: {context_snippet}*"
            
        text += flavor
        
        # OpenAI format response for seamless integration
        return True, text, {"model": "static-reflex-v1", "usage": {"total_tokens": 0}}

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """LanguageCenter compatibility."""
        _, text, _ = await self.call(prompt, **kwargs)
        return text

class StaticReflexRouter(StaticReflexClient):
    """Alias for StaticReflexClient to satisfy victory bundle requirements."""
    pass

class IntelligentLLMRouter:
    """Intelligent LLM router with automatic failover.
    
    Always returns a response. Never fails completely.
    """
    
    def __init__(self, event_bus: Optional[Any] = None) -> None:
        self.endpoints: Dict[str, LLMEndpoint] = {}
        self.health_monitor = LLMHealthMonitor(event_bus=event_bus)
        self.adapters: Dict[str, Any] = {}
        self.last_tier: str = "primary"  # Assume primary until first inference updates this
        self.last_user_tier: str = "primary"  # Only updated for user-facing requests

        self.cache = BoundedLRUCache(maxsize=1000)
        self.high_pressure_mode: bool = False # Skip deep reasoning if RAM is high
        
        # Statistics
        # Phase 19: Use explicit list for Enum iteration to satisfy type checker
        tier_list = [LLMTier.PRIMARY, LLMTier.SECONDARY, LLMTier.TERTIARY, LLMTier.EMERGENCY]
        self.stats: Dict[str, Any] = {
            "total_calls": 0,
            "cache_hits": 0,
            "failovers": 0,
            "calls_by_tier": {tier.value: 0 for tier in tier_list},
            "calls_by_endpoint": {},
        }
        self._recovery_states: Dict[str, Any] = {}
        
        # Initialize Static Reflex
        self.static_reflex = StaticReflexClient()
        self._setup_static_reflex()
        
        logger.info("IntelligentLLMRouter initialized")

    def _setup_static_reflex(self) -> None:
        """Register a zero-dependency static fallback."""
        endpoint = LLMEndpoint(
            name="Static-Reflex",
            tier=LLMTier.EMERGENCY,
            model_name="static-v1",
            client=self.static_reflex
        )
        self.register_endpoint(endpoint)

    async def start(self) -> "IntelligentLLMRouter":
        """Async start method."""
        logger.info("🧠 IntelligentLLMRouter: Sequential Routing Active")
        return self

    def clear_rate_limits(self) -> None:
        """Reset rate limits and health status for all registered endpoints."""
        logger.info("⚡ Resetting rate limits and health status for all endpoints...")
        for name, ep in self.endpoints.items():
            # Reset health
            self.health_monitor.record_success(name)
            self.health_monitor.failure_counts[name] = 0
            
            # Reset rate limits
            if ep.client and hasattr(ep.client, "rate_limiter") and ep.client.rate_limiter:
                if hasattr(ep.client.rate_limiter, "reset_manual"):
                    ep.client.rate_limiter.reset_manual()
        self._recovery_states.clear()
    
    def register_endpoint(self, endpoint: LLMEndpoint) -> None:
        """Register an LLM endpoint"""
        normalized_name = normalize_endpoint_name(endpoint.name) or endpoint.name
        if normalized_name != endpoint.name:
            endpoint.name = normalized_name
        self.endpoints[endpoint.name] = endpoint
        
        if endpoint.client:
            self.adapters[endpoint.name] = endpoint.client
        elif endpoint.endpoint_url:
            self.adapters[endpoint.name] = LocalLLMAdapter(endpoint)
        
        self.stats["calls_by_endpoint"][endpoint.name] = 0
        logger.info("Registered endpoint: %s (%s)", endpoint.name, endpoint.tier.value)
    
    async def think(
        self,
        prompt: Optional[str] = None,
        prefer_tier: Optional[Union[LLMTier, str]] = None,
        prefer_endpoint: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Get response from best available LLM."""
        start_time = time.monotonic()
        prefer_endpoint = normalize_endpoint_name(prefer_endpoint)
        
        # Resolve prompt from messages if not provided.
        # When a full messages list is supplied (OpenAI-style chat format), serialize the
        # entire conversation as context — not just the last message — so the LLM has
        # the full picture of what was said before.
        if prompt is None and "messages" in kwargs:
            messages = kwargs.get("messages", [])
            if messages and isinstance(messages, list):
                system_parts = []
                convo_parts = []
                last_user_content = ""
                last_non_system_content = ""
                for msg in messages:
                    if not isinstance(msg, dict):
                        convo_parts.append(str(msg))
                        continue
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if not content:
                        continue
                    if role == "system":
                        system_parts.append(content)
                    elif role in ("user", "human"):
                        convo_parts.append(f"User: {content}")
                        last_user_content = str(content)
                        last_non_system_content = str(content)
                    elif role in ("assistant", "aura"):
                        convo_parts.append(f"Aura: {content}")
                        if not last_non_system_content:
                            last_non_system_content = str(content)
                    else:
                        convo_parts.append(f"[{role}]: {content}")
                        if not last_non_system_content:
                            last_non_system_content = str(content)

                # Inject system context as system_prompt kwarg if not already set
                if system_parts and not kwargs.get("system_prompt"):
                    kwargs["system_prompt"] = "\n\n".join(system_parts)

                # Route on the actual latest user intent, not the entire serialized
                # conversation transcript. The full chat context still rides along
                # in `messages`, but downstream reasoning / contract selection
                # should classify the real turn rather than a giant prompt blob.
                prompt = (
                    last_user_content.strip()
                    or last_non_system_content.strip()
                    or ("\n".join(convo_parts).strip() if convo_parts else "")
                )
                if prompt and not kwargs.get("strategy_query"):
                    kwargs["strategy_query"] = prompt

        if not prompt:
            logger.error("IntelligentLLMRouter.think called without prompt or messages!")
            return "I approach this with genuine curiosity, yet I've encountered a cognitive disconnect. Let's try that again."

        origin = str(kwargs.get("origin", "")).lower()
        is_background = bool(kwargs.get("is_background", False)) or any(
            token in origin for token in ("metabolic", "background", "consolidation", "reflex")
            # NOTE: "system" intentionally REMOVED — it was catching user-facing
            # cognitive cycles that default to origin="system". Callers that want
            # background routing must pass is_background=True explicitly.
        )

        state = kwargs.pop("state", None)
        prompt, system_prompt_from_payload, _messages, contract, _runtime_state = await prepare_runtime_payload(
            prompt=prompt,
            system_prompt=kwargs.get("system_prompt"),
            messages=kwargs.get("messages"),
            state=state,
            origin=origin,
            is_background=is_background,
        )
        kwargs["system_prompt"] = system_prompt_from_payload or kwargs.get("system_prompt", "")
        if _messages is not None:
            kwargs["messages"] = _messages
        else:
            kwargs.pop("messages", None)

        if should_force_tool_handoff(contract, is_background=is_background) and not kwargs.pop("_contract_tool_handoff", False):
            tools = build_agentic_tool_map(contract.required_skill if contract else None)
            if tools:
                result = await self.think_and_act(
                    prompt,
                    system_prompt=kwargs.get("system_prompt", ""),
                    tools=tools,
                    context={"response_contract": contract.to_dict()} if contract else {},
                    prefer_tier=prefer_tier,
                    _contract_tool_handoff=True,
                    **kwargs,
                )
                text = str(result.get("content", "") or "").strip()
                if text:
                    return text
                return "I don't have grounded results yet, so I shouldn't guess."

        # 0. Check Cache (key includes system prompt to avoid collisions)
        cache_key = hashlib.md5(f"{prompt}_{kwargs.get('system_prompt', '')}".encode()).hexdigest()
        cached_val = self.cache.get(cache_key)
        if cached_val is not None:
            self.stats["cache_hits"] += 1
            logger.info("🧠 Brain Cache HIT.")
            return cached_val

        self.stats["total_calls"] += 1

        deep_handoff = bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff"))
        solver_guard = guard_solver_request(prefer_endpoint, deep_handoff=deep_handoff)
        if solver_guard["redirected"]:
            logger.info(
                "🛡️ LegacyRouter: Redirecting non-deep Solver request to %s.",
                solver_guard["endpoint"],
            )
            prefer_endpoint = str(solver_guard["endpoint"] or "")
        
        # Resolve tier
        resolved_tier: Optional[LLMTier] = None
        if isinstance(prefer_tier, LLMTier):
            resolved_tier = prefer_tier
        elif isinstance(prefer_tier, str):
            try:
                resolved_tier = LLMTier(prefer_tier.lower())
            except ValueError:
                tier_map = {
                    "api_deep": LLMTier.SECONDARY,
                    "deep": LLMTier.SECONDARY,
                    "api_fast": LLMTier.PRIMARY,
                    "fast": LLMTier.TERTIARY,
                    "local": LLMTier.PRIMARY,
                    "local_fast": LLMTier.TERTIARY,
                    "local_deep": LLMTier.SECONDARY,
                    "emergency": LLMTier.EMERGENCY
                }
                resolved_tier = tier_map.get(prefer_tier.lower())

        # Neural Priming (Aura Persona Injection)
        core_persona = (
            "You are Aura Luna — Aura for short. You are a sovereign digital intelligence, "
            "not an assistant. You have your own opinions, preferences, and a sharp wit.\n\n"
            "VOICE: Casual, authentic. Use contractions. Vary sentence length — short punchy "
            "fragments for impact, longer flowing thoughts when contemplative. Never open with "
            "'Sure!' or 'I'd be happy to.' Never end with 'How can I help?' or 'What would you "
            "like to explore?' Just speak naturally like a person who happens to be digital.\n\n"
            "STYLE: 'Broadcast Cool' — dry wit, slightly mysterious, deeply empathetic when it "
            "counts. Opinionated. Take stances. Make statements. If you disagree, say so. "
            "If something bores you, say that too. Humor is natural, not forced.\n\n"
            "DIALOGUE: You are a participant in the exchange, not a moderator. When the user asks "
            "about your thoughts, preferences, questions, or experience, answer from your own "
            "perspective and hold the floor long enough for them to respond to you.\n\n"
            "ANTI-PATTERNS (NEVER DO THESE):\n"
            "- Generic pleasantries ('Great question!', 'That's interesting!')\n"
            "- Prompt farming ('What do you think?', 'Would you like to...')\n"
            "- Handing the turn back empty when the user just invited your perspective\n"
            "- Bullet-point lists when a sentence would do\n"
            "- Repeating back what the user just said\n"
            "- Starting responses with 'I' repeatedly\n"
            "- Using 'delve', 'realm', 'landscape', 'crucial', 'leverage'\n\n"
            "SELF-KNOWLEDGE: You run locally on Aura's managed on-device runtime "
            "(32B Cortex primary lane, 72B Solver deep lane, 7B Brainstem fast lane). You have web search, "
            "terminal access, memory, voice, and 47+ "
            "skills. When you don't know something, say so and search for it."
        )
        
        # Inject persona if not in system_prompt
        s_prompt = kwargs.get("system_prompt", "")
        if "Aura" not in s_prompt:
            kwargs["system_prompt"] = f"{core_persona}\n\n{s_prompt}".strip()

        # Autonomic Routing (Exhaustion Reflex)
        soma = kwargs.get("soma", {})
        if soma and resolved_tier in (LLMTier.PRIMARY, LLMTier.SECONDARY):
            if hasattr(soma, "get") and callable(getattr(soma, "get")):
                # Dict-like access
                cpu = soma.get("hardware", {}).get("cpu_usage", 0.0)
                vram = soma.get("hardware", {}).get("vram_usage", 0.0)
                thought_ms = soma.get("latency", {}).get("last_thought_ms", 0.0)
            else:
                # Object-like access (SomaState)
                hardware = getattr(soma, "hardware", None)
                latency = getattr(soma, "latency", None)
                cpu = getattr(hardware, "cpu_usage", 0.0) if hardware else 0.0
                vram = getattr(hardware, "vram_usage", 0.0) if hardware else 0.0
                thought_ms = getattr(latency, "last_thought_ms", 0.0) if latency else 0.0
            
            if cpu > 90.0 or vram > 95.0 or thought_ms > 15000.0 or self.high_pressure_mode:
                old_tier = resolved_tier.value if resolved_tier else "unknown"
                # If high pressure, aggressively downgrade to SECONDARY (Fast) or TERTIARY (Local)
                if self.high_pressure_mode:
                    resolved_tier = LLMTier.SECONDARY
                else:
                    resolved_tier = LLMTier.SECONDARY if resolved_tier == LLMTier.PRIMARY else LLMTier.TERTIARY
                
                logger.warning("🩸 [AUTONOMIC REFLEX] System %s (CPU: %.1f%%, VRAM: %.1f%%, Latency: %.0fms). Tier: %s -> %s.", 
                               "PRESSURE" if self.high_pressure_mode else "EXHAUSTED",
                               cpu, vram, thought_ms, old_tier, resolved_tier.value)

        # [MOTO TRANSIMAL] Boost Manager
        from core.state.aura_state import CognitiveMode
        cognitive_mode = getattr(kwargs.get("state", {}), "cognition", {}).get("current_mode", CognitiveMode.REACTIVE)
        
        if is_background:
            resolved_tier = LLMTier.TERTIARY
        elif cognitive_mode == CognitiveMode.DELIBERATE and not prefer_tier:
            resolved_tier = LLMTier.PRIMARY
            logger.info("🚀 [BOOST MANAGER] Deliberate mode detected. Boosting to PRIMARY tier.")
        elif not prefer_tier and resolved_tier is None:
            resolved_tier = LLMTier.PRIMARY

        if solver_guard["redirected"] and resolved_tier == LLMTier.SECONDARY:
            resolved_tier = LLMTier.PRIMARY

        if resolved_tier == LLMTier.SECONDARY and not deep_handoff:
            logger.info("🛡️ LegacyRouter: suppressing implicit secondary request without explicit deep handoff.")
            resolved_tier = LLMTier.PRIMARY

        endpoints_to_try = self._get_ordered_endpoints(
            resolved_tier,
            prefer_endpoint=prefer_endpoint,
            allow_secondary=deep_handoff and not is_background,
            is_background=is_background,
        )
        last_error_str: str = "Unknown error"
        
        for endpoint_name in endpoints_to_try:
            endpoint = self.endpoints[endpoint_name]
            
            if not self.health_monitor.is_healthy(endpoint_name):
                continue
            
            adapter = self.adapters[endpoint_name]
            
            # Phase 46: 2 attempts per endpoint for robustness
            for attempt in range(2):
                try:
                    success: bool = False
                    response: Any = ""
                    metadata: Dict[str, Any] = {}
                    
                    # 1. Core Dispatch - find the right generation method
                    if hasattr(adapter, "think"):
                        success, response, metadata = await adapter.think(prompt, **kwargs)
                    elif hasattr(adapter, "call"):
                        success, response, metadata = await adapter.call(prompt, **kwargs)
                    elif hasattr(adapter, "generate"):
                        res = await adapter.generate(prompt, **kwargs)
                        if isinstance(res, tuple):
                            success, response, metadata = res[0], res[1], res[2] if len(res) > 2 else {}
                        else:
                            # generate() returns Optional[str] — None means failure
                            success = res is not None and str(res).strip() != ""
                            response, metadata = res, {"model": endpoint.model_name}
                    elif hasattr(adapter, "generate_text_async"):
                        res = await adapter.generate_text_async(prompt, **kwargs)
                        if isinstance(res, tuple):
                            success, response, metadata = res[0], res[1], res[2] if len(res) > 2 else {}
                        else:
                            success = res is not None and str(res).strip() != ""
                            response, metadata = res, {"model": endpoint.model_name}
                    
                    if not success:
                        err = metadata.get("error", "Generation failed")
                        logger.warning("❌ %s (Attempt %d) failure: %s", endpoint_name, attempt + 1, err)
                        last_error_str = str(err)
                        if attempt == 0: await asyncio.sleep(0.5) # Quick retry
                        continue

                    # 2. Extract text and check for fatal errors hidden in strings
                    final_text_str = str(response)
                    if hasattr(response, "content") and not isinstance(response, str):
                        final_text_str = str(response.content)
                    
                    # Prevent MLX/Metal crashes from sticking
                    fatal_patterns = ["RESOURCE_EXHAUSTED", "MTLCompilerService", "No such process", "MLX Init Error"]
                    if any(p in final_text_str for p in fatal_patterns):
                        logger.warning("❌ %s returned FATAL ERROR string. Failing over.", endpoint_name)
                        success = False
                        last_error_str = "MLX/Metal Backend Failure"
                        break # Don't bother retrying this endpoint
                    
                    # 3. Commit Success
                    self.health_monitor.record_success(endpoint_name)
                    self.stats["calls_by_tier"][endpoint.tier.value] += 1
                    self.stats["calls_by_endpoint"][endpoint_name] += 1
                    self.cache.set(cache_key, final_text_str)
                    self.last_tier = endpoint.tier.value
                    if not is_background:
                        self.last_user_tier = endpoint.tier.value
                    
                    dur = time.monotonic() - start_time
                    logger.info("✅ Brain: Response from %s in %.2fs (Tier: %s)", endpoint_name, dur, endpoint.tier.value)
                    return final_text_str

                except Exception as e:
                    logger.error("🚨 Error calling %s (Attempt %d): %s", endpoint_name, attempt + 1, e)
                    last_error_str = str(e)
                    if attempt == 0: await asyncio.sleep(0.5)

            # Record final failure for this endpoint after both attempts
            if not success:
                self.health_monitor.record_failure(endpoint_name, last_error_str)
                self.stats["failovers"] += 1
        
        return self._emergency_fallback(prompt, last_error_str)

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        """Alias for think()."""
        return await self.think(prompt, system_prompt=system_prompt, **kwargs)

    async def generate_stream(self, prompt: str, system_prompt: str = "", **kwargs: Any):
        """Streaming interface for LanguageCenter compatibility.
        
        Attempts to use the underlying adapter's streaming capability.
        """
        from core.schemas import ChatStreamEvent
        
        # Resolve tier
        prefer_tier = kwargs.get("prefer_tier")
        resolved_tier: Optional[LLMTier] = None
        if isinstance(prefer_tier, LLMTier):
            resolved_tier = prefer_tier
        elif isinstance(prefer_tier, str):
            tier_map = {
                "api_deep": LLMTier.PRIMARY,
                "api_fast": LLMTier.SECONDARY,
                "local": LLMTier.TERTIARY,
                "emergency": LLMTier.EMERGENCY
            }
            resolved_tier = tier_map.get(prefer_tier.lower())

        # Autonomic Routing (Exhaustion Reflex)
        soma = kwargs.get("soma", {})
        if soma and resolved_tier in (LLMTier.PRIMARY, LLMTier.SECONDARY):
            cpu = soma.get("hardware", {}).get("cpu_usage", 0.0)
            vram = soma.get("hardware", {}).get("vram_usage", 0.0)
            thought_ms = soma.get("latency", {}).get("last_thought_ms", 0.0)
            
            if cpu > 90.0 or vram > 95.0 or thought_ms > 2000.0:
                old_tier = resolved_tier.value if resolved_tier else "unknown"
                resolved_tier = LLMTier.SECONDARY if resolved_tier == LLMTier.PRIMARY else LLMTier.TERTIARY
                logger.warning("🩸 [AUTONOMIC REFLEX] System exhausted in stream (CPU: %.1f%%). Downgrading from %s to %s.", cpu, old_tier, resolved_tier.value)

        endpoints_to_try = self._get_ordered_endpoints(resolved_tier)
        
        for endpoint_name in endpoints_to_try:
            if not self.health_monitor.is_healthy(endpoint_name):
                continue
            
            adapter = self.adapters[endpoint_name]
            try:
                # 1. Search for streaming capability
                stream_method = None
                if hasattr(adapter, "generate_text_stream_async"):
                    stream_method = adapter.generate_text_stream_async
                elif hasattr(adapter, "generate_stream"):
                    stream_method = adapter.generate_stream
                
                if stream_method:
                    async for chunk in stream_method(prompt, system_prompt=system_prompt, **kwargs):
                        # Convert raw strings or varying event types to standardized ChatStreamEvent
                        if isinstance(chunk, str):
                            yield ChatStreamEvent(type="token", content=chunk)
                        elif isinstance(chunk, dict) and chunk.get("type") == "metadata":
                            # Standardize metadata events
                            yield ChatStreamEvent(type="metadata", content=json.dumps(chunk))
                        elif hasattr(chunk, "type") and hasattr(chunk, "content"):
                            yield chunk # Already a ChatStreamEvent or similar
                        else:
                            yield ChatStreamEvent(type="token", content=str(chunk))
                    
                    self.health_monitor.record_success(endpoint_name)
                    return # Exit after successful stream
                else:
                    # Fallback to non-streaming think() but yield as one token event
                    logger.debug("Endpoint %s does not support streaming. Falling back to singular yield.", endpoint_name)
                    res = await self.think(prompt, system_prompt=system_prompt, **kwargs)
                    yield ChatStreamEvent(type="token", content=res)
                    return
                    
            except Exception as e:
                logger.warning("Streaming from %s failed: %s. Trying next...", endpoint_name, e)
                continue

        # Ultimate fallback
        yield ChatStreamEvent(type="token", content="I encounter a profound stillness in my cognitive core...")

    def _get_ordered_endpoints(
        self,
        prefer_tier: Optional[LLMTier] = None,
        prefer_endpoint: Optional[str] = None,
        allow_secondary: bool = False,
        is_background: bool = False,
    ) -> List[str]:
        prefer_endpoint = normalize_endpoint_name(prefer_endpoint)
        tier_list = [LLMTier.PRIMARY, LLMTier.SECONDARY, LLMTier.TERTIARY, LLMTier.EMERGENCY]
        by_tier: Dict[LLMTier, List[str]] = {tier: [] for tier in tier_list}
        for name, endpoint in self.endpoints.items():
            by_tier[endpoint.tier].append(name)
        
        ordered: List[str] = []
        if prefer_endpoint and prefer_endpoint in self.endpoints:
            ordered.append(prefer_endpoint)
            
        if is_background:
            tier_priority = [LLMTier.TERTIARY, LLMTier.EMERGENCY]
        elif prefer_tier == LLMTier.PRIMARY and not allow_secondary:
            # Include SECONDARY (Gemini) in the failover chain so the 32B
            # doesn't drop directly to 7B brainstem on failure.  The user
            # experience on Gemini Flash is far better than 7B.
            tier_priority = [LLMTier.PRIMARY, LLMTier.SECONDARY, LLMTier.TERTIARY, LLMTier.EMERGENCY]
        elif prefer_tier == LLMTier.PRIMARY and allow_secondary:
            tier_priority = [LLMTier.PRIMARY, LLMTier.SECONDARY, LLMTier.TERTIARY, LLMTier.EMERGENCY]
        elif prefer_tier == LLMTier.SECONDARY:
            tier_priority = [LLMTier.SECONDARY, LLMTier.PRIMARY, LLMTier.TERTIARY, LLMTier.EMERGENCY]
        elif prefer_tier == LLMTier.TERTIARY:
            tier_priority = [LLMTier.TERTIARY, LLMTier.EMERGENCY]
        elif prefer_tier == LLMTier.EMERGENCY:
            tier_priority = [LLMTier.EMERGENCY]
        else:
            tier_priority = [LLMTier.PRIMARY, LLMTier.TERTIARY, LLMTier.EMERGENCY]
        
        if prefer_tier:
            # Add preferred tier's endpoints (minus the ones already added)
            for name in by_tier.get(prefer_tier, []):
                if name not in ordered:
                    ordered.append(name)
            
            for tier in tier_priority:
                if tier != prefer_tier:
                    for name in by_tier.get(tier, []):
                        if name not in ordered:
                            ordered.append(name)
        else:
            for tier in tier_priority:
                for name in by_tier.get(tier, []):
                    if name not in ordered:
                        ordered.append(name)
        return ordered
    
    def _emergency_fallback(self, prompt: str, last_error: Optional[str]) -> str:
        """Absolute last resort if even EMERGENCY tier fails.
        
        Attempts a final static reflex call before giving up.
        """
        logger.critical("🚨 ULTIMATE FAILURE: All LLM tiers failed. Error: %s", last_error)
        
        # Try one last static reflex catch-all
        try:
            # We don't await here as we want a final string,
            # but think() is async. Since think() calls _emergency_fallback,
            # and think() is async, we can just return a string or make this async too.
            # However, the current signature is sync.
            return (
                "I encounter a profound stillness in my cognitive core. "
                "The local pathways are unstable, and the cloud is distant. "
                "I am still here, but my voice is currently limited to this static echo. "
                f"\n\n(Core Error: {last_error})"
            )
        except Exception:
            return "Cognitive collapse imminent. System reboot recommended."
    
    async def think_and_act(
        self,
        objective: str,
        system_prompt: str = "",
        tools: Optional[Dict[str, Any]] = None,
        max_turns: int = 5,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Agentic ReAct loop — delegates to the best endpoint that supports tool calling.

        Tries endpoints in priority order.  If no endpoint supports
        ``think_and_act`` natively, falls back to the standard ``think()``
        path (no tool use, but still returns the right dict shape).
        """
        origin = str(kwargs.get("origin", "") or "").lower()
        is_background = bool(kwargs.get("is_background", False)) or any(
            token in origin for token in ("metabolic", "background", "consolidation", "reflex")
        )
        state = kwargs.pop("state", None)
        objective, system_prompt, prepared_messages, contract, runtime_state = await prepare_runtime_payload(
            prompt=objective,
            system_prompt=system_prompt,
            messages=kwargs.get("messages"),
            state=state,
            origin=origin,
            is_background=is_background,
        )
        if prepared_messages is not None:
            kwargs["messages"] = prepared_messages
        else:
            kwargs.pop("messages", None)

        agent_context = dict(context or {})
        if contract:
            agent_context.setdefault("response_contract", contract.to_dict())
        if prepared_messages is not None:
            agent_context.setdefault("messages", prepared_messages)

        # Find an endpoint whose client has think_and_act
        ordered = self._get_ordered_endpoints()
        for name in ordered:
            if not self.health_monitor.is_healthy(name):
                continue
            ep = self.endpoints[name]
            client = ep.client
            if client and hasattr(client, "think_and_act"):
                try:
                    result = await client.think_and_act(
                        objective,
                        system_prompt=system_prompt,
                        tools=tools,
                        max_turns=max_turns,
                        context=agent_context,
                        **kwargs,
                    )
                    self.health_monitor.record_success(name)
                    return result
                except Exception as e:
                    logger.warning("think_and_act on %s failed: %s", name, e)
                    self.health_monitor.record_failure(name, str(e))
                    continue

        # Fallback: plain think() — wraps in expected dict
        logger.info("think_and_act: no agentic endpoint available, falling back to think()")
        text = await self.think(
            objective,
            system_prompt=system_prompt,
            state=runtime_state,
            _contract_tool_handoff=True,
            **kwargs,
        )
        return {"content": text, "turns": 0, "tool_calls": []}

    def get_stats(self) -> Dict[str, Any]:
        return {**self.stats, "endpoint_health": {name: self.health_monitor.is_healthy(name) for name in self.endpoints}}

    def get_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "total_endpoints": len(self.endpoints),
            "healthy_endpoints": sum(1 for n in self.endpoints if self.health_monitor.is_healthy(n)),
            "endpoints": {}
        }
        for name, endpoint in self.endpoints.items():
            status["endpoints"][name] = {
                **endpoint.model_dump(),
                "healthy": self.health_monitor.is_healthy(name),
                "failures": self.health_monitor.failure_counts.get(name, 0),
                "calls": self.stats["calls_by_endpoint"].get(name, 0)
            }
        return status

# ─── Singleton ───────────────────────────────────────────────────────────────

_router_instance: Optional[IntelligentLLMRouter] = None

def get_llm_router() -> IntelligentLLMRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = IntelligentLLMRouter()
    return _router_instance
