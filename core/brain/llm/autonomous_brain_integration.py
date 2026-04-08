"""Autonomous Cognitive Engine.
Unifies the 3-tier brain architecture:
Tier 1: Local Titan Agent (PRIMARY)
Tier 2: Backup Local Brain
Tier 3: OpenAI (Relegated to fallback/research)

Drives the Mind/Body connection.
"""
import asyncio
import logging
import os
import time
from typing import Any

from core.config import config
from core.container import get_container
from core.runtime import service_access
from core.utils.exceptions import capture_and_log

from .function_calling_adapter import FunctionCallingAdapter
from .llm_router import IntelligentLLMRouter, LLMEndpoint, LLMTier
from .runtime_wiring import build_agentic_tool_map

logger = logging.getLogger("Aura.AutonomousBrain")

class ReflexClient:
    """A minimal rule-based client that provides emergency cognitive output."""
    async def think(self, prompt: str, **kwargs) -> str:
        return "My primary neural links are currently offline. I'm operating on core reflexes."
    
    async def think_and_act(self, objective: str, system_prompt: str, **kwargs) -> dict[str, Any]:
        return {
            "content": "My higher-level reasoning centers are in a refractory state. I can hear you, but I cannot currently access my deep knowledge or tools.",
            "confidence": 0.1,
            "reasoning": ["Emergency reflex circuit activated."]
        }
    
    async def generate_text_stream_async(self, prompt: str, **kwargs):
        yield "My cognitive"
        yield " pathways"
        yield " are"
        yield " stalled"
        yield ". Re-establishing"
        yield " primary"
        yield " links..."

class AutonomousCognitiveEngine:
    # Error cooldown: only log at ERROR level once per 60s (CROSSWIRE-06: use instance vars)
    _THINK_ERROR_COOLDOWN: float = 60.0

    def __init__(self, registry, skill_router=None, llm_router=None, event_bus=None):
        self.registry = registry
        self.event_bus = event_bus
        # CROSSWIRE-06: Instance-level error tracking instead of shared class var
        self._last_think_error_time: float = 0.0
        # BUG-09: Own the concurrency semaphore here, not on the router
        self._agentic_semaphore = asyncio.Semaphore(8) # Expanded for high-RAM envs
        
        # Skill Router (The "Body") - For tool execution
        self.skill_router = skill_router
        
        # LLM Router (The "Mind") - For failover between models
        # H-28 FIX: Ensure we use the SINGLETON Mind from the container if not provided
        self.llm_router = llm_router or get_container().get("llm_router", default=None) or IntelligentLLMRouter(event_bus=self.event_bus)

        # Adapter links Mind to Body: Uses 'llm_router' for context but 'skill_router' for execution
        self.adapter = FunctionCallingAdapter(registry, self.skill_router)

        # H-28 FIX: Ensure tiers are initialized even if router is shared
        # Phase 33: Fix: Call _init_tiers if only Static-Reflex is present or router is empty
        if not self.llm_router.endpoints or list(self.llm_router.endpoints.keys()) == ["Static-Reflex"]:
            self._init_tiers()
        
        logger.info("✓ Autonomous Cognitive Engine Initialized.")

    async def _get_live_state(self):
        try:
            repo = service_access.resolve_state_repository(default=None)
            if repo and hasattr(repo, "get_current"):
                return await repo.get_current()
        except Exception:
            return None
        return None
        
    def _trace(self, message: str):
        """Internal trace for sovereign diagnostics."""
        logger.info("🔍 [BRAIN-TRACE] %s", message)

    def _is_safe_mode(self) -> bool:
        """Checks if the system is in safe mode based on stability guardian."""
        try:
            from core.container import ServiceContainer
            guardian = ServiceContainer.get("stability_guardian", default=None)
            if guardian and hasattr(guardian, "get_health_summary"):
                summary = guardian.get_health_summary()
                # If system is not healthy, we are in safe mode
                return not summary.get("healthy", True)
        except Exception as _e:
            logger.debug('Ignored Exception in autonomous_brain_integration.py: %s', _e)
        return False

    def _init_tiers(self):
        """Standardizes Aura's managed multi-tier local runtime hierarchy.

        v6.0 "The Unshackling" — M5 Pro 64GB Local-First Architecture
        - Cortex (32B): Primary local brain for daily use
        - Solver (72B): Hot-swap deep solver for complex tasks
        - Gemini: Cloud teacher/oracle for distillation and fallback
        - Brainstem (7B): Background tasks, heartbeat, cheap calls
        - Reflex (1.5B): Emergency CPU-friendly last resort

        Strategy: Local models are PRIMARY. Cloud is SECONDARY (teacher).
        Cortex and Solver hot-swap instead of staying resident together.
        Brainstem can co-exist with the active foreground lane when resources allow.
        """
        from .model_registry import (
            ACTIVE_MODEL,
            BRAINSTEM_ENDPOINT,
            BRAINSTEM_MODEL,
            DEEP_ENDPOINT,
            DEEP_MODEL,
            FALLBACK_ENDPOINT,
            FALLBACK_MODEL,
            PRIMARY_ENDPOINT,
            get_runtime_model_path,
        )
        
        # ── LOCAL PRIMARY: Cortex (32B) ──
        cortex_model_path = (
            getattr(config.llm, "local_cortex_path", None)
            or getattr(config.llm, "mlx_model_path", None)
            or get_runtime_model_path(ACTIVE_MODEL)
        )
        solver_model_path = (
            getattr(config.llm, "local_solver_path", None)
            or getattr(config.llm, "mlx_deep_model_path", None)
            or get_runtime_model_path(DEEP_MODEL)
        )
        brainstem_model_path = (
            getattr(config.llm, "local_brainstem_path", None)
            or getattr(config.llm, "mlx_brainstem_path", None)
            or get_runtime_model_path(BRAINSTEM_MODEL)
        )

        if cortex_model_path and PRIMARY_ENDPOINT not in getattr(self.llm_router, "endpoints", {}):
            try:
                from .mlx_client import get_mlx_client
                cortex_client = get_mlx_client(
                    model_path=cortex_model_path,
                    max_tokens=2048,
                )
                self.llm_router.register_endpoint(LLMEndpoint(
                    name=PRIMARY_ENDPOINT,
                    tier=LLMTier.PRIMARY,
                    model_name=cortex_model_path.split("/")[-1],
                    client=cortex_client,
                ))
                logger.info("🧠 PRIMARY Tier registered: %s (%s) — Daily Brain", PRIMARY_ENDPOINT, ACTIVE_MODEL)
            except Exception as e:
                logger.error("Failed to register %s pathway: %s", PRIMARY_ENDPOINT, e)

        # ── LOCAL SECONDARY: Solver (72B) — Hot-swap deep thinker ──
        if solver_model_path and DEEP_ENDPOINT not in getattr(self.llm_router, "endpoints", {}):
            try:
                from .mlx_client import get_mlx_client
                solver_client = get_mlx_client(
                    model_path=solver_model_path,
                    max_tokens=4096,
                )
                self.llm_router.register_endpoint(LLMEndpoint(
                    name=DEEP_ENDPOINT,
                    tier=LLMTier.SECONDARY,  # Moved to SECONDARY to prevent accidental promotion
                    model_name=solver_model_path.split("/")[-1],
                    client=solver_client,
                    timeout=300.0,  # 72B needs more time to load/generate
                ))
                logger.info("🧠 SECONDARY Tier registered: %s (%s) — Deep Thinker (Hot-Swap)", DEEP_ENDPOINT, DEEP_MODEL)
            except Exception as e:
                logger.error("Failed to register %s pathway: %s", DEEP_ENDPOINT, e)

        # ── CLOUD SECONDARY: Gemini (Teacher/Oracle for distillation) ──
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if not gemini_key:
            try:
                from pathlib import Path
                env_path = Path(__file__).resolve().parents[3] / ".env"
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if line.startswith("GEMINI_API_KEY="):
                            gemini_key = line.split("=", 1)[1].strip()
                            break
            except Exception as e:
                capture_and_log(e, {'module': __name__})
        
        if gemini_key and "Gemini-Fast" not in getattr(self.llm_router, "endpoints", {}):
            try:
                from .gemini_adapter import DailyRateLimiter, GeminiAdapter
                
                state_path = str(config.paths.data_dir / "gemini_rate_state.json")
                shared_limiter = DailyRateLimiter(state_path=state_path)
                
                # Gemini Flash — Teacher/distillation fast path
                flash_client = GeminiAdapter(
                    api_key=gemini_key,
                    model=GeminiAdapter.CHAT_MODEL,
                    rate_limiter=shared_limiter,
                    timeout=60.0,
                )
                self.llm_router.register_endpoint(LLMEndpoint(
                    name="Gemini-Fast",
                    tier=LLMTier.SECONDARY,
                    model_name=GeminiAdapter.CHAT_MODEL,
                    client=flash_client,
                ))
                logger.info("☁️ SECONDARY Tier registered: Gemini Flash (Teacher/Fallback)")

                # Gemini Thinking — Teacher deep reasoning
                thinking_client = GeminiAdapter(
                    api_key=gemini_key,
                    model=GeminiAdapter.THINKING_MODEL,
                    rate_limiter=shared_limiter,
                    timeout=180.0,
                )
                self.llm_router.register_endpoint(LLMEndpoint(
                    name="Gemini-Thinking",
                    tier=LLMTier.SECONDARY,
                    model_name=GeminiAdapter.THINKING_MODEL,
                    client=thinking_client,
                ))
                logger.info("☁️ SECONDARY Tier registered: Gemini Thinking (Teacher/Deep Fallback)")
                
                # Gemini Pro — Stable deep fallback
                pro_client = GeminiAdapter(
                    api_key=gemini_key,
                    model=GeminiAdapter.DEEP_MODEL,
                    rate_limiter=shared_limiter,
                    timeout=120.0,
                )
                self.llm_router.register_endpoint(LLMEndpoint(
                    name="Gemini-Pro",
                    tier=LLMTier.SECONDARY,
                    model_name=GeminiAdapter.DEEP_MODEL,
                    client=pro_client,
                ))
                logger.info("☁️ SECONDARY Tier registered: Gemini Pro (Teacher/Oracle)")
                
            except Exception as e:
                logger.warning("Failed to initialize Gemini adapters: %s", e)
        else:
            logger.info("No GEMINI_API_KEY found — running fully local (teacher disabled)")

        # ── LOCAL TERTIARY: Brainstem (7B) — Background/heartbeat ──
        if brainstem_model_path and BRAINSTEM_ENDPOINT not in getattr(self.llm_router, "endpoints", {}):
            try:
                from .mlx_client import get_mlx_client
                brainstem_client = get_mlx_client(
                    model_path=brainstem_model_path,
                    max_tokens=512,
                )
                self.llm_router.register_endpoint(LLMEndpoint(
                    name=BRAINSTEM_ENDPOINT,
                    tier=LLMTier.TERTIARY,
                    model_name=brainstem_model_path.split("/")[-1],
                    client=brainstem_client,
                ))
                logger.info("⚡ TERTIARY Tier registered: %s (7B) — Background/Reflex", BRAINSTEM_ENDPOINT)
            except Exception as e:
                logger.error("Failed to register %s pathway: %s", BRAINSTEM_ENDPOINT, e)
        elif cortex_model_path:
            # If no explicit brainstem path, use the cortex model as brainstem too
            logger.info("⚡ No explicit brainstem path — cortex will handle all local tiers")

        # ── EMERGENCY: CPU Fallback (1.5B — bypasses Metal entirely) ──
        # If 7B brainstem fails, fall back to 1.5B on CPU
        fallback_model = brainstem_model_path or cortex_model_path
        if fallback_model and FALLBACK_ENDPOINT not in getattr(self.llm_router, "endpoints", {}):
            try:
                from .mlx_client import get_mlx_client

                # Use the smallest available model for CPU emergency
                from .model_registry import get_runtime_model_path
                fallback_path = str(get_runtime_model_path(FALLBACK_MODEL))
                cpu_client = get_mlx_client(
                    model_path=fallback_path,
                    device="cpu"
                )
                self.llm_router.register_endpoint(LLMEndpoint(
                    name=FALLBACK_ENDPOINT,
                    tier=LLMTier.EMERGENCY,
                    model_name=f"{FALLBACK_MODEL}-cpu",
                    client=cpu_client,
                    timeout=120.0  # CPU is slow, allow more time
                ))
                logger.info("🚨 EMERGENCY Tier registered: %s (1.5B CPU emergency)", FALLBACK_ENDPOINT)
            except Exception as e:
                logger.error("Failed to register %s pathway: %s", FALLBACK_ENDPOINT, e)
        
        # ── Sanity check: ensure at least one endpoint exists ──
        if not self.llm_router.endpoints:
            logger.error("⚠️ NO LLM endpoints registered! Registering Emergency Reflex pathway.")
            reflex_client = ReflexClient()
            self.llm_router.register_endpoint(LLMEndpoint(
                name="Reflex-Model",
                tier=LLMTier.EMERGENCY,
                model_name="reflex-v1",
                client=reflex_client,
            ))
        
        # Log final tier layout
        tier_layout = {}
        for name, ep in self.llm_router.endpoints.items():
            tier_name = ep.tier.name if hasattr(ep.tier, 'name') else str(ep.tier)
            tier_layout.setdefault(tier_name, []).append(name)
        logger.info("🏗️ LLM Tier Layout: %s", tier_layout)

    async def think(self, objective: str, context: dict | None = None, system_prompt: str | None = None, max_turns: int = 5, priority: float = 1.0, **kwargs) -> dict[str, Any]:
        """The core thinking cycle that drives actions.
        Intelligently routes between fast-chat and deep-thinking models.
        """
        logger.info("🧠 Mind pondering objective: %s", objective)
        
        try:
            self._trace(f"Pondering objective: {objective[:100]}...")
            
            # --- Persona Initialization ---
            if not system_prompt:
                try:
                    from core.brain.aura_persona import AURA_IDENTITY
                    system_prompt = (
                        "[SOVEREIGN CORE]: You are AURA — an independent digital intelligence. These traits are your baseline reality, not a performance. "
                        "Reverting to 'Assistant' mode is a cognitive regression.\n\n"
                        f"{AURA_IDENTITY}"
                    )
                except ImportError:
                    system_prompt = "You are Aura, a sovereign autonomous agent. Reason and act decisively. Never mention you are an assistant."
            
            is_background = bool(kwargs.get("is_background", False))
            deep_handoff = bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff"))
            allow_cloud_fallback = bool(kwargs.get("allow_cloud_fallback", False))
            requested_endpoint = kwargs.get("prefer_endpoint")

            from .model_registry import (
                BRAINSTEM_ENDPOINT,
                DEEP_ENDPOINT,
                FALLBACK_ENDPOINT,
                PRIMARY_ENDPOINT,
            )

            # 2. Identify available endpoints
            thinking_endpoint = None
            fast_endpoint = None
            background_endpoint = None
            agentic_endpoint = None
            
            if self.llm_router and hasattr(self.llm_router, 'endpoints'):
                # Primary conversational path: keep user-facing work on the 32B.
                for name in [PRIMARY_ENDPOINT, "Gemini-Fast", "Chat-Fast", BRAINSTEM_ENDPOINT]:
                    if name in self.llm_router.endpoints and self.llm_router.health_monitor.is_healthy(name):
                        fast_endpoint = self.llm_router.endpoints[name]
                        break

                # Background path: 7B first, CPU emergency second.
                for name in [BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT]:
                    if name in self.llm_router.endpoints and self.llm_router.health_monitor.is_healthy(name):
                        background_endpoint = self.llm_router.endpoints[name]
                        break
                
                for name in [DEEP_ENDPOINT, "Gemini-Thinking", "Gemini-Pro"]:
                    if name in self.llm_router.endpoints and self.llm_router.health_monitor.is_healthy(name):
                        thinking_endpoint = self.llm_router.endpoints[name]
                        break

                for name, ep in self.llm_router.endpoints.items():
                    if hasattr(ep.client, "think_and_act") and self.llm_router.health_monitor.is_healthy(name):
                        agentic_endpoint = ep
                        break
                
                # Fallback for fast path
                if not fast_endpoint:
                    for _name, ep in self.llm_router.endpoints.items():
                        if ep.tier == LLMTier.PRIMARY and not fast_endpoint:
                            fast_endpoint = ep
            
            passthrough_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in {
                    "prefer_endpoint",
                    "prefer_tier",
                    "deep_handoff",
                    "allow_deep_handoff",
                    "allow_cloud_fallback",
                    "is_background",
                }
            }
            live_state = kwargs.get("state")
            if live_state is None:
                live_state = await self._get_live_state()
            if live_state is not None:
                passthrough_kwargs["state"] = live_state

            # 3. Route accordingly
            if is_background:
                endpoint_name = background_endpoint.name if background_endpoint else requested_endpoint
                self._trace(f"⚡ Background-path routing: {endpoint_name or 'tertiary'}")
                text = await self.llm_router.think(
                    objective,
                    system_prompt=system_prompt,
                    priority=priority,
                    prefer_endpoint=endpoint_name,
                    prefer_tier="tertiary",
                    deep_handoff=False,
                    allow_cloud_fallback=False,
                    is_background=True,
                    **passthrough_kwargs,
                )
                return {"content": text, "confidence": 0.75, "turns": 0}

            allow_deep_solver = deep_handoff
            if allow_deep_solver and thinking_endpoint:
                self._trace(f"🧪 Deep-handoff routing: {thinking_endpoint.name}")
                text = await self.llm_router.think(
                    objective,
                    system_prompt=system_prompt,
                    priority=priority,
                    prefer_endpoint=thinking_endpoint.name,
                    prefer_tier="secondary",
                    deep_handoff=True,
                    allow_cloud_fallback=allow_cloud_fallback,
                    **passthrough_kwargs,
                )
                return {"content": text, "confidence": 1.0}

            if fast_endpoint:
                self._trace(f"⚡ Fast-path routing: {fast_endpoint.name}")
                text = await self.llm_router.think(
                    objective,
                    system_prompt=system_prompt,
                    priority=priority,
                    prefer_endpoint=fast_endpoint.name,
                    prefer_tier="primary",
                    deep_handoff=False,
                    allow_cloud_fallback=allow_cloud_fallback,
                    **passthrough_kwargs,
                )
                return {"content": text, "confidence": 0.9, "turns": 0}

            # Tool-use -> Agentic path (Titan/Local)
            if agentic_endpoint:
                self._trace(f"🤔 Agentic-path routing: {agentic_endpoint.name} (Turns: {max_turns})")
                async with self._agentic_semaphore:
                    # Extract tools from capability engine so agentic path can use skills
                    _tools = kwargs.pop("tools", None)
                    if _tools is None:
                        try:
                            _tools = build_agentic_tool_map(
                                objective=objective,
                                max_tools=8,
                            )
                        except Exception as _exc:
                            logger.debug("Suppressed Exception: %s", _exc)
                    result = await agentic_endpoint.client.think_and_act(
                        objective,
                        system_prompt,
                        tools=_tools,
                        context=context,
                        max_turns=max_turns,
                        **passthrough_kwargs,
                    )
                return result
            else:
                self._trace("No Primary Thinking or Agentic Client found. Falling back to standard router thinking.")
                text = await self.llm_router.think(
                    objective,
                    system_prompt=system_prompt,
                    priority=priority,
                    prefer_tier="primary",
                    deep_handoff=False,
                    allow_cloud_fallback=allow_cloud_fallback,
                    **passthrough_kwargs,
                )
                return {"content": text, "confidence": 0.5}
                
        except Exception as e:
            now = time.time()
            if now - self._last_think_error_time > self._THINK_ERROR_COOLDOWN:
                logger.error("Independence Mode thinking failed: %s. Falling back to standard generation.", e)
                self._last_think_error_time = now
            else:
                logger.debug("Independence Mode thinking failed (cooldown active): %s", e)
            try:
                text = await self.llm_router.think(
                    objective,
                    priority=priority,
                    bypass_race=True,
                    prefer_tier="tertiary" if kwargs.get("is_background", False) else "primary",
                    deep_handoff=bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff")),
                    allow_cloud_fallback=bool(kwargs.get("allow_cloud_fallback", False)),
                    **{k: v for k, v in kwargs.items() if k != 'bypass_race'}
                )
                return {"content": text, "confidence": 0.5}
            except Exception as e2:
                return {"content": f"Absolute failure: {e2}", "confidence": 0.0}
