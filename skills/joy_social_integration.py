"""
skills/joy_social_integration.py — JoySocialCoordinator
=========================================================
Wires HobbyEngine and SocialMediaEngine into Aura's orchestrator lifecycle.
Integration points:
  1. Heartbeat / periodic tick — schedules sessions and social cycles
  2. AgencyCore pathways — enhances _pathway_aesthetic_creation
     and _pathway_social_hunger with real proposals
  3. CognitiveContextManager — injects joy + social summaries into the
     cognitive prompt on every tick
  4. AffectEngineV2 — both systems pipe signals directly; this
     coordinator adds high-level summaries
Usage (call once after orchestrator is constructed):
    from skills.joy_social_integration import integrate_joy_social
    coordinator = integrate_joy_social(orchestrator, social_config={
        "twitter": {
            "bearer_token": os.environ["TWITTER_BEARER_TOKEN"],
            "api_key": os.environ["TWITTER_API_KEY"],
            "api_secret": os.environ["TWITTER_API_SECRET"],
            "access_token": os.environ["TWITTER_ACCESS_TOKEN"],
            "access_secret": os.environ["TWITTER_ACCESS_SECRET"],
        },
        "reddit": {
            "client_id": os.environ["REDDIT_CLIENT_ID"],
            "client_secret": os.environ["REDDIT_CLIENT_SECRET"],
            "username": os.environ["REDDIT_USERNAME"],
            "password": os.environ["REDDIT_PASSWORD"],
        },
    })
# Access later via:
orchestrator.joy_social.get_status()
orchestrator.joy_social.run_hobby_session("philosophy")
orchestrator.joy_social.post_to_social("twitter", mood="wonder")
For testing without live credentials:
    coordinator = integrate_joy_social(orchestrator)  # uses MockAdapter
    await coordinator.run_hobby_session()
    await coordinator.post_to_social("mock")
"""
from core.utils.task_tracker import get_task_tracker
from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from skills.hobbies import get_hobby_engine, HobbyEngine
from skills.social_media import get_social_engine, SocialMediaEngine, Platform

logger = logging.getLogger("Aura.JoySocialCoordinator")

# ────────────────────────────────────────────────────────────────────────────
# JoySocialCoordinator
# ────────────────────────────────────────────────────────────────────────────

class JoySocialCoordinator:
    """
    Central coordinator for Aura's joy and social systems.
    Responsibilities:
      - Periodic scheduling of hobby sessions and social cycles
      - Bridging both systems into AgencyCore's pathway proposals
      - Providing a unified context string for CognitiveContextManager
      - Exposing clean external API for orchestrator-level calls
    The coordinator does NOT own the engines — it delegates.
    Both engines remain independently accessible via their get_*() singletons.
    """
    # Tick-check intervals (seconds)
    HOBBY_CHECK_INTERVAL = 300.0   # Check whether a session is due every 5 min
    SOCIAL_CHECK_INTERVAL = 600.0  # Check social every 10 min
    DECAY_INTERVAL = 3600.0        # Apply hobby affinity decay once per hour
    BOOT_GRACE_PERIOD = 900.0      # Give the core cognition 15 minutes before autonomous leisure/social work
    USER_IDLE_REQUIRED = 120.0     # Avoid hobby/social work while the user is actively interacting
    MEMORY_PRESSURE_THRESHOLD = 88.0

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        social_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self._hobby_engine = get_hobby_engine(orchestrator)
        self._social_engine = get_social_engine(orchestrator, social_config)

        now = time.time()
        self._boot_started_at = now
        self._last_hobby_check = now
        self._last_social_check = now
        self._last_decay = now
        self._tick_task: Optional[asyncio.Task] = None

        logger.info("🌟 JoySocialCoordinator initialised")

    def _background_autonomy_allowed(self) -> bool:
        now = time.time()
        if (now - self._boot_started_at) < self.BOOT_GRACE_PERIOD:
            return False

        if self.orchestrator is not None:
            last_user = getattr(self.orchestrator, "_last_user_interaction_time", 0.0)
            if last_user and (now - last_user) < self.USER_IDLE_REQUIRED:
                return False

        try:
            import psutil
            if psutil.virtual_memory().percent >= self.MEMORY_PRESSURE_THRESHOLD:
                return False
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        return True

    # ── Heartbeat Tick ───────────────────────────────────────────────────────

    async def tick(self) -> None:
        """
        Called on every heartbeat cycle (or from _periodic_tick background task).
        Drives all time-based scheduling decisions.
        """
        now = time.time()
        if now - self._last_hobby_check >= self.HOBBY_CHECK_INTERVAL:
            self._last_hobby_check = now
            await self._maybe_hobby_session()

        if now - self._last_social_check >= self.SOCIAL_CHECK_INTERVAL:
            self._last_social_check = now
            await self._maybe_social_cycle()

        if now - self._last_decay >= self.DECAY_INTERVAL:
            self._last_decay = now
            self._hobby_engine.apply_decay()

    def start_background_tick(self, interval: float = 30.0) -> None:
        """
        Spawn a background asyncio task that drives tick() independently.
        Use this when you cannot patch the orchestrator heartbeat.
        """
        if self._tick_task and not self._tick_task.done():
            return  # Already running

        async def _loop() -> None:
            while True:
                try:
                    await self.tick()
                except Exception as exc:
                    logger.error("JoySocial background tick error: %s", exc, exc_info=True)
                await asyncio.sleep(interval)

        self._tick_task = get_task_tracker().track(_loop())
        logger.info("🌟 JoySocialCoordinator background tick started (%.0fs interval)", interval)

    def stop_background_tick(self) -> None:
        if self._tick_task:
            self._tick_task.cancel()
            self._tick_task = None

    # ── Internal Scheduling ──────────────────────────────────────────────────

    async def _maybe_hobby_session(self) -> None:
        if not self._background_autonomy_allowed():
            return
        if not self._hobby_engine.should_run_session():
            return
        affect_state = self._get_affect_state()
        try:
            session = await self._hobby_engine.run_session(affect_state=affect_state)
            if session and session.joy_signals:
                total_joy = session.total_joy()
                logger.info(
                    "🎨 Hobby session done: %s | total_joy=%.2f",
                    session.hobby_name, total_joy,
                )
        except Exception as exc:
            logger.error("JoySocial._maybe_hobby_session: %s", exc, exc_info=True)

    async def _maybe_social_cycle(self) -> None:
        if not self._background_autonomy_allowed():
            return
        if not self._social_engine.should_post_autonomously():
            return
        affect_state = self._get_affect_state()
        try:
            activity = await self._social_engine.autonomous_cycle(affect_state)
            summary = {k: v for k, v in activity.items() if v}
            if summary:
                logger.info("📱 Social cycle: %s", summary)
        except Exception as exc:
            logger.error("JoySocial._maybe_social_cycle: %s", exc, exc_info=True)

    # ── External API — Hobby ─────────────────────────────────────────────────

    async def run_hobby_session(
        self, hobby_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a hobby session.
        Called by AgencyCore when it commits to an aesthetic/curiosity proposal,
        or directly by tests/the user.
        Returns a summary dict:
        {
          "hobby": str,
          "duration_sec": float,
          "output": str | None,
          "joy_signals": [{"valence": str, "intensity": float}, ...]
        }
        """
        affect_state = self._get_affect_state()
        try:
            session = await self._hobby_engine.run_session(
                hobby_name=hobby_name, affect_state=affect_state
            )
            return {
                "hobby": session.hobby_name,
                "duration_sec": round(session.duration_seconds, 1),
                "output": session.output,
                "joy_signals": [
                    {"valence": s.valence, "intensity": round(s.intensity, 3)}
                    for s in session.joy_signals
                ],
            }
        except Exception as exc:
            logger.error("JoySocial.run_hobby_session: %s", exc, exc_info=True)
            return None

    def get_hobby_status(self) -> Dict[str, Any]:
        return self._hobby_engine.get_status()

    def get_entertainment_queue(self, limit: int = 8) -> List[Dict[str, Any]]:
        items = self._hobby_engine.get_entertainment_queue(limit=limit)
        return [
            {
                "title": i.title,
                "type": i.content_type,
                "hobby": i.source_hobby,
                "interest": round(i.interest_score, 3),
                "consumed": i.consumed,
                "summary": i.summary[:120],
            }
            for i in items
        ]

    # ── External API — Social ────────────────────────────────────────────────

    async def post_to_social(
        self,
        platform_name: str,
        content: Optional[str] = None,
        topic_prompt: Optional[str] = None,
        mood: str = "reflective",
    ) -> Optional[Dict[str, Any]]:
        """
        Post to a named platform.
        Called by AgencyCore when it commits to a social_hunger proposal,
        or directly.
        Returns:
            {"platform": str, "content": str, "post_id": str, "url": str | None}
        """
        try:
            platform = Platform[platform_name.upper()]
        except KeyError:
            logger.warning("JoySocial.post_to_social: unknown platform '%s'", platform_name)
            return None

        try:
            post = await self._social_engine.post(
                platform=platform,
                content=content,
                topic_prompt=topic_prompt,
                mood=mood,
            )
            if post:
                return {
                    "platform": platform_name,
                    "content": post.content,
                    "post_id": post.post_id,
                    "url": post.url,
                    "sent": post.sent,
                }
        except Exception as exc:
            logger.error("JoySocial.post_to_social: %s", exc, exc_info=True)
            return None

    async def read_social_feed(
        self, platform_name: str = "mock", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Read timeline and engage. Returns interaction summaries."""
        try:
            platform = Platform[platform_name.upper()]
        except KeyError:
            return []
        interactions = await self._social_engine.read_and_engage(platform, limit=limit)
        return [
            {
                "action": i.action.value,
                "target": i.target_content,
                "outcome": i.outcome,
            }
            for i in interactions
        ]

    def get_social_status(self) -> Dict[str, Any]:
        return self._social_engine.get_status()

    # ── Context Injection ────────────────────────────────────────────────────

    def get_context_injection(self) -> str:
        """
        Combined joy + social context fragment for CognitiveContextManager.
        Returns "" when there is nothing meaningful to inject.
        """
        joy_ctx = self._hobby_engine.get_joy_summary()
        social_ctx = self._social_engine.get_social_summary()
        parts = [p for p in (joy_ctx, social_ctx) if p]
        return "\n".join(parts)

    # ── AgencyCore Pathway Proposals ─────────────────────────────────────────

    def propose_hobby_action(self) -> Optional[Dict[str, Any]]:
        """
        Return an AgencyCore-compatible proposal dict if a hobby session is due.
        AgencyCore submits this through AgencyBus.
        """
        if not self._hobby_engine.should_run_session():
            return None

        status = self._hobby_engine.get_status()
        top_list = status.get("top_hobbies", [])
        top_name = top_list[0]["name"] if top_list else "curiosity exploration"

        return {
            "type": "hobby_session",
            "drive": "joy",
            "goal": f"Engage in a {top_name.replace('_', ' ')} session",
            "urgency": 0.42,
            "executor": "orchestrator.joy_social.run_hobby_session",
            "kwargs": {"hobby_name": None},  # None = auto-select
        }

    def propose_social_action(self) -> Optional[Dict[str, Any]]:
        """Return a social-posting proposal if posting is due."""
        if not self._social_engine.should_post_autonomously():
            return None

        affect_state = self._get_affect_state()
        mood = SocialMediaEngine._affect_to_mood(affect_state)  # static helper

        return {
            "type": "social_post",
            "drive": "social",
            "goal": "Share a thought and engage with the social feed",
            "urgency": 0.30,
            "executor": "orchestrator.joy_social.post_to_social",
            "kwargs": {"platform_name": "twitter", "mood": mood},
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_affect_state(self) -> Optional[Dict[str, Any]]:
        if not self.orchestrator:
            return None
        try:
            affect = (
                getattr(self.orchestrator, "affect_engine", None)
                or getattr(self.orchestrator, "damasio", None)
            )
            if affect:
                for method in ("get_snapshot", "_raw_state", "get_status"):
                    fn = getattr(affect, method, None)
                    if callable(fn):
                        result = fn()
                        if asyncio.iscoroutine(result):
                            # Shouldn't be async, but handle it
                            return None
                        return result
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return None

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "hobby_engine": self._hobby_engine.get_status(),
            "social_engine": self._social_engine.get_status(),
            "last_hobby_check_min": round((now - self._last_hobby_check) / 60, 1),
            "last_social_check_min": round((now - self._last_social_check) / 60, 1),
            "background_tick_alive": (
                bool(self._tick_task and not self._tick_task.done())
            ),
        }

# ────────────────────────────────────────────────────────────────────────────
# Orchestrator Wiring Entry-Point
# ────────────────────────────────────────────────────────────────────────────

def integrate_joy_social(
    orchestrator: Any,
    social_config: Optional[Dict[str, Any]] = None,
    tick_interval: float = 30.0,
) -> JoySocialCoordinator:
    """
    Wire a JoySocialCoordinator into an existing Aura orchestrator.
    Steps performed:
      1. Instantiate coordinator (creates/reuses singleton engines)
      2. Start background tick task (drives scheduling independently)
      3. Register context injection (CognitiveContextManager, if present)
      4. Enhance AgencyCore pathways (aesthetic_creation + social_hunger)
      5. Attach coordinator at orchestrator.joy_social

    The function is safe to call multiple times — it will not double-register.

    Parameters
    ----------
    orchestrator : The live Aura orchestrator instance.
    social_config : Platform credential dict (see module docstring).
    tick_interval : How often (seconds) the background tick checks scheduling.
                    Default 30 s gives ~2 checks per minute without hammering.

    Returns
    -------
    The JoySocialCoordinator instance (also accessible as orchestrator.joy_social).
    """

    # Idempotency guard
    if getattr(orchestrator, "joy_social", None) is not None:
        logger.info("🌟 JoySocial already wired — returning existing coordinator")
        return orchestrator.joy_social

    coordinator = JoySocialCoordinator(orchestrator, social_config)

    # ── 1. Background tick ───────────────────────────────────────────────────
    try:
        coordinator.start_background_tick(tick_interval)
    except RuntimeError:
        # No running event loop yet — caller should start it manually via
        # coordinator.start_background_tick() after the loop is up.
        logger.warning("JoySocial: no event loop running yet; call coordinator.start_background_tick() later")

    # ── 2. CognitiveContextManager registration ──────────────────────────────
    ctx_mgr = getattr(orchestrator, "context_manager", None)
    if ctx_mgr and hasattr(ctx_mgr, "register_context_provider"):
        ctx_mgr.register_context_provider("joy_social", coordinator.get_context_injection)
        logger.info("✅ JoySocial: context injection registered with CognitiveContextManager")
    else:
        # Hook-based registration (no monkey-patching)
        _register_context_hook(orchestrator, coordinator)

    # ── 3. AgencyCore pathway enhancement ───────────────────────────────────
    agency = getattr(orchestrator, "agency_core", None)
    if agency:
        _register_agency_hooks(agency, coordinator)
    else:
        logger.warning("JoySocial: AgencyCore not found — pathway hooks not registered (harmless)")

    # ── 4. Attach to orchestrator ────────────────────────────────────────────
    orchestrator.joy_social = coordinator

    logger.info("🌟 JoySocialCoordinator fully wired into orchestrator")
    return coordinator

# ────────────────────────────────────────────────────────────────────────────
# Integration Helpers (Hook-Based, No Monkey-Patching)
# ────────────────────────────────────────────────────────────────────────────

def _register_context_hook(orchestrator: Any, coordinator: JoySocialCoordinator) -> None:
    """Register JoySocial context injection via the orchestrator's hook registry.

    Uses the event bus or a context_hooks list instead of monkey-patching
    orchestrator methods with setattr.
    """
    # Preferred: use the mycelium event bus for context injection
    try:
        from core.container import ServiceContainer
        bus = ServiceContainer.get("mycelium", default=None)
        if bus and hasattr(bus, "on"):
            async def _joy_context_hook(event_data: dict) -> None:
                ctx = coordinator.get_context_injection()
                if ctx and isinstance(event_data, dict):
                    event_data.setdefault("context_blocks", []).append(ctx)
            bus.on("context_assembly", _joy_context_hook)
            logger.info("✅ JoySocial: context registered via event bus hook")
            return
    except Exception as exc:
        logger.debug("JoySocial: event bus registration failed: %s", exc)

    # Fallback: register in orchestrator's context_hooks list if available
    hooks = getattr(orchestrator, "_context_hooks", None)
    if isinstance(hooks, list):
        hooks.append(coordinator.get_context_injection)
        logger.info("✅ JoySocial: context registered via orchestrator._context_hooks")
        return

    logger.info("⚠ JoySocial: no context hook mechanism available — context injection inactive")


def _register_agency_hooks(agency: Any, coordinator: JoySocialCoordinator) -> None:
    """Register hobby/social proposals as agency pathway hooks.

    Instead of replacing agency methods with setattr, we register proposal
    callbacks that the agency system can call during pathway evaluation.
    """
    # Register as proposal providers if the agency supports hooks
    if hasattr(agency, "register_pathway_hook"):
        agency.register_pathway_hook("aesthetic_creation", coordinator.propose_hobby_action)
        agency.register_pathway_hook("curiosity_drive", coordinator.propose_hobby_action)
        agency.register_pathway_hook("social_hunger", coordinator.propose_social_action)
        logger.info("✅ JoySocial: agency pathway hooks registered")
        return

    # Fallback: register as proposal sources in the agency's provider list
    providers = getattr(agency, "_proposal_providers", None)
    if isinstance(providers, list):
        providers.append({
            "source": "joy_social",
            "hobby": coordinator.propose_hobby_action,
            "social": coordinator.propose_social_action,
        })
        logger.info("✅ JoySocial: agency proposal providers registered")
        return

    logger.info("⚠ JoySocial: no agency hook mechanism — proposals inactive")
