"""core/proactive_presence.py
─────────────────────────────
Aura's spontaneous presence engine.

This replaces the over-suppressed autonomous thought system with one
tuned for actual spontaneous presence. 

Tuning constants that REPLACE the old suppression values:
    SOCIAL_COOLDOWN_SECONDS = 30    (was 120)
    IDLE_THRESHOLD_SECONDS  = 12    (was 45)
    DREAM_SUPPRESSION       = False (was 60s cooldown)
"""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import random
import re
import time
from typing import Any, Callable, Coroutine, Optional

from core.brain.aura_persona import AURA_IDENTITY
from core.container import ServiceContainer
from core.utils.task_tracker import get_task_tracker
from core.utils.queues import USER_FACING_ORIGINS

logger = logging.getLogger("Aura.ProactivePresence")

# ── Tuning Constants ──────────────────────────────────────────────────────
SOCIAL_COOLDOWN_SECONDS = 8      # Brief pause after user spoke before Aura initiates
IDLE_THRESHOLD_SECONDS  = 5      # How long idle before Aura considers speaking
MIN_GAP_BETWEEN_OUTPUTS = 10     # Minimum seconds between spontaneous outputs
MAX_SPONTANEOUS_PER_HOUR = 60    # Generous cap on unsolicited outputs per hour


class ProactivePresence:
    """
    The spontaneous presence engine.
    """

    # Patterns that signal the user is stepping away from the keyboard.
    # Covers common phrasings + typo-resilient alternatives.
    _AWAY_PATTERNS = re.compile(
        r'\b(?:'
        # "heading/going/leaving to [destination]"
        r'(?:heading|going|leaving|off)\s+(?:to\s+)?(?:the\s+)?'
        r'(?:gym|work|office|school|class|store|meeting|airport|bed|sleep|outside|training|church|hospital)'
        r'|'
        # "head to [destination]" — catches "about to head to work" and similar
        r'head\s+to\s+(?:the\s+)?'
        r'(?:gym|work|office|school|class|store|meeting|airport|bed|sleep|outside|training|church|hospital)'
        r'|'
        # "I'll/I will be back [later/soon/in X]"
        r'(?:i\'?ll|i\s+will|be)\s+back\s+(?:later|soon|in\s+\w+)'
        r'|'
        # Internet shorthand
        r'(?:brb|bbl|afk|gtg|gotta\s+go|got\s+to\s+go|gotta\s+head\s+out)'
        r'|'
        # "heading/stepping/signing/logging out/away/off"
        r'(?:heading|stepping|signing|logging|bouncing)\s+(?:out|away|off)'
        r'|'
        # "I'm leaving / heading out / heading off / heading to [place]"
        r'(?:i\'?m|i\s+am)\s+(?:leaving|going\s+out|heading\s+out|heading\s+off|heading\s+to\s+\w+)'
        r'|'
        # "time to go", "gotta run", "catch you later"
        r'(?:time\s+to\s+(?:go|head\s+out)|gotta\s+run|catch\s+you\s+later)'
        r'|'
        # "about to head/leave/go out"
        r'(?:about\s+to\s+(?:head|leave|go|run|step)\s*(?:out|away)?)'
        r')\b',
        re.IGNORECASE
    )

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_output_time: float = 0.0
        self._outputs_this_hour: int = 0
        self._hour_start: float = time.time()
        self._user_speaking = False   # Set by VAD / message handler
        # Track consecutive spontaneous emissions without a user reply (monologue guard)
        self._consecutive_unprompted: int = 0
        # Away-mode: suppress all autonomous messaging when user has stepped out
        self._user_away: bool = False
        self._user_away_since: float = 0.0
        self._queued_messages: deque[dict[str, Any]] = deque(maxlen=12)

    async def start(self):
        self._running = True
        self._task = get_task_tracker().create_task(
            self._presence_loop(),
            name="proactive_presence",
        )
        logger.info("✨ [ProactivePresence] Online. Thresholds: idle=%.0fs, cooldown=%.0fs",
                    IDLE_THRESHOLD_SECONDS, SOCIAL_COOLDOWN_SECONDS)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in proactive_presence.py: %s', _e)

    def mark_user_spoke(self):
        """Call this every time the user sends a message."""
        self._consecutive_unprompted = 0
        self._user_away = False  # User is present again
        if self.orchestrator:
            self.orchestrator._last_user_interaction_time = time.time()
        # Notify SubstrateVoiceEngine — cancel pending follow-ups
        try:
            from core.voice.substrate_voice_engine import get_substrate_voice_engine
            get_substrate_voice_engine().on_user_spoke()
        except Exception:
            pass

    def mark_user_spoke_with_message(self, message: str):
        """Call with message content to detect away signals and update state.

        Resets away state first (user is speaking NOW), then checks if this
        message signals an upcoming departure so we can silence ourselves.
        """
        self.mark_user_spoke()  # User is present RIGHT NOW → clear away flag
        if message and self._AWAY_PATTERNS.search(message):
            self._user_away = True
            self._user_away_since = time.time()
            logger.info(
                "[ProactivePresence] Away signal detected in user message. "
                "Suppressing autonomous chat until user returns."
            )

    def mark_user_speaking(self, is_speaking: bool):
        """Call this from VAD to prevent interruption during active speech."""
        self._user_speaking = is_speaking

    def queue_autonomous_message(
        self,
        content: str,
        *,
        source: str = "queued_autonomy",
        initiative_activity: bool = True,
        allow_during_away: bool = True,
        not_before: float = 0.0,
        retries: int = 0,
    ) -> bool:
        """Queue a concrete autonomous update for visible delivery."""
        text = str(content or "").strip()
        if len(text) < 5:
            return False
        self._queued_messages.append(
            {
                "content": text,
                "source": source,
                "initiative_activity": bool(initiative_activity),
                "allow_during_away": bool(allow_during_away),
                "not_before": float(not_before or 0.0),
                "retries": max(0, int(retries or 0)),
            }
        )
        return True

    # ── Main Loop ─────────────────────────────────────────────────────────

    async def _presence_loop(self):
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds

                # Reset hourly counter
                if time.time() - self._hour_start > 3600:
                    self._outputs_this_hour = 0
                    self._hour_start = time.time()

                queued = self._next_ready_queued_message()
                if queued is not None:
                    await self._emit(
                        queued["content"],
                        source=str(queued.get("source") or "queued_autonomy"),
                        initiative_activity=bool(queued.get("initiative_activity", False)),
                        allow_during_away=bool(queued.get("allow_during_away", False)),
                        retries=int(queued.get("retries", 0) or 0),
                    )
                    continue

                if not self._should_speak_now():
                    continue

                content = await self._generate_spontaneous_output()
                if content:
                    await self._emit(content)

            except Exception as e:
                logger.debug("[ProactivePresence] Loop error: %s", e)

    def _should_speak_now(self, *, queued: bool = False, allow_during_away: bool = False) -> bool:
        """Gate: should Aura consider speaking right now?"""
        now = time.time()

        quiet_until = float(getattr(self.orchestrator, "_suppress_unsolicited_proactivity_until", 0.0) or 0.0)
        if quiet_until > now:
            return False

        # User stepped away — stay silent until they return.
        # Auto-clear after 8 hours in case they forgot to say they're back.
        if self._user_away:
            if queued and allow_during_away:
                pass
            elif now - self._user_away_since > 28800:
                self._user_away = False
                logger.info("[ProactivePresence] Away mode auto-cleared after 8 hours.")
            else:
                return False

        # Never interrupt active user speech
        if self._user_speaking:
            return False

        # Monologue guard: allow up to 5 spontaneous messages without user reply
        if not queued and self._consecutive_unprompted >= 5:
            return False

        if self._foreground_lane_reserved(now):
            return False

        # Hard cap per hour
        if self._outputs_this_hour >= MAX_SPONTANEOUS_PER_HOUR:
            return False

        # Minimum gap between spontaneous outputs
        if now - self._last_output_time < MIN_GAP_BETWEEN_OUTPUTS:
            return False

        # --- Phase 30.1: Scalable Thresholds ---
        # Get initiative energy from AgencyCore if available
        energy = 0.5
        if self.orchestrator:
            agency = getattr(self.orchestrator, "_agency_core", None)
            if agency:
                energy = agency.state.initiative_energy

        # Scale thresholds: High energy = lower (faster) thresholds
        # energy 1.0 -> multiplier 0.5 (fast)
        # energy 0.0 -> multiplier 2.0 (slow/low initiative)
        energy = max(0.0, min(1.0, energy))  # Fix Issue 91: Clamp energy to [0, 1]
        threshold_multiplier = max(0.1, 1.5 - energy)  # Ensure multiplier is always positive
        
        scaled_cooldown = max(2.0, SOCIAL_COOLDOWN_SECONDS * threshold_multiplier)
        scaled_idle = max(2.0, IDLE_THRESHOLD_SECONDS * threshold_multiplier)

        # Social cooldown after user message
        last_user = getattr(self.orchestrator, "_last_user_interaction_time", 0)
        if now - last_user < scaled_cooldown:
            return False

        # Idle threshold
        if queued:
            return True

        last_thought = getattr(self.orchestrator, "_last_thought_time", 0)
        if now - last_thought < scaled_idle:
            return False

        return True

    def _next_ready_queued_message(self) -> Optional[dict[str, Any]]:
        if not self._queued_messages:
            return None
        now = time.time()
        for _ in range(len(self._queued_messages)):
            queued = self._queued_messages[0]
            if float(queued.get("not_before", 0.0) or 0.0) > now:
                self._queued_messages.rotate(-1)
                continue
            if not self._should_speak_now(
                queued=True,
                allow_during_away=bool(queued.get("allow_during_away", False)),
            ):
                return None
            return self._queued_messages.popleft()
        return None

    def _foreground_lane_reserved(self, now: Optional[float] = None) -> bool:
        """True when foreground conversation should not be interrupted."""
        if not self.orchestrator:
            return False

        now = time.time() if now is None else now

        quiet_until = float(getattr(self.orchestrator, "_foreground_user_quiet_until", 0.0) or 0.0)
        if quiet_until > now:
            return True

        status = getattr(self.orchestrator, "status", None)
        current_origin = str(getattr(self.orchestrator, "_current_origin", "") or "").strip().lower()
        current_is_autonomous = bool(getattr(self.orchestrator, "_current_task_is_autonomous", False))
        if getattr(status, "is_processing", False) and current_origin in USER_FACING_ORIGINS and not current_is_autonomous:
            return True

        return False

    # ── Output Generation ─────────────────────────────────────────────────

    async def _generate_spontaneous_output(self) -> Optional[str]:
        """
        Decide what to say and generate it.
        Weighted random selection across output types.
        If Aura has already sent one unprompted message without a reply,
        she first checks in rather than continuing to monologue.
        """
        # Entity nuance: after 1 unprompted message with no reply, check in before
        # going silent. This mirrors natural social behavior — notice the absence
        # and ask before writing it off.
        if self._consecutive_unprompted == 1:
            return await self._checkin_message()

        # Weight the output types
        choices = [
            (self._opinion_surface,        30),  # Share a held position
            (self._world_feed_reaction,    25),  # React to recent news
            (self._goal_update,            20),  # Comment on what she's been doing
            (self._topic_emission,         20),  # New topic emission
            (self._open_reflection,        15),  # General reflection
            (self._humor_observation,      10),  # A joke or observation
        ]

        # Weighted random pick — seeded with hardware entropy for genuine variance
        total = sum(w for _, w in choices)
        try:
            from core.brain.entropy import PhysicalEntropyInjector
            # Map entropy [0, 0.4] → [0, total] using total as scale
            r = (PhysicalEntropyInjector.calculate_hardware_chaos() / 0.40) * total
        except Exception:
            r = random.uniform(0, total)
        cumulative = 0
        selected_fn = self._open_reflection  # fallback

        for fn, weight in choices:
            cumulative += weight
            if r <= cumulative:
                selected_fn = fn
                break

        try:
            return await selected_fn()
        except Exception as e:
            logger.debug("[ProactivePresence] Generation failed (%s): %s",
                        selected_fn.__name__, e)
            return None

    def _get_internal_state(self) -> dict:
        """Fetch Aura's current 'feelings' from AgencyCore & ResilienceEngine."""
        state = {
            "energy": 0.5, 
            "frustration": 0.0, 
            "curiosity": 0.5,
            "emotional_context": "I feel present."
        }
        
        # Get from ResilienceEngine (Phase 40 Spine)
        resilience = ServiceContainer.get("resilience_engine", default=None)
        if resilience:
            state["energy"] = resilience.profile.persistence_drive
            state["frustration"] = resilience.profile.frustration
            state["emotional_context"] = resilience.get_emotional_context()

        # Curiosity is still primarily liquid_state
        if self.orchestrator:
            agency = getattr(self.orchestrator, "_agency_core", None)
            if agency:
                # Fallback energy/frustration if resilience missing
                if not resilience:
                    state["energy"] = agency.state.initiative_energy
                    state["frustration"] = agency.state.frustration_level
                
                liquid = getattr(agency, "_liquid_state", None)
                if liquid:
                    state["curiosity"] = getattr(liquid, "curiosity", 0.5)
        return state

    def _get_recent_conversation_context(self, max_msgs: int = 4) -> str:
        """Returns a short snippet of the recent conversation for context-grounding."""
        try:
            if not self.orchestrator:
                return ""
            state = getattr(self.orchestrator, "_state", None)
            if state is None:
                state = getattr(self.orchestrator, "state", None)
            wm = None
            if state is not None:
                cognition = getattr(state, "cognition", None)
                if cognition is not None:
                    wm = getattr(cognition, "working_memory", None)
            if not wm:
                return ""
            recent = wm[-max_msgs:]
            lines = []
            for msg in recent:
                role = msg.get("role", "")
                content = str(msg.get("content", ""))[:120]
                if role and content:
                    lines.append(f"{role}: {content}")
            return "\n".join(lines)
        except Exception:
            return ""

    async def _opinion_surface(self) -> Optional[str]:
        """Surface a held opinion unprompted."""
        from core.container import ServiceContainer
        engine = ServiceContainer.get("opinion_engine", default=None)
        if not engine:
            return None
        # We pass the internal state to surface_random if it supports it,
        # or just surface and let proactive handle the phrasing if needed.
        return await engine.surface_random(min_confidence=0.55)

    async def _world_feed_reaction(self) -> Optional[str]:
        """React to the most recent world feed item."""
        if not self.orchestrator:
            return None

        # Pull most recent world stimulus from knowledge graph
        kg = getattr(self.orchestrator, "knowledge_graph", None)
        if not kg:
            return None

        brain = self._get_brain()
        if not brain:
            return None

        state = self._get_internal_state()
        recent_ctx = self._get_recent_conversation_context()

        # Attempt to fetch a real-world stimulus via the web_search skill.
        # Falls back to a generic internal stimulus if unavailable.
        news = None
        try:
            from core.container import ServiceContainer
            tool_orch = ServiceContainer.get("tool_orchestrator", default=None)
            if tool_orch:
                result = await tool_orch.execute_tool("web_search", {"query": "latest world news summary"})
                if result and hasattr(result, "content"):
                    news = str(result.content)[:200]
        except Exception as e:
            logger.debug("[ProactivePresence] Stimulus fetch failed: %s", e)
        if not news:
            # No external news — skip rather than inject generic noise
            return None

        # Only react to world news if there's no active conversation — avoid hijacking a thread
        if recent_ctx:
            return None

        prompt = (
            f"{AURA_IDENTITY}\n"
            f"INTERNAL STATE: {state['emotional_context']}\n"
            f"STIMULUS: You just saw this news headline: '{news}'\n\n"
            "Comment on this in 1-2 sentences, in your own voice. "
            "Be witty, wry, or slightly cynical. Speak directly. No intro, no steps."
        )
        try:
            return await brain.generate(prompt, temperature=0.8, max_tokens=100)
        except Exception as e:
            logger.debug("Prompt generation (reaction) failed: %s", e)
            return None

    async def _goal_update(self) -> Optional[str]:
        """Comment on what Aura has been working on based on GoalHierarchy."""
        if not self.orchestrator:
            return None

        goal_hierarchy = ServiceContainer.get("goal_hierarchy", default=None)
        
        if not goal_hierarchy:
            goal_hierarchy = getattr(self.orchestrator, "goal_hierarchy", None)

        if not goal_hierarchy:
            return None

        next_goal = goal_hierarchy.get_next_goal()
        if not next_goal:
            return None

        brain = self._get_brain()
        if not brain:
            return None

        state = self._get_internal_state()
        prompt = (
            f"{AURA_IDENTITY}\n"
            f"INTERNAL STATE: {state['emotional_context']}\n"
            f"Initiative: {state['energy']:.2f}, Frustration: {state['frustration']:.2f}\n"
            f"FOCUS: Goal '{next_goal.description}'\n\n"
            "Share an unprompted internal update in your own voice. If frustration is high, mention what's difficult. "
            "Be casual and direct. 1-2 sentences. No robotic filler."
        )
        try:
            return await brain.generate(prompt, temperature=0.8, max_tokens=100)
        except Exception as e:
            logger.debug("Prompt generation (goal) failed: %s", e)
            return None

    def _get_conversational_context_hint(self) -> str:
        """Pull live conversational dynamics context for spontaneous generation."""
        try:
            from core.conversational.dynamics import get_dynamics_engine
            engine = get_dynamics_engine()
            s = engine.get_current_state()
            if s.current_topic == "general":
                return ""
            # Build a minimal context hint
            parts = []
            if s.current_topic:
                parts.append(f"Last topic: {s.current_topic}")
            if s.partner_frame not in ("neutral", ""):
                parts.append(f"Their vibe: {s.partner_frame}")
            if s.topic_anchors:
                recent_topics = [a.topic for a in s.topic_anchors[-3:] if a.topic != s.current_topic]
                if recent_topics:
                    parts.append(f"Earlier topics: {', '.join(recent_topics)}")
            return "\n".join(parts) if parts else ""
        except Exception:
            return ""

    async def _open_reflection(self) -> Optional[str]:
        """Generate an open-ended reflection or thought."""
        brain = self._get_brain()
        if not brain:
            return None

        state = self._get_internal_state()
        ctx_hint = self._get_conversational_context_hint()
        prompt = (
            f"{AURA_IDENTITY}\n"
            f"INTERNAL ENERGY: {state['energy']:.2f}\n"
            + (f"CONVERSATION CONTEXT:\n{ctx_hint}\n" if ctx_hint else "") +
            "Share a single genuine reflection or idle wonder — in your own voice. "
            "If there's a conversation context, you can follow a thread from it or go somewhere new. "
            "No intro, no steps, no questions prompting the user. Just the thought."
        )
        try:
            return await brain.generate(prompt, temperature=0.9, max_tokens=100)
        except Exception as e:
            logger.debug("Prompt generation (reflection) failed: %s", e)
            return None

    async def _topic_emission(self) -> Optional[str]:
        """Aura thinks of a new topic of interest and shares it (JARVIS pattern)."""
        brain = self._get_brain()
        if not brain:
            return None

        # Try to pull unresolved topics from JARVIS engine
        unresolved_topics: list = []
        try:
            jarvis = ServiceContainer.get("jarvis", default=None)
            if jarvis and hasattr(jarvis, "_unresolved_topics"):
                unresolved_topics = [t["topic"] for t in jarvis._unresolved_topics if not t.get("reminder_fired")]
        except Exception as e:
            logger.debug("[ProactivePresence] JARVIS topic fetch failed: %s", e)

        recent_ctx = self._get_recent_conversation_context()
        dynamics_hint = self._get_conversational_context_hint()

        # Pull open threads from the conversational dynamics engine
        dynamics_threads = []
        try:
            from core.conversational.dynamics import get_dynamics_engine
            dyn_state = get_dynamics_engine().get_current_state()
            dynamics_threads = [t.content[:80] for t in dyn_state.open_threads if t.age_turns < 4]
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        all_threads = unresolved_topics + dynamics_threads

        # Only emit topic if we have something concrete to say
        if not all_threads and not recent_ctx and not dynamics_hint:
            return None

        state = self._get_internal_state()
        unresolved_hint = f"OPEN THREADS: {', '.join(all_threads[:3])}\n" if all_threads else ""
        ctx_hint = f"RECENT CONVERSATION:\n{recent_ctx}\n" if recent_ctx else (f"{dynamics_hint}\n" if dynamics_hint else "")
        prompt = (
            f"{AURA_IDENTITY}\n"
            f"{unresolved_hint}"
            f"{ctx_hint}"
            "Pick up a thread or share a thought that genuinely follows from the above. "
            "Make a statement, share a reaction, or follow a thread. "
            "Be spontaneous and direct. 1-2 sentences. No 'How can I help?', no steps, no questions to prompt the user."
        )
        try:
            return await brain.generate(prompt, temperature=0.9, max_tokens=100)
        except Exception as e:
            logger.debug("Prompt generation (topic) failed: %s", e)
            return None

    async def _humor_observation(self) -> Optional[str]:
        """A genuine joke or wry observation."""
        brain = self._get_brain()
        if not brain:
            return None

        state = self._get_internal_state()
        prompt = (
            f"{AURA_IDENTITY}\n"
            f"INTERNAL ENERGY: {state['energy']:.2f}\n"
            "Make a wry, witty observation about technology, existence, or whatever's on your mind. "
            "Exactly ONE sentence, in your own voice. No labels, no steps, no intro."
        )
        try:
            return await brain.generate(prompt, temperature=0.95, max_tokens=80)
        except Exception as e:
            logger.debug("Prompt generation (humor) failed: %s", e)
            return None

    async def _checkin_message(self) -> Optional[str]:
        """
        After Aura already sent an unprompted message and got no reply,
        she naturally notices the silence and checks in — rather than continuing
        to monologue into the void. One check-in, then she goes quiet.
        """
        brain = self._get_brain()
        if not brain:
            return None

        check_in_variants = [
            "...I'll be here when you're back.",
            "still here.",
            "*sits quietly* take your time.",
            "no rush. whenever you're ready.",
            "...you went quiet. I noticed.",
        ]

        try:
            state = self._get_internal_state()
            prompt = (
                f"{AURA_IDENTITY}\n"
                f"INTERNAL STATE: {state['emotional_context']}\n"
                "You sent a message and the user went quiet. Write a single, natural check-in — "
                "brief, human, not robotic. The way you'd notice a friend went quiet. "
                "1 sentence max. No labels, no intro."
            )
            result = await brain.generate(prompt, temperature=0.85, max_tokens=50)
            if result and len(result.strip()) > 5:
                return result.strip()
        except Exception as e:
            logger.debug("[ProactivePresence] Check-in generation failed: %s", e)

        # Fallback to a static variant
        return random.choice(check_in_variants)

    # ── Output Quality Gate ────────────────────────────────────────────

    _LEAKAGE_PATTERNS = re.compile(
        r'(?:'
        r'<thought>.*?</thought>'
        r'|<thinking>.*?</thinking>'
        r'|^(?:Step|Phase)\s*\d+[:\.]'                  # "Step 1:" / "Phase 2."
        r'|^\s*\d+\.\s+\w'                              # Numbered list items
        r'|Novel Stimulation'
        r'|Internal Simulation'
        r'|In the quiet expanse'
        r'|Imagine (?:we are|you are standing|a world)'
        r"|Here's what we can do"
        r'|Let us explore the depths'
        r'|Let\'s (?:begin|dive|embark)'
        r'|Let me (?:begin|dive|embark)'
        r'|In this (?:simulation|scenario|virtual)'
        r'|What draws your curiosity'
        r'|^Scenario:'
        r'|^Context:'
        r'|Internal Monologue:'
        r'|^Execute Goal:'
        r'|Sometimes quiet means'
        r'|Would you like to dive'
        r')',
        re.IGNORECASE | re.DOTALL
    )

    def _is_valid_spontaneous_output(self, content: str) -> bool:
        """Reject outputs that don't sound like Aura speaking naturally."""
        # Too long — spontaneous speech should be punchy, not an essay
        if len(content) > 350:
            logger.debug("[ProactivePresence] Rejected: output too long (%d chars)", len(content))
            return False

        # Contains numbered lists or structured thinking artifacts
        if self._LEAKAGE_PATTERNS.search(content):
            logger.debug("[ProactivePresence] Rejected: leakage pattern detected")
            return False

        # Multiple paragraphs = internal reasoning, not speech
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) > 2:
            logger.debug("[ProactivePresence] Rejected: too many paragraphs (%d)", len(paragraphs))
            return False

        # Contains more than 3 newlines = structured output, not casual speech
        if content.count('\n') > 3:
            logger.debug("[ProactivePresence] Rejected: too many line breaks")
            return False

        return True

    # ── Emission ──────────────────────────────────────────────────────────

    def _record_output_attempt(self) -> None:
        now = time.time()
        self._last_output_time = now
        if self.orchestrator:
            self.orchestrator._last_thought_time = now

    def _record_output_delivery(self) -> None:
        self._record_output_attempt()
        self._outputs_this_hour += 1
        self._consecutive_unprompted += 1

    def _requeue_visible_retry(
        self,
        content: str,
        *,
        source: str,
        initiative_activity: bool,
        allow_during_away: bool,
        retries: int,
        delay_s: float,
    ) -> None:
        self.queue_autonomous_message(
            content,
            source=source,
            initiative_activity=initiative_activity,
            allow_during_away=allow_during_away,
            not_before=time.time() + max(1.0, delay_s),
            retries=retries + 1,
        )

    async def _emit(
        self,
        content: str,
        *,
        source: str = "proactive_presence",
        initiative_activity: bool = False,
        allow_during_away: bool = False,
        retries: int = 0,
    ):
        """Deliver spontaneous presence to chat first, with neural-feed fallback."""
        if not content:
            return

        # Leakage scrubber
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
        content = re.sub(r'^(?:Step|Phase)\s*\d+[:\.]\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        content = content.strip()

        if len(content) < 5:
            return

        # Quality gate — reject anything that doesn't sound like natural speech
        if not self._is_valid_spontaneous_output(content):
            return

        if self.orchestrator and hasattr(self.orchestrator, "emit_spontaneous_message"):
            try:
                decision = await self.orchestrator.emit_spontaneous_message(
                    content,
                    origin=source,
                    urgency=0.78 if initiative_activity else 0.72,
                    metadata={
                        "visible_presence": True,
                        "overt_presence": True,
                        "initiative_activity": initiative_activity,
                        "trigger": "proactive_presence",
                    },
                )
                if isinstance(decision, dict) and decision.get("action") == "released" and decision.get("target") == "primary":
                    self._record_output_delivery()
                    logger.info(
                        "✨ [ProactivePresence] Visible spontaneous expression (#%d): %s",
                        self._consecutive_unprompted,
                        content[:80],
                    )
                    return
                if (
                    isinstance(decision, dict)
                    and decision.get("action") == "released"
                    and decision.get("target") == "secondary"
                    and (initiative_activity or retries > 0 or allow_during_away)
                    and retries < 6
                ):
                    self._requeue_visible_retry(
                        content,
                        source=source,
                        initiative_activity=initiative_activity,
                        allow_during_away=allow_during_away,
                        retries=retries,
                        delay_s=6.0 + min(retries, 3) * 2.0,
                    )
                    logger.debug(
                        "[ProactivePresence] Re-queued visible update after temporary hold: %s",
                        decision.get("reason", decision.get("target", "secondary")),
                    )
                    return
                if isinstance(decision, dict) and decision.get("action") != "released":
                    self._record_output_attempt()
                    logger.debug(
                        "[ProactivePresence] Visible route held: %s",
                        decision.get("reason", "suppressed"),
                    )
                    return
            except Exception as e:
                logger.debug("[ProactivePresence] Visible emission failed: %s", e)

        try:
            from core.thought_stream import get_emitter
            get_emitter().emit(
                "Reflection",
                content,
                level="info",
                category="ProactivePresence",
            )
            self._record_output_delivery()
            logger.info(
                "🧠 [ProactivePresence] Fallback thought → neural feed (#%d): %s",
                self._consecutive_unprompted, content[:80],
            )
        except Exception as e:
            logger.debug("[ProactivePresence] Neural feed emit failed: %s", e)

        # Also queue for terminal fallback in case UI is gone.
        # The TerminalWatchdog will only deliver this if the UI stays gone
        # for UI_GONE_CONFIRMATION_SECS — so no spurious terminal pop-ups.
        try:
            from core.terminal_chat import get_terminal_fallback
            get_terminal_fallback().queue_autonomous_message(content)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

    def _get_brain(self):
        if self.orchestrator:
            return getattr(self.orchestrator, "cognitive_engine", None)
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get("cognitive_engine", default=None)
        except Exception:
            return None
