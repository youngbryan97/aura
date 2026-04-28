"""core/brain/discourse_tracker.py
─────────────────────────────────────────────
Discourse State Tracker.

Updates AuraState.cognition discourse fields after each user message:
- discourse_topic: what the conversation is currently about
- discourse_depth: how many turns on this thread
- discourse_branches: adjacent topics that could branch naturally
- user_emotional_trend: warming_up | engaged | cooling_off | neutral
- conversation_energy: 0-1 momentum gauge

Uses a fast heuristic path for every message + occasional LLM deep analysis
(every 6th message or on significant topic shift).

Register in ServiceContainer as "discourse_tracker".
Call update(state, user_message) after each incoming message.
"""

from core.runtime.errors import record_degradation
import logging
import time
from typing import Optional

logger = logging.getLogger("Aura.DiscourseTracker")

# Emotion signals for fast heuristic trend detection
_POSITIVE_SIGNALS = frozenset(["haha", "lol", "love", "great", "nice", "!!",
                                "yesss", "true", "exactly", "agreed", "wow",
                                "let's", "lets", "go", "yes", "perfect"])
_NEGATIVE_SIGNALS = frozenset(["nah", "no", "wrong", "boring", "stop",
                                "whatever", "ugh", "meh", "bad", "not really"])


class DiscourseTracker:
    """
    Lightweight tracker that keeps AuraState's discourse fields current.
    """

    def __init__(self, cognitive_engine=None):
        self._brain = cognitive_engine
        self._turn_count = 0
        self._last_topic: Optional[str] = None
        self._topic_turn_start: int = 0
        self._positive_streak: int = 0
        self._negative_streak: int = 0

    def _get_brain(self):
        if self._brain:
            return self._brain
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get("cognitive_engine", default=None)
        except Exception:
            return None

    # ── Fast heuristic update (every message) ────────────────────────────

    def _fast_update(self, state, message: str):
        """Apply keyword heuristics without an LLM call."""
        msg_lower = message.lower()
        words = set(msg_lower.split())

        # Trend detection
        positive_hits = len(words & _POSITIVE_SIGNALS)
        negative_hits = len(words & _NEGATIVE_SIGNALS)

        if positive_hits > negative_hits:
            self._positive_streak += 1
            self._negative_streak = 0
        elif negative_hits > positive_hits:
            self._negative_streak += 1
            self._positive_streak = 0
        else:
            self._positive_streak = max(0, self._positive_streak - 1)
            self._negative_streak = max(0, self._negative_streak - 1)

        # Derive trend
        if self._positive_streak >= 2:
            trend = "engaged" if self._positive_streak >= 3 else "warming_up"
        elif self._negative_streak >= 2:
            trend = "cooling_off"
        else:
            trend = "neutral"
        state.cognition.user_emotional_trend = trend

        # Conversation energy — proxy: message length + punctuation intensity
        energy = min(1.0, len(message) / 150)  # normalise by ~150 char ideal
        if "!" in message or "?" in message:
            energy = min(1.0, energy + 0.2)
        # Smooth with existing value
        existing = getattr(state.cognition, "conversation_energy", 0.5)
        state.cognition.conversation_energy = existing * 0.6 + energy * 0.4

        # Discourse depth — increment if same topic, reset handled in deep update
        state.cognition.discourse_depth = getattr(state.cognition, "discourse_depth", 0) + 1

    # ── Deep LLM analysis (every N messages) ─────────────────────────────

    async def _deep_update(self, state, message: str):
        """Use LLM for topic detection, branch suggestions, trend verification."""
        brain = self._get_brain()
        if not brain:
            return

        # Gather recent history for context
        recent = []
        wm = getattr(state.cognition, "working_memory", [])
        for msg in wm[-6:]:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))[:100]
            if role in ("user", "assistant"):
                recent.append(f"{role}: {content}")
        history_str = "\n".join(recent) if recent else message

        prompt = (
            "Analyze this conversation fragment.\n"
            f"RECENT:\n{history_str}\n"
            f"LATEST MESSAGE: {message}\n\n"
            "Return JSON only:\n"
            '{"topic": "2-5 word description of current thread", '
            '"branches": ["topic1", "topic2"], '  # 2-3 natural branch topics
            '"topic_changed": true/false, '  # vs previous topic
            '"user_trend": "warming_up|engaged|cooling_off|neutral"}'
        )

        try:
            from core.container import ServiceContainer
            router = ServiceContainer.get("llm_router", default=None)
            if router:
                from core.brain.llm.llm_router import LLMTier
                result = await router.think(
                    prompt=prompt,
                    prefer_tier=LLMTier.TERTIARY,
                    is_background=True,
                    origin="discourse_tracker",
                    allow_cloud_fallback=False,
                )
            else:
                result = await brain.generate(prompt, temperature=0.3, max_tokens=80)

            if result:
                from core.utils.json_utils import extract_json
                data = extract_json(result)
                if isinstance(data, dict):
                    new_topic = data.get("topic")
                    if new_topic:
                        if data.get("topic_changed") and new_topic != self._last_topic:
                            # Topic changed -- record the OLD topic as unresolved
                            # if the conversation was still engaged (depth > 2)
                            old_depth = getattr(state.cognition, "discourse_depth", 0)
                            if self._last_topic and old_depth > 2:
                                self._record_unresolved_topic(self._last_topic, old_depth)
                            state.cognition.discourse_depth = 1
                            self._topic_turn_start = self._turn_count
                        state.cognition.discourse_topic = new_topic
                        self._last_topic = new_topic

                    branches = data.get("branches", [])
                    if isinstance(branches, list):
                        state.cognition.discourse_branches = [str(b) for b in branches[:3]]

                    trend = data.get("user_trend", "")
                    if trend in ("warming_up", "engaged", "cooling_off", "neutral"):
                        state.cognition.user_emotional_trend = trend

        except Exception as e:
            record_degradation('discourse_tracker', e)
            logger.debug("DiscourseTracker deep update failed: %s", e)

    # ── Public API ────────────────────────────────────────────────────────

    async def update(self, state, message: str):
        """
        Call this after each incoming user message.
        Fast path every message; deep LLM path every 6th message.
        """
        self._turn_count += 1
        self._fast_update(state, message)

        # Deep analysis on 1st message and then every 6th
        if self._turn_count == 1 or self._turn_count % 6 == 0:
            try:
                await self._deep_update(state, message)
            except Exception as e:
                record_degradation('discourse_tracker', e)
                logger.debug("DiscourseTracker async deep update failed: %s", e)

    def _record_unresolved_topic(self, topic: str, depth: int) -> None:
        """Record a topic that was left mid-conversation as an unresolved tension.

        Called when a topic change is detected while discourse depth > 2,
        indicating the conversation moved on before the topic was resolved.
        """
        try:
            from core.initiative_synthesis import get_initiative_synthesizer
            synth = get_initiative_synthesizer()
            synth.record_tension(
                content=f"Conversation topic left unresolved: {topic}",
                source="discourse_tracker",
                category="topic",
                urgency=min(0.6, 0.2 + depth * 0.05),
                discourse_depth=depth,
            )
        except Exception as e:
            record_degradation('discourse_tracker', e)
            logger.debug("Failed to record unresolved topic tension: %s", e)

    def get_status(self) -> dict:
        return {
            "turn_count": self._turn_count,
            "last_topic": self._last_topic,
            "positive_streak": self._positive_streak,
        }
