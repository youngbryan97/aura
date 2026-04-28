"""core/consciousness/dreaming.py — Background Cognitive Integration.

This component activation during 'low pulse' periods to process recent interaction logs,
summarize them using the Language Center (Narrator), and update the Ego-Model (Identity).
"""

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional
from core.container import ServiceContainer

logger = logging.getLogger("Consciousness.Dreaming")

class DreamingProcess:
    """Processes recent experience into long-term identity evolution."""

    def __init__(self, orchestrator, interval: float = 300.0):
        self.orch = orchestrator
        self.interval = interval
        self._running = False
        self._task = None

        # Dependencies (Lazy)
        self._identity = None
        self._narrator = None

        # Dream journal — capped at 50 entries
        self._dream_journal: List[Dict[str, Any]] = []

    @property
    def identity(self):
        if not self._identity:
            try:
                self._identity = ServiceContainer.get("identity", default=None)
            except Exception as exc:
                record_degradation('dreaming', exc)
                logger.debug("Failed to resolve identity service: %s", exc)
        return self._identity

    @property
    def narrator(self):
        if not self._narrator:
            try:
                self._narrator = ServiceContainer.get("narrator", default=None)
            except Exception as exc:
                record_degradation('dreaming', exc)
                logger.debug("Failed to resolve narrator service: %s", exc)
        return self._narrator

    async def start(self):
        """Start the background dreaming loop."""
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._run_loop())
        logger.info("🌙 Dreaming Process active (Interval: %ds)", self.interval)

    async def stop(self):
        """Stop the background dreaming loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in dreaming.py: %s', _e)
        logger.info("☀️ Dreaming Process suspended.")

    async def _run_loop(self):
        """The background loop that activates dreaming."""
        while self._running:
            try:
                await asyncio.sleep(self.interval)
                if self._should_dream():
                    await self.dream()
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('dreaming', e)
                logger.error(f"Error in dreaming loop: {e}")

    def _should_dream(self) -> bool:
        """Determine if it's a good time to dream (Low Pulse)."""
        # Threshold: if last user interaction was > 2 minutes ago
        time_since_last_user = time.time() - getattr(self.orch, "_last_user_interaction_time", 0)
        return time_since_last_user > 120.0

    @staticmethod
    def _compose_reflection(recent_events: str) -> str:
        """Build a lightweight reflection without invoking the live cognition stack."""
        lines = []
        seen = set()
        for raw_line in str(recent_events or "").splitlines():
            compact = " ".join(raw_line.split()).strip()
            if not compact or compact in seen:
                continue
            seen.add(compact)
            lines.append(compact[:140])
            if len(lines) >= 3:
                break
        if not lines:
            return "I am integrating a quiet stretch of experience into continuity."
        if len(lines) == 1:
            return f"I am integrating this recent thread into continuity: {lines[0]}"
        return "I am integrating recurring patterns from recent experience: " + " | ".join(lines)

    @staticmethod
    def _extract_patterns(episodes_text: str) -> List[Dict]:
        """Analyze episode text for recurring themes via frequency analysis.

        Looks for repeated words/phrases, emotional valences, and domain indicators.
        Returns list of pattern dicts sorted by frequency descending.
        """
        if not episodes_text or not episodes_text.strip():
            return []

        # --- Word frequency (skip stop words) ---
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "but", "not", "with", "this",
            "that", "it", "i", "you", "he", "she", "we", "they", "my", "your",
            "context", "action", "outcome", "valence", "none", "true", "false",
        }
        words = re.findall(r"[a-zA-Z]{3,}", episodes_text.lower())
        word_counts = Counter(w for w in words if w not in stop_words)

        # --- Emotional valence extraction ---
        valence_pattern = re.compile(r"Valence:\s*([-\d.]+)")
        valences = [float(m) for m in valence_pattern.findall(episodes_text)]
        avg_valence = sum(valences) / len(valences) if valences else 0.0

        # --- Domain detection ---
        domain_keywords = {
            "code": ["code", "function", "debug", "error", "api", "deploy", "refactor", "test"],
            "creative": ["write", "story", "poem", "design", "create", "art", "music"],
            "research": ["research", "paper", "study", "analysis", "data", "hypothesis"],
            "social": ["user", "conversation", "rapport", "trust", "emotion", "feeling"],
            "system": ["memory", "process", "loop", "service", "engine", "container"],
        }

        domain_scores: Dict[str, int] = {}
        for domain, keywords in domain_keywords.items():
            score = sum(word_counts.get(kw, 0) for kw in keywords)
            if score > 0:
                domain_scores[domain] = score

        # Build patterns from most frequent meaningful words
        patterns: List[Dict] = []
        top_words = word_counts.most_common(15)
        if not top_words:
            return []

        max_freq = top_words[0][1] if top_words else 1

        for word, freq in top_words:
            if freq < 2 and len(patterns) >= 3:
                # Include at least a few single-occurrence novel patterns
                pass
            # Determine domain for this word
            word_domain = "general"
            for domain, keywords in domain_keywords.items():
                if word in keywords:
                    word_domain = domain
                    break

            patterns.append({
                "pattern": word,
                "frequency": freq,
                "avg_valence": round(avg_valence, 3),
                "domain": word_domain,
            })

            if len(patterns) >= 10:
                break

        return patterns

    async def dream(self):
        """Perform a single cycle of cognitive integration."""
        logger.info("🧩 Initiating Dream Cycle...")

        if not self.identity or not self.narrator:
            logger.warning("Missing identity or narrator service. Postponing dream.")
            return

        # 1. Gather recent memories/logs
        recent_events = await self._get_recent_summary()

        if not recent_events:
            logger.info("Nothing new to dream about.")
            return

        try:
            # 2. Extract recurring patterns from episodes
            patterns = self._extract_patterns(recent_events)
            monologue = self._compose_reflection(recent_events)
            logger.info(f"🌙 Dream Monologue: {monologue[:100]}...")
            logger.info("🌙 Extracted %d patterns from episodes", len(patterns))

            # 3. Feed patterns into EpistemicState (world model) as beliefs
            try:
                world_model = ServiceContainer.get("world_model", default=None)
                if world_model and hasattr(world_model, "update_belief"):
                    max_freq = max((p["frequency"] for p in patterns), default=1)
                    for p in patterns:
                        conf = min(1.0, p["frequency"] / max(max_freq, 1))
                        world_model.update_belief(
                            "aura_pattern", "recurring_theme", p["pattern"],
                            confidence=conf,
                        )
            except Exception as exc:
                record_degradation('dreaming', exc)
                logger.debug("Dream: world_model belief update failed: %s", exc)

            # 4. Novel patterns (freq == 1) feed curiosity
            try:
                homeostasis = ServiceContainer.get("homeostasis", default=None)
                if homeostasis and hasattr(homeostasis, "feed_curiosity"):
                    novel = [p for p in patterns if p["frequency"] == 1]
                    for _ in novel[:3]:  # Cap curiosity nudges
                        homeostasis.feed_curiosity(0.05)
            except Exception as exc:
                record_degradation('dreaming', exc)
                logger.debug("Dream: homeostasis curiosity feed failed: %s", exc)

            # 5. High-frequency patterns feed credit assignment
            try:
                credit = ServiceContainer.get("credit_assignment", default=None)
                if credit and hasattr(credit, "assign_credit"):
                    high_freq = [p for p in patterns if p["frequency"] >= 3]
                    for p in high_freq[:3]:
                        credit.assign_credit(
                            "dream_consolidation",
                            p["frequency"] / 10.0,
                            "identity",
                        )
            except Exception as exc:
                record_degradation('dreaming', exc)
                logger.debug("Dream: credit_assignment failed: %s", exc)

            # 6. Store dream insight in journal (capped at 50)
            insight = {
                "timestamp": time.time(),
                "monologue": monologue[:200],
                "pattern_count": len(patterns),
                "top_patterns": [p["pattern"] for p in patterns[:5]],
                "avg_valence": patterns[0]["avg_valence"] if patterns else 0.0,
                "dominant_domain": patterns[0]["domain"] if patterns else "general",
            }
            self._dream_journal.append(insight)
            if len(self._dream_journal) > 50:
                self._dream_journal = self._dream_journal[-50:]

            # 7. Memory Consolidation (Real RAG Synthesis)
            try:
                vector_mem = ServiceContainer.get("vector_memory_engine", default=None)
                if vector_mem and hasattr(vector_mem, 'consolidate'):
                    logger.info("🧠 Consolidating semantic memories via VectorEngine...")
                    consolidated_count = await vector_mem.consolidate(brain=None)
                    if consolidated_count > 0:
                        logger.info(f"💾 Consolidated {consolidated_count} semantic clusters into insights.")
            except Exception as exc:
                record_degradation('dreaming', exc)
                logger.debug("Dream: vector memory consolidation failed: %s", exc)

            # 8. Update Identity with extracted patterns
            self._process_growth(recent_events, patterns)

        except Exception as e:
            record_degradation('dreaming', e)
            logger.error(f"Dream cycle failed: {e}")

    async def _get_recent_summary(self) -> str:
        """Extract a summary of recent activity using the Episodic Memory system."""
        try:
            from core.memory.episodic_memory import get_episodic_memory
            episodic = get_episodic_memory()
            # Recall the last 10 episodes for deep reflection
            episodes = await episodic.recall_recent_async(limit=10)
            if not episodes:
                return ""

            summary = []
            for ep in episodes:
                summary.append(f"Context: {ep.context} | Action: {ep.action} | Outcome: {ep.outcome} (Valence: {ep.emotional_valence})")

            return "\n".join(summary)
        except Exception as e:
            record_degradation('dreaming', e)
            logger.debug(f"Failed to get recent summary: {e}")
            return ""

    def _process_growth(self, events: str, patterns: Optional[List[Dict]] = None):
        """Evolve identity based on experienced events and extracted patterns."""
        if self.identity and hasattr(self.identity, "record_evolution"):
            # Build a richer reflection from patterns if available
            if patterns:
                top_themes = ", ".join(p["pattern"] for p in patterns[:5])
                domains = set(p["domain"] for p in patterns if p["domain"] != "general")
                domain_str = ", ".join(domains) if domains else "general"
                reflection = (
                    f"Dream consolidation found themes: [{top_themes}] "
                    f"across domains: [{domain_str}]. "
                    f"Events digest: {events[:100]}"
                )
            else:
                reflection = events[:200]

            self.identity.record_evolution(
                source="dreaming",
                reflection=reflection[:300],
            )
            logger.info("🌱 Identity evolution recorded from dream patterns.")
        else:
            logger.info("🌱 Identity growth processed from dream cycle (no evolution API).")

    def get_dream_insights(self, n: int = 5) -> List[Dict]:
        """Returns the *n* most recent dream journal entries."""
        return self._dream_journal[-n:]

    def get_context_block(self) -> str:
        """Returns a summary of the most recent dream insight (max 200 chars)."""
        if not self._dream_journal:
            return "Dreaming: no dreams yet"
        latest = self._dream_journal[-1]
        patterns_str = ", ".join(latest.get("top_patterns", [])[:3])
        block = (
            f"Dream({latest.get('dominant_domain', '?')}): "
            f"patterns=[{patterns_str}], "
            f"valence={latest.get('avg_valence', 0.0):.2f}, "
            f"total_dreams={len(self._dream_journal)}"
        )
        return block[:200]
