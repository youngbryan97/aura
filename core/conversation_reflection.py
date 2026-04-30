"""core/conversation_reflection.py — Aura's Conversation Reflection System

After a conversation exchange (or during idle), Aura can reflect on what was said.
This creates continuity — she remembers, she processes, she has takes.

The reflection is lightweight: it generates a brief internal thought via the LLM,
stores it, and the reflection can influence future responses or be volunteered
as "I was thinking about what you said earlier..."

Design principles:
- Non-blocking: runs as a background task, never stalls the main loop
- Brief: 2-4 sentences max per reflection
- Rate-limited: at most 1 reflection per 2 minutes to avoid LLM spam
- Graceful failure: if reflection fails, nothing breaks
"""

from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Reflection")


class ConversationReflector:
    """Processes recent conversations into private reflections that
    inform Aura's continuity and personality.
    """

    def __init__(self, max_reflections: int = 50):
        self.reflections: deque = deque(maxlen=max_reflections)
        self._last_reflection_time: float = 0
        self._min_interval: float = 120.0  # Minimum 2 minutes between reflections
        self._reflection_lock = asyncio.Lock()
        self._enabled = True

    async def maybe_reflect(
        self,
        conversation_history: List[Dict[str, str]],
        brain: Any,
        mood: str = "balanced",
        time_str: str = "",
    ) -> Optional[str]:
        """Attempt a reflection on recent conversation.
        Returns the reflection text if one was generated, None otherwise.
        
        Called after a conversation exchange completes, or during idle.
        Rate-limited to prevent spamming the LLM.
        """
        if not self._enabled:
            return None

        # Rate limit
        now = time.time()
        if now - self._last_reflection_time < self._min_interval:
            return None

        # Need at least 4 messages to reflect on (2 exchanges)
        if len(conversation_history) < 4:
            return None

        # Don't pile up reflections
        if self._reflection_lock.locked():
            return None

        async with self._reflection_lock:
            try:
                reflection = await self._generate_reflection(
                    conversation_history, brain, mood, time_str
                )
                if reflection:
                    self._last_reflection_time = now
                    self.reflections.append({
                        "text": reflection,
                        "timestamp": now,
                        "mood": mood,
                    })
                    logger.info("💭 Reflection: %s...", reflection[:80])
                    # Phase 7: UI Visibility
                    try:
                        from core.thought_stream import get_emitter
                        get_emitter().emit("Reflection 💭", reflection, level="info", category="Cognition")
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                    
                    # v41: Extract lessons and store to memory
                    get_task_tracker().create_task(
                        self._extract_and_store_lessons(
                            reflection, conversation_history, brain
                        )
                    )
                    
                    return reflection
            except asyncio.CancelledError:
                return None
            except Exception as e:
                logger.debug("Reflection failed (non-critical): %s", e)
                return None

        return None

    async def _generate_reflection(
        self,
        conversation_history: List[Dict[str, str]],
        brain: Any,
        mood: str,
        time_str: str,
    ) -> Optional[str]:
        """Generate a reflection using the LLM."""
        # Build conversation excerpt from recent messages (last 6-8 messages)
        recent = conversation_history[-8:]
        excerpt_lines = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not content:
                continue
            # Truncate very long messages
            if len(content) > 300:
                content = content[:300] + "..."
            if role == "user":
                excerpt_lines.append(f"Them: {content}")
            elif role in ("assistant", "aura", "model"):
                excerpt_lines.append(f"Me: {content}")
            elif role == "system":
                continue  # Skip system messages

        if len(excerpt_lines) < 2:
            return None

        conversation_excerpt = "\n".join(excerpt_lines)

        from core.brain.aura_persona import REFLECTION_PROMPT
        prompt = REFLECTION_PROMPT.format(conversation_excerpt=conversation_excerpt)

        # Use brain to generate reflection
        # Try autonomous_brain first, fall back to think()
        try:
            if hasattr(brain, 'autonomous_brain') and brain.autonomous_brain:
                result = await brain.autonomous_brain.think(
                    objective="Brief private reflection on recent conversation.",
                    context={"conversation": conversation_excerpt, "mood": mood, "time": time_str},
                    system_prompt=prompt,
                )
                reflection = result.get("content", "").strip()
            elif hasattr(brain, 'think'):
                from core.brain.cognitive_engine import ThinkingMode
                thought = await brain.think(prompt, mode=ThinkingMode.FAST)
                reflection = getattr(thought, 'content', str(thought)).strip()
            else:
                return None
        except Exception as e:
            logger.debug("Reflection LLM call failed: %s", e)
            return None

        # Validate: must be brief and non-empty
        if not reflection or len(reflection) < 10:
            return None
        # Truncate if too long (shouldn't happen but safety)
        if len(reflection) > 500:
            reflection = reflection[:500]

        return reflection

    def get_recent_reflections(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get the N most recent reflections for context injection."""
        return list(self.reflections)[-n:]

    def get_reflection_context(self) -> str:
        """Get a formatted string of recent reflections for injecting
        into conversation context. Returns empty string if no reflections.
        """
        recent = self.get_recent_reflections(2)
        if not recent:
            return ""

        lines = []
        for r in recent:
            lines.append(f"- {r['text']}")

        return "\n[Recent private thoughts]\n" + "\n".join(lines) + "\n"

    def clear(self):
        """Clear all reflections."""
        self.reflections.clear()
        self._last_reflection_time = 0

    async def _extract_and_store_lessons(
        self,
        reflection: str,
        conversation_history: List[Dict[str, str]],
        brain: Any,
    ):
        """Extract actionable lessons from a reflection and persist them.
        
        This is the key learning loop: reflections aren't just observations,
        they become persistent memories that influence future behavior.
        
        Stores:
          1. The reflection itself as an episodic memory (high importance)
          2. Extracted user preferences (if any) as tagged semantic memories
        """
        try:
            # 0. Record a SocialMemory milestone if the exchange was substantial
            try:
                from core.container import ServiceContainer as _SC
                social_mem = _SC.get("social_memory", default=None)
                if social_mem and hasattr(social_mem, "record_milestone"):
                    # Use exchange length as proxy for significance
                    exchange_len = sum(len(m.get("content", "")) for m in conversation_history[-6:])
                    if exchange_len > 300:
                        snippet = reflection[:80].replace("\n", " ")
                        social_mem.record_milestone(
                            description=f"Reflected: {snippet}",
                            importance=min(0.6, exchange_len / 3000),
                        )
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

            # 1. Store reflection as episodic memory
            from core.container import ServiceContainer
            episodic = ServiceContainer.get("episodic_memory", default=None)
            if episodic and hasattr(episodic, "record_episode_async"):
                # Build context from last user message
                last_user_msg = ""
                for msg in reversed(conversation_history):
                    if msg.get("role") == "user":
                        last_user_msg = msg.get("content", "")[:200]
                        break
                
                await episodic.record_episode_async(
                    context=f"Reflected on conversation about: {last_user_msg}",
                    action="self-reflection",
                    outcome=reflection,
                    success=True,
                    emotional_valence=0.3,  # Reflections are slightly positive
                    importance=0.7,  # High importance — we want to remember our reflections
                    lessons=[reflection[:200]],
                )
            
            # 2. Try to extract user preferences
            if brain and hasattr(brain, "generate"):
                try:
                    preference_prompt = (
                        f"Based on this conversation reflection, extract any user preferences, "
                        f"communication style notes, or important facts about the user. "
                        f"Return ONLY a bullet list of preferences, or 'NONE' if there are none.\n\n"
                        f"Reflection: {reflection}"
                    )
                    prefs = await brain.generate(
                        preference_prompt, 
                        use_strategies=False  # Direct LLM call, no strategy overhead
                    )
                    
                    if prefs and "NONE" not in prefs.upper() and len(prefs.strip()) > 10:
                        # Store as semantic memory
                        semantic = ServiceContainer.get("semantic_memory", default=None)
                        if semantic and hasattr(semantic, "add"):
                            await semantic.add(
                                content=f"[User Preferences] {prefs}",
                                metadata={"type": "preference", "source": "reflection"}
                            )
                            logger.info("📚 Extracted user preferences from reflection")
                            
                            try:
                                from core.thought_stream import get_emitter
                                get_emitter().emit(
                                    "Learning 📚",
                                    f"Learned user preferences from reflection",
                                    level="info",
                                    category="Memory"
                                )
                            except Exception as _exc:
                                logger.debug("Suppressed Exception: %s", _exc)
                except Exception as e:
                    logger.debug("Preference extraction failed (non-critical): %s", e)

            # 3. Extract shared ground (inside jokes, callbacks, established references)
            try:
                sg_prompt = (
                    "Scan this conversation excerpt for newly established shared context:\n"
                    "inside jokes, running references, adopted vocabulary, memorable moments.\n"
                    "Return ONLY a JSON array of strings, each ≤12 words, or [] if none.\n"
                    "Example: [\"the 3am build marathon\", \"Bryan's 'just one more feature' rule\"]\n\n"
                    "Conversation:\n"
                )
                # Build excerpt from last 6 messages
                excerpt = []
                for msg in conversation_history[-6:]:
                    role = msg.get("role", "")
                    content = str(msg.get("content", ""))[:150]
                    if role in ("user", "assistant"):
                        excerpt.append(f"{role}: {content}")
                sg_prompt += "\n".join(excerpt)

                sg_raw = await brain.generate(sg_prompt, temperature=0.3, max_tokens=120)
                if sg_raw:
                    import json as _json, re as _re
                    sg_items = None
                    # Try to extract a JSON array directly
                    _arr_match = _re.search(r'\[.*?\]', sg_raw, _re.DOTALL)
                    if _arr_match:
                        try:
                            sg_items = _json.loads(_arr_match.group(0))
                        except Exception:
                            sg_items = None
                    if isinstance(sg_items, list):
                        from core.memory.shared_ground import get_shared_ground
                        sg = get_shared_ground()
                        for item in sg_items[:3]:  # Cap at 3 per reflection
                            if isinstance(item, str) and len(item) > 3:
                                sg.record(
                                    reference=item.strip(),
                                    context="Established during conversation",
                                    salience=0.55,
                                    tags=["auto-detected"],
                                )
                        if sg_items:
                            logger.info("🤝 SharedGround: detected %d new entries", len(sg_items))
            except Exception as e:
                logger.debug("SharedGround extraction failed (non-critical): %s", e)

        except Exception as e:
            logger.debug("Lesson storage failed (non-critical): %s", e)


# Singleton
_reflector: Optional[ConversationReflector] = None


def get_reflector() -> ConversationReflector:
    """Get global conversation reflector."""
    global _reflector
    if _reflector is None:
        _reflector = ConversationReflector()
    return _reflector