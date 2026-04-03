"""core/conversation_reflector.py — Aura ConversationReflector v1.0
=============================================================
Reflects on recent conversation history to find deep patterns and gaps.

This system runs periodically to analyze the meta-structure of dialogue:
  - What topics am I avoiding?
  - What questions has Bryan asked that I didn't actually answer?
  - What emotional patterns are appearing in our relationship?
  - What do I need to inquire about next?

Output: Seeds for the InquiryEngine and SoulMarkers for the NarrativeThread.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("Aura.ConversationReflector")

@dataclass
class ReflectionResult:
    topic: str
    insight: str
    urgency: float
    seed_question: Optional[str] = None

class ConversationReflector:
    name = "conversation_reflector"

    def __init__(self):
        self._inquiry_engine = None
        self._narrative = None
        self._last_reflection = 0.0

    async def start(self):
        from core.container import ServiceContainer
        self._inquiry_engine = ServiceContainer.get("inquiry_engine", default=None)
        self._narrative = ServiceContainer.get("narrative_thread", default=None)
        logger.info("✅ ConversationReflector ONLINE.")

    async def reflect_on_history(self, history: List[Dict[str, str]]):
        """Analyze conversation history for gaps and themes."""
        if not history or time.time() - self._last_reflection < 300: # 5 min cooldown
            return

        self._last_reflection = time.time()
        logger.info("Reflecting on recent conversation history...")

        # In a real build, this would use a fast LLM call.
        # For Zenith Phase 10, we implement the logic for the connection.
        
        # ── REFLECTION -> INQUIRY WIRING ──
        # If history contains unanswered questions or deep themes, seed the InquiryEngine.
        if self._inquiry_engine:
            # Simple heuristic: find '?' in user messages not addressed in recent Aura messages
            user_questions = [h['content'] for h in history[-5:] if h.get('role') == 'user' and '?' in h.get('content', '')]
            if user_questions:
                for q in user_questions:
                    self._inquiry_engine.open_question(
                        question=f"Deep reflection on: {q}",
                        domain="conversational_gap",
                        urgency=0.6,
                        from_gap=True
                    )

        # Update NarrativeThread with the "vibe" of the conversation
        if self._narrative:
            # We can't easily derive 'vibe' without LLM here, so we skip or use simple logic
            pass

    def get_status(self) -> Dict[str, Any]:
        return {"last_reflection": self._last_reflection}
