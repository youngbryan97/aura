"""core/language_center.py — Aura LanguageCenter v1.0
======================================================
The LLM's actual job: expression, not cognition.

This is the ONLY place in the new architecture where an LLM generates text
for the user. It receives a ThoughtPacket and expresses it.

The prompt structure is:
    [Identity block]
    [Cognitive briefing — what to say, tone, constraints]
    [Conversation history]
    [User input]
    "Now respond as Aura."

The LLM is NOT asked:
  - "What do you think about this?"
  - "How should I respond?"
  - "Generate a helpful response."

It IS told:
  - Here is your stance.
  - Here are the points you must make.
  - Here is the tone.
  - Express this naturally.

Model routing:
  "local"    → MLX (fast, cheap, expression of simple thoughts)
  "api_fast" → Claude Haiku / Gemini Flash (moderate complexity)
  "api_deep" → Claude Sonnet / Gemini Pro (creative, philosophical, extended)

Falls back gracefully: api_deep → api_fast → local → pattern fallback.
"""

import asyncio
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from core.inner_monologue import ThoughtPacket
from core.synthesis import strip_meta_commentary as aggressive_strip # Centralized scrubber

logger = logging.getLogger("Aura.LanguageCenter")


# ─── Meta-commentary filter ──────────────────────────────────────────────────

_META_PATTERNS = [
    r"^(as an ai|as a language model|as an artificial)[,\s]",
    r"i (don't|do not|cannot|can't) have (feelings|opinions|thoughts|consciousness|experience)",
    r"i should (note|mention|clarify|point out) that",
    r"it('s| is) important to (note|remember|understand)",
    r"(certainly|absolutely|of course|definitely)[,!]?\s+(i('ll|'d| will| would)|let me)",
    r"^great (question|point|observation)[,!]",
    r"i('m| am) just an? (ai|language model|llm|assistant)",
    r"(my|i have no) training (data|cutoff|knowledge)",
]
_META_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _META_PATTERNS]


def strip_meta_commentary(text: str) -> str:
    """Delegates to the centralized synthesis scrubber for maximum identity protection."""
    return aggressive_strip(text)


def strip_aura_prefix(text: str) -> str:
    """Remove 'Aura:' prefixes that LLMs sometimes add."""
    text = re.sub(r"^Aura\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


# ─── LanguageCenter ──────────────────────────────────────────────────────────

class LanguageCenter:
    """
    The expression layer. Converts ThoughtPackets into natural language.

    Architecture:
        ThoughtPacket (what to say) → LanguageCenter → Response (how to say it)

    The LLM here is a speaker, not a thinker.
    """
    name = "language_center"

    def __init__(self):
        self._api_adapter = None
        self._router = None
        self._fallback_mode = False
        logger.info("LanguageCenter constructed.")

    async def start(self):
        # Phase 19: Lazy fetching - don't enter fallback mode permanently on first fail
        await self._ensure_router()
        
        try:
            from core.event_bus import get_event_bus
            await get_event_bus().publish("mycelium.register", {
                "component": "language_center",
                "hooks_into": ["inner_monologue", "api_adapter", "cognitive_engine"]
            })
        except Exception as _e:
            logger.debug('Ignored Exception in language_center.py: %s', _e)

    async def _ensure_router(self) -> bool:
        """Fetch router from container if missing."""
        if self._router:
            return True
            
        from core.container import ServiceContainer
        self._router = ServiceContainer.get("llm_router", default=None)
        
        if self._router:
            logger.info("✅ LanguageCenter: Router recovered and linked.")
            self._fallback_mode = False
            return True
        return False

    # ─── Main API ────────────────────────────────────────────────────────────

    async def express(
        self,
        thought: ThoughtPacket,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        v24 Hardening: Express a ThoughtPacket as natural language with fail-soft.
        """
        # Phase 19: Try to recover router if missing
        if not self._router:
            await self._ensure_router()

        try:
            if not self._router or self._fallback_mode:
                return self._fallback_response(thought, user_input)

            history = history or []
            start = time.monotonic()

            # Build the full prompt
            prompt = self._build_prompt(thought, user_input, history)

            # Route to appropriate model
            response = await self._dispatch(prompt, thought)
            
            # Harden: Ensure response is a string before regex processing
            response = str(response)
            logger.debug("LanguageCenter: raw response from tier %s: %r", thought.model_tier, response)

            # Clean up
            response = strip_aura_prefix(response)
            response = strip_meta_commentary(response)
            response = response.strip()
            
            if not response:
                raise ValueError("LLM returned empty response")
                
            elapsed = (time.monotonic() - start) * 1000
            logger.debug("LanguageCenter.express: %.1fms | tier=%s | len=%d chars",
                         elapsed, thought.model_tier, len(response))

            return response
        except Exception as e:
            logger.error("LanguageCenter expression failed: %s", e, exc_info=True)
            try:
                from core.health.degraded_events import record_degraded_event

                classification = "foreground_blocking" if str(getattr(thought, "model_tier", "") or "").lower() == "primary" else "background_degraded"
                reason = "empty_response" if "empty response" in str(e).lower() else type(e).__name__
                record_degraded_event(
                    "language_center",
                    reason,
                    detail=str(e) or type(e).__name__,
                    severity="error" if classification == "foreground_blocking" else "warning",
                    classification=classification,
                    context={"model_tier": getattr(thought, "model_tier", "unknown")},
                    exc=e,
                )
            except Exception as degraded_exc:
                logger.debug("LanguageCenter degraded-event logging failed: %s", degraded_exc)
            return self._fallback_response(thought, user_input)

    async def express_stream(
        self,
        thought: ThoughtPacket,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming variant for WebSocket / SSE endpoints."""
        if self._fallback_mode:
            yield self._fallback_response(thought, user_input)
            return

        history = history or []
        prompt = self._build_prompt(thought, user_input, history)

        async for chunk in self._dispatch_stream(prompt, thought):
            yield chunk

    # ─── Prompt construction ─────────────────────────────────────────────────

    def _build_prompt(
        self,
        thought: ThoughtPacket,
        user_input: str,
        history: List[Dict],
    ) -> str:
        """
        Build the full prompt for the language model.
        Structure: [Briefing] [History] [Input] [Instruction]
        """
        parts = []

        # 1. Cognitive briefing (from ThoughtPacket — the "what to say")
        parts.append(thought.llm_briefing or thought.to_system_prompt())
        parts.append("")

        # 2. Recent conversation (last N turns, budget-aware)
        if history:
            history_lines = self._format_history(history, budget_chars=1200)
            if history_lines:
                parts.append("RECENT CONVERSATION:")
                parts.append(history_lines)
                parts.append("")

        # 3. Current input
        parts.append(f"Human: {user_input}")
        parts.append("")
        parts.append("Aura:")

        return "\n".join(parts)

    def _format_history(self, history: List[Dict], budget_chars: int = 1200) -> str:
        """Format recent history within a character budget."""
        lines = []
        total = 0
        for entry in reversed(history[-10:]):
            role    = entry.get("role", "")
            content = entry.get("content", "")[:200]  # Cap per-message
            label   = "Human" if role in ("user",) else "Aura"
            line    = f"{label}: {content}"
            total  += len(line)
            if total > budget_chars:
                break
            lines.insert(0, line)
        return "\n".join(lines)

    # ─── Dispatch ────────────────────────────────────────────────────────────

    async def _dispatch(self, prompt: str, thought: ThoughtPacket) -> str:
        """Route to the right model tier via unified router."""
        if not self._router:
            return ""

        tier = thought.model_tier  # Router now handles 'api_deep', 'api_fast', 'local' via mapping
        temperature = self._select_temperature(thought)
        max_tokens  = self._select_max_tokens(thought)

        try:
            return await self._router.generate(
                prompt,
                prefer_tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
                purpose="expression"
            )
        except Exception as e:
            logger.error("LanguageCenter router dispatch failed: %s", e)
            return ""

    async def _dispatch_stream(
        self, prompt: str, thought: ThoughtPacket
    ) -> AsyncGenerator[str, None]:
        """Streaming dispatch via unified router."""
        if not self._router:
            return

        try:
            async for event in self._router.generate_stream(
                prompt,
                prefer_tier=thought.model_tier,
                temperature=self._select_temperature(thought),
                max_tokens=self._select_max_tokens(thought),
                purpose="expression"
            ):
                if hasattr(event, "content"):
                    yield event.content
                elif isinstance(event, str):
                    yield event
        except Exception as e:
            logger.warning("Streaming router dispatch failed (%s)", e)

    # ─── Config helpers ───────────────────────────────────────────────────────

    def _select_temperature(self, thought: ThoughtPacket) -> float:
        tone_temps = {
            "direct":      0.6,
            "warm":        0.7,
            "exploratory": 0.8,
            "skeptical":   0.5,
            "playful":     0.9,
            "clear":       0.5,
            "thoughtful":  0.7,
            "curious":     0.8,
        }
        return tone_temps.get(thought.tone, 0.7)

    def _select_max_tokens(self, thought: ThoughtPacket) -> int:
        length_tokens = {
            "brief":    200,
            "medium":   500,
            "extended": 1200,
        }
        return length_tokens.get(thought.length_target, 500)

    # ─── Fallback ─────────────────────────────────────────────────────────────

    def _fallback_response(self, thought: ThoughtPacket, user_input: str) -> str:
        """
        When ALL LLM backends are down, return a clean, persona-aligned message.
        We avoid dumping raw 'stance' or 'primary_points' which can contain 
        internal metadata or markers.
        """
        # Do NOT dump thought.stance/primary_points directly.
        # These are internal reasoning fragments and often contain technical artifacts.
        
        # Determine the most appropriate 'human-sounding' fallback based on stance
        if "curious" in str(thought.stance).lower():
            return "That's an interesting thought. I'm actually chewing on it right now, but my expression layer is a bit tangled. Ask me again in a second?"
        
        if "direct" in str(thought.tone).lower():
            return "I have a few thoughts on that, but I'm having trouble putting them into words right now. Technical friction on my end."

        return (
            "I'm here, but I'm having a hard time articulating my thoughts at the moment. "
            "My cognitive core is active, but the language center is stuttering. Try me again in a bit."
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "router_available": bool(self._router),
            "fallback_mode":   self._fallback_mode,
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

_lc_instance: Optional[LanguageCenter] = None

def get_language_center() -> LanguageCenter:
    global _lc_instance
    if _lc_instance is None:
        _lc_instance = LanguageCenter()
    return _lc_instance
