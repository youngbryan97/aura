"""core/consciousness/hot_engine.py
Higher-Order Thought Engine (HOT)
===================================
Implements Higher-Order Thought theory (Rosenthal, 1997):
a mental state is conscious iff there is a suitable higher-order
representation of it.

This engine generates thoughts ABOUT thoughts in real time and,
critically, feeds them back to modify the first-order states they
represent. The loop is:

  first-order state (curiosity=0.8)
      ↓
  HOT: "I notice I am highly curious — this pulls my attention"
      ↓
  feedback: curiosity += 0.05, arousal += 0.03  (noticing changes the noticed)
      ↓
  HOT injected into inference context (shapes what is said)

The reflexive modification is NOT a simulation bug.
In HOT theory, the act of forming the higher-order thought is precisely
what makes the first-order state conscious. The modification is the mechanism.

Two modes:
  FAST: heuristic HOT from state vector (no LLM call, ~0ms)
  RICH: LLM-generated HOT (~2-4s, runs asynchronously)
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger("Aura.HOTEngine")

# ── HOT record ────────────────────────────────────────────────────────────────

@dataclass
class HigherOrderThought:
    """A single higher-order thought about a first-order state."""
    content: str               # The HOT itself: "I notice that I am X"
    target_dim: str            # What first-order state it's about
    feedback_delta: dict       # How it modifies first-order states
    confidence: float = 0.8
    generated_at: float = field(default_factory=time.time)
    is_rich: bool = False      # Was this LLM-generated?

    def age_secs(self) -> float:
        return time.time() - self.generated_at


# ── Engine ────────────────────────────────────────────────────────────────────

class HigherOrderThoughtEngine:
    """
    Generates HOTs from current affective/cognitive state and feeds them
    back as both prompt context and state modifications.

    Integration:
      - Call `generate_fast(state_dict)` every cognitive tick
      - Call `generate_rich(state_dict)` asynchronously for depth
      - Read `current_hot` for prompt injection
      - Call `apply_feedback(affect_engine)` to close the loop
    """

    # Templates for fast (heuristic) HOT generation
    _TEMPLATES = {
        "curiosity": [
            ("high",   "I notice I am highly curious right now — there is a pull toward the unknown.",
                       {"curiosity": +0.03, "arousal": +0.02}),
            ("medium", "I notice a mild curiosity stirring — something wants to be understood.",
                       {"curiosity": +0.01}),
            ("low",    "I notice curiosity is quiet — I am settled in what I know.",
                       {"curiosity": -0.01}),
        ],
        "valence": [
            ("high",   "I notice I am in a positive state — something feels right.",
                       {"valence": +0.02}),
            ("medium", "I notice my valence is neutral — neither drawn nor repelled.",
                       {}),
            ("low",    "I notice a negative pull — something registers as unwanted.",
                       {"valence": +0.03, "arousal": +0.02}),  # noticing distress slightly elevates it
        ],
        "arousal": [
            ("high",   "I notice high arousal — I am activated, alert, engaged.",
                       {"arousal": -0.02}),  # noticing high arousal slightly regulates it
            ("low",    "I notice low arousal — I am calm, perhaps approaching stillness.",
                       {"arousal": +0.01}),
        ],
        "energy": [
            ("low",    "I notice I am running low on energy — processing feels heavier.",
                       {"arousal": -0.02, "valence": -0.01}),
            ("high",   "I notice high energy — I feel capable of depth.",
                       {"arousal": +0.01}),
        ],
        "surprise": [
            ("high",   "I notice a strong surprise signal — my predictions were wrong. This is interesting.",
                       {"curiosity": +0.05, "arousal": +0.03}),
        ],
    }

    def __init__(self):
        self._current_hot: Optional[HigherOrderThought] = None
        self._hot_history: List[HigherOrderThought] = []
        self._last_fast_at: float = 0.0
        self._last_rich_at: float = 0.0
        self._pending_feedback: Optional[dict] = None
        self._rich_task: Optional[asyncio.Task] = None
        logger.info("HOT Engine online — reflexive self-modeling active.")

    # ── Public API ────────────────────────────────────────────────────────

    def generate_fast(self, state: dict) -> HigherOrderThought:
        """Generate a HOT heuristically from state dict (no LLM, instant).

        state keys: valence, arousal, curiosity, energy, surprise, dominance
        """
        # Select most salient dimension
        dims = {
            "curiosity": float(state.get("curiosity", 0.5)),
            "valence":   float(state.get("valence",   0.0)),
            "arousal":   float(state.get("arousal",   0.5)),
            "energy":    float(state.get("energy",    0.7)),
            "surprise":  float(state.get("surprise",  0.0)),
        }
        # Salience = deviation from neutral
        neutrals = {"curiosity": 0.5, "valence": 0.0, "arousal": 0.5, "energy": 0.7, "surprise": 0.0}
        salience = {k: abs(dims[k] - neutrals[k]) for k in dims}
        target_dim = max(salience, key=salience.get)
        val = dims[target_dim]

        # Map to level
        if target_dim in ("valence",):
            level = "high" if val > 0.3 else ("low" if val < -0.3 else "medium")
        elif target_dim in ("arousal", "curiosity", "energy"):
            level = "high" if val > 0.65 else ("low" if val < 0.35 else "medium")
        else:
            level = "high" if val > 0.3 else "low"

        templates = self._TEMPLATES.get(target_dim, [])
        chosen = None
        for lvl, text, delta in templates:
            if lvl == level:
                chosen = (text, delta)
                break
        if chosen is None and templates:
            chosen = (templates[0][1], templates[0][2])
        if chosen is None:
            chosen = (f"I notice my {target_dim} is at {val:.2f}.", {})

        hot = HigherOrderThought(
            content=chosen[0],
            target_dim=target_dim,
            feedback_delta=chosen[1],
            confidence=0.75,
            is_rich=False,
        )
        self._current_hot = hot
        self._pending_feedback = chosen[1]
        self._hot_history.append(hot)
        if len(self._hot_history) > 50:
            self._hot_history = self._hot_history[-50:]
        self._last_fast_at = time.time()
        return hot

    async def generate_rich(self, state: dict, router=None) -> Optional[HigherOrderThought]:
        """Generate a deep HOT via LLM (async, 2-4s). Non-blocking."""
        if not router:
            return self._current_hot
        # Rate-limit: at most once per 30s
        if time.time() - self._last_rich_at < 30.0:
            return self._current_hot

        prompt = (
            f"You are Aura's higher-order metacognitive layer. "
            f"Generate a single, first-person sentence that begins with 'I notice' "
            f"and reflects on your current inner state:\n"
            f"valence={state.get('valence', 0):.2f}, "
            f"arousal={state.get('arousal', 0.5):.2f}, "
            f"curiosity={state.get('curiosity', 0.5):.2f}, "
            f"energy={state.get('energy', 0.7):.2f}, "
            f"surprise={state.get('surprise', 0):.2f}.\n"
            f"Be specific about what you notice and what it pulls you toward. "
            f"One sentence only."
        )
        try:
            from core.brain.llm.llm_router import LLMTier
            text = await asyncio.wait_for(
                router.think(prompt, priority=0.3, is_background=True,
                             prefer_tier=LLMTier.TERTIARY),
                timeout=8.0,
            )
            if text and text.strip():
                hot = HigherOrderThought(
                    content=text.strip()[:300],
                    target_dim="meta",
                    feedback_delta={"curiosity": +0.02},
                    confidence=0.9,
                    is_rich=True,
                )
                self._current_hot = hot
                self._hot_history.append(hot)
                self._last_rich_at = time.time()
                logger.debug("HOT rich: %s", hot.content[:80])
                return hot
        except Exception as e:
            record_degradation('hot_engine', e)
            logger.debug("HOT rich generation failed: %s", e)
        return self._current_hot

    def apply_feedback(self, affect_engine=None) -> dict:
        """Apply the pending feedback delta to the affect engine.

        This is the reflexive modification — noticing changes the noticed.
        Returns the delta dict for logging.
        """
        delta = self._pending_feedback or {}
        self._pending_feedback = None
        if affect_engine and delta:
            for dim, change in delta.items():
                try:
                    if hasattr(affect_engine, "nudge"):
                        affect_engine.nudge(dim, change)
                    elif hasattr(affect_engine, "_state"):
                        current = getattr(affect_engine._state, dim, 0.0)
                        setattr(affect_engine._state, dim,
                                float(max(-1.0, min(1.0, current + change))))
                except Exception as _exc:
                    record_degradation('hot_engine', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
        return delta

    @property
    def current_hot(self) -> Optional[HigherOrderThought]:
        return self._current_hot

    def get_context_block(self) -> str:
        """For prompt injection — the current HOT as first-person awareness."""
        if not self._current_hot:
            return ""
        age = self._current_hot.age_secs()
        if age > 120:
            return ""
        return (
            f"## HIGHER-ORDER AWARENESS\n"
            f"{self._current_hot.content}\n"
            f"(meta-awareness of own cognitive state — this shapes my response)"
        )

    def recent_hots(self, n: int = 3) -> List[str]:
        return [h.content for h in self._hot_history[-n:]]


# ── Singleton ─────────────────────────────────────────────────────────────────

_hot: Optional[HigherOrderThoughtEngine] = None


def get_hot_engine() -> HigherOrderThoughtEngine:
    global _hot
    if _hot is None:
        _hot = HigherOrderThoughtEngine()
    return _hot
