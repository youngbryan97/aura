"""Mode Router — Thinking Mode Selection

Routes incoming messages to the appropriate processing tier:
  REFLEX  → instant pattern-matched responses (no LLM)
  LIGHT   → fast LLM call with minimal scaffolding
  DEEP    → full structured reasoning + optional critic loop

Decision factors: message length, keyword analysis, complexity signals.
"""
import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger("Ops.ModeRouter")


class ThinkingTier(Enum):
    REFLEX = "reflex"
    LIGHT = "light"
    DEEP = "deep"


# Keywords that signal complex reasoning is needed
DEEP_KEYWORDS = {
    "analyze", "explain", "compare", "evaluate", "design",
    "implement", "debug", "optimize", "refactor", "architect",
    "why", "how does", "what if", "trade-off", "tradeoff",
    "strategy", "plan", "review", "critique", "improve",
    "investigate", "research", "summarize", "synthesize",
}

# Short messages that are almost always reflexive
REFLEX_MAX_WORDS = 4

# Messages shorter than this in words get LIGHT treatment
LIGHT_MAX_WORDS = 15


class ModeRouter:
    """Routes messages to REFLEX, LIGHT, or DEEP processing tiers.

    Parameters
    ----------
    reflex_engine : ReflexEngine, optional
        The reflex engine for instant pattern matches.
    deep_word_threshold : int
        Messages with more words than this default to DEEP.

    """

    def __init__(
        self,
        reflex_engine=None,
        deep_word_threshold: int = 40,
    ):
        self.reflex_engine = reflex_engine
        self.deep_word_threshold = deep_word_threshold

    def route(self, message: str) -> ThinkingTier:
        """Determine the thinking tier for a message.

        Returns ThinkingTier.REFLEX, LIGHT, or DEEP.
        """
        clean = message.strip()
        if not clean:
            return ThinkingTier.REFLEX

        words = clean.split()
        word_count = len(words)
        lower = clean.lower()

        # 1. Check reflex patterns first
        if self.reflex_engine:
            reflex_result = self.reflex_engine.check(clean)
            if reflex_result is not None:
                logger.debug("Routed to REFLEX: pattern match")
                return ThinkingTier.REFLEX

        # 2. Very short messages → REFLEX (greetings, one-word commands)
        if word_count <= REFLEX_MAX_WORDS:
            # Check for deep keywords even in short messages
            if any(kw in lower for kw in DEEP_KEYWORDS):
                logger.debug("Routed to DEEP: short but contains deep keyword")
                return ThinkingTier.DEEP
            logger.debug("Routed to REFLEX: very short message (%d words)", word_count)
            return ThinkingTier.LIGHT  # Short but not a known pattern → LIGHT

        # 3. Check for deep keywords
        if any(kw in lower for kw in DEEP_KEYWORDS):
            logger.debug("Routed to DEEP: keyword match")
            return ThinkingTier.DEEP

        # 4. Long messages → DEEP
        if word_count >= self.deep_word_threshold:
            logger.debug("Routed to DEEP: long message (%d words)", word_count)
            return ThinkingTier.DEEP

        # 5. Check for code blocks or technical content
        if "```" in clean or re.search(r"def |class |import |function |const |var ", clean):
            logger.debug("Routed to DEEP: contains code")
            return ThinkingTier.DEEP

        # 6. Questions → LIGHT or DEEP based on length
        if clean.endswith("?"):
            if word_count > LIGHT_MAX_WORDS:
                return ThinkingTier.DEEP
            return ThinkingTier.LIGHT

        # 7. Default: medium messages → LIGHT
        logger.debug("Routed to LIGHT: default (%d words)", word_count)
        return ThinkingTier.LIGHT