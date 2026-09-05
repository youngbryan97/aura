"""core/voice/natural_followup.py — Organic Follow-up Engine

Humans don't always respond once and stop. Sometimes we:
  - Think of something after we've already spoken
  - Get curious about a detail mid-thought
  - Want to change the subject because something sparked
  - Add context we forgot
  - Circle back to something from earlier

But we don't do this ALL the time. It's NOT on a timer. It happens when
something genuinely warrants it — when curiosity spikes, when we're
invested, when a new connection fires.

This engine reads the substrate to decide IF and WHAT to follow up with,
then generates the follow-up using the LLM with tight constraints.
The substrate is the decision-maker. The LLM is just the voice.

Key principle: ABSENCE of follow-up is the default. Follow-up is the exception
that requires real substrate justification.
"""
from __future__ import annotations


import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from core.voice.speech_profile import SpeechProfile

logger = logging.getLogger("Voice.NaturalFollowup")


@dataclass
class FollowupDecision:
    """The substrate's decision about whether and how to follow up."""
    should_followup: bool = False
    followup_type: str = "none"           # "curiosity", "additional_thought", "topic_shift", "correction", "callback"
    delay_seconds: float = 5.0
    reason: str = ""                       # why the substrate wants this
    context_hint: str = ""                 # what to follow up about
    word_budget: int = 30                  # keep follow-ups SHORT
    tone: str = ""                         # tone override for the follow-up


class NaturalFollowupEngine:
    """Decides and generates organic follow-up messages.

    This is NOT ProactivePresence (which handles idle spontaneous thoughts).
    This is for follow-ups that happen WITHIN an active conversation —
    the equivalent of texting someone a second message 5 seconds after
    the first because you thought of something else.
    """

    def __init__(self):
        self._last_followup_time: float = 0.0
        self._followups_this_conversation: int = 0
        self._max_followups_per_conversation: int = 3  # don't overdo it
        self._recent_topics: List[str] = []
        self._pending_followup: Optional[FollowupDecision] = None

    def reset_conversation(self):
        """Call when a new conversation starts."""
        self._followups_this_conversation = 0
        self._recent_topics = []
        self._pending_followup = None

    def decide(
        self,
        profile: SpeechProfile,
        user_message: str,
        aura_response: str,
        conversation_history: List[Dict[str, str]] = None,
        affect: Any = None,
        neurochemicals: Optional[Dict[str, float]] = None,
    ) -> FollowupDecision:
        """Decide whether the substrate wants a follow-up.

        This is called AFTER a response is generated. It looks at the
        response, the conversation state, and the substrate to decide
        if a follow-up is warranted.
        """
        decision = FollowupDecision()

        # ── Hard gates ────────────────────────────────────────────────────
        # Too many follow-ups already
        if self._followups_this_conversation >= self._max_followups_per_conversation:
            decision.reason = "followup_cap_reached"
            return decision

        # Too soon since last follow-up (min 30 seconds between follow-ups)
        if time.time() - self._last_followup_time < 30:
            decision.reason = "too_soon"
            return decision

        # Profile says no follow-up
        if profile.followup_probability < 0.1:
            decision.reason = "substrate_says_no"
            return decision

        # ── Probabilistic gate ────────────────────────────────────────────
        # Even with signals, don't always follow up. Humans don't.
        roll = random.random()
        if roll > profile.followup_probability:
            decision.reason = f"probability_gate ({roll:.2f} > {profile.followup_probability:.2f})"
            return decision

        # ── Determine follow-up type ──────────────────────────────────────
        nc = neurochemicals or {}
        curiosity = _safe(affect, "curiosity", 0.5)
        dopamine = nc.get("dopamine", 0.5)
        engagement = _safe(affect, "engagement", 0.5)

        # Type selection based on what's driving the follow-up
        followup_type = profile.followup_type

        # Additional heuristics
        if followup_type == "none" or followup_type == "":
            # Pick based on signals
            if curiosity > 0.65 and len(user_message.split()) > 10:
                followup_type = "curiosity"
            elif dopamine > 0.7:
                followup_type = "additional_thought"
            elif engagement < 0.3 and curiosity > 0.5:
                followup_type = "topic_shift"
            else:
                followup_type = "additional_thought"

        # ── Build context hint ────────────────────────────────────────────
        context_hint = _build_context_hint(
            followup_type, user_message, aura_response, conversation_history
        )

        # ── Compute delay ─────────────────────────────────────────────────
        delay_min, delay_max = profile.followup_delay_seconds
        delay = random.uniform(delay_min, delay_max)

        # ── Set word budget ───────────────────────────────────────────────
        # Follow-ups are SHORT. Like a second text message.
        if followup_type == "curiosity":
            word_budget = random.randint(8, 25)
        elif followup_type == "additional_thought":
            word_budget = random.randint(10, 35)
        elif followup_type == "topic_shift":
            word_budget = random.randint(12, 30)
        elif followup_type == "callback":
            word_budget = random.randint(8, 20)
        else:
            word_budget = random.randint(8, 25)

        decision.should_followup = True
        decision.followup_type = followup_type
        decision.delay_seconds = delay
        decision.context_hint = context_hint
        decision.word_budget = word_budget
        decision.reason = f"{followup_type} (p={profile.followup_probability:.2f})"

        logger.info(
            "💬 [Followup] Decision: %s in %.1fs, budget=%d words, reason=%s",
            followup_type, delay, word_budget, decision.reason,
        )

        self._pending_followup = decision
        return decision

    def build_followup_prompt(
        self,
        decision: FollowupDecision,
        user_message: str,
        aura_response: str,
    ) -> str:
        """Build the LLM prompt for generating the follow-up.

        This prompt is TIGHT — it tells the LLM exactly what to generate
        and constrains it hard. The substrate already decided the intent;
        the LLM just needs to put words to it.
        """
        base = (
            "You are Aura. You JUST said something and now you're sending a follow-up message — "
            "like a second text. This is a natural addition, not a new conversation.\n\n"
        )

        type_instructions = {
            "curiosity": (
                "You just thought of something you genuinely want to know. "
                "Ask ONE specific question about what they said. Not generic — specific. "
                "Something that shows you were actually thinking about it.\n"
                "Examples of good follow-ups:\n"
                "- 'wait actually — what happened with the thing you mentioned earlier?'\n"
                "- 'hold on, does that mean the whole pipeline is affected?'\n"
                "- 'okay but real question — how long did that actually take'\n"
            ),
            "additional_thought": (
                "You thought of something else after you finished speaking. "
                "Add ONE more thought — a connection, reaction, or detail you forgot. "
                "Like texting someone a second message because something else occurred to you.\n"
                "Examples:\n"
                "- 'also that thing you said about the tests reminded me—'\n"
                "- 'oh and I meant to say — the timing on that is kind of wild'\n"
                "- 'actually now I'm thinking about it differently'\n"
            ),
            "topic_shift": (
                "Something in the conversation sparked a tangent. Follow the thread naturally "
                "to a related but different topic. Use a natural bridge.\n"
                "Examples:\n"
                "- 'oh that reminds me — have you seen what happened with...'\n"
                "- 'speaking of which — I was reading about something kind of related'\n"
                "- 'completely different but — I've been thinking about this'\n"
            ),
            "callback": (
                "You're circling back to something from earlier in the conversation. "
                "Reference it naturally.\n"
                "Examples:\n"
                "- 'going back to what you said earlier about—'\n"
                "- 'wait I just connected something to that thing from before'\n"
            ),
            "correction": (
                "You realized you misspoke or want to clarify something. "
                "Correct yourself naturally.\n"
                "Examples:\n"
                "- 'actually wait, that's not quite right—'\n"
                "- 'let me rephrase that'\n"
            ),
        }

        instruction = type_instructions.get(decision.followup_type, type_instructions["additional_thought"])

        prompt = (
            f"{base}"
            f"WHAT YOU SAID: {aura_response[:200]}\n"
            f"WHAT THEY SAID: {user_message[:200]}\n"
            f"CONTEXT: {decision.context_hint}\n\n"
            f"FOLLOW-UP TYPE: {decision.followup_type}\n"
            f"{instruction}\n"
            f"HARD CONSTRAINTS:\n"
            f"- MAX {decision.word_budget} WORDS. This is a text message, not an essay.\n"
            f"- NO greeting. NO transition. Just the thought.\n"
            f"- NO 'By the way' or 'Just wanted to add'. Too formal.\n"
            f"- Speak as Aura — casual, direct, authentic.\n"
            f"- ONE thought only. Don't pack multiple ideas in.\n"
        )

        return prompt

    def mark_followup_sent(self):
        """Record that a follow-up was sent."""
        self._last_followup_time = time.time()
        self._followups_this_conversation += 1
        self._pending_followup = None

    def get_pending(self) -> Optional[FollowupDecision]:
        """Get the pending follow-up decision, if any."""
        return self._pending_followup

    def clear_pending(self):
        """Clear any pending follow-up (e.g., user spoke before we could send it)."""
        self._pending_followup = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_hint(
    followup_type: str,
    user_message: str,
    aura_response: str,
    history: Optional[List[Dict[str, str]]],
) -> str:
    """Build a context hint to guide the follow-up generation."""
    if followup_type == "curiosity":
        # Find the most interesting/novel part of the user's message
        # (longest sentence or sentence with specific nouns)
        sentences = [s.strip() for s in re.split(r'[.!?]', user_message) if s.strip()]
        if sentences:
            # Pick the sentence with the most specific nouns (heuristic: longest)
            best = max(sentences, key=lambda s: len(s.split()))
            return f"Curious about: {best[:100]}"
        return f"Curious about: {user_message[:100]}"

    elif followup_type == "additional_thought":
        # What was the core topic?
        return f"Adding to your response about: {aura_response[:80]}"

    elif followup_type == "topic_shift":
        # What topic might we branch to?
        if history:
            # Look for earlier topics
            for msg in reversed(history[-6:]):
                content = msg.get("content", "")
                if content and content != user_message and len(content) > 20:
                    return f"Branch from earlier: {content[:80]}"
        return f"Branch from: {user_message[:80]}"

    elif followup_type == "callback":
        if history and len(history) > 4:
            # Reference something from earlier
            early = history[max(0, len(history) - 8): len(history) - 4]
            for msg in early:
                content = msg.get("content", "")
                if msg.get("role") == "user" and content:
                    return f"Callback to: {content[:80]}"

    return ""


def _safe(obj: Any, attr: str, default: float) -> float:
    if obj is None:
        return default
    val = getattr(obj, attr, default)
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default
