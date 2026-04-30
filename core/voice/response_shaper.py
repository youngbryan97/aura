"""core/voice/response_shaper.py — Post-LLM Response Shaping

The LLM generates raw language. This module enforces the substrate's will on it.

Think of it like this: the LLM is vocal cords vibrating. This module is the
tongue, lips, and jaw that shape those vibrations into the specific sounds
the brain intended. The substrate decided how to speak — this ensures
the voicebox obeys.

Operations:
  1. Word budget enforcement (hard truncation with grace)
  2. Trailing question suppression
  3. Punctuation reshaping (mood-driven)
  4. Capitalization enforcement
  5. Multi-message splitting
  6. Fragment injection (break long sentences when substrate demands fragments)
  7. Acknowledgment-only reduction
  8. Filler/hedge injection (post-hoc naturalization)
"""
from __future__ import annotations


import logging
import random
import re
from typing import List, Optional, Tuple

from core.voice.speech_profile import SpeechProfile

logger = logging.getLogger("Voice.ResponseShaper")

# Common trailing question patterns that betray chatbot origins
_TRAILING_QUESTION_PATTERNS = re.compile(
    r'(?:'
    r'(?:what|how)\s+(?:about|do)\s+you'
    r'|what\s+(?:are\s+your|do\s+you)\s+thoughts'
    r'|(?:does|did)\s+that\s+(?:make\s+sense|resonate|help)'
    r'|(?:want|like)\s+(?:me\s+)?to\s+'
    r'|anything\s+(?:else|you\s+want)'
    r'|shall\s+(?:I|we)'
    r'|would\s+you\s+like'
    r'|how\s+(?:was|did)\s+(?:it|that)'
    r'|(?:you\s+)?know\s+what\s+I\s+mean'
    r'|what\s+do\s+you\s+think'
    r'|thoughts\??'
    r'|right\?\s*$'
    r')',
    re.IGNORECASE
)

# Generic closer questions
_GENERIC_CLOSERS = re.compile(
    r'(?:^|\.\s+)(?:'
    r'What do you think\??'
    r'|How about you\??'
    r'|What are your thoughts\??'
    r'|Does that make sense\??'
    r'|Let me know (?:if|what) you think\.?'
    r'|Curious to hear your (?:thoughts|take)\.?'
    r')\s*$',
    re.IGNORECASE
)

# Assistant-isms to scrub — these patterns are matched and removed entirely
_ASSISTANT_ISMS = [
    re.compile(r'\bCertainly[!,.]?\s*', re.IGNORECASE),
    re.compile(r"\bI'd be happy to\b[^.!?]*[.!?]?\s*", re.IGNORECASE),
    re.compile(r"\bThat's a (?:great|good|interesting|wonderful) (?:question|point|thought)[!.]?\s*", re.IGNORECASE),
    re.compile(r'\bAbsolutely[!,.]?\s*', re.IGNORECASE),
    re.compile(r'\bGreat question[!.]?\s*', re.IGNORECASE),
    re.compile(r'\bI appreciate (?:you sharing|that|your)[^.!?]*[.!?]?\s*', re.IGNORECASE),
    re.compile(r'\bThank you for sharing[^.!?]*[.!?]?\s*', re.IGNORECASE),
    re.compile(r"\bWould you like (?:me to|to)\b[^.!?]*\??\s*$", re.IGNORECASE),
]

# Acknowledgment vocabulary
_ACK_WORDS = [
    "yeah", "mm", "right", "fair", "hm", "got it",
    "true", "okay yeah", "makes sense", "yep", "mhm",
]


class ResponseShaper:
    """Enforces the SpeechProfile on raw LLM output.

    This is not optional. Every response passes through this.
    The substrate said how to speak. This makes it so.
    """

    @staticmethod
    def shape(raw: str, profile: SpeechProfile) -> str | List[str]:
        """Shape a raw LLM response according to the SpeechProfile.

        Returns:
            str for single message, List[str] for multi-message.
        """
        if not raw or not raw.strip():
            return raw

        text = raw.strip()

        # ── 0. Scrub assistant-isms ──────────────────────────────────────
        for pattern in _ASSISTANT_ISMS:
            text = pattern.sub("", text)
        text = text.strip()

        # ── 1. Acknowledgment-only check ─────────────────────────────────
        if random.random() < profile.acknowledgment_only_probability:
            # Substrate says: just acknowledge, don't elaborate
            ack = random.choice(_ACK_WORDS)
            if profile.capitalization == "lowercase":
                ack = ack.lower()
            logger.debug("🗣️ [Shaper] Acknowledgment-only triggered: '%s'", ack)
            return ack

        # ── 2. Trailing question suppression ─────────────────────────────
        if profile.trailing_question_banned:
            text = _strip_trailing_question(text)

        # ── 3. Generic closer removal ────────────────────────────────────
        text = _GENERIC_CLOSERS.sub("", text).strip()

        # ── 4. Word budget enforcement ───────────────────────────────────
        text = _enforce_word_budget(text, profile.word_budget)

        # ── 5. Punctuation reshaping ─────────────────────────────────────
        text = _reshape_punctuation(text, profile)

        # ── 6. Capitalization enforcement ────────────────────────────────
        text = _enforce_capitalization(text, profile.capitalization)

        # ── 7. Fragment injection ────────────────────────────────────────
        if profile.fragment_ratio > 0.25:
            text = _inject_fragments(text, profile.fragment_ratio)

        # ── 8. Ellipsis injection ────────────────────────────────────────
        if random.random() < profile.ellipsis_probability:
            text = _maybe_add_ellipsis(text)

        # ── 9. Multi-message splitting ───────────────────────────────────
        if profile.multi_message and profile.multi_message_count > 1:
            messages = _split_into_messages(text, profile.multi_message_count)
            if len(messages) > 1:
                logger.debug(
                    "🗣️ [Shaper] Split into %d messages: %s",
                    len(messages),
                    [m[:30] for m in messages],
                )
                return messages

        # Final cleanup
        text = text.strip()
        if not text:
            text = random.choice(_ACK_WORDS)

        return text


# ─────────────────────────────────────────────────────────────────────────────
# Internal shaping operations
# ─────────────────────────────────────────────────────────────────────────────

def _strip_trailing_question(text: str) -> str:
    """Remove chatbot-style trailing questions."""
    # Check if the LAST sentence is a trailing question
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return text  # Don't strip if it's the only sentence

    last = sentences[-1].strip()
    if _TRAILING_QUESTION_PATTERNS.search(last):
        # Remove the trailing question
        result = " ".join(sentences[:-1]).strip()
        # Clean up trailing punctuation
        result = re.sub(r'\s+$', '', result)
        if result and result[-1] not in ".!?…":
            result += "."
        logger.debug("🗣️ [Shaper] Stripped trailing question: '%s'", last[:50])
        return result

    return text


def _enforce_word_budget(text: str, budget: int) -> str:
    """Hard-enforce word budget with graceful sentence-boundary truncation."""
    words = text.split()
    if len(words) <= budget:
        return text

    # Try to cut at sentence boundary near the budget
    truncated_words = words[:budget]
    truncated = " ".join(truncated_words)

    # Find the last sentence boundary
    last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    last_dash = truncated.rfind("—")

    # If we have a sentence boundary in the last 40% of the budget, cut there
    cutoff_threshold = len(truncated) * 0.6
    if last_period > cutoff_threshold:
        truncated = truncated[: last_period + 1]
    elif last_dash > cutoff_threshold:
        truncated = truncated[: last_dash + 1]
    # Otherwise just use the word-count cut (better than losing content at a bad boundary)

    logger.debug("🗣️ [Shaper] Budget enforced: %d → %d words", len(words), len(truncated.split()))
    return truncated.strip()


def _reshape_punctuation(text: str, profile: SpeechProfile) -> str:
    """Adjust punctuation to match mood."""
    # Remove excess exclamation marks
    if not profile.exclamation_allowed:
        text = text.replace("!", ".")
    elif profile.exclamation_max >= 0:
        count = 0
        result = []
        for char in text:
            if char == "!":
                count += 1
                if count <= profile.exclamation_max:
                    result.append(char)
                else:
                    result.append(".")
            else:
                result.append(char)
        text = "".join(result)

    # Period weight: reduce periods for fragment style
    if profile.period_weight < 0.3:
        # Sometimes drop periods at end of short sentences
        sentences = _split_sentences(text)
        rebuilt = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s.split()) < 6 and s.endswith(".") and random.random() > profile.period_weight:
                s = s[:-1]  # Drop the period
            rebuilt.append(s)
        text = " ".join(rebuilt)

    return text


def _enforce_capitalization(text: str, mode: str) -> str:
    """Enforce capitalization rules."""
    if mode == "lowercase":
        # Lowercase everything except proper nouns (approximation: keep words that
        # appear mid-sentence as capitalized if they look like names)
        sentences = _split_sentences(text)
        result_parts = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # Lowercase the first character, keep rest
            words = s.split()
            lowered = []
            for i, w in enumerate(words):
                # Keep capitalized if it looks like a proper noun (mid-sentence capital)
                if i > 0 and w[0:1].isupper() and len(w) > 1 and not w.isupper():
                    lowered.append(w)  # Keep — probably a name
                else:
                    lowered.append(w.lower())
            result_parts.append(" ".join(lowered))
        text = " ".join(result_parts)
    elif mode == "emphatic":
        # Allow existing caps, maybe add some
        pass  # LLM already handles this; we just don't suppress it

    return text


def _inject_fragments(text: str, fragment_ratio: float) -> str:
    """Break some sentences into fragments for naturalness."""
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return text

    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        # Long sentences can be broken
        if len(words) > 10 and random.random() < fragment_ratio:
            # Find a natural break point (conjunction, comma)
            break_points = []
            for i, w in enumerate(words):
                wl = w.lower().rstrip(",.;:")
                if wl in ("and", "but", "so", "because", "though", "—") and 3 < i < len(words) - 3:
                    break_points.append(i)
            if break_points:
                bp = random.choice(break_points)
                first_part = " ".join(words[:bp]).rstrip(",;:")
                second_part = " ".join(words[bp:])
                result.append(first_part)
                result.append(second_part)
                continue
        result.append(s)

    return " ".join(result)


def _maybe_add_ellipsis(text: str) -> str:
    """Possibly add a trailing ellipsis for contemplative mood."""
    text = text.rstrip()
    if text.endswith("...") or text.endswith("…"):
        return text
    # Replace final punctuation with ellipsis
    if text and text[-1] in ".!":
        text = text[:-1] + "..."
    elif text and text[-1] not in "?":
        text += "..."
    return text


def _split_into_messages(text: str, target_count: int) -> List[str]:
    """Split response into multiple messages like natural texting."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    # Distribute sentences across messages
    messages = []
    per_msg = max(1, len(sentences) // target_count)

    current = []
    for i, s in enumerate(sentences):
        current.append(s.strip())
        if len(current) >= per_msg and len(messages) < target_count - 1:
            messages.append(" ".join(current))
            current = []

    if current:
        messages.append(" ".join(current))

    # Filter empty messages
    messages = [m.strip() for m in messages if m.strip()]
    return messages if messages else [text]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling common edge cases."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r'(?<=[.!?…])\s+', text)
    return [p for p in parts if p.strip()]
