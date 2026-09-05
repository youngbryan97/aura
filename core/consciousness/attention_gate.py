"""core/consciousness/attention_gate.py
=======================================
Causal Attention Gate — Active Context Pruning.

The Problem:
    The AttentionSchema tracks what Aura is attending to and generates
    HOT meta-representations. But this is purely descriptive. The LLM
    still sees the entire context window — every memory retrieval, every
    system block, every conversation turn. The attention schema says
    "I am focusing on X" but doesn't prevent the model from attending to Y.

    In biological brains, attention is causally restrictive: unattended
    stimuli are actively suppressed at the neural level. They don't reach
    conscious processing. The current system has no equivalent — attention
    is a report, not a filter.

The Solution:
    An active gating mechanism that sits between the context assembler
    and the LLM, reading the attention schema's current focus and salience
    map, and REMOVING or COMPRESSING context entries that fall below the
    attention threshold.

    This is not an optional annotation. It literally prevents the LLM
    from seeing irrelevant context, forcing the generative mind to operate
    within the bounds of the attention schema's focus.

    The gate operates on the final message list produced by
    ContextAssembler.build_messages():
    1. Read current attentional focus and salience map
    2. Score each non-system message for relevance to the current focus
    3. Messages below the attention threshold are compressed or removed
    4. The gate respects a minimum context floor (never removes all context)

    Modulation by internal state:
    - High arousal → narrower gate (tunnel vision under stress)
    - High curiosity → wider gate (open attention for exploration)
    - Existential threat → maximally narrow gate (survival focus)

    This creates genuine selective attention: the LLM's outputs are
    causally different because it literally cannot see what the gate
    removed.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Consciousness.AttentionGate")

# ── Configuration ─────────────────────────────────────────────────────────────

# Minimum messages to retain (never strip below this)
MIN_CONTEXT_MESSAGES = 4

# Base attention threshold (messages scoring below this get compressed)
BASE_ATTENTION_THRESHOLD = 0.3

# Arousal modulation: high arousal narrows the gate
AROUSAL_THRESHOLD_BOOST = 0.25  # Max additional threshold from arousal

# Curiosity modulation: high curiosity widens the gate
CURIOSITY_THRESHOLD_REDUCTION = 0.15  # Max threshold reduction from curiosity

# Threat modulation: existential threat maximally narrows
THREAT_THRESHOLD_BOOST = 0.35

# Maximum fraction of user/assistant messages that can be gated out
MAX_GATE_FRACTION = 0.5

# Roles that are never gated
PROTECTED_ROLES = frozenset({"system"})

# Words that boost relevance scoring
_RELEVANCE_BOOST_RE = re.compile(
    r"\b(?:you|your|aura|feel|think|opinion|believe|remember|experience)\b",
    re.IGNORECASE,
)


class AttentionGate:
    """Active context pruning based on the attention schema's current focus.

    Call gate_context() on the final message list from ContextAssembler
    before passing to the LLM.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._total_gated: int = 0
        self._total_passed: int = 0
        self._total_calls: int = 0
        self._last_threshold: float = BASE_ATTENTION_THRESHOLD
        self._last_gate_report: dict[str, Any] = {}

        logger.info("AttentionGate ONLINE — causal context pruning active")

    def gate_context(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Apply attentional gating to a message list.

        Messages are scored for relevance to the current attentional focus.
        Below-threshold messages are compressed (summarized to one line)
        or removed entirely.

        Returns a new list — the input is not modified.
        """
        with self._lock:
            self._total_calls += 1

            if not messages or len(messages) <= MIN_CONTEXT_MESSAGES:
                self._total_passed += len(messages or [])
                return list(messages or [])

            # 1. Get current attention state
            focus, salience_map = self._get_attention_state()
            if not focus:
                # No attentional focus → pass everything through
                self._total_passed += len(messages)
                return list(messages)

            # 2. Compute dynamic threshold
            threshold = self._compute_threshold()
            self._last_threshold = threshold

            # 3. Score each message for relevance
            scored = []
            for i, msg in enumerate(messages):
                role = str(msg.get("role", "")).strip().lower()
                content = str(msg.get("content", ""))

                if role in PROTECTED_ROLES:
                    scored.append((msg, 1.0, True))  # Always pass system messages
                    continue

                relevance = self._score_relevance(content, focus, salience_map)
                scored.append((msg, relevance, False))

            # 4. Apply gating
            # Count non-protected messages
            gateable = [(msg, rel, prot) for msg, rel, prot in scored if not prot]
            max_removable = int(len(gateable) * MAX_GATE_FRACTION)

            # Sort gateable by relevance (ascending) to identify candidates
            gateable_sorted = sorted(
                enumerate(gateable), key=lambda x: x[1][1]
            )

            # Mark low-relevance messages for gating
            gate_indices = set()
            gated_count = 0
            for orig_idx, (msg, rel, prot) in gateable_sorted:
                if gated_count >= max_removable:
                    break
                if rel < threshold:
                    gate_indices.add(orig_idx)
                    gated_count += 1

            # 5. Build output
            # Always keep the last few messages (recency protection)
            result = []
            gateable_idx = 0
            gated_this_call = 0
            passed_this_call = 0

            for msg, relevance, protected in scored:
                if protected:
                    result.append(msg)
                    passed_this_call += 1
                else:
                    if gateable_idx in gate_indices:
                        # Compress instead of remove for context continuity
                        compressed = self._compress_message(msg, relevance)
                        if compressed:
                            result.append(compressed)
                        gated_this_call += 1
                    else:
                        result.append(msg)
                        passed_this_call += 1
                    gateable_idx += 1

            # Ensure minimum context
            if len(result) < MIN_CONTEXT_MESSAGES:
                result = list(messages[-MIN_CONTEXT_MESSAGES:])
                gated_this_call = 0
                passed_this_call = len(result)

            # Always keep the last 2 user/assistant messages intact
            # (the immediate conversation context must survive)
            if len(messages) > 2:
                last_two = messages[-2:]
                for lt in last_two:
                    if lt not in result:
                        result.append(lt)

            self._total_gated += gated_this_call
            self._total_passed += passed_this_call

            self._last_gate_report = {
                "threshold": round(threshold, 3),
                "gated": gated_this_call,
                "passed": passed_this_call,
                "focus": focus[:80] if focus else "",
                "total_messages": len(messages),
                "output_messages": len(result),
            }

            if gated_this_call > 0:
                logger.debug(
                    "AttentionGate: gated %d/%d messages (threshold=%.2f, focus='%s')",
                    gated_this_call,
                    len(messages),
                    threshold,
                    focus[:50],
                )

            return result

    # ── Relevance Scoring ─────────────────────────────────────────────────

    def _score_relevance(
        self,
        content: str,
        focus: str,
        salience_map: dict[str, float],
    ) -> float:
        """Score a message's relevance to the current attentional focus.

        Uses keyword overlap + salience map + structural heuristics.
        """
        if not content or not focus:
            return 0.5

        score = 0.3  # Base relevance

        # Keyword overlap with focus
        focus_words = set(focus.lower().split())
        content_words = set(content.lower().split())
        if focus_words:
            overlap = len(focus_words & content_words) / len(focus_words)
            score += overlap * 0.35

        # Salience map matching
        for topic, salience in salience_map.items():
            if topic.lower() in content.lower():
                score += salience * 0.2

        # Self-referential content is always relevant
        if _RELEVANCE_BOOST_RE.search(content):
            score += 0.15

        # Recency bias: shorter messages in the middle of context
        # are less relevant than longer substantive ones
        word_count = len(content.split())
        if word_count < 5:
            score -= 0.1
        elif word_count > 50:
            score += 0.05

        return max(0.0, min(1.0, score))

    # ── Message Compression ───────────────────────────────────────────────

    @staticmethod
    def _compress_message(
        msg: dict[str, str], relevance: float
    ) -> dict[str, str] | None:
        """Compress a gated message to a minimal summary.

        Very low relevance messages are removed entirely.
        Moderate-low relevance messages get compressed to one line.
        """
        if relevance < 0.1:
            return None  # Remove entirely

        role = msg.get("role", "user")
        content = str(msg.get("content", ""))

        # Extract first meaningful line
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if not lines:
            return None

        summary = lines[0][:80]
        if len(content) > len(summary):
            summary += " [...]"

        return {"role": role, "content": f"[gated: {summary}]"}

    # ── Threshold Computation ─────────────────────────────────────────────

    def _compute_threshold(self) -> float:
        """Compute the dynamic attention threshold based on internal state.

        Higher threshold = narrower gate = less context passes through.
        """
        threshold = BASE_ATTENTION_THRESHOLD

        try:
            from core.container import ServiceContainer

            # Arousal → narrows attention (tunnel vision under stress)
            affect = ServiceContainer.get("affective_circumplex", default=None)
            if affect and hasattr(affect, "_sample_raw_axes"):
                _, arousal = affect._sample_raw_axes()
                if arousal > 0.5:
                    threshold += (arousal - 0.5) * AROUSAL_THRESHOLD_BOOST * 2.0

            # Curiosity → widens attention (open to everything)
            substrate = ServiceContainer.get("conscious_substrate", default=None)
            if substrate and hasattr(substrate, "x"):
                curiosity_idx = getattr(substrate, "idx_curiosity", 1)
                curiosity = float(substrate.x[curiosity_idx])
                if curiosity > 0.3:
                    threshold -= curiosity * CURIOSITY_THRESHOLD_REDUCTION

            # Existential threat → maximally narrow gate
            stakes = ServiceContainer.get("existential_stakes", default=None)
            if stakes:
                threat = stakes.get_existential_threat()
                if threat > 0.3:
                    threshold += threat * THREAT_THRESHOLD_BOOST

        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("attention_gate", exc)

        return max(0.1, min(0.8, threshold))

    # ── Attention State ───────────────────────────────────────────────────

    @staticmethod
    def _get_attention_state():
        """Read the current attentional focus and salience map."""
        try:
            from core.container import ServiceContainer

            schema = ServiceContainer.get("attention_schema", default=None)
            if schema is None:
                return "", {}

            focus = ""
            if hasattr(schema, "current_focus") and schema.current_focus:
                focus = str(getattr(schema.current_focus, "content", ""))

            salience_map = {}
            if hasattr(schema, "salience_map"):
                salience_map = dict(schema.salience_map or {})

            return focus, salience_map
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return "", {}

    # ── Telemetry ─────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            total = self._total_gated + self._total_passed
            return {
                "total_calls": self._total_calls,
                "total_gated": self._total_gated,
                "total_passed": self._total_passed,
                "gate_rate": round(
                    self._total_gated / max(1, total), 4
                ),
                "last_threshold": round(self._last_threshold, 3),
                "last_report": self._last_gate_report,
            }

    def is_ready(self) -> bool:
        """Synchronous liveness probe for runtime health."""
        with self._lock:
            return bool(
                0.1 <= self._last_threshold <= 0.8
                and self._total_calls >= 0
                and self._total_gated >= 0
                and self._total_passed >= 0
            )

    def get_context_block(self) -> str:
        """Minimal context block about gating state."""
        with self._lock:
            if self._total_calls == 0:
                return ""
            total = self._total_gated + self._total_passed
            gate_pct = (self._total_gated / max(1, total)) * 100
            return (
                f"## ATTENTION GATE\n"
                f"- {self._total_gated}/{total} context entries actively suppressed "
                f"({gate_pct:.0f}% gating rate)\n"
                f"- Current threshold: {self._last_threshold:.2f}"
            )


# ── Singleton ─────────────────────────────────────────────────────────────────

_GATE: AttentionGate | None = None


def get_attention_gate() -> AttentionGate:
    global _GATE
    if _GATE is None:
        _GATE = AttentionGate()
    return _GATE
