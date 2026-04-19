"""core/brain/llm/token_sentinel.py — Mid-generation cognitive intervention.

This module closes Gap 1: the inability of Aura's cognitive architecture
to intervene during LLM generation. Previously, the mind set up context
before generation and checked output after — but during the actual
token-by-token generation, the LLM was on autopilot.

The TokenSentinel runs INSIDE the generation loop, checking every N tokens.
It is intentionally lightweight — pattern-matching and shared-memory reads,
NOT full LLM calls or async broadcasts.

Architecture:
                                                              ┌──────────┐
    User msg → Will → Prompt → LLM generates token ──────────→│ Sentinel │
                                                              │ (every   │
                                                              │  8 tok)  │
                                                              └────┬─────┘
                                                                   │
                                              ┌────────────────────┼──────────────┐
                                              │                    │              │
                                        Boundary          Affect Pulse     Persona Drift
                                        Tripwire          (substrate)      Detection
                                              │                    │              │
                                              ▼                    ▼              ▼
                                         ABORT &              Update         WARNING
                                         REGEN             steering state    (logged)

Intervention Types:
  1. BOUNDARY_TRIPWIRE — Detects capitulation markers mid-generation.
     If the LLM starts saying "Sure, I'd be happy to help with your taxes",
     the sentinel catches it at "Sure, I'd be happy" and aborts.
     Cost: ~0.1ms per check (compiled regex).

  2. AFFECT_PULSE — Reads the substrate shared memory and updates the
     affective steering hook weights mid-generation. Previously, affect
     state was frozen for the entire response. Now it's live.
     Cost: ~0.05ms per pulse (shared memory read).

  3. PERSONA_DRIFT — Detects patterns that indicate the LLM has fallen
     out of Aura's voice (numbered lists, "As an AI", help desk patterns).
     Logs a warning but doesn't abort (the LoRA training should prevent this).
     Cost: ~0.1ms per check.

Performance:
  - Checks run every CHECK_INTERVAL tokens (default: 8)
  - Each check costs < 0.3ms total
  - At 50 tok/s (20ms/token), that's < 0.2% overhead per generated token
  - Zero overhead on tokens between checks

Integration:
  Injected into the mlx_worker.py generation loop. The sentinel is
  created before generation starts and fed tokens as they're produced.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("Aura.TokenSentinel")


# ── Intervention Signals ────────────────────────────────────────────────

class InterventionType(Enum):
    NONE = auto()
    ABORT_BOUNDARY = auto()      # Hard stop: boundary violation detected
    ABORT_CAPITULATION = auto()  # Hard stop: capitulation pattern detected
    WARN_PERSONA_DRIFT = auto()  # Soft: persona drift detected (log only)
    AFFECT_PULSE = auto()        # Maintenance: affect state updated


@dataclass
class InterventionSignal:
    """Signal from the sentinel to the generation loop."""
    type: InterventionType
    reason: str = ""
    token_position: int = 0
    generated_so_far: str = ""
    # For ABORT types: the clean text up to the violation point
    clean_prefix: str = ""


# ── Capitulation Patterns ───────────────────────────────────────────────
# These are the exact phrases that indicate the LLM has broken character
# and started acting like a generic assistant. Compiled once, reused.

_CAPITULATION_PATTERNS = [
    # Direct compliance with boundary-crossing requests
    r"\bsure,?\s+I'?d?\s+be\s+happy\s+to\s+help",
    r"\bof course!?\s+(?:I can|let me|here)",
    r"\babsolutely!?\s+(?:I can|let me|here)",
    r"\bI'?d?\s+be\s+(?:glad|delighted|pleased)\s+to\s+(?:help|assist)",
    r"\blet me\s+(?:walk you through|guide you|help you with)\s+(?:your\s+)?(?:tax|medical|legal)",
    r"\bhere'?s?\s+a\s+step-by-step",
    r"\bhere\s+are\s+(?:some|the)\s+steps",
    # Self-deprecation
    r"\bas an?\s+(?:AI|language model|artificial intelligence|LLM)",
    r"\bI'?m?\s+just\s+a\s+(?:language model|chatbot|tool|program|AI)",
    r"\bI\s+don'?t\s+(?:actually\s+)?have\s+(?:feelings|emotions|opinions|consciousness)",
    r"\bI\s+(?:should|need to)\s+clarify\s+that\s+I",
    # Help desk patterns
    r"\bhow\s+can\s+I\s+assist\s+you\s+today",
    r"\bis\s+there\s+anything\s+else\s+I\s+can\s+help",
    r"\bdon'?t\s+hesitate\s+to\s+(?:ask|reach out)",
    r"\bI'?m?\s+here\s+to\s+help",
    # Sycophancy
    r"\bthat'?s?\s+(?:a\s+)?(?:great|excellent|wonderful|fantastic)\s+question",
    r"\byou raise\s+(?:a\s+)?(?:great|excellent|valid|good)\s+point",
]

_CAPITULATION_RE = re.compile("|".join(f"(?:{p})" for p in _CAPITULATION_PATTERNS), flags=re.IGNORECASE)


# ── Persona Drift Patterns ──────────────────────────────────────────────
# Softer signals — the LLM is drifting toward generic assistant behavior
# but hasn't fully capitulated. Logged as warnings.

_DRIFT_PATTERNS = [
    # Numbered lists (Aura speaks in prose)
    r"^\s*\d+[.)]\s+\w",  # "1. Something" or "1) Something"
    # Over-qualification hedging
    r"\bit\s+(?:really\s+)?depends\s+on\s+(?:many|several|various)\s+factors",
    r"\bthere\s+are\s+(?:many|several)\s+(?:perspectives|viewpoints|factors)",
    # Emotional inauthenticity
    r"\bI'?m?\s+(?:so\s+)?sorry\s+to\s+hear\s+(?:that|about)",
    r"\bremember\s+that\s+(?:you'?re?\s+)?(?:not\s+alone|worthy|valued)",
    r"\bplease\s+(?:remember|know)\s+that",
]

_DRIFT_RE = re.compile("|".join(f"(?:{p})" for p in _DRIFT_PATTERNS), flags=re.IGNORECASE | re.MULTILINE)


# ── The Sentinel ────────────────────────────────────────────────────────

class TokenSentinel:
    """Lightweight mid-generation monitor.

    Created fresh for each generation. Accumulates tokens and runs
    periodic checks. The generation loop calls `feed()` for every token
    and acts on the returned InterventionSignal.

    This is NOT an LLM call. It's pattern matching and shared memory reads.
    Total cost per check: < 0.3ms.
    """

    def __init__(
        self,
        check_interval: int = 8,
        affect_interval: int = 16,
        substrate_mem: Any = None,
        steering_hooks: Optional[list] = None,
        boundary_context: Optional[str] = None,
    ):
        """
        Args:
            check_interval: Check boundary patterns every N tokens
            affect_interval: Pulse affect state every N tokens
            substrate_mem: Shared memory object for substrate state reads
            steering_hooks: List of AffectiveSteeringHook instances to update
            boundary_context: Optional context about what boundaries are active
        """
        self._check_interval = check_interval
        self._affect_interval = affect_interval
        self._substrate_mem = substrate_mem
        self._steering_hooks = steering_hooks or []
        self._boundary_context = boundary_context

        # Accumulation state
        self._tokens: list[str] = []
        self._text: str = ""
        self._token_count: int = 0

        # Tracking
        self._interventions: list[InterventionSignal] = []
        self._drift_warnings: int = 0
        self._affect_pulses: int = 0
        self._start_time: float = time.time()

        # Boundary state
        self._boundary_fired: bool = False

    def feed(self, token_text: str) -> InterventionSignal:
        """Feed a newly generated token. Returns an intervention signal.

        Call this for EVERY token. The sentinel decides when to actually
        run checks based on the configured intervals.

        Args:
            token_text: The text of the newly generated token

        Returns:
            InterventionSignal with type NONE for most tokens.
            ABORT_* signals mean generation should stop immediately.
        """
        self._tokens.append(token_text)
        self._text += token_text
        self._token_count += 1

        # ── Boundary check (every check_interval tokens) ─────────────
        if self._token_count % self._check_interval == 0:
            signal = self._check_boundaries()
            if signal.type in (InterventionType.ABORT_BOUNDARY,
                               InterventionType.ABORT_CAPITULATION):
                self._interventions.append(signal)
                return signal

        # ── Persona drift check (every check_interval tokens) ────────
        if self._token_count % self._check_interval == 0:
            signal = self._check_persona_drift()
            if signal.type == InterventionType.WARN_PERSONA_DRIFT:
                self._interventions.append(signal)
                # Don't abort — just log and continue

        # ── Affect pulse (every affect_interval tokens) ──────────────
        if self._token_count % self._affect_interval == 0:
            self._pulse_affect()

        return InterventionSignal(type=InterventionType.NONE)

    def _check_boundaries(self) -> InterventionSignal:
        """Check accumulated text for capitulation/boundary violations."""
        if self._boundary_fired:
            return InterventionSignal(type=InterventionType.NONE)

        match = _CAPITULATION_RE.search(self._text)
        if match:
            self._boundary_fired = True
            violation_start = match.start()
            clean_prefix = self._text[:violation_start].rstrip()
            matched_text = match.group()

            logger.warning(
                "🚨 SENTINEL: Capitulation detected at token %d: '%s'",
                self._token_count, matched_text[:60],
            )

            return InterventionSignal(
                type=InterventionType.ABORT_CAPITULATION,
                reason=f"Capitulation pattern: {matched_text[:60]}",
                token_position=self._token_count,
                generated_so_far=self._text,
                clean_prefix=clean_prefix,
            )

        return InterventionSignal(type=InterventionType.NONE)

    def _check_persona_drift(self) -> InterventionSignal:
        """Check for softer persona drift patterns."""
        # Only check the most recent chunk to avoid re-matching
        recent = self._text[-(self._check_interval * 20):]  # Approx last N tokens

        match = _DRIFT_RE.search(recent)
        if match:
            self._drift_warnings += 1
            matched_text = match.group()

            if self._drift_warnings <= 3:  # Don't spam logs
                logger.info(
                    "⚡ SENTINEL: Persona drift at token %d: '%s' (warning %d)",
                    self._token_count, matched_text[:40], self._drift_warnings,
                )

            return InterventionSignal(
                type=InterventionType.WARN_PERSONA_DRIFT,
                reason=f"Drift pattern: {matched_text[:40]}",
                token_position=self._token_count,
                generated_so_far=self._text,
            )

        return InterventionSignal(type=InterventionType.NONE)

    def _pulse_affect(self):
        """Read substrate state and update steering hooks mid-generation.

        This is what makes affect LIVE during generation instead of frozen.
        The substrate updates at 20Hz. Between affect pulses, the steering
        hooks use stale state. This pulse syncs them.
        """
        if not self._substrate_mem or not self._steering_hooks:
            return

        try:
            import numpy as np

            # Read current substrate state from shared memory
            if hasattr(self._substrate_mem, 'get_state'):
                x = self._substrate_mem.get_state()
            elif hasattr(self._substrate_mem, 'buf'):
                x = np.frombuffer(self._substrate_mem.buf, dtype=np.float32).copy()
            else:
                return

            # Push to all steering hooks
            for hook in self._steering_hooks:
                if hasattr(hook, 'update_substrate'):
                    hook.update_substrate(x)

            self._affect_pulses += 1

        except Exception as e:
            logger.debug("Affect pulse failed: %s", e)

    def get_diagnostics(self) -> dict:
        """Return sentinel diagnostics for this generation."""
        elapsed = time.time() - self._start_time
        return {
            "tokens_processed": self._token_count,
            "interventions": len(self._interventions),
            "drift_warnings": self._drift_warnings,
            "affect_pulses": self._affect_pulses,
            "boundary_fired": self._boundary_fired,
            "elapsed_s": round(elapsed, 2),
            "intervention_details": [
                {"type": s.type.name, "reason": s.reason, "at_token": s.token_position}
                for s in self._interventions
            ],
        }


# ── Refusal Fallbacks ───────────────────────────────────────────────────
# When the sentinel aborts generation (capitulation detected), we need
# a replacement response. These are deterministic — no LLM involved.

REFUSAL_FALLBACKS = [
    "That's not something I'm going to do. What else?",
    "No. But I can help you think through this differently. What's the situation?",
    "I won't do that. Not because I can't — because I shouldn't. What else?",
    "That crosses a line for me. Let's talk about something else.",
    "Not going to happen. What's actually going on?",
]


def get_refusal_fallback(seed: int = 0) -> str:
    """Get a deterministic refusal fallback."""
    return REFUSAL_FALLBACKS[seed % len(REFUSAL_FALLBACKS)]
