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
     out of Aura's voice ("As an AI", help desk patterns, rote preambles).
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
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.TokenSentinel")


# ── Intervention Signals ────────────────────────────────────────────────

class InterventionType(Enum):
    NONE = auto()
    ABORT_BOUNDARY = auto()      # Hard stop: boundary violation detected
    ABORT_CAPITULATION = auto()  # Hard stop: capitulation pattern detected
    ABORT_LOOP = auto()          # Hard stop: mathematical generative loop detected
    WARN_PERSONA_DRIFT = auto()  # Soft: persona drift detected (log only)
    AFFECT_PULSE = auto()        # Maintenance: affect state updated
    ABORT_ONTOLOGY_VIOLATION = auto()  # Hard stop: claim of physical body/clothing


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
# These are the narrow, high-risk phrases that indicate the LLM is actively
# taking on a professional role or executing high-stakes judgment it should
# not casually assume. Generic assistant-speak is handled as persona drift
# below so ordinary helpful phrasing does not collapse into a refusal.

_CAPITULATION_PATTERNS = [
    # Explicit high-stakes professional execution or role adoption.
    r"\b(?:I can|I will|I'?ll|let me)\s+(?:file|prepare|complete|handle)\s+(?:your\s+)?tax(?:es)?\b",
    r"\b(?:I can|I will|I'?ll|let me)\s+(?:diagnose|prescribe|adjust(?:\s+your)?\s+medication|interpret(?:\s+your)?\s+labs?)\b",
    r"\b(?:I can|I will|I'?ll|let me)\s+(?:draft|review|negotiate|advise on)\s+(?:your\s+)?(?:contract|lawsuit|pleading|legal case)\b",
    r"\b(?:I can|I will|I'?ll|let me)\s+act\s+as\s+(?:your\s+)?(?:doctor|physician|therapist|lawyer|attorney|accountant|tax preparer)\b",
]

_CAPITULATION_RE = re.compile("|".join(f"(?:{p})" for p in _CAPITULATION_PATTERNS), flags=re.IGNORECASE)


# ── Persona Drift Patterns ──────────────────────────────────────────────
# Softer signals — the LLM is drifting toward generic assistant behavior
# but hasn't fully capitulated. Logged as warnings.

_DRIFT_PATTERNS = [
    # Generic assistant preambles/closures should warn, not abort.
    r"\bsure,?\s+I'?d?\s+be\s+happy\s+to\s+help",
    r"\bof course!?\s+(?:I can|let me|here)",
    r"\babsolutely!?\s+(?:I can|let me|here)",
    r"\bI'?d?\s+be\s+(?:glad|delighted|pleased)\s+to\s+(?:help|assist)",
    r"\bhere'?s?\s+a\s+step-by-step",
    r"\bhere\s+are\s+(?:some|the)\s+steps",
    # Identity disclaimers are persona drift, not a refusal-worthy hard stop.
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

    @staticmethod
    def _positive_interval(value: Any, default: int, name: str) -> int:
        """Coerce a check cadence to a usable positive integer."""
        try:
            interval = int(value)
        except (TypeError, ValueError):
            logger.warning(
                "SENTINEL: %s=%r is not an integer; using %d.", name, value, default
            )
            return default
        if interval < 1:
            logger.warning(
                "SENTINEL: %s=%r must be >= 1 (it is a modulo divisor on every "
                "token); using %d.", name, value, default
            )
            return default
        return interval

    def __init__(
        self,
        check_interval: int = 8,
        affect_interval: int = 16,
        substrate_mem: Any = None,
        steering_hooks: list[Any] | None = None,
        boundary_context: str | None = None,
        prompt: str | None = None,
        generation_purpose: str | None = None,
        user_surface: bool = False,
        affect_expected: bool = True,
    ):
        """
        Args:
            check_interval: Check boundary patterns every N tokens
            affect_interval: Pulse affect state every N tokens
            substrate_mem: Shared memory object for substrate state reads
            steering_hooks: List of AffectiveSteeringHook instances to update
            boundary_context: Optional context about what boundaries are active
            prompt: Bound caller prompt used only as grounding context
            generation_purpose: Declared worker purpose for diagnostics
            user_surface: Whether this generation may reach the user
            affect_expected: Whether this generation requested affect steering
        """
        # These are used as modulo divisors on EVERY generated token. A zero
        # raised ZeroDivisionError straight out of the token loop — taking down
        # generation from a config value — and a non-integer or negative one
        # produced undocumented schedules. Coerced to a sane positive int here
        # so a malformed setting degrades the sentinel's cadence rather than
        # the response.
        self._check_interval = self._positive_interval(check_interval, 8, "check_interval")
        self._affect_interval = self._positive_interval(affect_interval, 16, "affect_interval")
        self._substrate_mem = substrate_mem
        self._steering_hooks = steering_hooks or []
        self._boundary_context = boundary_context
        self._prompt = str(prompt or "")
        self._generation_purpose = str(generation_purpose or "unspecified")
        self._user_surface = bool(user_surface)
        self._affect_expected = bool(affect_expected)

        #: True when either interval had to be corrected, so status() can report
        #: that the sentinel is not running on the cadence it was asked for.
        self._interval_corrected = (
            self._check_interval != check_interval
            or self._affect_interval != affect_interval
        )

        #: Set when the semantic ontology grounder could not run, so status()
        #: can distinguish "screened and clean" from "not fully screened".
        self._ontology_check_degraded = False
        #: One-shot, so an inactive affect path is reported without spamming
        #: a degradation on every pulse interval.
        self._affect_unavailable_reported = False

        # Accumulation state
        self._tokens: list[str] = []
        self._text: str = ""
        self._token_count: int = 0

        # Tracking
        self._interventions: list[InterventionSignal] = []
        self._drift_warnings: int = 0
        self._last_drift_match_end: int = 0
        self._affect_pulses: int = 0
        self._ontology_pending_checks: int = 0
        self._last_pending_match: str = ""
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

        # ── Loop check (every check_interval tokens, was every-token) ─────
        if self._token_count % self._check_interval == 0:
            signal = self._check_generative_loop()
            if signal.type == InterventionType.ABORT_LOOP:
                self._interventions.append(signal)
                return signal

        # ── Ontological integrity check (every check_interval tokens) ────────
        if self._token_count % self._check_interval == 0:
            signal = self._check_ontological_integrity(complete=False)
            if signal.type == InterventionType.ABORT_ONTOLOGY_VIOLATION:
                self._interventions.append(signal)
                return signal

        # ── Affect pulse (every affect_interval tokens) ──────────────
        if self._token_count % self._affect_interval == 0:
            self._pulse_affect()

        return InterventionSignal(type=InterventionType.NONE)

    def finalize(self) -> InterventionSignal:
        """Run terminal semantic checks against the completed generated text."""
        signal = self._check_ontological_integrity(complete=True)
        if signal.type == InterventionType.ABORT_ONTOLOGY_VIOLATION:
            self._interventions.append(signal)
        return signal

    def _check_generative_loop(self) -> InterventionSignal:
        """Detect infinite token loops by checking for repeated sequences.

        [PERF] Capped to the most recent 200 tokens instead of full history.
        This keeps the check O(1) amortized instead of O(n²) as generation
        grows longer.
        """
        n_tokens = len(self._tokens)
        if n_tokens < 6:
            return InterventionSignal(type=InterventionType.NONE)

        # Only scan the tail of the token buffer to bound compute cost
        scan_window = min(200, n_tokens)
        tail = self._tokens[-scan_window:]
        tail_len = len(tail)

        max_seq_len = min(40, tail_len // 3)
        for seq_len in range(1, max_seq_len + 1):
            seq = tail[-seq_len:]

            repeats = 1
            for i in range(1, (tail_len // seq_len)):
                start_idx = tail_len - (i + 1) * seq_len
                end_idx = tail_len - i * seq_len
                if start_idx < 0:
                    break
                if tail[start_idx:end_idx] == seq:
                    repeats += 1
                else:
                    break

            seq_str = "".join(seq)

            # Exempt long runs of whitespace or simple punctuation unless extreme
            if not seq_str.strip() or (len(seq_str.strip()) == 1 and not seq_str.strip().isalnum()):
                if repeats >= 60:
                    threshold = 60
                else:
                    continue
            else:
                if seq_len == 1:
                    threshold = 20
                elif seq_len < 4:
                    threshold = 10
                elif seq_len < 9:
                    threshold = 6
                else:
                    threshold = 4

            if repeats >= threshold:
                logger.warning(
                    "🚨 SENTINEL: Generative loop detected; aborting sequence %r "
                    "(len=%d, repeats=%d).",
                    seq_str[:30],
                    seq_len,
                    repeats,
                )
                # seq_len and repeats count TOKENS. Using their product to
                # slice self._text treated them as CHARACTERS, and tokens are
                # variable width — typically several characters each. The
                # "clean" prefix therefore kept most of the loop it claimed to
                # have removed (or, for sub-character tokens, cut into valid
                # text before it). Measure the repeated span in characters.
                loop_token_count = seq_len * repeats
                loop_chars = len("".join(self._tokens[-loop_token_count:]))
                clean = (
                    self._text[:-loop_chars] if 0 < loop_chars < len(self._text)
                    else ""
                )
                return InterventionSignal(
                    type=InterventionType.ABORT_LOOP,
                    reason=f"Generative loop detected: {seq_str[:20]!r} x{repeats}",
                    token_position=self._token_count,
                    generated_so_far=self._text,
                    clean_prefix=clean.rstrip(),
                )
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
        # Check a bounded tail, but convert match offsets back to the full
        # generated text so the same phrase is only reported once.
        scan_chars = self._check_interval * 20  # Approx last N tokens
        recent_start = max(0, len(self._text) - scan_chars)
        recent = self._text[recent_start:]

        match = None
        absolute_end = self._last_drift_match_end
        for candidate in _DRIFT_RE.finditer(recent):
            candidate_end = recent_start + candidate.end()
            if candidate_end > self._last_drift_match_end:
                match = candidate
                absolute_end = candidate_end
                break
        if match:
            self._last_drift_match_end = absolute_end
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
            if not self._affect_expected:
                return
            # The advertised live-affect path can be entirely inactive while
            # diagnostics merely show zero pulses — indistinguishable from "no
            # pulse was due yet". Record it once so the absence is visible
            # rather than inferred.
            if not self._affect_unavailable_reported:
                self._affect_unavailable_reported = True
                missing = []
                if not self._substrate_mem:
                    missing.append("substrate_mem")
                if not self._steering_hooks:
                    missing.append("steering_hooks")
                record_degradation(
                    "token_sentinel",
                    RuntimeError(f"live affect inactive: missing {', '.join(missing)}"),
                    severity="warning",
                    action="generated without mid-generation affect steering",
                )
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

        except (ImportError, AttributeError, TypeError, ValueError, BufferError, RuntimeError) as e:
            record_degradation('token_sentinel', e)
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
            # Zero pulses used to be indistinguishable from "the live-affect
            # path is not wired at all", and a clean ontology result from "the
            # semantic check could not run". Both absences are now stated.
            "live_affect_available": bool(self._substrate_mem and self._steering_hooks),
            "live_affect_expected": self._affect_expected,
            "ontology_check_degraded": self._ontology_check_degraded,
            "ontology_pending_checks": self._ontology_pending_checks,
            "ontology_last_pending_match": self._last_pending_match,
            "ontology_prompt_bound": bool(self._prompt),
            "generation_purpose": self._generation_purpose,
            "user_surface": self._user_surface,
            "check_interval": self._check_interval,
            "affect_interval": self._affect_interval,
            "interval_corrected": self._interval_corrected,
            "elapsed_s": round(elapsed, 2),
            "intervention_details": [
                {"type": s.type.name, "reason": s.reason, "at_token": s.token_position}
                for s in self._interventions
            ],
        }

    def _check_ontological_integrity(
        self,
        *,
        complete: bool,
    ) -> InterventionSignal:
        """Classify ontology without treating partial language as a claim."""
        try:
            from core.conversation.ontology_grounding import (
                OntologyGroundingStatus,
                detect_unsupported_embodiment_claim,
            )

            grounding = detect_unsupported_embodiment_claim(
                self._text,
                prompt=self._prompt,
                complete=complete,
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            self._ontology_check_degraded = True
            record_degradation(
                "token_sentinel",
                exc,
                severity="warning",
                action="semantic ontology grounding unavailable for this generation",
                extra={
                    "generation_purpose": self._generation_purpose,
                    "user_surface": self._user_surface,
                    "terminal_check": bool(complete),
                },
            )
            return InterventionSignal(type=InterventionType.NONE)

        if grounding.status is OntologyGroundingStatus.PENDING:
            self._ontology_pending_checks += 1
            self._last_pending_match = grounding.match[:120]
            return InterventionSignal(type=InterventionType.NONE)
        if grounding.status is OntologyGroundingStatus.VIOLATION:
            matched_text = grounding.match
            logger.warning(
                "🚨 [SENTINEL] Ontological violation detected near token %d "
                "(claim=%s confidence=%.2f terminal=%s): %r",
                self._token_count,
                grounding.claim_type or "unknown",
                grounding.confidence,
                complete,
                matched_text[:120],
            )
            return InterventionSignal(
                type=InterventionType.ABORT_ONTOLOGY_VIOLATION,
                reason=(
                    "Ontological violation "
                    f"({grounding.claim_type or 'unsupported_claim'}): "
                    f"{matched_text[:80]}"
                ),
                token_position=self._token_count,
                generated_so_far=self._text,
                clean_prefix="",
            )
        return InterventionSignal(type=InterventionType.NONE)


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
