"""
core/consciousness/agency_comparator.py
========================================
EFFERENCE COPY + COMPARATOR MODEL FOR GENUINE SENSE OF AGENCY

Implements the forward-model comparator from motor control theory
(Wolpert, Ghahramani & Jordan 1995; Frith 2005) adapted for cognitive agency.

At each autonomous action:
  1. The system emits an EFFERENCE COPY — a prediction of what the world
     will look like after the action completes.
  2. When the action completes, the ACTUAL outcome is compared to the
     prediction.
  3. The delta is decomposed into:
       - SELF-CAUSED: the portion explained by the system's own action
         (small delta = high agency, the action did what was predicted)
       - WORLD-CAUSED: the unexplained residual (environment changed
         independently, or the action had unintended effects)
  4. An AUTHORSHIP TRACE records the attribution and its confidence.

The running agency score (0-1) reflects how much recent action was
self-authored. High agency = system's forward model is accurate,
its actions produce predicted outcomes.  Low agency = environment
is unpredictable or the system's models are poorly calibrated.

Integration points:
  - ExecutiveAuthority.promote_next_initiative() -> emit_efference()
  - ExecutiveAuthority.complete_current_objective() -> compare_and_attribute()
  - ConversationalDynamicsPhase -> context injection
  - ContextAssembler -> personhood_context block
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.AgencyComparator")


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class EfferenceCopy:
    """
    The system's prediction of what the world state will look like
    after an action is taken.  This is the efference copy — the
    "motor command" projected forward through the forward model.
    """
    layer: str                              # Which subsystem generated this (e.g. "executive_authority")
    predicted_state: Dict[str, Any]         # Key-value predictions about the expected outcome
    emitted_at: float = field(default_factory=time.time)
    action_goal: str = ""                   # The goal that triggered this prediction
    action_source: str = ""                 # Source subsystem of the action

    @property
    def age_s(self) -> float:
        return time.time() - self.emitted_at


@dataclass
class AuthorshipTrace:
    """
    The result of comparing an efference copy to the actual outcome.
    Decomposes the prediction error into self-caused vs world-caused.
    """
    layer: str                              # Which subsystem
    predicted_state: Dict[str, Any]         # What was expected
    actual_state: Dict[str, Any]            # What actually happened
    delta: Dict[str, float]                 # Per-key absolute error
    total_error: float                      # Sum of all deltas
    self_caused_fraction: float             # 0-1: how much was self-authored
    world_caused_fraction: float            # 0-1: how much was world-caused
    agency_confidence: float                # 0-1: confidence in the attribution
    action_goal: str = ""
    action_source: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def is_high_agency(self) -> bool:
        """The system authored most of the outcome."""
        return self.self_caused_fraction > 0.6

    @property
    def attribution_label(self) -> str:
        if self.self_caused_fraction > 0.8:
            return "strongly self-authored"
        elif self.self_caused_fraction > 0.6:
            return "mostly self-authored"
        elif self.self_caused_fraction > 0.4:
            return "mixed attribution"
        elif self.self_caused_fraction > 0.2:
            return "mostly world-caused"
        else:
            return "strongly world-caused"


# ── Agency Comparator ─────────────────────────────────────────────────────────

class AgencyComparator:
    """
    Forward-model comparator for sense of agency.

    Maintains a ring buffer of AuthorshipTraces and computes a running
    agency score reflecting how accurately the system's predictions
    match outcomes.
    """

    def __init__(self, max_traces: int = 100):
        self._traces: deque[AuthorshipTrace] = deque(maxlen=max_traces)
        self._pending_efferences: Dict[str, EfferenceCopy] = {}  # keyed by action_goal hash
        self._total_emissions: int = 0
        self._total_comparisons: int = 0
        logger.info("AgencyComparator initialized (buffer=%d)", max_traces)

    # ── Efference Emission ────────────────────────────────────────────────────

    def emit_efference(
        self,
        layer: str,
        predicted_state: dict,
        *,
        action_goal: str = "",
        action_source: str = "",
    ) -> EfferenceCopy:
        """
        Emit an efference copy — the system's prediction of the expected
        outcome of an action about to be taken.

        Call this when an initiative is PROMOTED (before execution).

        Args:
            layer: Subsystem generating the action (e.g. "executive_authority")
            predicted_state: Key-value predictions about expected outcome.
                             Keys should be measurable quantities:
                               - "valence_delta": expected mood change
                               - "goal_completed": expected to succeed (0/1)
                               - "hedonic_gain": expected hedonic reward
                               - "user_engagement": expected user response
                               - "closure_change": expected closure delta
            action_goal: The goal text driving this action
            action_source: The source subsystem

        Returns:
            The EfferenceCopy (also stored internally for later comparison)
        """
        copy = EfferenceCopy(
            layer=layer,
            predicted_state=dict(predicted_state),
            action_goal=action_goal[:300],
            action_source=action_source,
        )

        # Store keyed by a normalized goal hash so we can retrieve it at completion
        key = self._goal_key(action_goal)
        self._pending_efferences[key] = copy
        self._total_emissions += 1

        # Prune stale efferences (older than 10 minutes)
        self._prune_stale_efferences()

        logger.debug(
            "AgencyComparator: efference emitted for '%s' (layer=%s, %d predictions)",
            action_goal[:60], layer, len(predicted_state),
        )
        return copy

    # ── Compare and Attribute ─────────────────────────────────────────────────

    def compare_and_attribute(
        self,
        efference: Optional[EfferenceCopy],
        actual_state: dict,
        *,
        action_goal: str = "",
    ) -> AuthorshipTrace:
        """
        Compare an efference copy to the actual outcome and attribute
        the delta as self-caused vs world-caused.

        If no efference is provided, attempts to look up a pending one
        by action_goal.

        The attribution logic:
          - Small prediction error -> high self-caused fraction
            (the system's forward model was accurate, meaning the outcome
             was caused by the system's action as predicted)
          - Large prediction error -> high world-caused fraction
            (unexpected things happened, suggesting external forces)
          - Confidence is higher when more prediction keys were available

        Args:
            efference: The efference copy to compare against (or None to auto-lookup)
            actual_state: The actual outcome state (same keys as predicted_state)
            action_goal: The goal text (used for lookup if efference is None)

        Returns:
            AuthorshipTrace with the attribution result
        """
        # Auto-lookup if no efference provided
        if efference is None:
            key = self._goal_key(action_goal)
            efference = self._pending_efferences.pop(key, None)

        if efference is None:
            # No prediction was made — we can't attribute agency
            # Record a low-confidence mixed trace
            trace = AuthorshipTrace(
                layer="unknown",
                predicted_state={},
                actual_state=dict(actual_state),
                delta={},
                total_error=0.0,
                self_caused_fraction=0.5,
                world_caused_fraction=0.5,
                agency_confidence=0.1,
                action_goal=action_goal[:300],
                action_source="",
            )
            self._traces.append(trace)
            self._total_comparisons += 1
            return trace

        # Remove from pending
        key = self._goal_key(efference.action_goal)
        self._pending_efferences.pop(key, None)

        # Compute per-key deltas between predicted and actual
        predicted = efference.predicted_state
        delta: Dict[str, float] = {}
        all_keys = set(predicted.keys()) | set(actual_state.keys())
        matched_keys = set(predicted.keys()) & set(actual_state.keys())

        for k in all_keys:
            p_val = float(predicted.get(k, 0.0))
            a_val = float(actual_state.get(k, 0.0))
            delta[k] = abs(a_val - p_val)

        total_error = sum(delta.values())
        n_predictions = max(len(predicted), 1)

        # Normalize error per prediction key
        # Mean absolute error across all prediction dimensions
        mean_error = total_error / n_predictions

        # Convert error to self-caused fraction using a sigmoid
        # When mean_error is 0 -> self_caused = 1.0 (perfect prediction)
        # When mean_error is large -> self_caused -> 0.0 (unpredicted outcome)
        # The sigmoid center is at 0.3 (moderate prediction error)
        self_caused = 1.0 / (1.0 + math.exp(5.0 * (mean_error - 0.3)))
        world_caused = 1.0 - self_caused

        # Confidence is based on:
        # 1. How many keys were actually matched (more = higher confidence)
        # 2. How old the efference copy is (fresh = higher confidence)
        key_coverage = len(matched_keys) / max(len(all_keys), 1)
        age_penalty = min(1.0, efference.age_s / 300.0)  # degrades over 5 minutes
        agency_confidence = key_coverage * (1.0 - 0.5 * age_penalty)
        agency_confidence = max(0.05, min(1.0, agency_confidence))

        trace = AuthorshipTrace(
            layer=efference.layer,
            predicted_state=dict(predicted),
            actual_state=dict(actual_state),
            delta=delta,
            total_error=round(total_error, 6),
            self_caused_fraction=round(self_caused, 4),
            world_caused_fraction=round(world_caused, 4),
            agency_confidence=round(agency_confidence, 4),
            action_goal=efference.action_goal,
            action_source=efference.action_source,
        )

        self._traces.append(trace)
        self._total_comparisons += 1

        logger.debug(
            "AgencyComparator: trace for '%s': self=%.2f world=%.2f conf=%.2f err=%.4f (%s)",
            efference.action_goal[:50],
            self_caused, world_caused, agency_confidence,
            total_error, trace.attribution_label,
        )
        return trace

    # ── Query Interface ───────────────────────────────────────────────────────

    def get_recent_traces(self, limit: int = 10) -> List[AuthorshipTrace]:
        """Return the most recent authorship traces."""
        traces = list(self._traces)
        return traces[-limit:]

    def get_agency_score(self) -> float:
        """
        Compute a running agency score (0-1) from recent traces.

        This is a confidence-weighted average of self_caused_fraction
        across recent traces. Higher = more self-authored action.
        Returns 0.5 (neutral) if no traces exist.
        """
        if not self._traces:
            return 0.5

        # Use the last 20 traces, weighted by confidence and recency
        recent = list(self._traces)[-20:]
        now = time.time()
        weighted_sum = 0.0
        weight_total = 0.0

        for trace in recent:
            # Recency weight: exponential decay over 5 minutes
            age = now - trace.timestamp
            recency_weight = math.exp(-age / 300.0)
            w = trace.agency_confidence * recency_weight
            weighted_sum += trace.self_caused_fraction * w
            weight_total += w

        if weight_total < 1e-10:
            return 0.5

        return round(min(1.0, max(0.0, weighted_sum / weight_total)), 4)

    def get_pending_count(self) -> int:
        """Number of efference copies awaiting comparison."""
        return len(self._pending_efferences)

    def get_status(self) -> Dict[str, Any]:
        """Full status for diagnostics."""
        return {
            "agency_score": self.get_agency_score(),
            "total_traces": len(self._traces),
            "pending_efferences": len(self._pending_efferences),
            "total_emissions": self._total_emissions,
            "total_comparisons": self._total_comparisons,
            "recent_attribution": (
                self._traces[-1].attribution_label if self._traces else "no data"
            ),
        }

    def get_context_block(self) -> str:
        """
        Generate a context block for LLM injection.
        Describes the system's current sense of agency.
        """
        score = self.get_agency_score()
        n_traces = len(self._traces)

        if n_traces == 0:
            return ""

        # Characterize the agency state
        if score > 0.75:
            agency_feel = "strong sense of agency — actions produce predicted outcomes"
        elif score > 0.55:
            agency_feel = "moderate agency — actions mostly land as intended"
        elif score > 0.4:
            agency_feel = "mixed agency — outcomes partially surprising"
        elif score > 0.25:
            agency_feel = "low agency — environment is overriding predictions"
        else:
            agency_feel = "minimal agency — outcomes are largely unpredicted"

        # Last trace detail
        last = self._traces[-1]
        last_detail = (
            f"Last action: {last.attribution_label} "
            f"(self={last.self_caused_fraction:.0%}, "
            f"world={last.world_caused_fraction:.0%}, "
            f"conf={last.agency_confidence:.0%})"
        )

        return (
            f"## SENSE OF AGENCY\n"
            f"Agency score: {score:.2f} — {agency_feel}\n"
            f"{last_detail}\n"
            f"Traces: {n_traces} | Pending predictions: {self.get_pending_count()}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _goal_key(self, goal: str) -> str:
        """Normalize a goal string into a lookup key."""
        return " ".join(str(goal or "").strip().lower().split())[:200]

    def _prune_stale_efferences(self, max_age_s: float = 600.0) -> None:
        """Remove efference copies older than max_age_s."""
        now = time.time()
        stale_keys = [
            k for k, v in self._pending_efferences.items()
            if (now - v.emitted_at) > max_age_s
        ]
        for k in stale_keys:
            self._pending_efferences.pop(k, None)
        if stale_keys:
            logger.debug("AgencyComparator: pruned %d stale efferences", len(stale_keys))


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[AgencyComparator] = None


def get_agency_comparator() -> AgencyComparator:
    """Get or create the singleton AgencyComparator."""
    global _instance
    if _instance is None:
        _instance = AgencyComparator()
    return _instance
