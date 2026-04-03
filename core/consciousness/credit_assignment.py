"""core/consciousness/credit_assignment.py
Credit Assignment System — Distributes reward signals across cognitive pathways.

Deepened Implementation:
  - Full domain performance tracking with trend analysis
  - Self-modifying domain weights based on sustained performance
  - Module influence scoring for GWT priority modulation
  - Context block for LLM prompt injection
  - Integration with Hedonic Gradient via weight adjustment signals
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("Consciousness.CreditAssignment")


@dataclass
class CreditEvent:
    action_id: str
    outcome_score: float
    domain: str  # e.g., "chat", "identity", "navigation", "autonomy", "dream"
    timestamp: float = field(default_factory=time.time)


class CreditAssignmentSystem:
    """
    Distributes "reward" signals across competing cognitive pathways.

    Closed-loop integration:
      - Receives credit events from InferenceGate, CounterfactualEngine, DreamingProcess
      - Computes per-domain performance with exponential moving average
      - Self-adjusts domain weights: well-performing domains get amplified
      - Feeds module influence scores to Hedonic Gradient for resource allocation
      - Provides context block showing cognitive performance landscape to LLM
    """

    _MAX_HISTORY = 1000
    _WEIGHT_ADAPTATION_RATE = 0.005  # How fast domain weights self-modify
    _TREND_WINDOW = 20               # Events to compute trend over

    def __init__(self):
        self.history: List[CreditEvent] = []
        self.domain_weights: Dict[str, float] = {
            "chat": 1.0,
            "identity": 1.2,   # Priority for sovereign self
            "logic": 1.0,
            "metabolic": 0.8,
            "autonomy": 1.0,   # Autonomous actions
            "dream": 0.7,      # Dream consolidation
            "social": 0.9,     # Theory of mind / user modeling
            "creative": 1.0,   # Creative generation
        }

        # ── Exponential Moving Averages per domain ────────────────────────
        self._domain_ema: Dict[str, float] = {d: 0.5 for d in self.domain_weights}
        self._ema_alpha = 0.1  # Smoothing factor

        # ── Per-domain event history for trend analysis ───────────────────
        self._domain_history: Dict[str, Deque[Tuple[float, float]]] = {
            d: deque(maxlen=self._TREND_WINDOW) for d in self.domain_weights
        }

        # ── Cumulative stats ──────────────────────────────────────────────
        self._total_events = 0
        self._total_positive = 0
        self._total_negative = 0
        self._best_domain: Optional[str] = None
        self._worst_domain: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────────
    # Core API (existing — enhanced)
    # ──────────────────────────────────────────────────────────────────────

    def assign_credit(self, action_id: str, outcome: float, domain: str):
        """
        Record an outcome and map it back to a cognitive domain.
        Now also updates EMA, self-modifies weights, and tracks trends.
        """
        # Ensure domain exists in tracking
        if domain not in self.domain_weights:
            self.domain_weights[domain] = 1.0
            self._domain_ema[domain] = 0.5
            self._domain_history[domain] = deque(maxlen=self._TREND_WINDOW)

        # Apply domain weight gating
        effective_outcome = outcome * self.domain_weights.get(domain, 1.0)

        # Cross-context penalty: chat actions affecting identity are down-weighted
        if domain == "chat" and "identity" in action_id.lower():
            effective_outcome *= 0.5
            logger.debug("Cross-context penalty applied to credit assignment.")

        event = CreditEvent(
            action_id=action_id,
            outcome_score=effective_outcome,
            domain=domain
        )
        self.history.append(event)
        if len(self.history) > self._MAX_HISTORY:
            self.history.pop(0)

        # ── Update EMA ────────────────────────────────────────────────────
        old_ema = self._domain_ema.get(domain, 0.5)
        self._domain_ema[domain] = (
            self._ema_alpha * effective_outcome
            + (1 - self._ema_alpha) * old_ema
        )

        # ── Track history for trends ──────────────────────────────────────
        self._domain_history[domain].append((time.time(), effective_outcome))

        # ── Cumulative stats ──────────────────────────────────────────────
        self._total_events += 1
        if effective_outcome > 0:
            self._total_positive += 1
        elif effective_outcome < 0:
            self._total_negative += 1

        # ── Self-modify domain weights ────────────────────────────────────
        self._adapt_weights()

        # ── Update best/worst tracking ────────────────────────────────────
        self._update_extremes()

        logger.debug(
            "Credit assigned: %s -> %.2f (%s) | EMA: %.3f | weight: %.3f",
            action_id, effective_outcome, domain,
            self._domain_ema[domain], self.domain_weights[domain]
        )

    # ──────────────────────────────────────────────────────────────────────
    # Domain Performance (existing — enhanced)
    # ──────────────────────────────────────────────────────────────────────

    def get_domain_performance(self, domain: str) -> float:
        """Calculate recent success rate for a specific cognitive domain.
        Now uses EMA for smoother, more responsive tracking.
        """
        return self._domain_ema.get(domain, 0.5)

    def get_all_domain_performance(self) -> Dict[str, float]:
        """Returns performance scores for all tracked domains."""
        return {d: round(v, 3) for d, v in self._domain_ema.items()}

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Trend Analysis
    # ──────────────────────────────────────────────────────────────────────

    def get_performance_trend(self, domain: str) -> str:
        """Returns 'rising', 'falling', or 'stable' for a domain."""
        history = self._domain_history.get(domain)
        if not history or len(history) < 5:
            return "stable"
        values = [v for _, v in history]
        first_half = sum(values[:len(values)//2]) / max(1, len(values)//2)
        second_half = sum(values[len(values)//2:]) / max(1, len(values) - len(values)//2)
        delta = second_half - first_half
        if delta > 0.05:
            return "rising"
        if delta < -0.05:
            return "falling"
        return "stable"

    def get_all_trends(self) -> Dict[str, str]:
        """Returns trend for every tracked domain."""
        return {d: self.get_performance_trend(d) for d in self.domain_weights}

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Module Influence Scores (for Hedonic Gradient)
    # ──────────────────────────────────────────────────────────────────────

    def get_influence_scores(self) -> Dict[str, float]:
        """Returns normalized influence scores per domain (0-1).
        Higher performance = higher influence allocation.
        Used by Hedonic Gradient to weight resource distribution.
        """
        perfs = self.get_all_domain_performance()
        if not perfs:
            return {}
        total = sum(perfs.values())
        if total == 0:
            return {d: 1.0 / len(perfs) for d in perfs}
        return {d: round(v / total * len(perfs), 3) for d, v in perfs.items()}

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Self-Modifying Weights
    # ──────────────────────────────────────────────────────────────────────

    def _adapt_weights(self):
        """Slowly adjust domain weights based on sustained performance.
        Well-performing domains get slightly amplified; poor ones get
        slightly dampened. This creates a feedback loop where success
        breeds more influence.
        """
        for domain, ema in self._domain_ema.items():
            if domain not in self.domain_weights:
                continue
            current_weight = self.domain_weights[domain]
            # Target: weight proportional to performance
            # If EMA > 0.5 (above average), nudge weight up
            # If EMA < 0.5, nudge weight down
            adjustment = (ema - 0.5) * self._WEIGHT_ADAPTATION_RATE
            new_weight = max(0.3, min(2.0, current_weight + adjustment))
            self.domain_weights[domain] = new_weight

    def _update_extremes(self):
        """Track best and worst performing domains."""
        if not self._domain_ema:
            return
        self._best_domain = max(self._domain_ema, key=self._domain_ema.get)
        self._worst_domain = min(self._domain_ema, key=self._domain_ema.get)

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Context Block for Inference Gate
    # ──────────────────────────────────────────────────────────────────────

    def get_context_block(self) -> str:
        """Returns a concise context block for LLM prompt injection."""
        if not self.history:
            return ""
        best = self._best_domain or "none"
        worst = self._worst_domain or "none"
        best_perf = self._domain_ema.get(best, 0.0)
        worst_perf = self._domain_ema.get(worst, 0.0)
        best_trend = self.get_performance_trend(best)
        positive_rate = (self._total_positive / max(1, self._total_events))
        return (
            f"## COGNITIVE CREDIT\n"
            f"Best: {best}={best_perf:.2f}({best_trend}) | "
            f"Weakest: {worst}={worst_perf:.2f} | "
            f"Success: {positive_rate:.0%} over {self._total_events} events"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Status / Snapshot
    # ──────────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Full status for diagnostics."""
        return {
            "total_events": self._total_events,
            "total_positive": self._total_positive,
            "total_negative": self._total_negative,
            "domain_performance": self.get_all_domain_performance(),
            "domain_weights": {d: round(w, 3) for d, w in self.domain_weights.items()},
            "domain_trends": self.get_all_trends(),
            "best_domain": self._best_domain,
            "worst_domain": self._worst_domain,
            "influence_scores": self.get_influence_scores(),
        }
