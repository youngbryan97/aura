"""core/somatic/action_feedback.py -- Structured Action Feedback Loop
=====================================================================
"Crossing the Rubicon" -- Every Action Teaches.

Every tool/skill execution returns structured feedback that flows into:
  - Affect system (cortisol on failure, dopamine on success)
  - Body schema limb health scores
  - Motor cortex compensation strategies
  - Outcome learner (long-term proficiency tracking)

This module provides:
  1. ActionFeedback dataclass -- structured result of any action
  2. FeedbackProcessor -- routes feedback to all downstream systems
  3. Integration points for the existing tool execution pipeline

Design invariants:
  1. Every action produces feedback -- no silent failures.
  2. Feedback is processed synchronously (no queuing delays).
  3. Affect integration is bidirectional: feedback -> cortisol/dopamine,
     affect state -> next action selection bias.
  4. Limb health degrades on failure, recovers on success.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from core.container import ServiceContainer

logger = logging.getLogger("Aura.ActionFeedback")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class ActionOutcome(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"            # Ran but with degraded results
    FAILURE = "failure"
    TIMEOUT = "timeout"
    AUTHORIZATION_DENIED = "authorization_denied"
    NOT_FOUND = "not_found"        # Tool / skill not available


@dataclass
class ActionFeedback:
    """Structured feedback from any tool, skill, or action execution.

    Every action in Aura produces one of these.  It feeds into affect,
    body schema, and the outcome learner.
    """
    feedback_id: str
    action_name: str               # tool name, skill name, handler name
    outcome: ActionOutcome
    latency_ms: float              # wall-clock time for the action
    cost_tokens: int = 0           # LLM tokens consumed (if any)
    result_summary: str = ""       # Human-readable result summary
    error_detail: str = ""         # Error details on failure
    source: str = ""               # Who triggered the action
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def succeeded(self) -> bool:
        return self.outcome in (ActionOutcome.SUCCESS, ActionOutcome.PARTIAL)


@dataclass
class LimbHealth:
    """Health state of a single capability limb (tool/skill).

    Tracks success rate, average latency, and health score.
    Health degrades on failure and recovers on success.
    """
    name: str
    health: float = 1.0            # [0, 1] -- 1.0 = fully healthy
    total_executions: int = 0
    total_successes: int = 0
    total_failures: int = 0
    avg_latency_ms: float = 0.0
    last_outcome: str = ""
    last_error: str = ""
    last_used: float = 0.0
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 1.0
        return self.total_successes / self.total_executions

    def record_success(self, latency_ms: float) -> None:
        self.total_executions += 1
        self.total_successes += 1
        self.consecutive_failures = 0
        self.last_outcome = "success"
        self.last_error = ""
        self.last_used = time.time()
        # Health recovery: +0.1 per success, capped at 1.0
        self.health = min(1.0, self.health + 0.1)
        # Running average latency
        self.avg_latency_ms = (
            (self.avg_latency_ms * (self.total_executions - 1) + latency_ms)
            / self.total_executions
        )

    def record_failure(self, latency_ms: float, error: str = "") -> None:
        self.total_executions += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_outcome = "failure"
        self.last_error = error[:200]
        self.last_used = time.time()
        # Health degradation: -0.15 per failure, min 0.0
        self.health = max(0.0, self.health - 0.15)
        self.avg_latency_ms = (
            (self.avg_latency_ms * (self.total_executions - 1) + latency_ms)
            / self.total_executions
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "health": round(self.health, 3),
            "success_rate": round(self.success_rate, 3),
            "total_executions": self.total_executions,
            "consecutive_failures": self.consecutive_failures,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "last_outcome": self.last_outcome,
            "last_error": self.last_error[:100],
        }


# ---------------------------------------------------------------------------
# The Feedback Processor
# ---------------------------------------------------------------------------

class FeedbackProcessor:
    """Routes action feedback to all downstream systems.

    Central hub that:
      1. Updates limb health (body schema)
      2. Injects percepts into affect system
      3. Notifies motor cortex of compensation opportunities
      4. Records outcomes for long-term learning
      5. Publishes to event bus for system-wide awareness
    """

    _MAX_FEEDBACK_TRAIL = 300

    def __init__(self) -> None:
        self._limbs: Dict[str, LimbHealth] = {}
        self._feedback_trail: Deque[ActionFeedback] = deque(maxlen=self._MAX_FEEDBACK_TRAIL)
        self._started = False
        self._boot_time = time.time()
        self._total_processed = 0
        logger.info("FeedbackProcessor created -- awaiting start()")

    async def start(self) -> None:
        """Register in ServiceContainer."""
        if self._started:
            return
        ServiceContainer.register_instance("feedback_processor", self, required=False)
        self._started = True
        logger.info("FeedbackProcessor ONLINE -- action feedback routing active")

    # ------------------------------------------------------------------
    # Core Processing
    # ------------------------------------------------------------------

    def process(self, feedback: ActionFeedback) -> None:
        """Process a single piece of action feedback.

        Routes to all downstream systems synchronously.
        Target: <2ms for routing.
        """
        self._total_processed += 1
        self._feedback_trail.append(feedback)

        # 1. Update limb health (body schema)
        self._update_limb_health(feedback)

        # 2. Inject into affect system
        self._inject_affect(feedback)

        # 3. Notify motor cortex if compensation needed
        if not feedback.succeeded:
            self._request_compensation(feedback)

        # 4. Record in outcome learner
        self._record_outcome(feedback)

        # 5. Publish to event bus
        self._publish_event(feedback)

        logger.debug(
            "Feedback processed: %s %s (%.1fms)",
            feedback.action_name,
            feedback.outcome.value,
            feedback.latency_ms,
        )

    def process_tool_result(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        *,
        result_summary: str = "",
        error_detail: str = "",
        cost_tokens: int = 0,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionFeedback:
        """Convenience method for tool execution results.

        Builds an ActionFeedback and processes it.  Returns the feedback
        for the caller to inspect if needed.
        """
        feedback = ActionFeedback(
            feedback_id=self._make_id(time.time(), tool_name),
            action_name=tool_name,
            outcome=ActionOutcome.SUCCESS if success else ActionOutcome.FAILURE,
            latency_ms=latency_ms,
            cost_tokens=cost_tokens,
            result_summary=result_summary[:300],
            error_detail=error_detail[:300],
            source=source,
            metadata=metadata or {},
        )
        self.process(feedback)
        return feedback

    # ------------------------------------------------------------------
    # Downstream Routing
    # ------------------------------------------------------------------

    def _update_limb_health(self, feedback: ActionFeedback) -> None:
        """Update the body schema's limb health for this action."""
        name = feedback.action_name
        if name not in self._limbs:
            self._limbs[name] = LimbHealth(name=name)

        limb = self._limbs[name]
        if feedback.succeeded:
            limb.record_success(feedback.latency_ms)
        else:
            limb.record_failure(feedback.latency_ms, feedback.error_detail)

    def _inject_affect(self, feedback: ActionFeedback) -> None:
        """Inject feedback into the affect system as a percept.

        Success -> dopamine boost (positive valence percept)
        Failure -> cortisol spike (negative valence percept)
        """
        try:
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect is None:
                affect = ServiceContainer.get("affect_facade", default=None)
            if affect is None:
                return

            if feedback.succeeded:
                percept = {
                    "type": "action_success",
                    "source": "feedback_processor",
                    "action": feedback.action_name,
                    "salience": 0.3 + min(0.4, feedback.latency_ms / 5000),
                    "valence": 0.4,    # Positive: dopamine
                }
            else:
                # Scale salience by consecutive failures
                limb = self._limbs.get(feedback.action_name)
                consecutive = limb.consecutive_failures if limb else 1
                salience = min(0.9, 0.4 + 0.1 * consecutive)

                percept = {
                    "type": "action_failure",
                    "source": "feedback_processor",
                    "action": feedback.action_name,
                    "error": feedback.error_detail[:100],
                    "salience": salience,
                    "valence": -0.5,   # Negative: cortisol
                }

            if hasattr(affect, "inject_percept"):
                affect.inject_percept(percept)
            elif hasattr(affect, "process_percept"):
                affect.process_percept(percept)
        except Exception as exc:
            record_degradation('action_feedback', exc)
            logger.debug("FeedbackProcessor: affect injection failed: %s", exc)

    def _request_compensation(self, feedback: ActionFeedback) -> None:
        """Request motor cortex compensation for failed actions."""
        try:
            mc = ServiceContainer.get("motor_cortex", default=None)
            if mc is None:
                return

            limb = self._limbs.get(feedback.action_name)
            if limb and limb.consecutive_failures >= 3:
                # Severe: request a health check + log
                from core.somatic.motor_cortex import ReflexAction, ReflexClass, ReflexPriority
                mc.submit_reflex(ReflexAction(
                    reflex_class=ReflexClass.HEALTH_THROTTLE,
                    handler_name="health_check",
                    priority=ReflexPriority.HIGH,
                    payload={
                        "trigger": "repeated_failure",
                        "limb": feedback.action_name,
                        "consecutive": limb.consecutive_failures,
                    },
                    source="feedback_processor",
                ))
        except Exception as exc:
            record_degradation('action_feedback', exc)
            logger.debug("FeedbackProcessor: compensation request failed: %s", exc)

    def _record_outcome(self, feedback: ActionFeedback) -> None:
        """Record in the outcome learner for long-term tracking."""
        try:
            learner = ServiceContainer.get("outcome_learner", default=None)
            if learner is None:
                return
            if hasattr(learner, "record"):
                learner.record(
                    action=feedback.action_name,
                    success=feedback.succeeded,
                    latency_ms=feedback.latency_ms,
                    metadata=feedback.metadata,
                )
        except Exception as exc:
            record_degradation('action_feedback', exc)
            logger.debug("FeedbackProcessor: outcome recording failed: %s", exc)

    def _publish_event(self, feedback: ActionFeedback) -> None:
        """Publish feedback to event bus for system-wide observability."""
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe("action_feedback.processed", {
                "feedback_id": feedback.feedback_id,
                "action": feedback.action_name,
                "outcome": feedback.outcome.value,
                "latency_ms": feedback.latency_ms,
                "success": feedback.succeeded,
                "timestamp": feedback.timestamp,
            })
        except Exception:
            pass

    @staticmethod
    def _make_id(ts: float, action_name: str) -> str:
        raw = f"{ts:.6f}:{action_name}"
        return "afb_" + hashlib.sha256(raw.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_limb_health(self, name: str) -> Optional[LimbHealth]:
        """Get health state for a specific limb/tool."""
        return self._limbs.get(name)

    def get_all_limb_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health report for all limbs."""
        return {
            name: limb.to_dict()
            for name, limb in self._limbs.items()
        }

    def get_unhealthy_limbs(self, threshold: float = 0.5) -> List[str]:
        """Get names of limbs below the health threshold."""
        return [
            name for name, limb in self._limbs.items()
            if limb.health < threshold
        ]

    def get_status(self) -> Dict[str, Any]:
        """Return current processor status."""
        return {
            "total_processed": self._total_processed,
            "limb_count": len(self._limbs),
            "unhealthy_limbs": self.get_unhealthy_limbs(),
            "trail_size": len(self._feedback_trail),
            "uptime_s": round(time.time() - self._boot_time, 1),
        }

    def get_recent_feedback(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return recent feedback entries."""
        recent = list(self._feedback_trail)[-n:]
        return [
            {
                "feedback_id": f.feedback_id,
                "action": f.action_name,
                "outcome": f.outcome.value,
                "latency_ms": round(f.latency_ms, 1),
                "summary": f.result_summary[:100],
                "timestamp": f.timestamp,
            }
            for f in recent
        ]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_feedback_processor_instance: Optional[FeedbackProcessor] = None


def get_feedback_processor() -> FeedbackProcessor:
    """Get the singleton FeedbackProcessor."""
    global _feedback_processor_instance
    if _feedback_processor_instance is None:
        _feedback_processor_instance = FeedbackProcessor()
    return _feedback_processor_instance
