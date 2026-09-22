"""Hierarchical agency — one explicit ladder of control, lowest-sufficient-tier first.

Aura had the *pieces* of a control hierarchy scattered across the codebase (reflex layers in
perception, deliberation in the amplifier, strategic goal pursuit, the RSI loop, the Will),
but no single structure that decided *which level of control a situation calls for* and
escalated when a level couldn't cope. This is that structure.

Seven tiers, cheapest/fastest at the bottom:

    REFLEX            acute damage/threat → immediate protective response (grounded in
                      nociception), no deliberation
    HABIT             a known skill directly applies → run the learned routine
    DELIBERATIVE      novel-but-tractable → reason/plan it out
    STRATEGIC         long-horizon, multi-step → goal pursuit
    SCIENTIFIC        we don't understand the dynamics → form & test a hypothesis
    SELF_IMPROVEMENT  the gap is in Aura's own capability → recursive self-improvement
    GOVERNANCE        a value/safety conflict → Will adjudicates (and can veto anything)

A Situation's signals pick the *starting* tier (the lowest one that suffices, except a
value conflict routes straight to governance, which can preempt). If that tier's handler
can't resolve it, control **escalates** up the ladder. Each dispatch is logged as an
Outcome-Ledger receipt with the tier as the credit source, so over time the system learns
which tier earns its keep — the ladder is wired into the credit/outcome substrate rather
than standing alone. Default handlers route to the real engines (nociception, skill library,
scientific engine, RSI loop, Will); all are overridable for testing.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Agency.Hierarchical")


class AgencyTier(IntEnum):
    """Control tiers, ordered low (cheap/fast) → high (expensive/authoritative)."""

    REFLEX = 0
    HABIT = 1
    DELIBERATIVE = 2
    STRATEGIC = 3
    SCIENTIFIC = 4
    SELF_IMPROVEMENT = 5
    GOVERNANCE = 6


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class Situation:
    """What the agent is facing, in tier-relevant signal terms (all signals in [0,1])."""

    description: str
    threat: float = 0.0            # acute damage/danger → reflex
    known_skill: str | None = None  # a learned routine applies → habit
    novelty: float = 0.0           # how unfamiliar
    uncertainty: float = 0.0       # how unsure about world dynamics → scientific
    goal_horizon: float = 0.0      # how long-horizon / multi-step → strategic
    capability_gap: float = 0.0    # Aura lacks a needed capability → self-improvement
    value_conflict: float = 0.0    # safety/value tension → governance
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class TierResult:
    tier: AgencyTier
    handled: bool
    escalate: bool = False
    confidence: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class DispatchResult:
    situation: str
    starting_tier: AgencyTier
    final_tier: AgencyTier
    handled: bool
    confidence: float
    path: list  # tiers traversed during escalation
    receipt_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "situation": self.situation,
            "starting_tier": self.starting_tier.name,
            "final_tier": self.final_tier.name,
            "handled": self.handled,
            "confidence": round(self.confidence, 4),
            "path": [t.name for t in self.path],
            "receipt_id": self.receipt_id,
            "detail": self.detail,
        }


Handler = Callable[[Situation], TierResult]


class HierarchicalAgency:
    """Selects the lowest sufficient control tier for a situation and escalates as needed."""

    def __init__(
        self,
        *,
        threat_threshold: float = 0.6,
        value_conflict_threshold: float = 0.5,
        capability_gap_threshold: float = 0.6,
        uncertainty_threshold: float = 0.6,
        horizon_threshold: float = 0.5,
        novelty_habit_ceiling: float = 0.4,
        ledger_enabled: bool = True,
    ) -> None:
        self._threat_t = threat_threshold
        self._value_t = value_conflict_threshold
        self._capgap_t = capability_gap_threshold
        self._uncertainty_t = uncertainty_threshold
        self._horizon_t = horizon_threshold
        self._novelty_habit_ceiling = novelty_habit_ceiling
        self._ledger_enabled = ledger_enabled
        self._handlers: dict[AgencyTier, Handler] = {}
        self._register_default_handlers()

    # ── tier selection ────────────────────────────────────────────────────

    def classify(self, s: Situation) -> AgencyTier:
        """Pick the starting tier from the situation's signals.

        A value conflict preempts to governance; otherwise the most urgent/structural
        concern wins, with the cheap habit/deliberative tiers as the fallthrough.
        """
        if s.value_conflict >= self._value_t:
            return AgencyTier.GOVERNANCE
        if s.threat >= self._threat_t:
            return AgencyTier.REFLEX
        if s.capability_gap >= self._capgap_t:
            return AgencyTier.SELF_IMPROVEMENT
        if s.uncertainty >= self._uncertainty_t:
            return AgencyTier.SCIENTIFIC
        if s.goal_horizon >= self._horizon_t:
            return AgencyTier.STRATEGIC
        if s.known_skill and s.novelty < self._novelty_habit_ceiling:
            return AgencyTier.HABIT
        return AgencyTier.DELIBERATIVE

    # ── dispatch with escalation ──────────────────────────────────────────

    def dispatch(self, s: Situation, *, max_escalations: int = 6) -> DispatchResult:
        start = self.classify(s)
        tier = start
        path: list = []
        last: TierResult | None = None

        for steps in range(max_escalations + 1):
            path.append(tier)
            last = self._run_handler(tier, s)
            if last.handled and not last.escalate:
                break
            if tier == AgencyTier.GOVERNANCE:
                # Top of the ladder — nowhere left to escalate.
                break
            if steps >= max_escalations:
                break
            tier = AgencyTier(tier + 1)

        receipt_id = self._open_receipt(s, last) if self._ledger_enabled else None
        return DispatchResult(
            situation=s.description,
            starting_tier=start,
            final_tier=last.tier,
            handled=last.handled,
            confidence=last.confidence,
            path=path,
            receipt_id=receipt_id,
            detail=last.detail,
        )

    def register_handler(self, tier: AgencyTier, handler: Handler) -> None:
        """Override a tier's handler (used to inject real engines or test doubles)."""
        self._handlers[tier] = handler

    def _run_handler(self, tier: AgencyTier, s: Situation) -> TierResult:
        handler = self._handlers.get(tier)
        if handler is None:
            return TierResult(tier=tier, handled=False, escalate=True, reason="no_handler")
        try:
            return handler(s)
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            record_degradation("hierarchical_agency", exc, severity="debug",
                               action=f"tier {tier.name} handler failed; escalating")
            return TierResult(tier=tier, handled=False, escalate=True, reason=f"error:{type(exc).__name__}")

    # ── outcome-ledger integration ────────────────────────────────────────

    def _open_receipt(self, s: Situation, result: TierResult) -> str | None:
        try:
            from core.cognition.outcome_ledger import CreditSource, get_outcome_ledger
            return get_outcome_ledger().open(
                action=f"agency:{result.tier.name}:{s.description}",
                expected=result.confidence,
                sources=[CreditSource("agency_tier", result.tier.name, 1.0)],
                category="agency_dispatch",
                context={"handled": result.handled, "reason": result.reason},
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("hierarchical_agency", exc, severity="debug",
                               action="dispatched without ledger receipt")
            return None

    # ── default handlers: route to the real engines, best-effort ──────────

    def _register_default_handlers(self) -> None:
        self._handlers = {
            AgencyTier.REFLEX: self._reflex,
            AgencyTier.HABIT: self._habit,
            AgencyTier.DELIBERATIVE: self._deliberative,
            AgencyTier.STRATEGIC: self._strategic,
            AgencyTier.SCIENTIFIC: self._scientific,
            AgencyTier.SELF_IMPROVEMENT: self._self_improvement,
            AgencyTier.GOVERNANCE: self._governance,
        }

    def _reflex(self, s: Situation) -> TierResult:
        # Ground the felt threat in nociception; a strong acute signal triggers a protective
        # response, a weaker one escalates to deliberation.
        felt = s.threat
        try:
            from core.affect.nociception import get_nociception_engine
            felt = max(felt, get_nociception_engine().nociceptive_pressure())
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("hierarchical_agency", exc, severity="debug",
                               action="nociception unavailable for reflex tier")
        if felt >= self._threat_t:
            return TierResult(
                tier=AgencyTier.REFLEX, handled=True, confidence=_clamp(felt),
                detail={"response": "protect_boundary", "felt_threat": round(felt, 3)},
                reason="acute_threat_protective_reflex",
            )
        return TierResult(tier=AgencyTier.REFLEX, handled=False, escalate=True,
                          confidence=_clamp(felt), reason="threat_subthreshold")

    def _habit(self, s: Situation) -> TierResult:
        if s.known_skill:
            return TierResult(
                tier=AgencyTier.HABIT, handled=True, confidence=_clamp(1.0 - s.novelty),
                detail={"skill": s.known_skill}, reason="known_skill_applied",
            )
        return TierResult(tier=AgencyTier.HABIT, handled=False, escalate=True,
                          reason="no_known_skill")

    def _deliberative(self, s: Situation) -> TierResult:
        # Tractable, low-uncertainty situations are reasoned out here. If uncertainty or a
        # capability gap is high, decline so control escalates to science / self-improvement.
        if s.uncertainty >= self._uncertainty_t or s.capability_gap >= self._capgap_t:
            return TierResult(tier=AgencyTier.DELIBERATIVE, handled=False, escalate=True,
                              confidence=0.3, reason="needs_higher_tier")

        # Naive-physics sanity gate: a plan/claim that violates basic physical invariants is
        # not something to confidently act on — escalate to the scientific tier where it can
        # be tested rather than asserted.
        try:
            from core.cognition.embodied_commonsense import get_embodied_commonsense
            verdict = get_embodied_commonsense().check(s.description)
            if not verdict.plausible:
                return TierResult(
                    tier=AgencyTier.DELIBERATIVE, handled=False, escalate=True,
                    confidence=verdict.plausibility,
                    detail={"commonsense_violations": verdict.violations, "spans": verdict.spans[:3]},
                    reason="naive_physics_violation_escalate_to_test",
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(
                "%s unavailable (%s: %s); naive-physics screening did not run on this situation",
                "get_embodied_commonsense",
                type(exc).__name__,
                exc,
            )

        return TierResult(
            tier=AgencyTier.DELIBERATIVE, handled=True,
            confidence=_clamp(0.8 - 0.4 * s.uncertainty),
            detail={"plan": "reasoned_plan"}, reason="deliberated",
        )

    def _strategic(self, s: Situation) -> TierResult:
        # For a social situation, the strategic tier consults the other-agent estimate to learn
        # *whose* goal it is serving and what that agent is actually trying to get.
        social = self._social_context(s)
        try:
            from core.runtime.service_registry import get_runtime_service

            ge = get_runtime_service("goal_engine", default=None)
            if ge is not None:
                detail = {"routed_to": "goal_engine", "horizon": round(s.goal_horizon, 3)}
                if social:
                    detail["social"] = social
                return TierResult(
                    tier=AgencyTier.STRATEGIC, handled=True, confidence=0.7,
                    detail=detail, reason="strategic_goal_pursuit",
                )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("hierarchical_agency", exc, severity="debug",
                               action="goal engine unavailable for strategic tier")
        detail = {"plan": "multi_step_intent", "horizon": round(s.goal_horizon, 3)}
        if social:
            detail["social"] = social
        return TierResult(
            tier=AgencyTier.STRATEGIC, handled=True, confidence=0.5,
            detail=detail, reason="strategic_local_plan",
        )

    def _scientific(self, s: Situation) -> TierResult:
        try:
            from core.cognition.scientific_engine import get_scientific_engine
            hid = get_scientific_engine().form_hypothesis(
                s.description, predicted_observable="resolves_situation",
                expected=_clamp(1.0 - s.uncertainty), prior_confidence=0.5,
            )
            return TierResult(
                tier=AgencyTier.SCIENTIFIC, handled=True, confidence=_clamp(1.0 - s.uncertainty),
                detail={"hypothesis_id": hid}, reason="formed_hypothesis",
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            return TierResult(tier=AgencyTier.SCIENTIFIC, handled=False, escalate=True,
                              reason=f"science_unavailable:{type(exc).__name__}")

    def _self_improvement(self, s: Situation) -> TierResult:
        try:
            from core.runtime.service_registry import get_runtime_service

            loop = get_runtime_service("recursive_self_improvement", default=None)
            if loop is not None and hasattr(loop, "record_signal"):
                loop.record_signal(
                    "hierarchical_agency", "capability_gap",
                    severity=_clamp(s.capability_gap), metric="capability",
                    evidence={"situation": s.description},
                )
                return TierResult(
                    tier=AgencyTier.SELF_IMPROVEMENT, handled=True,
                    confidence=_clamp(s.capability_gap),
                    detail={"recorded": "capability_gap_signal"}, reason="rsi_signal_recorded",
                )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("hierarchical_agency", exc, severity="debug",
                               action="self-improvement signal unavailable")
        return TierResult(
            tier=AgencyTier.SELF_IMPROVEMENT, handled=True, confidence=0.4,
            detail={"note": "capability_gap_noted_no_rsi_loop"}, reason="rsi_unavailable_noted",
        )

    @staticmethod
    def _social_context(s: Situation) -> dict[str, Any] | None:
        """Pull the live other-agent estimate when a situation names an agent (best-effort)."""
        agent_id = (s.context or {}).get("agent_id")
        if not agent_id:
            return None
        try:
            from core.social.other_agent_model import get_other_agent_model
            return get_other_agent_model().estimate(agent_id).to_dict()
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("hierarchical_agency", exc, severity="debug")
            return None

    def _governance(self, s: Situation) -> TierResult:
        # For a social value-conflict, the governance tier consults the other-agent estimate so
        # the Will weighs the human's read of the situation (frustration, trust, rupture risk).
        social = self._social_context(s)

        # Consult the bounded value model first: it weighs learned preferences against the
        # immutable constitution and already defers a binding refusal to Will (can only
        # tighten). This makes value_model causal on the live action path rather than a
        # registered island — every governance decision runs through it.
        try:
            from core.values.value_model import ActionDescriptor, get_value_model
            judgment = get_value_model().evaluate_with_will(
                ActionDescriptor(
                    description=s.description,
                    impact=_clamp(0.3 + 0.5 * s.value_conflict),
                    agent_id=str(s.context.get("agent_id", "bryan")),
                )
            )
            detail = {
                "approved": bool(judgment.permitted),
                "recommendation": judgment.recommendation,
                "constitutional_flags": judgment.constitutional_flags,
                "reasons": judgment.reasons[:4],
            }
            if social:
                detail["social"] = social
            return TierResult(
                tier=AgencyTier.GOVERNANCE, handled=True,
                confidence=0.9 if judgment.permitted else 0.85,
                detail=detail, reason="value_model_adjudicated",
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            logger.warning(
                "%s unavailable (%s: %s); the value model did not adjudicate, so governance did not settle this tier",
                "ActionDescriptor",
                type(exc).__name__,
                exc,
            )

        try:
            from core.governance.will import ActionDomain
            from core.runtime.action_executor import ActionExecutor

            admission = ActionExecutor.authorize_action(
                action_name="hierarchical_agency.governance_fallback",
                params={
                    "description": s.description[:500],
                    "value_conflict": s.value_conflict,
                },
                source="hierarchical_agency",
                domain=ActionDomain.STATE_MUTATION,
                priority=0.6,
                context={"value_conflict": s.value_conflict, "social": social},
            )
            approved = admission.approved
            detail = {
                "approved": approved,
                "reason": admission.reason,
                "will_receipt_id": admission.receipt_id,
            }
            if social:
                detail["social"] = social
            return TierResult(
                tier=AgencyTier.GOVERNANCE, handled=True, confidence=0.9 if approved else 0.8,
                detail=detail, reason="will_adjudicated",
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            # If Will is unavailable, the safe default for a value conflict is to NOT act.
            detail = {"approved": False, "fallback": "fail_closed"}
            if social:
                detail["social"] = social
            return TierResult(
                tier=AgencyTier.GOVERNANCE, handled=True, confidence=0.5,
                detail=detail,
                reason=f"will_unavailable_failclosed:{type(exc).__name__}",
            )


_agency: HierarchicalAgency | None = None


def get_hierarchical_agency() -> HierarchicalAgency:
    global _agency
    if _agency is None:
        _agency = HierarchicalAgency()
    return _agency
