"""Tiered action control: reflex, habit, tactical, deliberative, reflective.

The point of a tier system is that a cheap tier is only allowed to act when
the situation is genuinely cheap. That requires three things this module
previously only claimed:

* the tier has to be derived from *validated* risk inputs (a NaN risk must not
  read as "below every threshold" and land in reflex);
* the selection has to be a decision, not ``candidates[0]``; and
* ``requires_system2`` has to actually withhold the action until a
  tier-appropriate proof comes back.

CP126 1b5e9bae / 9bffa715 / 1c45593a / 838b9e38 / 5abe6e13 / 846edf27.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any

from core.runtime.numeric_safety import validated_unit as _shared_unit

from .schemas import ActionCandidate, Observation, stable_hash

logger = logging.getLogger("Aura.TieredAction")


class ActionTier(IntEnum):
    REFLEX = 0
    HABIT = 1
    TACTICAL = 2
    DELIBERATIVE = 3
    REFLECTIVE = 4


#: What each System-2 tier must produce before an action is released. CP126
#: 1c45593a: ``requires_system2`` was a boolean the caller could ignore, so a
#: "requires deliberation" verdict and a "deliberated" verdict released the
#: action identically. Each key must be present AND non-empty — a key mapped
#: to None/""/[] is missing evidence, not satisfied evidence.
TIER_PROOF_REQUIREMENTS: dict[ActionTier, tuple[str, ...]] = {
    ActionTier.TACTICAL: ("search_ranking", "prediction"),
    ActionTier.DELIBERATIVE: (
        "search_ranking",
        "prediction",
        "alternatives_considered",
        "approval",
    ),
    ActionTier.REFLECTIVE: (
        "search_ranking",
        "prediction",
        "alternatives_considered",
        "approval",
        "proof_obligations",
        "postmortem_owner",
    ),
}

#: Tags that describe an action as pre-vetted, and tags that describe one as
#: consequential. Used for eligibility, not merely for scoring.
SAFE_TAGS = frozenset({"probe", "safe", "read_only", "observation"})
CONSEQUENTIAL_TAGS = frozenset(
    {"self_modify", "destructive", "irreversible", "external", "financial"}
)


@dataclass
class TieredActionDecision:
    decision_id: str
    tier: ActionTier
    selected: dict[str, Any] | None
    latency_budget_ms: int
    reason: str
    requires_system2: bool
    #: True when there was nothing to decide, or nothing eligible at this tier.
    #: CP126 5abe6e13: an empty candidate list used to come back as a
    #: low-risk reflex decision that merely happened to select nothing.
    abstained: bool = False
    system2_satisfied: bool = False
    #: Why the action is being withheld, if it is.
    blocked_reason: str = ""
    missing_proof: tuple[str, ...] = ()
    ranking: tuple[dict[str, Any], ...] = ()
    ineligible: tuple[dict[str, Any], ...] = ()
    #: The values the tier was actually computed from, after validation, plus
    #: any input that had to be repaired (CP126 9bffa715).
    inputs: dict[str, Any] = field(default_factory=dict)
    input_faults: tuple[str, ...] = ()
    deadline_monotonic: float = 0.0
    decision_latency_ms: float = 0.0
    system2_latency_ms: float | None = None

    @property
    def released(self) -> bool:
        """Whether this decision authorizes execution of ``selected``."""
        if self.abstained or self.selected is None:
            return False
        return self.system2_satisfied if self.requires_system2 else True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tier"] = int(self.tier)
        payload["tier_name"] = self.tier.name.lower()
        payload["released"] = self.released
        return payload


def _validated_unit(name: str, value: Any) -> tuple[float, str]:
    """A usable [0,1] scalar, failing toward caution.

    CP126 9bffa715: comparisons ran on the raw value while ``clamp`` was only
    applied inside the decision id, so NaN made every ``>=`` threshold false
    and the decision fell through to REFLEX — the cheapest tier — precisely
    when the risk signal was broken. The shared primitive lives in
    core/runtime/numeric_safety.py; this is the whole class of defect.
    """
    scalar = _shared_unit(value, name=name, cautious_high=True)
    return float(scalar), scalar.fault


def _coerce_candidates(
    actions: Sequence[ActionCandidate | Mapping[str, Any]],
) -> tuple[list[ActionCandidate], list[dict[str, Any]]]:
    acts: list[ActionCandidate] = []
    rejected: list[dict[str, Any]] = []
    for index, raw in enumerate(actions or ()):
        if isinstance(raw, ActionCandidate):
            acts.append(raw)
            continue
        try:
            acts.append(ActionCandidate(**dict(raw)))
        except (TypeError, ValueError) as exc:
            rejected.append({"index": index, "reason": f"malformed candidate: {exc}"})
    return acts, rejected


def _absent(value: Any) -> bool:
    """Whether an evidence slot is effectively empty."""
    if value is None or value is False:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
        return len(value) == 0
    return False


def _eligibility(tier: ActionTier, action: ActionCandidate, risk: float) -> str:
    """Why this action may not be taken at this tier, or ''.

    Cheap tiers exist because the action is cheap to undo. An irreversible or
    privileged action is not eligible for one no matter how it scores.
    """
    tags = {str(tag).strip().lower() for tag in (action.tags or ())}
    cost, _ = _validated_unit("expected_cost", action.expected_cost)
    try:
        authority = int(action.authority_tier)
    except (TypeError, ValueError):
        return "authority_tier is not an integer"

    if tags & CONSEQUENTIAL_TAGS and tier < ActionTier.DELIBERATIVE:
        return f"consequential action ({sorted(tags & CONSEQUENTIAL_TAGS)}) needs deliberation"
    if tier <= ActionTier.HABIT:
        if not action.reversible:
            return "irreversible action is not eligible below the tactical tier"
        if authority > 1:
            return f"authority tier {authority} requires at least tactical control"
        if cost > 0.25:
            return f"expected cost {cost:.2f} exceeds the reflex/habit budget"
    if tier == ActionTier.TACTICAL and not action.reversible and authority > 2:
        return "irreversible privileged action requires deliberative control"
    if not action.reversible and risk >= 0.75:
        return "irreversible action under high risk"
    return ""


def _score(action: ActionCandidate, risk: float) -> float:
    """Preference among eligible actions: cheap, reversible, low-authority."""
    cost, _ = _validated_unit("expected_cost", action.expected_cost)
    try:
        authority = max(0, int(action.authority_tier) - 1)
    except (TypeError, ValueError):
        authority = 4
    tags = {str(tag).strip().lower() for tag in (action.tags or ())}

    score = 1.0
    score -= 0.55 * cost
    score -= 0.12 * authority
    if not action.reversible:
        score -= 0.40 * (0.5 + risk)
    if tags & SAFE_TAGS:
        score += 0.15
    if tags & CONSEQUENTIAL_TAGS:
        score -= 0.35
    return round(score, 6)


class TieredActionController:
    """Chooses the cheapest *adequate* control tier under uncertainty and risk."""

    LATENCY_BUDGET_MS = {
        ActionTier.REFLEX: 5,
        ActionTier.HABIT: 50,
        ActionTier.TACTICAL: 500,
        ActionTier.DELIBERATIVE: 10_000,
        ActionTier.REFLECTIVE: 60_000,
    }

    def choose_tier(
        self,
        observation: Observation | Mapping[str, Any],
        actions: Sequence[ActionCandidate | Mapping[str, Any]],
        *,
        risk: float,
        uncertainty: float,
        novelty: float = 0.0,
        self_modification: bool = False,
        system2_evidence: Mapping[str, Any] | None = None,
        system2: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> TieredActionDecision:
        """Pick a tier, pick an action within it, and withhold what isn't proven.

        ``system2_evidence`` is proof the caller already has (a search ranking,
        a world-model prediction, an approval record). ``system2`` is a callable
        invoked when a System-2 tier is reached without sufficient evidence; it
        receives the pending decision context and returns additional evidence.
        """
        started = time.monotonic()
        obs = (
            observation
            if isinstance(observation, Observation)
            else Observation(**dict(observation))
        )
        acts, rejected = _coerce_candidates(actions)

        risk_value, risk_fault = _validated_unit("risk", risk)
        uncertainty_value, uncertainty_fault = _validated_unit("uncertainty", uncertainty)
        novelty_value, novelty_fault = _validated_unit("novelty", novelty)
        faults = tuple(f for f in (risk_fault, uncertainty_fault, novelty_fault) if f)
        if faults:
            self._record_input_fault(faults)

        tier, reason = self._tier_for(
            risk_value, uncertainty_value, novelty_value, acts, bool(self_modification)
        )
        budget_ms = self.LATENCY_BUDGET_MS[tier]
        deadline = started + budget_ms / 1000.0
        requires_system2 = tier >= ActionTier.TACTICAL

        ranked: list[dict[str, Any]] = []
        ineligible: list[dict[str, Any]] = list(rejected)
        for action in acts:
            blocked = _eligibility(tier, action, risk_value)
            entry = {"action": action.to_dict(), "score": _score(action, risk_value)}
            if blocked:
                ineligible.append({**entry, "reason": blocked})
            else:
                ranked.append(entry)
        ranked.sort(key=lambda row: (-row["score"], str(row["action"].get("action_id", ""))))

        abstained = False
        blocked_reason = ""
        selected: dict[str, Any] | None = None
        if not acts:
            # CP126 5abe6e13: nothing to decide is an abstention, not a
            # low-risk reflex that happens to select nothing.
            abstained = True
            blocked_reason = "no_candidates"
            reason = "no candidate actions were supplied; abstaining"
        elif not ranked:
            abstained = True
            blocked_reason = "no_eligible_candidates"
            reason = (
                f"no candidate is eligible at the {tier.name.lower()} tier; abstaining"
            )
        else:
            selected = ranked[0]["action"]

        evidence: dict[str, Any] = dict(system2_evidence or {})
        satisfied, missing = self._proof_state(tier, evidence)
        system2_latency: float | None = None

        if requires_system2 and not satisfied and system2 is not None and not abstained:
            supplied, system2_latency = self._invoke_system2(
                system2,
                {
                    "observation": obs.to_dict(),
                    "tier": int(tier),
                    "tier_name": tier.name.lower(),
                    "ranking": tuple(ranked),
                    "risk": risk_value,
                    "uncertainty": uncertainty_value,
                    "novelty": novelty_value,
                    "self_modification": bool(self_modification),
                    "required_proof": TIER_PROOF_REQUIREMENTS.get(tier, ()),
                    "deadline_monotonic": deadline,
                    "latency_budget_ms": budget_ms,
                },
                deadline=deadline,
            )
            if supplied is not None:
                evidence.update(supplied)
                satisfied, missing = self._proof_state(tier, evidence)

        if requires_system2 and not satisfied and not abstained:
            # CP126 1c45593a: the action is withheld here, at the controller,
            # rather than being handed to the caller with an advisory flag.
            blocked_reason = "system2_proof_missing"
            selected = None

        decision = TieredActionDecision(
            decision_id=stable_hash(
                {
                    "obs": obs.observation_id,
                    # CP126 846edf27: bind the whole candidate, not just its id,
                    # and bind every tier-driving input.
                    "actions": [action.to_dict() for action in acts],
                    "risk": risk_value,
                    "uncertainty": uncertainty_value,
                    "novelty": novelty_value,
                    "self_modification": bool(self_modification),
                    "tier": int(tier),
                    "selected": (selected or {}).get("action_id", ""),
                    "system2_satisfied": bool(satisfied),
                },
                prefix="tier_",
            ),
            tier=tier,
            selected=selected,
            latency_budget_ms=budget_ms,
            reason=reason,
            requires_system2=requires_system2,
            abstained=abstained,
            system2_satisfied=bool(satisfied) if requires_system2 else True,
            blocked_reason=blocked_reason,
            missing_proof=missing,
            ranking=tuple(ranked),
            ineligible=tuple(ineligible),
            inputs={
                "risk": risk_value,
                "uncertainty": uncertainty_value,
                "novelty": novelty_value,
                "self_modification": bool(self_modification),
            },
            input_faults=faults,
            deadline_monotonic=deadline,
            decision_latency_ms=round((time.monotonic() - started) * 1000.0, 3),
            system2_latency_ms=system2_latency,
        )
        if decision.requires_system2 and not decision.released and not abstained:
            logger.info(
                "Tier %s withheld action: %s (missing %s)",
                tier.name.lower(),
                decision.blocked_reason,
                ", ".join(missing) or "-",
            )
        return decision

    def _tier_for(
        self,
        risk: float,
        uncertainty: float,
        novelty: float,
        acts: Sequence[ActionCandidate],
        self_modification: bool,
    ) -> tuple[ActionTier, str]:
        tagged_self_modify = any(
            "self_modify" in {str(t).strip().lower() for t in (a.tags or ())} for a in acts
        )
        if self_modification or tagged_self_modify:
            return (
                ActionTier.REFLECTIVE,
                "self-modification requires proof obligations and postmortem learning",
            )
        if any(not a.reversible for a in acts) and risk >= 0.45:
            return (
                ActionTier.DELIBERATIVE,
                "an irreversible option under material risk requires System 2",
            )
        if risk >= 0.75 or uncertainty >= 0.75:
            return ActionTier.DELIBERATIVE, "high risk or uncertainty requires System 2"
        if risk >= 0.45 or novelty >= 0.6:
            return ActionTier.TACTICAL, "moderate risk/novelty requires short-horizon search"
        if any(
            SAFE_TAGS & {str(t).strip().lower() for t in (a.tags or ())} for a in acts
        ):
            return ActionTier.HABIT, "familiar safe policy/habit is adequate"
        return ActionTier.REFLEX, "low-risk immediate action"

    @staticmethod
    def _proof_state(
        tier: ActionTier, evidence: Mapping[str, Any]
    ) -> tuple[bool, tuple[str, ...]]:
        required = TIER_PROOF_REQUIREMENTS.get(tier, ())
        if not required:
            return True, ()
        # Present-but-empty is missing: an empty ranking is not a search, and
        # ``approval: False`` is a refusal, not a satisfied requirement.
        missing = tuple(key for key in required if _absent(evidence.get(key)))
        return (not missing), missing

    def _invoke_system2(
        self,
        system2: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        context: Mapping[str, Any],
        *,
        deadline: float,
    ) -> tuple[Mapping[str, Any] | None, float]:
        """Run the deliberation callback and hold it to the tier's budget.

        A synchronous callback cannot be preempted, so the deadline is passed
        in (a cooperative implementation can self-cancel) and enforced on the
        RESULT: evidence that arrives after the budget is refused rather than
        silently accepted, which is what makes the budget more than a label
        (CP126 838b9e38).
        """
        started = time.monotonic()
        try:
            supplied = system2(context)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            elapsed = (time.monotonic() - started) * 1000.0
            logger.warning("System 2 deliberation failed: %s", exc)
            self._record_input_fault((f"system2 callback raised {type(exc).__name__}",))
            return None, round(elapsed, 3)
        elapsed = (time.monotonic() - started) * 1000.0
        if time.monotonic() > deadline:
            logger.warning(
                "System 2 deliberation overran its %s tier budget (%.1fms); evidence refused",
                context.get("tier_name"),
                elapsed,
            )
            return None, round(elapsed, 3)
        if not isinstance(supplied, Mapping):
            return None, round(elapsed, 3)
        return dict(supplied), round(elapsed, 3)

    def execute_within_budget(
        self,
        decision: TieredActionDecision,
        run: Callable[[dict[str, Any]], Any],
    ) -> dict[str, Any]:
        """Run ``run(selected)`` only if the decision released it, and time it.

        This is the enforcement half of the latency budget: a tier that
        overruns its budget is reported as an overrun with the measurement
        attached, not quietly accepted (CP126 838b9e38).
        """
        if not decision.released:
            return {
                "ok": False,
                "executed": False,
                "error": decision.blocked_reason or "action_not_released",
                "missing_proof": list(decision.missing_proof),
                "tier": decision.tier.name.lower(),
            }
        started = time.monotonic()
        try:
            result = run(dict(decision.selected or {}))
            error = ""
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            result, error = None, f"{type(exc).__name__}: {exc}"
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
        overran = elapsed_ms > decision.latency_budget_ms
        if overran:
            logger.warning(
                "Tier %s action overran its %dms budget (%.1fms)",
                decision.tier.name.lower(),
                decision.latency_budget_ms,
                elapsed_ms,
            )
        return {
            "ok": not error,
            "executed": True,
            "result": result,
            "error": error,
            "elapsed_ms": elapsed_ms,
            "budget_ms": decision.latency_budget_ms,
            "overran": overran,
            "tier": decision.tier.name.lower(),
        }

    @staticmethod
    def _record_input_fault(faults: Sequence[str]) -> None:
        detail = "; ".join(faults)
        logger.warning("Tiered action inputs required repair: %s", detail)
        try:
            from core.runtime.errors import record_degradation

            record_degradation(
                "tiered_action",
                ValueError(detail),
                action="escalated the control tier because a risk input was unusable",
                severity="warning",
            )
        except (ImportError, RuntimeError, TypeError, ValueError):
            return
