"""General action gateway for embodied environments.

Every proposed action should pass through a small deterministic gate before
it reaches the environment. The gate checks legality, modal state, risk,
uncertainty, loop/stagnation, and explicit invariants. It can approve,
replace, or veto without pretending the action happened.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from .belief_state import EnvironmentBeliefState
from .environment_parser import EnvironmentState
from .goal_manager import EmbodiedGoal
from .reflex_layer import RiskProfile
from .skill_graph import SkillOption


@dataclass
class ActionRequest:
    action: str
    source: str = "policy"
    reason: str = ""
    tags: List[str] = field(default_factory=list)
    expected_effect: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDecision:
    approved: bool
    action: Optional[str]
    original_action: str
    reason: str
    vetoes: List[str] = field(default_factory=list)
    replacements: List[str] = field(default_factory=list)
    risk_level: str = "safe"
    receipt_id: str = ""

    @property
    def replaced(self) -> bool:
        return bool(self.replacements) and self.action != self.original_action


class EnvironmentActionGateway:
    """Fast action approval/replacement surface."""

    def __init__(
        self,
        *,
        legal_actions: Optional[Iterable[str]] = None,
        prompt_actions: Optional[Dict[str, str]] = None,
        reversible_tags: Optional[Iterable[str]] = None,
    ) -> None:
        self.legal_actions: Optional[Set[str]] = set(legal_actions) if legal_actions else None
        self.prompt_actions = dict(prompt_actions or {})
        self.reversible_tags = {str(tag) for tag in (reversible_tags or {"inspect", "wait", "cancel", "retreat"})}
        self.decisions: List[ActionDecision] = []

    def approve(
        self,
        request: ActionRequest,
        *,
        state: EnvironmentState,
        risk: RiskProfile,
        goal: EmbodiedGoal,
        skill: SkillOption,
        belief: EnvironmentBeliefState,
    ) -> ActionDecision:
        action = str(request.action or "").strip()
        vetoes: List[str] = []
        replacements: List[str] = []

        if not action:
            vetoes.append("empty_action")

        if self.legal_actions is not None and action and action not in self.legal_actions:
            vetoes.append(f"illegal_action:{action}")

        if state.has_active_prompt():
            prompt_replacement = self._prompt_replacement(state)
            if prompt_replacement and action != prompt_replacement:
                replacements.append(prompt_replacement)
                action = prompt_replacement
            elif not prompt_replacement and "prompt_safe" not in request.tags:
                vetoes.append("active_prompt_requires_prompt_safe_action")

        risky_tags = {"irreversible", "unknown_use", "combat", "destructive"}
        if risk.critical and risky_tags.intersection(set(request.tags)):
            vetoes.append("critical_risk_blocks_risky_action")

        if belief.epistemic_uncertainty() >= 0.75 and "irreversible" in request.tags:
            vetoes.append("high_uncertainty_blocks_irreversible_action")

        # Inhibitory Gating: Block repetitive failures if we are in a loop or stagnation
        risk_tags = risk.tags()
        if "loop" in risk_tags or "stagnation" in risk_tags:
            # We look at the most recent executed outcomes (surprise >= 0.5 means environmental failure)
            recent_failures = [
                str(item.get("action", ""))
                for item in belief.action_outcomes[-6:]
                if item.get("surprise", 0.0) >= 0.5
            ]
            if action in recent_failures:
                vetoes.append(f"inhibitory_block:action_{action}_recently_failed_in_loop")

        if "loop" in risk_tags and "repeated" in request.tags:
            vetoes.append("loop_risk_blocks_repeated_action")

        approved = not vetoes
        if not approved and replacements:
            # A safe replacement can satisfy the gate if it is legal.
            replacement = replacements[-1]
            if self.legal_actions is None or replacement in self.legal_actions:
                action = replacement
                vetoes = [v for v in vetoes if not v.startswith("illegal_action")]
                approved = not any(v != "active_prompt_requires_prompt_safe_action" for v in vetoes)
                if vetoes == ["active_prompt_requires_prompt_safe_action"]:
                    vetoes = []
                    approved = True

        decision = ActionDecision(
            approved=approved,
            action=action if approved else None,
            original_action=request.action,
            reason=self._reason(approved, action, vetoes, replacements, risk, goal, skill),
            vetoes=vetoes,
            replacements=replacements,
            risk_level=risk.level,
            receipt_id=f"env-act-{int(time.time() * 1000)}",
        )
        self.decisions.append(decision)
        self.decisions = self.decisions[-1000:]
        return decision

    def _prompt_replacement(self, state: EnvironmentState) -> Optional[str]:
        prompt_text = " ".join(state.active_prompts).lower()
        for marker, action in self.prompt_actions.items():
            if marker.lower() in prompt_text:
                return action
        if "--more--" in prompt_text or "press return" in prompt_text:
            return self.prompt_actions.get("--more--") or self.prompt_actions.get("more")
        if "cancel" in self.prompt_actions and any(token in prompt_text for token in ("what do you want", "direction", "menu")):
            return self.prompt_actions["cancel"]
        return None

    @staticmethod
    def _reason(
        approved: bool,
        action: str,
        vetoes: List[str],
        replacements: List[str],
        risk: RiskProfile,
        goal: EmbodiedGoal,
        skill: SkillOption,
    ) -> str:
        if approved and replacements:
            return (
                f"Approved replacement action {action!r} under {risk.level} risk "
                f"for goal {goal.name} via {skill.name}."
            )
        if approved:
            return f"Approved action {action!r} under {risk.level} risk for goal {goal.name} via {skill.name}."
        return f"Rejected action under {risk.level} risk: {', '.join(vetoes)}"
