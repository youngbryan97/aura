"""Reusable embodied cognition runtime.

The runtime ties together fast perception, persistent belief, risk/reflex
assessment, hierarchical goals, option selection, affordance recall, action
gating, trace logging, and evaluation hooks. NetHack is only an adapter that
plugs raw terminal text and a key action space into this general loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.container import ServiceContainer

from .action_gateway import ActionDecision, ActionRequest, EnvironmentActionGateway
from .affordance_schema import AffordanceKnowledgeBase
from .belief_state import EnvironmentBeliefState
from .environment_parser import EnvironmentParser, EnvironmentState
from .evaluation_harness import EmbodiedEvaluationHarness
from .goal_manager import EmbodiedGoal, EnvironmentGoalManager
from .reflex_layer import EnvironmentReflexLayer, RiskProfile
from .skill_graph import EnvironmentSkillGraph, SkillOption
from .trace import EmbodiedTraceLogger


@dataclass
class EmbodiedCognitiveFrame:
    state: EnvironmentState
    belief: EnvironmentBeliefState
    risk: RiskProfile
    goal: EmbodiedGoal
    skill: SkillOption
    affordance_summary: str = ""
    reflex_percepts: List[str] = field(default_factory=list)
    due_intentions: List[str] = field(default_factory=list)

    def to_prompt(self, *, include_raw: Optional[str] = None, max_raw_chars: int = 4000) -> str:
        blocks = [
            self.state.to_structured_prompt(),
            "[BELIEF STATE]\n" + self.belief.get_strategic_summary(),
            "[RISK PROFILE]\n"
            + f"LEVEL: {self.risk.level}\nSCORE: {self.risk.score:.2f}\n"
            + "\n".join(f"- {reason}" for reason in self.risk.reasons()),
            self.goal_stack_prompt,
            self.skill_prompt,
        ]
        if self.affordance_summary:
            blocks.append(self.affordance_summary)
        if self.reflex_percepts:
            blocks.append("[REFLEX PERCEPTS]\n" + "\n".join(self.reflex_percepts))
        if self.due_intentions:
            blocks.append("[PROSPECTIVE MEMORY]\n" + "\n".join(f"- {i}" for i in self.due_intentions))
        if include_raw:
            blocks.append("[RAW OBSERVATION]\n" + include_raw[:max_raw_chars])
        return "\n\n".join(block for block in blocks if block)

    @property
    def goal_stack_prompt(self) -> str:
        manager = getattr(self, "_goal_manager", None)
        if manager is not None:
            return manager.to_prompt()
        return f"[GOAL]\n- {self.goal.name}: {self.goal.reason}"

    @property
    def skill_prompt(self) -> str:
        graph = getattr(self, "_skill_graph", None)
        if graph is not None:
            return graph.to_prompt(self.skill)
        return f"[SKILL]\nSELECTED OPTION: {self.skill.name} - {self.skill.description}"


class EmbodiedCognitionRuntime:
    """One environment's live perception-action cognition loop."""

    def __init__(
        self,
        *,
        domain: str,
        parser: EnvironmentParser,
        legal_actions: Optional[Iterable[str]] = None,
        prompt_actions: Optional[Dict[str, str]] = None,
        storage_root: Optional[Path] = None,
    ) -> None:
        self.domain = domain
        self.parser = parser
        storage_root = Path(storage_root) if storage_root else None
        affordance_path = storage_root / f"{domain}_affordances.json" if storage_root else None
        trace_path = storage_root / f"{domain}_trace.jsonl" if storage_root else None

        self.existing_world_state = ServiceContainer.get("world_state", default=None)
        self.existing_causal_world_model = ServiceContainer.get("causal_world_model", default=None)
        self.existing_skill_library = ServiceContainer.get("skill_library", default=None)
        self.existing_goal_engine = ServiceContainer.get("goal_engine", default=None)

        self.belief = EnvironmentBeliefState(session_id=domain)
        self.reflex = EnvironmentReflexLayer()
        self.goal_manager = EnvironmentGoalManager()
        self.skill_graph = EnvironmentSkillGraph(macro_library=self.existing_skill_library)
        self.action_gateway = EnvironmentActionGateway(
            legal_actions=legal_actions,
            prompt_actions=prompt_actions,
        )
        self.affordances = AffordanceKnowledgeBase(domain=domain, storage_path=affordance_path)
        self.trace = EmbodiedTraceLogger(path=trace_path)
        self.evaluation = EmbodiedEvaluationHarness(domain=domain)
        self.last_frame: Optional[EmbodiedCognitiveFrame] = None

    def observe(self, raw_input: Any, *, context_id: Optional[str] = None) -> EmbodiedCognitiveFrame:
        state = self.parser.parse(raw_input)
        state.domain = state.domain or self.domain
        if state.domain == "generic":
            state.domain = self.domain
        if context_id is not None:
            state.context_id = context_id
        state.refresh_observation_id()

        self.belief.update_from_observation(state, context_id=state.context_id)
        risk = self.reflex.assess_profile(state, self.belief)
        goal = self.goal_manager.update_from_state(state, risk, self.belief)
        skill = self.skill_graph.select(state, risk, goal)
        visible_entities = state.entity_labels() + [entity.get("type", "") for entity in state.entities]
        affordance_summary = self.affordances.get_summary_for_prompt(visible_entities)
        reflex_percepts = self.reflex.generate_reflex_percepts(risk.assessments)
        due = [item.intention for item in self.belief.due_intentions(state)]

        frame = EmbodiedCognitiveFrame(
            state=state,
            belief=self.belief,
            risk=risk,
            goal=goal,
            skill=skill,
            affordance_summary=affordance_summary,
            reflex_percepts=reflex_percepts,
            due_intentions=due,
        )
        setattr(frame, "_goal_manager", self.goal_manager)
        setattr(frame, "_skill_graph", self.skill_graph)
        self.last_frame = frame
        self.trace.record_observation(
            domain=self.domain,
            observation_id=state.observation_id,
            context_id=state.context_id,
            risk_level=risk.level,
            risk_score=risk.score,
            goal=goal.name,
            skill=skill.name,
            messages=state.messages,
            belief_uncertainty=self.belief.epistemic_uncertainty(),
        )
        self._publish_to_existing_organs(frame)
        return frame

    def approve_action(
        self,
        action: str,
        *,
        source: str = "policy",
        reason: str = "",
        tags: Optional[List[str]] = None,
        expected_effect: str = "",
    ) -> ActionDecision:
        if self.last_frame is None:
            raise RuntimeError("cannot approve action before first observation")
        request = ActionRequest(
            action=action,
            source=source,
            reason=reason,
            tags=list(tags or []),
            expected_effect=expected_effect,
        )
        frame = self.last_frame
        decision = self.action_gateway.approve(
            request,
            state=frame.state,
            risk=frame.risk,
            goal=frame.goal,
            skill=frame.skill,
            belief=self.belief,
        )
        # Record the INTENTIONAL outcome (pre-execution)
        self.belief.record_action_outcome(
            action,
            expected=expected_effect,
            observed=decision.reason,
            success=decision.approved,
            surprise=0.0 if decision.approved else 0.5,
        )
        self._publish_action_decision(decision)
        return decision

    def record_environmental_outcome(self, action: str, success: bool, message: str = "") -> None:
        """Update the belief state with the actual result of an executed action."""
        surprise = 0.0 if success else 0.5
        self.belief.record_action_outcome(
            action,
            observed=message,
            success=success,
            surprise=surprise,
        )
        if self.last_frame and self.last_frame.skill:
            self.last_frame.skill.record_outcome(success)

    def command_contract(
        self,
        *,
        action_marker: str,
        valid_actions: Iterable[str],
        extra_rules: Optional[Iterable[str]] = None,
    ) -> str:
        actions = ", ".join(str(action) for action in valid_actions)
        rules = [
            "Output exactly one action marker and no conversational prose.",
            f"Marker format: [ACTION:{action_marker}] <action>",
            f"Valid actions: {actions}",
            "Respect the current goal, skill constraints, risk warnings, and active prompt state.",
            "Prefer reversible information-gathering under high uncertainty.",
            "Interrupt progress-seeking when risk becomes critical.",
        ]
        rules.extend(extra_rules or [])
        return "[EMBODIED CONTROL CONTRACT]\n" + "\n".join(f"- {rule}" for rule in rules)

    def _publish_to_existing_organs(self, frame: EmbodiedCognitiveFrame) -> None:
        """Bridge the local embodied frame into Aura's canonical systems."""
        if self.existing_world_state is None:
            self.existing_world_state = ServiceContainer.get("world_state", default=None)
        if self.existing_world_state is not None:
            try:
                self.existing_world_state.record_event(
                    f"{self.domain} observation risk={frame.risk.level} goal={frame.goal.name}",
                    source=f"environment:{self.domain}",
                    salience=max(0.35, frame.risk.score),
                    ttl=900.0,
                    observation_id=frame.state.observation_id,
                    context_id=frame.state.context_id,
                    skill=frame.skill.name,
                )
                self.existing_world_state.set_belief(
                    f"environment.{self.domain}.risk_level",
                    frame.risk.level,
                    confidence=0.85,
                    source="embodied_cognition_runtime",
                    ttl=600.0,
                )
                self.existing_world_state.set_belief(
                    f"environment.{self.domain}.current_goal",
                    frame.goal.name,
                    confidence=0.8,
                    source="embodied_cognition_runtime",
                    ttl=600.0,
                )
            except Exception:
                pass

        if self.existing_causal_world_model is None:
            self.existing_causal_world_model = ServiceContainer.get("causal_world_model", default=None)
        if self.existing_causal_world_model is not None:
            try:
                if frame.risk.danger_or_worse:
                    self.existing_causal_world_model.add_observation(
                        f"{self.domain} {frame.risk.level} risk",
                        "need stabilize control",
                        0.8,
                    )
                if self.belief.epistemic_uncertainty() >= 0.6:
                    self.existing_causal_world_model.add_observation(
                        f"{self.domain} high uncertainty",
                        "need reversible information gathering",
                        0.75,
                    )
            except Exception:
                pass

        if self.existing_skill_library is None:
            self.existing_skill_library = ServiceContainer.get("skill_library", default=None)
            if self.existing_skill_library is not None:
                self.skill_graph.load_from_macro_library(self.existing_skill_library)

    def _publish_action_decision(self, decision: ActionDecision) -> None:
        if self.existing_world_state is None:
            self.existing_world_state = ServiceContainer.get("world_state", default=None)
        if self.existing_world_state is not None:
            try:
                self.existing_world_state.record_event(
                    f"{self.domain} action {'approved' if decision.approved else 'blocked'}: {decision.original_action}",
                    source=f"environment:{self.domain}",
                    salience=0.65 if decision.approved else 0.85,
                    ttl=600.0,
                    approved=decision.approved,
                    action=decision.action,
                    reason=decision.reason,
                    vetoes=decision.vetoes,
                )
            except Exception:
                pass
