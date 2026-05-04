"""Structured postmortem analysis for cross-run learning.

Postmortems are general failure analysis, not game-specific obituary text.
They classify what failed, extract affordance/risk lessons, and produce
policy corrections that future runs can test against held-out traces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from .belief_state import EnvironmentBeliefState
from .affordance_schema import AffordanceKnowledgeBase, Affordance

logger = logging.getLogger("Aura.PostmortemAnalyzer")

@dataclass
class Lesson:
    rule: str
    applies_to_entities: List[str]
    confidence: float
    action: str = ""
    effect: str = ""

@dataclass
class PostmortemReport:
    cause_of_death: str
    contributing_factors: List[str]
    lessons: List[Lesson]
    strategic_corrections: List[str]
    failure_class: str = "unknown"
    policy_updates: Dict[str, Any] = field(default_factory=dict)

class PostmortemAnalyzer:
    """
    Analyzes environment sessions that ended in failure to extract lessons.
    """

    def __init__(self, brain: Any = None, knowledge_base: Optional[AffordanceKnowledgeBase] = None):
        self.brain = brain
        self.kb = knowledge_base or AffordanceKnowledgeBase(domain="generic")

    async def analyze_failure(self,
                              death_message: str,
                              final_state_summary: str,
                              recent_events: List[str],
                              belief: EnvironmentBeliefState) -> PostmortemReport:
        """
        Uses the cognitive engine to reflect on the failure and extract actionable rules.
        """
        logger.info(f"Initiating postmortem analysis for death: {death_message}")

        prompt = f"""[POSTMORTEM ANALYSIS]
You are analyzing a failure/death in an embodied environment session to extract lessons for future runs.

DEATH MESSAGE: {death_message}
FINAL STATE: {final_state_summary}
RECENT EVENTS:
{chr(10).join(recent_events)}

Extract specific, actionable lessons about the entities involved.
Format your response exactly as follows (JSON-like but readable):
CAUSE: <brief cause>
FACTORS:
- <factor 1>
- <factor 2>
LESSONS:
- ENTITY: <entity name>
  ACTION: <action that caused death, e.g. melee, quaff>
  EFFECT: <what happened>
  RULE: <actionable rule, e.g. "Do not melee floating eyes">
CORRECTIONS:
- <high level strategy correction>
"""
        if self.brain is None:
            return self._heuristic_postmortem(death_message, final_state_summary, recent_events, belief)

        try:
            from core.brain.cognitive_engine import ThinkingMode
            # Use SLOW mode for deep reflection
            response = await self.brain.think(prompt, mode=ThinkingMode.SLOW)
            content = response.content

            return self._parse_llm_postmortem(content)

        except Exception as e:
            logger.error(f"Postmortem analysis failed: {e}")
            return self._heuristic_postmortem(death_message, final_state_summary, recent_events, belief)

    def _parse_llm_postmortem(self, content: str) -> PostmortemReport:
        """Naively parses the expected output format into a report and updates KB."""
        cause = "Unknown"
        factors = []
        lessons = []
        corrections = []

        lines = content.split('\n')
        current_section = None

        current_lesson = {}

        for line in lines:
            line = line.strip()
            if not line: continue

            if line.startswith("CAUSE:"):
                cause = line.replace("CAUSE:", "").strip()
            elif line.startswith("FACTORS:"):
                current_section = "factors"
            elif line.startswith("LESSONS:"):
                current_section = "lessons"
            elif line.startswith("CORRECTIONS:"):
                current_section = "corrections"
            elif current_section == "factors" and line.startswith("-"):
                factors.append(line[1:].strip())
            elif current_section == "corrections" and line.startswith("-"):
                corrections.append(line[1:].strip())
            elif current_section == "lessons":
                if line.startswith("- ENTITY:"):
                    if current_lesson and "entity" in current_lesson:
                        self._commit_lesson_to_kb(current_lesson)
                        lessons.append(Lesson(
                            current_lesson.get("rule", ""),
                            [current_lesson["entity"]],
                            0.8,
                            action=current_lesson.get("action", ""),
                            effect=current_lesson.get("effect", ""),
                        ))
                    current_lesson = {"entity": line.replace("- ENTITY:", "").strip()}
                elif line.startswith("ACTION:") and current_lesson:
                    current_lesson["action"] = line.replace("ACTION:", "").strip()
                elif line.startswith("EFFECT:") and current_lesson:
                    current_lesson["effect"] = line.replace("EFFECT:", "").strip()
                elif line.startswith("RULE:") and current_lesson:
                    current_lesson["rule"] = line.replace("RULE:", "").strip()

        if current_lesson and "entity" in current_lesson:
            self._commit_lesson_to_kb(current_lesson)
            lessons.append(Lesson(
                current_lesson.get("rule", ""),
                [current_lesson["entity"]],
                0.8,
                action=current_lesson.get("action", ""),
                effect=current_lesson.get("effect", ""),
            ))

        failure_class = self._classify_failure(" ".join([cause, *factors, *corrections]))
        return PostmortemReport(
            cause,
            factors,
            lessons,
            corrections,
            failure_class=failure_class,
            policy_updates=self._policy_updates_for_class(failure_class),
        )

    def _commit_lesson_to_kb(self, lesson_data: Dict[str, str]):
        """Commits an extracted lesson into the structured affordance KB."""
        if "entity" not in lesson_data or "action" not in lesson_data:
            return

        aff = Affordance(
            entity=lesson_data["entity"],
            action=lesson_data["action"],
            preconditions=[],
            effects=[lesson_data.get("effect", "causes death")],
            risk_level=1.0,  # Highly risky if it caused a postmortem
            confidence=0.8,  # High confidence because we just experienced it
            source="postmortem_experience"
        )
        self.kb.add_learned_affordance(aff)
        logger.info(f"Learned new affordance from death: {aff.entity} -> {aff.action} -> {aff.effects[0]}")

    def _heuristic_postmortem(
        self,
        death_message: str,
        final_state_summary: str,
        recent_events: List[str],
        belief: EnvironmentBeliefState,
    ) -> PostmortemReport:
        text = " ".join([death_message, final_state_summary, *recent_events]).lower()
        failure_class = self._classify_failure(text)
        factors: List[str] = []
        corrections: List[str] = []
        lessons: List[Lesson] = []

        if "hp" in text or "hit" in text or "killed" in text or "die" in text:
            factors.append("Immediate survival/resource risk was not neutralized in time.")
            corrections.append("Raise priority of stabilize/retreat skills when critical resources fall.")
            lessons.append(
                Lesson(
                    "When a critical resource is low, suspend progress and choose recovery, retreat, or safe escape.",
                    ["critical resource"],
                    0.75,
                    action="continue_progress",
                    effect="can convert manageable danger into failure",
                )
            )
        if "hungry" in text or "weak" in text or "faint" in text or "starv" in text:
            factors.append("Resource depletion/hunger was allowed to persist.")
            corrections.append("Treat worsening resource messages as strategic blockers, not background noise.")
        if "unknown" in text or belief.epistemic_uncertainty() > 0.6:
            factors.append("High uncertainty was present near failure.")
            corrections.append("Prefer reversible information gathering under high uncertainty.")
            lessons.append(
                Lesson(
                    "Do not use unknown irreversible actions while risk or uncertainty is high.",
                    ["unknown object"],
                    0.7,
                    action="use",
                    effect="may trigger unknown harmful effects",
                )
            )
        if not factors:
            factors.append("Failure did not match a known taxonomy; preserve trace for human/LLM review.")
            corrections.append("Create a regression trace and require explicit classification before policy promotion.")

        for lesson in lessons:
            self._commit_lesson_to_kb(
                {
                    "entity": lesson.applies_to_entities[0],
                    "action": lesson.action or "act",
                    "effect": lesson.effect or lesson.rule,
                    "rule": lesson.rule,
                }
            )

        return PostmortemReport(
            cause_of_death=death_message or "Unknown failure",
            contributing_factors=factors,
            lessons=lessons,
            strategic_corrections=corrections,
            failure_class=failure_class,
            policy_updates=self._policy_updates_for_class(failure_class),
        )

    @staticmethod
    def _classify_failure(text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("prompt", "menu", "--more--", "what do you want")):
            return "interface_modal_failure"
        if any(token in lowered for token in ("hungry", "weak", "faint", "starv")):
            return "resource_mismanagement"
        if any(token in lowered for token in ("unknown", "uncertain", "unidentified")):
            return "uncertainty_mismanagement"
        if any(token in lowered for token in ("hit", "damage", "killed", "die", "hp")):
            return "tactical_survival_failure"
        return "unknown"

    @staticmethod
    def _policy_updates_for_class(failure_class: str) -> Dict[str, Any]:
        updates = {
            "interface_modal_failure": {
                "prompt_resolution_priority": 0.98,
                "block_normal_actions_during_prompts": True,
            },
            "resource_mismanagement": {
                "resource_recovery_priority": 0.9,
                "progress_blocked_under_resource_depletion": True,
            },
            "uncertainty_mismanagement": {
                "irreversible_unknown_action_threshold": 0.25,
                "prefer_reversible_tests": True,
            },
            "tactical_survival_failure": {
                "critical_resource_interrupt_threshold": 0.5,
                "prefer_retreat_before_emergency": True,
            },
        }
        return updates.get(failure_class, {"requires_review": True})
