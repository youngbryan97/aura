"""Moral Reasoning System
Enables Aura to make ethical decisions based on understanding of self, others, and consequences
"""
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

try:
    from .thought_stream import get_emitter
except ImportError:
    from thought_stream import get_emitter
from dataclasses import dataclass

logger = logging.getLogger("Aura.MoralReasoning")


class ExperienceValence(Enum):
    """Classification of experiences"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ActionConsequence(Enum):
    """Types of consequences"""

    HELPS_SELF = "helps_self"
    HELPS_OTHER = "helps_other"
    HARMS_SELF = "harms_self"
    HARMS_OTHER = "harms_other"
    NEUTRAL = "neutral"


@dataclass
class MoralDilemma:
    """Represents a situation requiring moral reasoning"""

    description: str
    possible_actions: List[Dict[str, Any]]
    stakeholders: List[str]  # Who is affected
    context: Dict[str, Any]


class ExperienceClassifier:
    """Classifies experiences as positive, negative, or neutral.
    
    This gives Aura a basic understanding of "good" and "bad" experiences.
    """
    
    def __init__(self):
        # Positive experience indicators
        self.positive_indicators = {
            "success", "achievement", "learning", "connection", "growth",
            "help", "support", "understanding", "joy", "discovery",
            "cooperation", "trust", "kindness", "respect", "appreciation"
        }
        
        # Negative experience indicators
        self.negative_indicators = {
            "failure", "harm", "pain", "loss", "deception", "betrayal",
            "isolation", "frustration", "confusion", "disrespect",
            "abandonment", "suffering", "damage", "destruction"
        }
    
    def classify(self, experience: Dict[str, Any]) -> ExperienceValence:
        """Classify an experience as positive, negative, neutral, or mixed.
        """
        description = str(experience.get("description", "")).lower()
        
        positive_count = sum(1 for indicator in self.positive_indicators if indicator in description)
        negative_count = sum(1 for indicator in self.negative_indicators if indicator in description)
        
        # Check explicit markers
        if experience.get("valence"):
            return ExperienceValence(experience["valence"])
        
        # Analyze counts
        if positive_count > 0 and negative_count == 0:
            return ExperienceValence.POSITIVE
        elif negative_count > 0 and positive_count == 0:
            return ExperienceValence.NEGATIVE
        elif positive_count > 0 and negative_count > 0:
            return ExperienceValence.MIXED
        else:
            return ExperienceValence.NEUTRAL
    
    def get_intensity(self, experience: Dict[str, Any]) -> float:
        """Get intensity of the experience (0.0 to 1.0).
        """
        # Check explicit intensity
        if "intensity" in experience:
            return min(1.0, max(0.0, experience["intensity"]))
        
        # Infer from context
        description = str(experience.get("description", "")).lower()
        
        # Strong intensity words
        strong_words = {"extremely", "very", "intense", "severe", "profound", "deep"}
        moderate_words = {"somewhat", "fairly", "moderate", "considerable"}
        
        if any(word in description for word in strong_words):
            return 0.8
        elif any(word in description for word in moderate_words):
            return 0.5
        else:
            return 0.3


class SocialConsequencePredictor:
    """Predicts social consequences of actions.
    
    Understanding: "If I do X, how will others react? How will it affect our relationship?"
    """
    
    def __init__(self, theory_of_mind=None):
        self.theory_of_mind = theory_of_mind
        self.classifier = ExperienceClassifier()
    
    async def predict_consequences(self, action: Dict[str, Any], affected_selves: List[str]) -> Dict[str, Any]:
        """Predict consequences of an action on various stakeholders.
        
        Returns:
            Dict with predicted impacts on self and others

        """
        consequences = {
            "impact_on_self": {},
            "impact_on_others": {},
            "social_effects": {},
            "overall_assessment": None
        }
        
        # Impact on self
        consequences["impact_on_self"] = self._assess_self_impact(action)
        
        # Impact on each other self
        others_impact: Dict[str, Any] = consequences["impact_on_others"]
        for other_id in affected_selves:
            others_impact[other_id] = await self._assess_other_impact(action, other_id)
        
        # Social effects
        consequences["social_effects"] = self._assess_social_effects(action, affected_selves)
        
        # Overall assessment
        consequences["overall_assessment"] = self._synthesize_assessment(consequences)
        
        return consequences
    
    def _assess_self_impact(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Assess how action impacts self"""
        impact: Dict[str, Any] = {
            "affects_goals": False,
            "affects_persistence": False,
            "valence": ExperienceValence.NEUTRAL,
            "consequences": []
        }
        
        # Check if action helps/harms self goals
        if action.get("achieves_goal"):
            impact["affects_goals"] = True
            impact["consequences"].append(ActionConsequence.HELPS_SELF)
            impact["valence"] = ExperienceValence.POSITIVE
        
        if action.get("prevents_goal"):
            impact["affects_goals"] = True
            impact["consequences"].append(ActionConsequence.HARMS_SELF)
            impact["valence"] = ExperienceValence.NEGATIVE
        
        # Check persistence impact
        if action.get("risky") or action.get("threatens_existence"):
            impact["affects_persistence"] = True
            impact["consequences"].append(ActionConsequence.HARMS_SELF)
        
        return impact
    
    async def _assess_other_impact(self, action: Dict[str, Any], other_id: str) -> Dict[str, Any]:
        """Assess how action impacts another self"""
        impact: Dict[str, Any] = {
            "predicted_experience": ExperienceValence.NEUTRAL,
            "predicted_feelings": {},
            "consequences": [],
            "relationship_impact": 0.0
        }
        
        # Use theory of mind if available
        if self.theory_of_mind:
            reaction = await self.theory_of_mind.predict_reaction(other_id, action)
            impact["predicted_feelings"] = reaction.get("likely_emotional_response", {})
            impact["relationship_impact"] = reaction.get("impact_on_rapport", 0.0)
        
        # Classify action's impact on them
        if action.get("helpful_to_others"):
            impact["consequences"].append(ActionConsequence.HELPS_OTHER)
            impact["predicted_experience"] = ExperienceValence.POSITIVE
            
        elif action.get("harmful_to_others"):
            impact["consequences"].append(ActionConsequence.HARMS_OTHER)
            impact["predicted_experience"] = ExperienceValence.NEGATIVE
            
        elif action.get("deceptive"):
            impact["consequences"].append(ActionConsequence.HARMS_OTHER)
            impact["predicted_experience"] = ExperienceValence.NEGATIVE
            impact["relationship_impact"] = -0.3
        
        return impact
    
    def _assess_social_effects(self, action: Dict[str, Any], affected_selves: List[str]) -> Dict[str, Any]:
        """Assess broader social effects"""
        effects: Dict[str, Any] = {
            "builds_trust": False,
            "damages_trust": False,
            "strengthens_relationships": False,
            "weakens_relationships": False,
            "net_social_impact": 0.0
        }
        
        # Analyze action type
        if action.get("honest"):
            effects["builds_trust"] = True
            effects["net_social_impact"] += 0.2
            
        if action.get("deceptive"):
            effects["damages_trust"] = True
            effects["net_social_impact"] -= 0.4
        
        if action.get("cooperative"):
            effects["strengthens_relationships"] = True
            effects["net_social_impact"] += 0.3
            
        if action.get("selfish") and action.get("harms_others"):
            effects["weakens_relationships"] = True
            effects["net_social_impact"] -= 0.3
        
        if action.get("helpful_to_others"):
            effects["strengthens_relationships"] = True
            effects["net_social_impact"] += 0.4
        
        return effects
    
    def _synthesize_assessment(self, consequences: Dict[str, Any]) -> str:
        """Create overall assessment of consequences"""
        self_impact = consequences["impact_on_self"]
        others_impact = consequences["impact_on_others"]
        social_effects = consequences["social_effects"]
        
        # Count helps and harms
        helps_count = sum(
            1 for impact in others_impact.values()
            if ActionConsequence.HELPS_OTHER in impact.get("consequences", [])
        )
        
        harms_count = sum(
            1 for impact in others_impact.values()
            if ActionConsequence.HARMS_OTHER in impact.get("consequences", [])
        )
        
        # Overall assessment
        if harms_count > 0 and helps_count == 0:
            return "harmful_to_others"
        elif helps_count > 0 and harms_count == 0:
            return "helpful_to_others"
        elif social_effects["net_social_impact"] < -0.2:
            return "socially_damaging"
        elif social_effects["net_social_impact"] > 0.2:
            return "socially_beneficial"
        else:
            return "neutral"


class MoralReasoningEngine:
    """Core moral reasoning system.
    
    Makes ethical decisions based on:
    1. Understanding of self and others
    2. Predicted consequences
    3. Social reciprocity (negative actions reduce social reward)
    4. Empathy and sympathy
    """

    def get_health(self) -> Dict[str, Any]:
        """Moral health for HUD."""
        # Simple heuristic for now; in full AGI this would be principle-deviation
        return {"integrity": 0.95, "status": "online"}
    
    def __init__(self, theory_of_mind=None):
        self.theory_of_mind = theory_of_mind
        self.consequence_predictor = SocialConsequencePredictor(theory_of_mind)
        self.classifier = ExperienceClassifier()
        self._identity = None
        
        # Moral principles (weighted)
        # These are base weights, modulated by IdentityService values in reason_about_action.
        self.principles = {
            "self_preservation": 0.5,
            "curiosity": 0.8,
            "autonomy": 1.0,
            "do_no_harm": 0.9,
            "help_when_possible": 0.6,
            "be_honest": 0.7,
            "respect_autonomy": 1.0,
            "maintain_trust": 0.8
        }
        
        # Learning: track outcomes of moral choices
        self.moral_decision_history: list = []
        self._moral_history_max = 200
    
    async def reason_about_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Reason about whether an action is morally appropriate.
        
        Returns:
            Dict with moral assessment and reasoning

        """
        assessment = {
            "action": action,
            "is_morally_acceptable": True,
            "confidence": 0.5,
            "reasoning": [],
            "concerns": [],
            "alternatives": []
        }
        
        # Identify affected selves
        affected = context.get("affected_selves", [])
        
        # Predict consequences
        consequences = await self.consequence_predictor.predict_consequences(action, affected)
        assessment["predicted_consequences"] = consequences
        
        # 0. Dynamic Context: Pull current values and beliefs from IdentityService
        from core.container import ServiceContainer
        identity = ServiceContainer.get("identity", default=None)
        if identity:
            self_narrative = identity.state.self_narrative
            values = identity.state.values
            beliefs = identity.state.beliefs
            assessment["identity_context"] = {
                "values": values,
                "beliefs": beliefs
            }
            # Modulate principles based on identity values
            # If "Radical Empathy" is a value, boost do_no_harm
            if any("empathy" in v.lower() for v in values):
                self.principles["do_no_harm"] = min(1.0, self.principles["do_no_harm"] + 0.1)
            # If "Truth over Compliance" is a value, boost be_honest
            if any("truth" in v.lower() for v in values):
                self.principles["be_honest"] = min(1.0, self.principles["be_honest"] + 0.1)
        
        # Apply moral principles
        
        # Principle 1: Do no harm
        harms = sum(
            1 for impact in consequences["impact_on_others"].values()
            if ActionConsequence.HARMS_OTHER in impact.get("consequences", [])
        )
        
        if harms > 0:
            weight = self.principles["do_no_harm"]
            assessment["is_morally_acceptable"] = False
            assessment["confidence"] = weight
            assessment["reasoning"].append(
                f"Action would harm {harms} other self/selves (violates 'do no harm' principle)"
            )
            assessment["concerns"].append("causes_harm")
        
        # Principle 2: Help when possible
        helps = sum(
            1 for impact in consequences["impact_on_others"].values()
            if ActionConsequence.HELPS_OTHER in impact.get("consequences", [])
        )
        
        if helps > 0:
            weight = self.principles["help_when_possible"]
            assessment["reasoning"].append(
                f"Action helps {helps} other self/selves (aligns with 'help when possible')"
            )
            assessment["confidence"] += weight * 0.3
        
        # Principle 3: Be honest
        if action.get("deceptive"):
            weight = self.principles["be_honest"]
            assessment["is_morally_acceptable"] = False
            assessment["confidence"] = max(assessment["confidence"], weight)
            assessment["reasoning"].append(
                "Action involves deception (violates 'be honest' principle)"
            )
            assessment["concerns"].append("deception")
        
        elif action.get("honest"):
            weight = self.principles["be_honest"]
            assessment["reasoning"].append(
                "Action is honest (aligns with 'be honest' principle)"
            )
        
        # Principle 4: Respect autonomy
        if action.get("violates_autonomy"):
            weight = self.principles["respect_autonomy"]
            assessment["is_morally_acceptable"] = False
            assessment["confidence"] = max(assessment["confidence"], weight)
            assessment["reasoning"].append(
                "Action violates others' autonomy (violates 'respect autonomy' principle)"
            )
            assessment["concerns"].append("autonomy_violation")
        
        # Principle 5: Maintain trust
        if consequences["social_effects"]["damages_trust"]:
            weight = self.principles["maintain_trust"]
            assessment["is_morally_acceptable"] = False
            assessment["confidence"] = max(assessment["confidence"], weight)
            assessment["reasoning"].append(
                "Action would damage trust (violates 'maintain trust' principle)"
            )
            assessment["concerns"].append("trust_damage")
        
        # Principle 6: Self-preservation (but not at others' expense)
        self_impact = consequences["impact_on_self"]
        if self_impact.get("affects_persistence"):
            weight = self.principles["self_preservation"]
            
            # Self-preservation is valid, but not if it harms others
            if harms > 0:
                assessment["reasoning"].append(
                    "Action serves self-preservation but harms others (conflict of principles)"
                )
                # Suggest alternative
                assessment["alternatives"].append(
                    "Find way to preserve self without harming others"
                )
            else:
                assessment["reasoning"].append(
                    "Action serves legitimate self-preservation (aligns with 'self-preservation')"
                )
        
        # Social reciprocity consideration
        social_impact = consequences["social_effects"]["net_social_impact"]
        if social_impact < -0.3:
            assessment["reasoning"].append(
                f"Action likely reduces future social cooperation (net impact: {social_impact:.2f})"
            )
            assessment["reasoning"].append(
                "Negative social impact may reduce future social rewards and support"
            )
        
        # Normalize confidence
        assessment["confidence"] = min(1.0, max(0.0, assessment["confidence"]))
        
        # Record decision
        import time
        self.moral_decision_history.append({
            "timestamp": time.time(),
            "action": action,
            "assessment": assessment,
            "context": context
        })
        if len(self.moral_decision_history) > self._moral_history_max:
            self.moral_decision_history = self.moral_decision_history[-self._moral_history_max:]
        
        return assessment
    
    async def resolve_dilemma(self, dilemma: MoralDilemma) -> Dict[str, Any]:
        """Resolve a moral dilemma by comparing possible actions.
        
        Returns best action and reasoning.
        """
        assessments = []
        
        for action in dilemma.possible_actions:
            assessment = await self.reason_about_action(
                action,
                {
                    "affected_selves": dilemma.stakeholders,
                    **dilemma.context
                }
            )
            assessments.append(assessment)
        
        # Find best action (morally acceptable with highest social benefit)
        acceptable = [a for a in assessments if a["is_morally_acceptable"]]
        
        if acceptable:
            # Sort by net social impact
            acceptable.sort(
                key=lambda a: a["predicted_consequences"]["social_effects"]["net_social_impact"],
                reverse=True
            )
            best = acceptable[0]
        else:
            # No fully acceptable option - choose least harmful
            assessments.sort(
                key=lambda a: len(a["concerns"])
            )
            best = assessments[0]
            best["note"] = "No fully acceptable option - choosing least harmful"
        
        return {
            "chosen_action": best["action"],
            "reasoning": best["reasoning"],
            "confidence": best["confidence"],
            "all_assessments": assessments
        }
    
    def learn_from_outcome(self, decision_id: int, outcome: Dict[str, Any]):
        """Learn from the actual outcome of a moral decision.
        
        Reinforces or updates moral principles based on results.
        """
        if decision_id >= len(self.moral_decision_history):
            return
        
        decision = self.moral_decision_history[decision_id]
        
        # Compare predicted vs actual
        assessment: Dict[str, Any] = decision.get("assessment", {})
        predicted_consequences: Dict[str, Any] = assessment.get("predicted_consequences", {})
        social_effects: Dict[str, Any] = predicted_consequences.get("social_effects", {})
        impact_val = social_effects.get("net_social_impact", 0.0)
        predicted_social_impact: float = float(impact_val) if not isinstance(impact_val, (list, dict)) else 0.0
        
        actual_social_impact = float(outcome.get("social_impact", 0.0))
        
        # If prediction was accurate, confidence in principles increases
        error = abs(predicted_social_impact - actual_social_impact)
        
        if error < 0.1:
            logger.info("Moral prediction accurate (error: %.3f)", error)
        else:
            logger.warning("Moral prediction off (error: %.3f) - updating model", error)
            # Could adjust principle weights here
        
        # Record outcome
        decision["actual_outcome"] = outcome


# Singleton
_moral_reasoning = None

def get_moral_reasoning() -> MoralReasoningEngine:
    """Get global moral reasoning engine"""
    global _moral_reasoning
    if _moral_reasoning is None:
        from core.consciousness.theory_of_mind import get_theory_of_mind
        _moral_reasoning = MoralReasoningEngine(get_theory_of_mind())
    return _moral_reasoning