"""Causal Reasoning & Impact Assessment

Enables Aura to:
1. Understand cause-effect relationships
2. Model second-order effects (ripple effects)
3. Assess social and environmental impacts
4. Consider effects on other individuals
"""
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("Cognition.CausalReasoning")


@dataclass
class CausalRelationship:
    """A cause-effect relationship"""

    cause: str
    effect: str
    confidence: float  # 0.0-1.0
    mechanism: str  # How cause leads to effect
    conditions: List[str]  # Under what conditions
    evidence_count: int  # How many times observed
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ImpactAssessment:
    """Assessment of an action's impacts"""

    action: str
    direct_effects: List[str]  # Immediate consequences
    second_order_effects: List[str]  # Ripple effects
    affected_parties: List[Dict[str, Any]]  # Who/what is impacted
    positive_impacts: List[str]
    negative_impacts: List[str]
    neutral_impacts: List[str]
    severity: str  # 'minor', 'moderate', 'major', 'critical'
    reversibility: str  # 'reversible', 'difficult', 'irreversible'
    recommendation: str
    
    def to_dict(self):
        return asdict(self)


class CausalGraph:
    """Maintains a graph of cause-effect relationships.
    """
    
    def __init__(self):
        # Edges: cause -> [effects]
        self.graph: Dict[str, List[CausalRelationship]] = defaultdict(list)
        
        # Reverse index: effect -> [causes]
        self.reverse_graph: Dict[str, List[CausalRelationship]] = defaultdict(list)
        
        logger.info("CausalGraph initialized")
    
    def add_relationship(
        self,
        cause: str,
        effect: str,
        mechanism: str,
        confidence: float = 0.5,
        conditions: Optional[List[str]] = None
    ):
        """Add or update a causal relationship"""
        # Check if relationship already exists
        existing = None
        for rel in self.graph[cause]:
            if rel.effect == effect:
                existing = rel
                break
        
        if existing:
            # Update confidence (average with new observation)
            existing.confidence = (existing.confidence * existing.evidence_count + confidence) / (existing.evidence_count + 1)
            existing.evidence_count += 1
            logger.debug("Updated: %s -> %s (confidence: %.2f)", cause, effect, existing.confidence)
        else:
            # Create new relationship
            rel = CausalRelationship(
                cause=cause,
                effect=effect,
                confidence=confidence,
                mechanism=mechanism,
                conditions=conditions or [],
                evidence_count=1
            )
            
            self.graph[cause].append(rel)
            self.reverse_graph[effect].append(rel)
            logger.info("Added: %s -> %s", cause, effect)
    
    def predict_effects(self, cause: str, depth: int = 2) -> List[str]:
        """Predict effects of a cause, including ripple effects.
        
        Args:
            cause: The initial cause
            depth: How many levels of effects to explore
            
        Returns:
            List of predicted effects

        """
        effects = set()
        visited = set()
        
        def explore(current_cause: str, current_depth: int):
            if current_depth > depth or current_cause in visited:
                return
            
            visited.add(current_cause)
            
            for rel in self.graph[current_cause]:
                if rel.confidence > 0.3:  # Only high-confidence relationships
                    effects.add(rel.effect)
                    # Recurse for ripple effects
                    explore(rel.effect, current_depth + 1)
        
        explore(cause, 0)
        return list(effects)
    
    def explain_effect(self, effect: str) -> List[Dict[str, Any]]:
        """Explain why an effect occurred (backward reasoning).
        
        Args:
            effect: The effect to explain
            
        Returns:
            List of possible causal explanations

        """
        explanations = []
        
        for rel in self.reverse_graph[effect]:
            if rel.confidence > 0.3:
                explanations.append({
                    "cause": rel.cause,
                    "mechanism": rel.mechanism,
                    "confidence": rel.confidence,
                    "conditions": rel.conditions
                })
        
        # Sort by confidence
        explanations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return explanations
    
    def find_path(self, from_cause: str, to_effect: str) -> Optional[List[str]]:
        """Find causal path from cause to effect"""
        visited = set()
        queue = [(from_cause, [from_cause])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current == to_effect:
                return path
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for rel in self.graph[current]:
                if rel.confidence > 0.3:
                    new_path = path + [rel.effect]
                    queue.append((rel.effect, new_path))
        
        return None  # No path found
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph structure"""
        return {
            "nodes": list(set(list(self.graph.keys()) + list(self.reverse_graph.keys()))),
            "edges": [
                rel.to_dict()
                for rels in self.graph.values()
                for rel in rels
            ]
        }


class ImpactAssessmentEngine:
    """Assesses the potential impacts of actions on various parties.
    
    Considers:
    - Direct effects
    - Second-order effects (ripple effects)
    - Social impacts (other people/agents)
    - Environmental impacts (system/world state)
    """
    
    def __init__(
        self,
        cognitive_engine,
        causal_graph: CausalGraph
    ):
        self.brain = cognitive_engine
        self.causal_graph = causal_graph
        
        logger.info("ImpactAssessmentEngine initialized")
    
    def assess_impact(
        self,
        action: str,
        context: Dict[str, Any],
        stakeholders: Optional[List[str]] = None
    ) -> ImpactAssessment:
        """Comprehensive impact assessment of an action.
        
        Args:
            action: Action being considered
            context: Current situation/environment
            stakeholders: Who/what might be affected
            
        Returns:
            ImpactAssessment object

        """
        logger.info("Assessing impact of: %s", action)
        
        # Use causal graph to predict effects
        predicted_effects = self.causal_graph.predict_effects(action, depth=3)
        
        # Use LLM for deeper analysis
        impact_analysis = self._analyze_with_llm(action, context, stakeholders, predicted_effects)
        
        # Classify impacts
        positive, negative, neutral = self._classify_impacts(impact_analysis)
        
        # Assess severity
        severity = self._assess_severity(impact_analysis)
        
        # Assess reversibility
        reversibility = self._assess_reversibility(impact_analysis)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            action, positive, negative, severity, reversibility
        )
        
        assessment = ImpactAssessment(
            action=action,
            direct_effects=impact_analysis.get('direct_effects', []),
            second_order_effects=impact_analysis.get('second_order_effects', []),
            affected_parties=impact_analysis.get('affected_parties', []),
            positive_impacts=positive,
            negative_impacts=negative,
            neutral_impacts=neutral,
            severity=severity,
            reversibility=reversibility,
            recommendation=recommendation
        )
        
        # Update causal graph with new observations
        self._update_causal_graph(action, assessment)
        
        return assessment
    
    def compare_impacts(
        self,
        actions: List[str],
        context: Dict[str, Any],
        stakeholders: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare impacts of multiple possible actions.
        
        Returns ranking and recommendation.
        """
        logger.info("Comparing impacts of %s actions", len(actions))
        
        assessments = []
        for action in actions:
            assessment = self.assess_impact(action, context, stakeholders)
            assessments.append(assessment)
        
        # Rank by impact favorability
        ranked = self._rank_by_impact(assessments)
        
        return {
            "assessments": [a.to_dict() for a in assessments],
            "ranking": ranked,
            "recommendation": ranked[0]['action'] if ranked else None
        }
    
    def _analyze_with_llm(
        self,
        action: str,
        context: Dict[str, Any],
        stakeholders: Optional[List[str]],
        predicted_effects: List[str]
    ) -> Dict[str, Any]:
        """Use LLM to analyze impacts"""
        stakeholder_list = stakeholders if stakeholders else ["user", "system", "environment"]
        
        prompt = f"""Analyze the potential impacts of this action.

Action: {action}
Context: {json.dumps(context)}
Stakeholders: {', '.join(stakeholder_list)}

Predicted effects from causal model:
{', '.join(predicted_effects) if predicted_effects else 'None modeled yet'}

Identify:
1. Direct effects (immediate, first-order)
2. Second-order effects (ripple effects, consequences of consequences)
3. Affected parties (who/what is impacted and how)
4. Positive impacts
5. Negative impacts
6. Unintended consequences

Return JSON:
{{
  "direct_effects": ["effect1", "effect2"],
  "second_order_effects": ["ripple1", "ripple2"],
  "affected_parties": [
    {{"party": "user", "impact": "description", "sentiment": "positive/negative/neutral"}}
  ],
  "positive_impacts": ["good1", "good2"],
  "negative_impacts": ["bad1", "bad2"],
  "unintended_consequences": ["surprise1", "surprise2"]
}}"""
        
        try:
            thought = self.brain.think(prompt)
            response = thought.content.strip()
            # Hardening: Use robust extraction
            from core.utils.json_utils import extract_json
            return extract_json(response) or {}
        except Exception as e:
            logger.error("Impact analysis failed: %s", e)
            return {
                "direct_effects": [],
                "second_order_effects": [],
                "affected_parties": [],
                "positive_impacts": [],
                "negative_impacts": [],
                "unintended_consequences": []
            }
    
    def _classify_impacts(self, analysis: Dict[str, Any]) -> tuple:
        """Separate positive, negative, and neutral impacts"""
        positive = analysis.get('positive_impacts', [])
        negative = analysis.get('negative_impacts', [])
        
        # Classify affected parties
        for party in analysis.get('affected_parties', []):
            sentiment = party.get('sentiment', 'neutral')
            impact_desc = f"{party.get('party')}: {party.get('impact')}"
            
            if sentiment == 'positive':
                positive.append(impact_desc)
            elif sentiment == 'negative':
                negative.append(impact_desc)
        
        # Everything else is neutral
        direct = analysis.get('direct_effects', [])
        second_order = analysis.get('second_order_effects', [])
        
        neutral = [e for e in direct + second_order if e not in positive and e not in negative]
        
        return positive, negative, neutral
    
    def _assess_severity(self, analysis: Dict[str, Any]) -> str:
        """Assess overall severity of impacts"""
        negative = len(analysis.get('negative_impacts', []))
        positive = len(analysis.get('positive_impacts', []))
        affected = len(analysis.get('affected_parties', []))
        
        # Simple heuristic
        impact_score = abs(negative - positive) + affected
        
        if impact_score > 10:
            return 'critical'
        elif impact_score > 6:
            return 'major'
        elif impact_score > 3:
            return 'moderate'
        else:
            return 'minor'
    
    def _assess_reversibility(self, analysis: Dict[str, Any]) -> str:
        """Assess whether impacts can be reversed"""
        # Keywords indicating irreversibility
        irreversible_keywords = [
            'permanent', 'delete', 'remove', 'destroy', 'irreversible', 'cannot undo'
        ]
        
        difficult_keywords = [
            'difficult', 'hard to reverse', 'challenging to undo'
        ]
        
        # Check all impacts
        all_impacts = (
            analysis.get('direct_effects', []) +
            analysis.get('second_order_effects', []) +
            analysis.get('negative_impacts', [])
        )
        
        text = ' '.join(all_impacts).lower()
        
        if any(kw in text for kw in irreversible_keywords):
            return 'irreversible'
        elif any(kw in text for kw in difficult_keywords):
            return 'difficult'
        else:
            return 'reversible'
    
    def _generate_recommendation(
        self,
        action: str,
        positive: List[str],
        negative: List[str],
        severity: str,
        reversibility: str
    ) -> str:
        """Generate actionable recommendation"""
        pos_count = len(positive)
        neg_count = len(negative)
        
        if neg_count == 0 and pos_count > 0:
            return f"Recommended: No significant negative impacts detected"
        
        if neg_count > pos_count:
            if reversibility == 'irreversible':
                return f"Not recommended: Irreversible negative impacts outweigh benefits"
            elif severity in ['critical', 'major']:
                return f"Use caution: {severity} negative impacts identified"
            else:
                return f"Consider alternatives: Negative impacts present but reversible"
        
        if pos_count > neg_count:
            if severity == 'minor':
                return f"Recommended: Benefits outweigh minor risks"
            else:
                return f"Recommended with monitoring: Significant positive impacts expected"
        
        return f"Neutral: Impacts balanced, decision depends on priorities"
    
    def _rank_by_impact(self, assessments: List[ImpactAssessment]) -> List[Dict[str, Any]]:
        """Rank actions by favorability of their impacts"""
        ranked = []
        
        severity_scores = {
            'minor': 1,
            'moderate': 2,
            'major': 3,
            'critical': 4
        }
        
        for assessment in assessments:
            # Positive score
            score = len(assessment.positive_impacts) * 2
            
            # Negative penalty
            score -= len(assessment.negative_impacts) * 3
            
            # Severity penalty
            score -= severity_scores.get(assessment.severity, 0)
            
            # Reversibility bonus
            if assessment.reversibility == 'reversible':
                score += 2
            elif assessment.reversibility == 'irreversible':
                score -= 3
            
            ranked.append({
                "action": assessment.action,
                "score": score,
                "severity": assessment.severity,
                "recommendation": assessment.recommendation
            })
        
        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked
    
    def _update_causal_graph(self, action: str, assessment: ImpactAssessment):
        """Update causal graph with observed effects"""
        # Add direct effects
        for effect in assessment.direct_effects:
            self.causal_graph.add_relationship(
                cause=action,
                effect=effect,
                mechanism="direct",
                confidence=0.7
            )
        
        # Add second-order effects
        for effect in assessment.second_order_effects:
            # These are less certain
            self.causal_graph.add_relationship(
                cause=action,
                effect=effect,
                mechanism="indirect",
                confidence=0.4
            )