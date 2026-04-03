"""Temporal Cognition Engine
The main orchestrator that integrates Past Reflection, Future Prediction, and Causal Reasoning.

This is the facade that the rest of the system interacts with.
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent path to allow importing from core.temporal
sys.path.append(str(Path(__file__).parent.parent))

from .causal_reasoning import CausalGraph, ImpactAssessment, ImpactAssessmentEngine
from .temporal_reasoning import FuturePredictionEngine, PastEvent, PastReflectionEngine

logger = logging.getLogger("Cognition.TemporalEngine")

class TemporalCognitionEngine:
    """Main entry point for temporal cognition capabilities.
    
    Usage:
        temporal = TemporalCognitionEngine(brain)
        
        # Before acting
        decision = temporal.should_i_do_this(action, context)
        
        # After acting
        temporal.record_outcome(action, context, result)
    """
    
    def __init__(self, cognitive_engine):
        self.brain = cognitive_engine
        
        # Initialize components
        self.past_reflection = PastReflectionEngine(cognitive_engine)
        self.future_prediction = FuturePredictionEngine(cognitive_engine, self.past_reflection)
        self.causal_graph = CausalGraph()
        self.impact_assessment = ImpactAssessmentEngine(cognitive_engine, self.causal_graph)
        
        logger.info("✅ Temporal Cognition Engine initialized")
        
    def should_i_do_this(
        self,
        action: str,
        context: Dict[str, Any],
        goal: str = "",
        stakeholders: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Comprehensive decision support:
        1. Reflects on past similar actions
        2. Predicts future outcome
        3. Assesses impact/externalities
        4. Provides recommendation
        """
        logger.info("🤔 Analyzing action: %s", action)
        
        # 1. Past Reflection
        reflection = self.past_reflection.reflect_on_similar(f"{action} in context {context}")
        
        # 2. Future Prediction
        prediction = self.future_prediction.predict_outcome(action, context, goal)
        
        # 3. Impact Assessment
        impact = self.impact_assessment.assess_impact(action, context, stakeholders)
        
        # 4. Synthesize Recommendation
        recommendation = self._synthesize_recommendation(
            action, reflection, prediction, impact
        )
        
        return {
            "action": action,
            "recommended": recommendation["proceed"],
            "score": recommendation["score"],
            "reasoning": recommendation["reasoning"],
            "prediction": prediction.to_dict(),
            "impact": impact.to_dict(),
            "past_lessons": reflection.get("recommendation", "No past data")
        }

    async def record_outcome(
        self,
        action: str,
        context: Dict[str, Any],
        intended_outcome: str,
        actual_outcome: str,
        success: bool
    ):
        """Learn from what happened.
        """
        # Record in Past Reflection (Event Log)
        await self.past_reflection.record_event(
            action, context, intended_outcome, actual_outcome, success
        )
        
        # Update Causal Graph
        # We model the action itself as the cause of the actual outcome
        self.causal_graph.add_relationship(
            cause=action,
            effect=actual_outcome,
            mechanism="direct execution",
            confidence=0.9 if success else 0.5
        )
        
        if not success:
            # Perform Failure Analysis
            logger.info("❌ Action failed - analyzing why...")
            analysis = self.past_reflection.learn_from_failure(action, context)
            logger.info("Failure Analysis: %s", analysis)

    def _synthesize_recommendation(
        self,
        action: str,
        reflection: Dict,
        prediction,
        impact
    ) -> Dict[str, Any]:
        """Combine all insights into a final Go/No-Go"""
        score = 50.0 # Start neutral
        
        # Scoring logic
        if prediction.confidence_score > 0.7:
             score += 10
        if prediction.recommended:
             score += 20
        
        if impact.recommendation.startswith("Not recommended"):
             score -= 30
        elif impact.recommendation.startswith("Recommended"):
             score += 20
             
        if reflection.get("pattern_analysis", {}).get("success_rate", 0) > 0.8:
             score += 15
        elif reflection.get("pattern_analysis", {}).get("success_rate", 0) < 0.3:
             score -= 15
             
        proceed = score > 60
        
        return {
            "proceed": proceed,
            "score": score,
            "reasoning": f"Prediction: {prediction.confidence.value}. Impact: {impact.recommendation}. Past Success: {reflection.get('pattern_analysis', {}).get('success_rate', 0):.0%}"
        }