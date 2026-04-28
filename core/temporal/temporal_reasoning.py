"""Temporal Reasoning Engine - Past Reflection & Future Prediction

Enables Aura to:
1. Reflect on past events and learn from outcomes
2. Predict future outcomes of potential actions
3. Model confidence levels (certain/maybe/uncertain)
4. Consider externalities and impacts
"""
from core.runtime.errors import record_degradation
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Cognition.TemporalReasoning")


class ConfidenceLevel(Enum):
    """Confidence in predicted outcomes"""

    CERTAIN = "certain"        # 90%+ confidence
    LIKELY = "likely"          # 70-90% confidence
    MAYBE = "maybe"            # 40-70% confidence
    UNLIKELY = "unlikely"      # 10-40% confidence
    UNCERTAIN = "uncertain"    # <10% confidence


class OutcomeType(Enum):
    """Types of outcomes"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class PastEvent:
    """Record of a past action and its outcome"""

    timestamp: float
    action: str
    context: Dict[str, Any]
    intended_outcome: str
    actual_outcome: str
    outcome_type: OutcomeType
    success: bool
    externalities: List[str]  # Unintended consequences
    lessons_learned: List[str]
    
    def to_dict(self):
        d = asdict(self)
        d['outcome_type'] = self.outcome_type.value
        return d


@dataclass
class FuturePrediction:
    """Predicted outcome of a potential action"""

    action: str
    predicted_outcomes: List[Dict[str, Any]]  # Multiple possible outcomes
    confidence: ConfidenceLevel
    confidence_score: float  # 0.0-1.0
    expected_externalities: List[str]
    risks: List[str]
    opportunities: List[str]
    recommended: bool
    reasoning: str
    
    def to_dict(self):
        d = asdict(self)
        d['confidence'] = self.confidence.value
        return d


class PastReflectionEngine:
    """Analyzes past events to extract lessons and patterns.
    
    "Those who cannot remember the past are condemned to repeat it."
    """
    
    def __init__(
        self,
        cognitive_engine,
        memory_db: str = "autonomy_engine/data/temporal_memory.jsonl"
    ):
        self.brain = cognitive_engine
        self.memory_path = Path(memory_db)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.past_events: List[PastEvent] = []
        self.max_cache = 1000
        
        # Load existing memories
        self._load_past_events()
        
        logger.info("PastReflectionEngine initialized with %d past events", len(self.past_events))
    
    async def record_event(
        self,
        action: str,
        context: Dict[str, Any],
        intended_outcome: str,
        actual_outcome: str,
        success: bool
    ) -> PastEvent:
        """Record an event that just happened.
        
        Args:
            action: What action was taken
            context: Situation/environment when action occurred
            intended_outcome: What was hoped to happen
            actual_outcome: What actually happened
            success: Whether outcome matched intent
            
        Returns:
            PastEvent object

        """
        logger.info("Recording past event: %s -> %s", action, actual_outcome)
        
        # Determine outcome type
        outcome_type = self._classify_outcome(actual_outcome, success)
        
        # Substring matching for robustness against tool name variations
        # Substring matching for robustness against tool name variations
        skip_keywords = ["status", "cardio", "health", "diag", "check", "ping", "verify", "speak", "say", "convers", "search", "read", "lookup"]
        if any(k in action.lower() for k in skip_keywords):
             logger.info("Skipping deep temporal analysis for routine action: %s", action)
             externalities = []
             lessons = [f"Routine execution of {action} completed"]
        else:
            # Run deep analysis in parallel to save time
            # Note: We pass empty list for externalities to _extract_lessons to allow parallelism
            # This is a trade-off for speed (saving ~50% time)
            try:
                ext_coro = self._identify_externalities(action, intended_outcome, actual_outcome, context)
                less_coro = self._extract_lessons(action, intended_outcome, actual_outcome, success, [])
                
                externalities, lessons = await asyncio.gather(ext_coro, less_coro)
            except Exception as e:
                record_degradation('temporal_reasoning', e)
                logger.error("Error in temporal analysis: %s", e, exc_info=True)
                externalities = []
                lessons = []
        
        # Create event record
        event = PastEvent(
            timestamp=time.time(),
            action=action,
            context=context,
            intended_outcome=intended_outcome,
            actual_outcome=actual_outcome,
            outcome_type=outcome_type,
            success=success,
            externalities=externalities,
            lessons_learned=lessons
        )
        
        # Store
        self.past_events.append(event)
        if len(self.past_events) > self.max_cache:
            self.past_events = self.past_events[-self.max_cache:]
        
        await self._persist_event(event)
        
        return event
    
    async def reflect_on_similar(self, current_situation: str) -> Dict[str, Any]:
        """Reflect on past events similar to current situation.
        
        Args:
            current_situation: Description of current situation
            
        Returns:
            Reflection with relevant past events and lessons

        """
        logger.info("Reflecting on similar situations to: %s", current_situation[:100])
        
        # Find similar past events
        similar = self._find_similar_events(current_situation)
        
        if not similar:
            return {
                "found_similar": False,
                "recommendation": "No similar past experience to guide this decision"
            }
        
        # Analyze patterns
        analysis = self._analyze_event_pattern(similar)
        
        # Generate reflection using LLM
        reflection = await self._generate_reflection(current_situation, similar, analysis)
        
        return {
            "found_similar": True,
            "similar_events": [e.to_dict() for e in similar[:5]],
            "pattern_analysis": analysis,
            "reflection": reflection,
            "recommendation": self._extract_recommendation(reflection)
        }
    
    async def learn_from_failure(self, failed_action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deep analysis of why something failed.
        
        Args:
            failed_action: Action that failed
            context: Context of failure
            
        Returns:
            Analysis with corrective strategies

        """
        logger.info("Learning from failure: %s", failed_action)
        
        # Find all past failures of similar actions
        similar_failures = [
            e for e in self.past_events
            if not e.success and self._is_similar_action(e.action, failed_action)
        ]
        
        if not similar_failures:
            return {"lessons": ["First failure of this type - establish baseline"]}
        
        # Analyze failure pattern
        prompt = f"""Analyze this pattern of failures to identify root cause.

Current Failure:
Action: {failed_action}
Context: {json.dumps(context)}

Past Similar Failures:
{self._format_events(similar_failures[:5])}

Identify:
1. Common factors across failures
2. Root cause (not surface symptoms)
3. Corrective strategy
4. Success criteria for next attempt

Return JSON:
{{
  "root_cause": "explanation",
  "common_factors": ["factor1", "factor2"],
  "corrective_strategy": "specific approach",
  "success_criteria": ["criterion1", "criterion2"]
}}"""
        
        try:
            thought = await self.brain.think(prompt)
            response = thought.content.strip()
            analysis = json.loads(response.strip('```json').strip('```'))
            
            return {
                "failure_count": len(similar_failures),
                **analysis
            }
        except Exception as e:
            record_degradation('temporal_reasoning', e)
            logger.error("Failure analysis error: %s", e, exc_info=True)
            return {"error": str(e)}
    
    def _classify_outcome(self, outcome: str, success: bool) -> OutcomeType:
        """Classify whether outcome was positive/negative/neutral"""
        if not success:
            return OutcomeType.NEGATIVE
        
        # Use sentiment analysis on outcome description
        positive_words = ['success', 'good', 'better', 'improved', 'solved', 'fixed']
        negative_words = ['failed', 'worse', 'broken', 'error', 'problem']
        
        outcome_lower = outcome.lower()
        pos_count = sum(1 for word in positive_words if word in outcome_lower)
        neg_count = sum(1 for word in negative_words if word in outcome_lower)
        
        if pos_count > neg_count:
            return OutcomeType.POSITIVE
        elif neg_count > pos_count:
            return OutcomeType.NEGATIVE
        elif pos_count > 0 and neg_count > 0:
            return OutcomeType.MIXED
        else:
            return OutcomeType.NEUTRAL
    
    async def _identify_externalities(
        self,
        action: str,
        intended: str,
        actual: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Identify unintended consequences using LLM"""
        prompt = f"""Identify unintended consequences of this action.

Action: {action}
Intended outcome: {intended}
Actual outcome: {actual}
Context: {json.dumps(context)}

List any unintended consequences (good or bad) that weren't part of the original goal.
Return as JSON array: ["consequence1", "consequence2"]
Return [] if no externalities.
Ensure valid JSON."""
        
        try:
            thought = await self.brain.think(prompt)
            response = thought.content.strip()
            # Clean markdown
            response = response.strip('```json').strip('```').strip()
            # Handle potential non-json output
            if not response.startswith('['):
                 # Fallback if LLM chats
                 import re
                 match = re.search(r'\[.*\]', response, re.DOTALL)
                 if match: response = match.group(0)
                 else: return []

            externalities = json.loads(response)
            return externalities if isinstance(externalities, list) else []
        except Exception as e:
            record_degradation('temporal_reasoning', e)
            logger.debug("Failed to identify externalities: %s", e)
            return []
    
    async def _extract_lessons(
        self,
        action: str,
        intended: str,
        actual: str,
        success: bool,
        externalities: List[str]
    ) -> List[str]:
        """Extract actionable lessons from event"""
        prompt = f"""Extract actionable lessons from this event.

Action: {action}
Intended: {intended}
Actual: {actual}
Success: {success}
Externalities: {json.dumps(externalities)}

Generate 2-3 specific, actionable lessons learned.
Format as JSON array: ["lesson1", "lesson2"]"""
        
        try:
            thought = await self.brain.think(prompt)
            response = thought.content.strip()
            response = response.strip('```json').strip('```').strip()
            # Handle potential non-json output
            if not response.startswith('['):
                 import re
                 match = re.search(r'\[.*\]', response, re.DOTALL)
                 if match: response = match.group(0)
                 else: return [f"Document {'success' if success else 'failure'} of {action}"]

            lessons = json.loads(response)
            return lessons if isinstance(lessons, list) else []
        except Exception as e:
            record_degradation('temporal_reasoning', e)
            logger.debug("Failed to extract lessons: %s", e)
            return [f"Document {'success' if success else 'failure'} of {action}"]
    
    def _find_similar_events(self, situation: str, limit: int = 10) -> List[PastEvent]:
        """Find past events similar to current situation"""
        # Fast associative keyword clustering (heavy embeddings are deferred to Vector Memory)
        
        keywords = set(situation.lower().split())
        
        scored_events = []
        for event in self.past_events:
            event_text = f"{event.action} {event.context} {event.actual_outcome}".lower()
            event_keywords = set(event_text.split())
            
            # Jaccard similarity
            intersection = keywords & event_keywords
            union = keywords | event_keywords
            similarity = len(intersection) / len(union) if union else 0
            
            if similarity > 0.1:  # Threshold
                scored_events.append((similarity, event))
        
        # Sort by similarity
        scored_events.sort(reverse=True, key=lambda x: x[0])
        
        return [event for _, event in scored_events[:limit]]
    
    def _analyze_event_pattern(self, events: List[PastEvent]) -> Dict[str, Any]:
        """Analyze patterns in a set of events"""
        if not events:
            return {}
        
        total = len(events)
        successes = sum(1 for e in events if e.success)
        
        # Outcome type distribution
        outcome_dist = {}
        for event in events:
            outcome_dist[event.outcome_type.value] = outcome_dist.get(event.outcome_type.value, 0) + 1
        
        # Common externalities
        all_externalities = []
        for event in events:
            all_externalities.extend(event.externalities)
        
        externality_counts = {}
        for ext in all_externalities:
            externality_counts[ext] = externality_counts.get(ext, 0) + 1
        
        common_externalities = sorted(
            externality_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_events": total,
            "success_rate": successes / total if total > 0 else 0,
            "outcome_distribution": outcome_dist,
            "common_externalities": [ext for ext, _ in common_externalities]
        }
    
    async def _generate_reflection(
        self,
        current_situation: str,
        similar_events: List[PastEvent],
        analysis: Dict[str, Any]
    ) -> str:
        """Generate reflective analysis using LLM"""
        prompt = f"""Reflect on past experience to guide current decision.

Current Situation:
{current_situation}

Past Similar Experiences:
{self._format_events(similar_events[:3])}

Pattern Analysis:
- Success rate: {analysis.get('success_rate', 0)*100:.0f}%
- Common externalities: {', '.join(analysis.get('common_externalities', []))}

Based on this history, provide:
1. What worked in the past
2. What didn't work
3. Key lessons to apply now
4. Recommended approach

Be specific and actionable."""
        
        try:
            thought = await self.brain.think(prompt)
            return thought.content
        except Exception as e:
            record_degradation('temporal_reasoning', e)
            return f"Reflection unavailable: {e}"
    
    def _extract_recommendation(self, reflection: str) -> str:
        """Extract key recommendation from reflection"""
        lines = reflection.split('\n')
        for line in lines:
            if 'recommend' in line.lower():
                return line.strip()
        return "Consider past patterns when deciding"
    
    def _is_similar_action(self, action1: str, action2: str) -> bool:
        """Check if two actions are similar"""
        # Simple word overlap for now
        words1 = set(action1.lower().split())
        words2 = set(action2.lower().split())
        overlap = words1 & words2
        return len(overlap) >= 2
    
    def _format_events(self, events: List[PastEvent]) -> str:
        """Format events for LLM prompts"""
        formatted = []
        for i, event in enumerate(events, 1):
            formatted.append(f"""
Event {i}:
  Action: {event.action}
  Intended: {event.intended_outcome}
  Actual: {event.actual_outcome}
  Success: {event.success}
  Externalities: {', '.join(event.externalities) if event.externalities else 'None'}
  Lessons: {', '.join(event.lessons_learned)}
""")
        return '\n'.join(formatted)
    
    async def _persist_event(self, event: PastEvent):
        """Save event to disk without blocking the event loop"""
        def write_sync():
            try:
                with open(self.memory_path, 'a') as f:
                    f.write(json.dumps(event.to_dict()) + '\n')
            except Exception as e:
                record_degradation('temporal_reasoning', e)
                logger.error("Failed to persist event: %s", e)
        await asyncio.to_thread(write_sync)
    
    def _load_past_events(self):
        """Load past events from disk"""
        if not self.memory_path.exists():
            return
        
        try:
            with open(self.memory_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Convert outcome_type back to enum
                    if 'outcome_type' in data:
                         data['outcome_type'] = OutcomeType(data['outcome_type'])
                    event = PastEvent(**data)
                    self.past_events.append(event)
            
            # Keep only recent events in memory
            if len(self.past_events) > self.max_cache:
                self.past_events = self.past_events[-self.max_cache:]
            
        except Exception as e:
            record_degradation('temporal_reasoning', e)
            logger.error("Failed to load past events: %s", e)


class FuturePredictionEngine:
    """Predicts outcomes of potential actions before taking them.
    
    "The best way to predict the future is to invent it."
    But before inventing, simulate it.
    """
    
    def __init__(
        self,
        cognitive_engine,
        past_reflection: PastReflectionEngine
    ):
        self.brain = cognitive_engine
        self.past = past_reflection  # Use past to inform future
        
        logger.info("FuturePredictionEngine initialized")
    
    async def predict_outcome(
        self,
        action: str,
        context: Dict[str, Any],
        goal: Optional[str] = None
    ) -> FuturePrediction:
        """Predict what will happen if action is taken.
        
        Args:
            action: Action being considered
            context: Current situation
            goal: What's trying to be achieved (optional)
            
        Returns:
            FuturePrediction with confidence levels

        """
        logger.info("Predicting outcome for: %s", action)
        
        # Check past for similar situations
        past_reflection = self.past.reflect_on_similar(f"{action} in context {context}")
        
        # Generate prediction using LLM
        prediction_data = await self._generate_prediction(action, context, goal, past_reflection)
        
        # Calculate confidence
        confidence, confidence_score = self._calculate_confidence(prediction_data, past_reflection)
        
        # Assess recommendation
        recommended = self._should_recommend(prediction_data, confidence_score)
        
        prediction = FuturePrediction(
            action=action,
            predicted_outcomes=prediction_data.get('outcomes', []),
            confidence=confidence,
            confidence_score=confidence_score,
            expected_externalities=prediction_data.get('externalities', []),
            risks=prediction_data.get('risks', []),
            opportunities=prediction_data.get('opportunities', []),
            recommended=recommended,
            reasoning=prediction_data.get('reasoning', '')
        )
        
        logger.info("Prediction: %s confidence, recommended=%s", confidence.value, recommended)
        
        return prediction
    
    async def compare_options(
        self,
        options: List[str],
        context: Dict[str, Any],
        goal: str
    ) -> Dict[str, Any]:
        """Compare multiple possible actions.
        
        Args:
            options: List of possible actions
            context: Current situation
            goal: What's trying to be achieved
            
        Returns:
            Comparison with ranked recommendations

        """
        logger.info("Comparing %d options for goal: %s", len(options), goal)
        
        predictions = []
        for option in options:
            pred = await self.predict_outcome(option, context, goal)
            predictions.append(pred)
        
        # Rank by confidence and positive outcomes
        ranked = self._rank_options(predictions)
        
        # Generate comparison summary
        comparison = await self._generate_comparison(options, predictions, ranked, goal)
        
        return {
            "goal": goal,
            "options_considered": len(options),
            "predictions": [p.to_dict() for p in predictions],
            "ranking": ranked,
            "recommendation": comparison.get('recommendation'),
            "reasoning": comparison.get('reasoning')
        }
    
    async def _generate_prediction(
        self,
        action: str,
        context: Dict[str, Any],
        goal: Optional[str],
        past_reflection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate prediction using LLM"""
        past_info = ""
        if past_reflection.get('found_similar'):
            past_info = f"""
Past Experience:
{past_reflection.get('reflection', '')}
Success rate in similar situations: {past_reflection.get('pattern_analysis', {}).get('success_rate', 0)*100:.0f}%
"""
        
        prompt = f"""Predict the outcome of taking this action.

Action: {action}
Context: {json.dumps(context)}
Goal: {goal or 'Not specified'}

{past_info}

Predict:
1. Most likely outcome
2. Alternative possible outcomes
3. Confidence level (certain/likely/maybe/unlikely/uncertain)
4. Expected externalities (unintended consequences)
5. Risks
6. Opportunities

Return JSON:
{{
  "outcomes": [
    {{"scenario": "most likely", "description": "...", "probability": 0.7}},
    {{"scenario": "alternative", "description": "...", "probability": 0.2}}
  ],
  "confidence": "likely",
  "externalities": ["effect1", "effect2"],
  "risks": ["risk1", "risk2"],
  "opportunities": ["opportunity1", "opportunity2"],
  "reasoning": "why this prediction"
}}"""
        
        try:
            thought = await self.brain.think(prompt)
            response = thought.content.strip()
            # Hardening: Use robust extraction
            from core.utils.json_utils import extract_json
            return extract_json(response) or {}
        except Exception as e:
            record_degradation('temporal_reasoning', e)
            logger.error("Prediction generation failed: %s", e)
            return {
                "outcomes": [{"scenario": "unknown", "description": "Prediction unavailable", "probability": 0.5}],
                "confidence": "uncertain",
                "externalities": [],
                "risks": ["Prediction system error"],
                "opportunities": [],
                "reasoning": f"Error: {e}"
            }
    
    def _calculate_confidence(
        self,
        prediction_data: Dict[str, Any],
        past_reflection: Dict[str, Any]
    ) -> Tuple[ConfidenceLevel, float]:
        """Calculate confidence level and score"""
        # Start with LLM's stated confidence
        stated = prediction_data.get('confidence', 'maybe').lower()
        
        # Adjust based on past experience
        if past_reflection.get('found_similar'):
            success_rate = past_reflection.get('pattern_analysis', {}).get('success_rate', 0.5)
            
            # Higher past success = higher confidence
            if success_rate > 0.8:
                confidence_boost = 0.2
            elif success_rate > 0.6:
                confidence_boost = 0.1
            elif success_rate < 0.3:
                confidence_boost = -0.2
            else:
                confidence_boost = 0.0
        else:
            # No past data = less confident
            confidence_boost = -0.1
        
        # Map stated confidence to score
        confidence_map = {
            'certain': 0.95,
            'likely': 0.75,
            'maybe': 0.5,
            'unlikely': 0.25,
            'uncertain': 0.1
        }
        
        base_score = confidence_map.get(stated, 0.5)
        final_score = max(0.0, min(1.0, base_score + confidence_boost))
        
        # Map score to level
        if final_score >= 0.9:
            level = ConfidenceLevel.CERTAIN
        elif final_score >= 0.7:
            level = ConfidenceLevel.LIKELY
        elif final_score >= 0.4:
            level = ConfidenceLevel.MAYBE
        elif final_score >= 0.1:
            level = ConfidenceLevel.UNLIKELY
        else:
            level = ConfidenceLevel.UNCERTAIN
        
        return level, final_score
    
    def _should_recommend(self, prediction_data: Dict[str, Any], confidence_score: float) -> bool:
        """Decide whether to recommend this action"""
        # Don't recommend if confidence is too low
        if confidence_score < 0.4:
            return False
        
        # Check risk/opportunity balance
        risks = len(prediction_data.get('risks', []))
        opportunities = len(prediction_data.get('opportunities', []))
        
        # More opportunities than risks = recommend
        if opportunities > risks and confidence_score >= 0.5:
            return True
        
        # High confidence positive outcome
        outcomes = prediction_data.get('outcomes', [])
        if outcomes:
            primary = outcomes[0]
            if primary.get('probability', 0) > 0.6:
                description = primary.get('description', '').lower()
                positive_words = ['success', 'good', 'improve', 'solve', 'fix']
                if any(word in description for word in positive_words):
                    return True
        
        return False
    
    def _rank_options(self, predictions: List[FuturePrediction]) -> List[Dict[str, Any]]:
        """Rank options by desirability"""
        ranked = []
        
        for pred in predictions:
            # Score based on confidence, risks, opportunities
            score = pred.confidence_score
            score += len(pred.opportunities) * 0.1
            score -= len(pred.risks) * 0.1
            score += 0.2 if pred.recommended else 0.0
            
            ranked.append({
                "action": pred.action,
                "score": score,
                "confidence": pred.confidence.value,
                "recommended": pred.recommended
            })
        
        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked
    
    async def _generate_comparison(
        self,
        options: List[str],
        predictions: List[FuturePrediction],
        ranked: List[Dict[str, Any]],
        goal: str
    ) -> Dict[str, Any]:
        """Generate comparison summary"""
        prompt = f"""Compare these action options for the goal.

Goal: {goal}

Options (ranked by predicted success):
"""
        for i, item in enumerate(ranked, 1):
            pred = next(p for p in predictions if p.action == item['action'])
            prompt += f"""
{i}. {item['action']}
   Confidence: {item['confidence']}
   Risks: {', '.join(pred.risks)}
   Opportunities: {', '.join(pred.opportunities)}
"""
        
        prompt += """
Provide:
1. Top recommendation and why
2. What makes it better than alternatives
3. Any important caveats

Be concise (2-3 sentences)."""
        
        try:
            thought = await self.brain.think(prompt)
            return {
                "recommendation": ranked[0]['action'],
                "reasoning": thought.content
            }
        except Exception:
            return {
                "recommendation": ranked[0]['action'] if ranked else "No clear recommendation",
                "reasoning": "Comparison analysis unavailable"
            }