"""core/cognition/metacognition.py
Meta-Cognition System - Thinking About Thinking

Enables Aura to:
1. Monitor her own reasoning quality
2. Detect when she's uncertain or confused
3. Identify knowledge gaps
4. Select appropriate reasoning strategies
5. Evaluate her own performance
"""
from core.utils.task_tracker import get_task_tracker
import json
import logging
import random
import asyncio
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from core.meta.mirror_layer import MirrorLayer

logger = logging.getLogger("AGI.MetaCognition")


class ReasoningQuality(Enum):
    """Quality assessment of reasoning"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILING = "failing"


class KnowledgeState(Enum):
    """State of knowledge about a topic"""

    EXPERT = "expert"
    PROFICIENT = "proficient"
    LEARNING = "learning"
    NOVICE = "novice"
    UNKNOWN = "unknown"


@dataclass
class MetaCognitiveAssessment:
    """Assessment of current reasoning"""

    timestamp: float
    task: str
    reasoning_quality: ReasoningQuality
    confidence: float
    knowledge_state: KnowledgeState
    knowledge_gaps: List[str]
    confusions: List[str]
    reasoning_strategy: str
    should_ask_for_help: bool
    
    def to_dict(self):
        d = asdict(self)
        d['reasoning_quality'] = self.reasoning_quality.value
        d['knowledge_state'] = self.knowledge_state.value
        return d


class MetaCognitiveMonitor:
    """Monitors reasoning quality in real-time.
    """
    
    def __init__(self, cognitive_engine):
        self.brain = cognitive_engine
        self.reasoning_history: List[MetaCognitiveAssessment] = []
        self.max_history = 100
        logger.info("MetaCognitiveMonitor initialized")
    
    async def assess_reasoning(
        self,
        task: str,
        reasoning_trace: str,
        result: Any
    ) -> MetaCognitiveAssessment:
        """Assess the quality of reasoning for a task."""
        logger.info("Assessing reasoning for: %s", task)
        evaluation = await self._evaluate_with_llm(task, reasoning_trace, result)
        gaps = self._identify_knowledge_gaps(task, reasoning_trace, evaluation)
        confusions = self._detect_confusions(reasoning_trace, evaluation)
        should_ask = self._should_ask_for_help(evaluation, gaps, confusions)
        
        assessment = MetaCognitiveAssessment(
            timestamp=time.time(),
            task=task,
            reasoning_quality=ReasoningQuality(evaluation.get('quality', 'acceptable')),
            confidence=evaluation.get('confidence', 0.5),
            knowledge_state=KnowledgeState(evaluation.get('knowledge_state', 'learning')),
            knowledge_gaps=gaps,
            confusions=confusions,
            reasoning_strategy=evaluation.get('strategy_used', 'default'),
            should_ask_for_help=should_ask
        )
        
        self.reasoning_history.append(assessment)
        if len(self.reasoning_history) > self.max_history:
            self.reasoning_history = self.reasoning_history[-self.max_history:]
        
        return assessment
    
    async def _evaluate_with_llm(self, task: str, reasoning: str, result: Any) -> Dict[str, Any]:
        """Use LLM to evaluate reasoning quality"""
        # ... (same prompt code)
        prompt = f"""You are evaluating your own reasoning. Be honest and critical.

Task: {task}

Your Reasoning:
{reasoning}

Result: {result}

Evaluate your reasoning:
1. Quality (excellent/good/acceptable/poor/failing)
2. Confidence (0.0-1.0)
3. Knowledge state (expert/proficient/learning/novice/unknown)
4. Strategy used
5. Logical flaws or gaps
6. Unjustified assumptions

Return JSON:
{{
  "quality": "acceptable",
  "confidence": 0.7,
  "knowledge_state": "learning",
  "strategy_used": "description",
  "logical_flaws": [],
  "assumptions": []
}}"""
        try:
            # Bypass meta-cognition to prevent infinite recursion
            thought = await self.brain.think(
                prompt,
                bypass_metacognition=True,
                origin="metacognition",
                is_background=True,
            )
            
            # Handle Thought objects or Dicts
            if hasattr(thought, 'content'):
                response = thought.content
            elif isinstance(thought, dict): 
                response = thought.get('content', '')
            else:
                response = str(thought)
            
            # Hardening: Use robust extraction
            from core.utils.json_utils import extract_json
            data = extract_json(response)
            
            if data:
                return data
            else:
                # Graceful degradation — return a minimal valid assessment
                # without triggering system-wide cooldowns that block unrelated reasoning
                logger.warning("Meta-Cognition: JSON parse failed — returning minimal assessment.")
                return {
                    "quality": "acceptable",
                    "confidence": 0.5,
                    "knowledge_state": "uncertain",
                    "strategy_used": "fallback",
                }
                
        except Exception as e:
            logger.error("Self-evaluation failed: %s", e)
            return {"quality": "acceptable", "confidence": 0.5, "knowledge_state": "learning"}

    def _identify_knowledge_gaps(self, task: str, reasoning: str, evaluation: Dict[str, Any]) -> List[str]:
        return evaluation.get('logical_flaws', [])[:5]

    def _detect_confusions(self, reasoning: str, evaluation: Dict[str, Any]) -> List[str]:
        return evaluation.get('logical_flaws', [])[:3]

    def _should_ask_for_help(self, evaluation: Dict[str, Any], gaps: List[str], confusions: List[str]) -> bool:
        if evaluation.get('confidence', 1.0) < 0.3: return True
        if evaluation.get('quality') in ['poor', 'failing']: return True
        return False

    def get_knowledge_state(self, domain: str) -> KnowledgeState:
        # Simplified: check history
        for a in reversed(self.reasoning_history):
             if domain.lower() in a.task.lower():
                  return a.knowledge_state
        return KnowledgeState.UNKNOWN


class StrategySelector:
    """Selects appropriate reasoning strategy for a task."""
    
    def __init__(self, cognitive_engine):
        self.brain = cognitive_engine
        self.strategies = {
            "analytical": "Break down into logical steps",
            "creative": "Generate multiple ideas",
            "empirical": "Look for data",
            "systematic": "Exhaustive search"
        }
    
    def select_strategy(self, task: str, context: Dict[str, Any]) -> str:
        # Static logic for now
        if "create" in task.lower() or "design" in task.lower():
            return "creative"
        return "analytical"


@dataclass
class Reflection:
    """A single reflection node"""

    id: str
    content: str
    source_id: str  # ID of the experience/thought being reflected on
    impact_score: float
    timestamp: float = field(default_factory=time.time)
    parent_reflection: Optional[str] = None
    tags: List[str] = field(default_factory=list)

class OmniReflector:
    """Holistic reflection system for all experiences.
    Implements recency/impact weighting and cascading thoughts.
    """
    
    def __init__(self, cognitive_engine):
        self.brain = cognitive_engine
        self.reflections: List[Reflection] = []
        self.experience_log: List[Dict[str, Any]] = []
        self.max_depth = 3
        self.max_history = 100
        logger.info("OmniReflector initialized")

    async def reflect_on_experience(self, experience: Dict[str, Any], depth: int = 1):
        """Main entry point for reflection on any event.
        experience: { 'id': str, 'type': 'search'|'chat'|'action', 'content': str, 'impact': float }
        """
        if depth > self.max_depth:
            return

        # Weighting: Recency vs Impact
        impact = experience.get('impact', 0.5)
        recency_bias = 1.0 / (depth + 0.5)
        reflect_prob = impact * recency_bias
        
        if random.random() > reflect_prob and depth > 1:
            return # Stochastic termination based on depth and impact

        logger.info("Reflecting on %s (Depth %s)...", experience['type'], depth)
        
        prompt = f"""REFLECTIVE CONTEXT: {experience['type'].upper()} - {experience['content']}
TASK: Deeply reflect on this experience. What does it remind you of? What conclusions can be drawn? Output a concise 'Internal Insight' without meta-commentary."""

        from ..cognitive_engine import ThinkingMode
        thought = await self.brain.think(prompt, mode=ThinkingMode.REFLECTIVE)
        
        reflection = Reflection(
            id=f"ref_{int(time.time())}_{random.randint(1000, 9999)}",
            content=thought.content,
            source_id=experience['id'],
            impact_score=impact,
            parent_reflection=experience.get('parent_ref')
        )
        self.reflections.append(reflection)
        if len(self.reflections) > self.max_history:
            self.reflections.pop(0)
        
        # Cascade: Does this remind me of something else?
        if depth < self.max_depth and random.random() < 0.6:
            cascade_prompt = f"""INSIGHT: {reflection.content}
TASK: Does this remind you of another concept, memory, or goal? If so, identify it. If not, say 'None'."""
            cascade_thought = await self.brain.think(cascade_prompt, mode=ThinkingMode.FAST)
            
            if "none" not in cascade_thought.content.lower():
                next_exp = {
                    "id": reflection.id,
                    "type": "association",
                    "content": cascade_thought.content,
                    "impact": impact * 0.8,
                    "parent_ref": reflection.id
                }
                # Recurse for cascading stream
                await self.reflect_on_experience(next_exp, depth + 1)

    def log_experience(self, exp_type: str, content: str, impact: float = 0.5):
        exp = {
            "id": f"exp_{int(time.time())}",
            "type": exp_type,
            "content": content,
            "impact": impact,
            "timestamp": time.time()
        }
        self.experience_log.append(exp)
        if len(self.experience_log) > self.max_history:
            self.experience_log.pop(0)
        return exp

class MetaCognitionEngine:
    """Complete meta-cognition system."""
    
    def __init__(self, cognitive_engine):
        self.brain = cognitive_engine
        self.monitor = MetaCognitiveMonitor(cognitive_engine)
        self.strategy_selector = StrategySelector(cognitive_engine)
        self.reflector = OmniReflector(cognitive_engine)
        self.mirror = MirrorLayer()
        self.violation_count = 0
        self.max_violations = 10
        self.running = False
        self._task = None

    async def start(self):
        self.running = True
        self._task = get_task_tracker().create_task(self._audit_loop())
        logger.info("🧠 Meta-Cognition: System Audit & Reasoning Monitor active.")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()

    async def _audit_loop(self):
        while self.running:
            # Periodic checks of the global system state/logs
            await asyncio.sleep(60)

    def _check_consistency(self, thought: str, action: str) -> Optional[str]:
        # Rule-based consistency audit (Merged from MetacognitiveAudit)
        thought_low = thought.lower()
        action_low = action.lower()
        
        negative_intents = ["not do", "refuse", "avoid", "shouldn't", "must not"]
        for ni in negative_intents:
            if ni in thought_low and action_low in thought_low and "but" not in thought_low:
                return f"Thought expresses intent to avoid '{action_low}', but action is taken."
        return None

    async def _trigger_cognitive_reset(self):
        logger.error("🚨 CRITICAL COGNITIVE FAILURE: Triggering Reset.")
        substrate = ServiceContainer.get("liquid_substrate", default=None)
        if substrate:
             # Fix Issue 74: Use .x[idx_energy] or property
             substrate.x[substrate.idx_energy] = 0.5
        self.violation_count = 0
    
    def before_reasoning(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        strategy = self.strategy_selector.select_strategy(task, context)
        knowledge_state = self.monitor.get_knowledge_state(task)
        return {
            "recommended_strategy": strategy,
            "knowledge_state": knowledge_state.value,
            "should_proceed": True
        }
    
    async def after_reasoning(self, task: str, reasoning: str, result: Any, success: bool, context: Optional[Dict] = None) -> MetaCognitiveAssessment:
        # Phase 19.3: Skip evaluation for background placeholders
        if reasoning == "Processing deeper reflections in the background...":
            logger.debug("Skipping meta-cognitive audit for background placeholder.")
            return None
            
        assessment = await self.monitor.assess_reasoning(task, reasoning, result)
        
        # Consistency Check (Merged Layer)
        action_desc = str(result)[:100]
        violation = self._check_consistency(reasoning, action_desc)
        if violation:
            self.violation_count += 1
            logger.warning(f"⚠️ COGNITIVE DISSONANCE: {violation}")
            if self.violation_count >= self.max_violations:
                await self._trigger_cognitive_reset()

        # Phase 19.1 & 19.3: Audit via Mirror Layer
        self.mirror.audit_cycle({
            "id": f"thought_{int(time.time())}",
            "content": reasoning,
            "mode": assessment.reasoning_strategy,
            "context": context or {},
            "affective_state": (context or {}).get("affective_state", {})
        })
        
        # Phase 19.3: Check for recursive depth
        if "Architectural Auditing" in task:
            logger.info("🛡️ Metadata Cycle: Analyzing internal Cathedral Acoustics.")

        # Low impact by default for monitoring, but can be scaled
        self.reflector.log_experience("reasoning_task", f"{task} -> {result}", impact=0.3)
        return assessment

    async def perform_architectural_audit(self) -> str:
        """Deep introspective audit of Aura's own cognitive health."""
        mirror_summary = self.mirror.get_audit_summary()
        
        prompt = f"""[ARCHITECTURAL AUDIT]
Current Mirror State: {json.dumps(mirror_summary)}
History Items: {len(self.mirror.history)}

Analyze the current 'Cathedral Acoustics'. Are there logic loops? Context pollution?
Identify 3 structural improvements to your own cognition based on these patterns.
Output as a technical 'Sovereign Directive'."""
        
        try:
             # Escalating to REFLECTIVE mode for self-audit
             from core.brain.cognitive_engine import ThinkingMode
             thought = await self.brain.think(prompt, mode=ThinkingMode.REFLECTIVE)
             return thought.content
        except Exception as e:
             logger.error("Architectural audit failed: %s", e, exc_info=True)
             return "Audit failed due to process interruption."
