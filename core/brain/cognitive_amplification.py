import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("CognitiveAmplification")

# ── Confidence Constants ──
_CONFIDENCE_DIRECT = 0.80
_CONFIDENCE_COT    = 0.90

class ReasoningStrategy(Enum):
    DIRECT = "direct"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    SELF_CONSISTENCY = "self_consistency"
    SYMBOLIC = "symbolic"
    TOOL_AUGMENTED = "tool_augmented"

@dataclass
class ReasoningResult:
    answer: str
    confidence: float
    reasoning_steps: List[str] = field(default_factory=list)
    strategy_used: ReasoningStrategy = ReasoningStrategy.DIRECT
    alternatives_considered: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "strategy_used": self.strategy_used.value,
            "alternatives_considered": self.alternatives_considered,
            "tools_used": self.tools_used
        }

class CognitiveAmplificationSystem:
    """Amplifies intelligence through architectural reasoning patterns.
    """
    
    def __init__(self, brain):
        self.brain = brain
        logger.info("Cognitive Amplification System initialized")
    
    async def solve(self, problem: str, strategy: Optional[ReasoningStrategy] = None) -> ReasoningResult:
        if strategy is None:
            strategy = self._select_strategy(problem)
        
        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            return await self._chain_of_thought(problem)
        elif strategy == ReasoningStrategy.SELF_CONSISTENCY:
            return await self._self_consistency(problem)
        # Fallback to direct for now, others can be implemented as needed
        return await self._direct_reasoning(problem)

    def _select_strategy(self, problem: str) -> ReasoningStrategy:
        if any(word in problem.lower() for word in ['solve', 'calculate', 'why', 'how']):
            return ReasoningStrategy.CHAIN_OF_THOUGHT
        return ReasoningStrategy.DIRECT

    async def _direct_reasoning(self, question: str) -> ReasoningResult:
        response = await self.brain.think(question)
        return ReasoningResult(answer=response, confidence=_CONFIDENCE_DIRECT)

    async def _chain_of_thought(self, question: str) -> ReasoningResult:
        prompt = f"Let's solve this step-by-step.\n\nQuestion: {question}\n\nReasoning:"
        response = await self.brain.think(prompt)
        steps = [s.strip() for s in response.split('\n') if s.strip()]
        answer = steps[-1] if steps else response
        return ReasoningResult(
            answer=answer,
            confidence=_CONFIDENCE_COT,
            reasoning_steps=steps,
            strategy_used=ReasoningStrategy.CHAIN_OF_THOUGHT
        )

    async def _self_consistency(self, question: str, num_attempts: int = 3) -> ReasoningResult:
        import asyncio
        tasks = [self.brain.think(question) for _ in range(num_attempts)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        responses = [r for r in responses if isinstance(r, str)]
        if not responses:
            return ReasoningResult(answer="Unable to reach consensus.", confidence=0.0, strategy_used=ReasoningStrategy.SELF_CONSISTENCY)
        from collections import Counter
        counts = Counter(responses)
        consensus = counts.most_common(1)[0][0]
        confidence = counts[consensus] / len(responses)
        return ReasoningResult(
            answer=consensus,
            confidence=confidence,
            strategy_used=ReasoningStrategy.SELF_CONSISTENCY,
            alternatives_considered=[r for r in responses if r != consensus]
        )

def integrate_cognitive_amplification(orchestrator):
    orchestrator.cognition = CognitiveAmplificationSystem(orchestrator.cognitive_engine)
    logger.info("Cognitive Amplification integrated into Orchestrator")