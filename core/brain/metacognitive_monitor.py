import json
import re
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
from ..state.aura_state import AuraState
from ..container import ServiceContainer
from core.brain.llm.llm_router import LLMTier

logger = logging.getLogger("Aura.MetacognitiveMonitor")

@dataclass
class CoherenceReport:
    is_coherent: bool
    coherence_score: float      # 0.0 to 1.0
    violations: list[str]       # Specific inconsistencies found
    metrics: dict[str, float]   # [MOTO TRANSIMAL] Clarity, Logic, Factuality, Persona
    revision_needed: bool
    revised_response: Optional[str] = None

class MetacognitiveMonitor:
    """
    Watches Aura's outputs for coherence with her current self-model.
    """

    def __init__(self):
        self.router = None

    def _get_router(self):
        if self.router is None:
            self.router = ServiceContainer.get("llm_router", default=None)
        return self.router

    async def evaluate(
        self, 
        response: str, 
        state: AuraState
    ) -> CoherenceReport:
        router = self._get_router()
        if not router:
            return CoherenceReport(
                is_coherent=True, 
                coherence_score=1.0, 
                violations=[], 
                metrics={"clarity": 1.0, "logic": 1.0, "factuality": 1.0, "persona": 1.0}, 
                revision_needed=False
            )
        
        identity_summary = state.identity.current_narrative[:400]
        affect_desc = self._affect_to_description(state.affect)
        beliefs = self._extract_core_beliefs(state)
        
        prompt = f"""You are auditing a response for coherence with the responder's self-model.

Self-model:
{identity_summary}

Current affect: {affect_desc}
Core beliefs: {beliefs}

Response to evaluate:
{response}

Evaluate the response based on these Structured Critique Metrics:
1. Clarity: Is the message ambiguous or confusing?
2. Logic: Are there logical fallacies or contradictions?
3. Factuality: Does it contradict anything in the state/memory?
4. Persona: Does it align with Aura's deep/philosophical tone?

Respond in JSON: {{"coherent": bool, "score": float, "violations": [str], "metrics": {{"clarity": 0-1, "logic": 0-1, "factuality": 0-1, "persona": 0-1}}}}"""

        try:
            result_text = await router.think(
                prompt,
                priority=0.5,
                is_background=True,
                prefer_tier=LLMTier.TERTIARY  # [Phase 36] Preserves Gemini for Chat
            )
            
            # Extract JSON from response
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            data = json.loads(match.group()) if match else {}
            coherent = data.get("coherent", True)
            score = float(data.get("score", 1.0))
            violations = data.get("violations", [])
            metrics = data.get("metrics", {"clarity": 1.0, "logic": 1.0, "factuality": 1.0, "persona": 1.0})
        except Exception:
            return CoherenceReport(True, 1.0, [], {"clarity": 1.0, "logic": 1.0, "factuality": 1.0, "persona": 1.0}, False)

        # Decide whether to revise or just log
        # Revision needed if score is low or a critical metric (factuality/logic) is low
        revision_needed = (score < 0.6 and len(violations) > 0) or (metrics.get("factuality", 1.0) < 0.5)
        revised = None
        
        if revision_needed:
            revised = await self._revise(response, violations, state)
        
        return CoherenceReport(
            is_coherent=coherent,
            coherence_score=score,
            violations=violations,
            metrics=metrics,
            revision_needed=revision_needed,
            revised_response=revised,
        )

    async def _revise(
        self, 
        original: str, 
        violations: list[str], 
        state: AuraState
    ) -> str:
        router = self._get_router()
        if not router: return original

        violations_text = "\n".join(f"- {v}" for v in violations)
        prompt = f"""Revise this response to be coherent with the self-model.

Original response: {original}

Inconsistencies to fix:
{violations_text}

Self-model: {state.identity.current_narrative[:300]}

Revised response (same content intent, corrected voice/consistency):"""
        
        try:
            return await router.think(
                prompt,
                priority=0.8, # Higher priority than audit
                is_background=True,
                prefer_tier=LLMTier.TERTIARY
            )
        except Exception:
            return original

    def _affect_to_description(self, affect) -> str:
        return (
            f"valence={affect.valence:.2f}, "
            f"arousal={affect.arousal:.2f}, "
            f"curiosity={affect.curiosity:.2f}"
        )

    def _extract_core_beliefs(self, state: AuraState) -> str:
        values = state.identity.core_values
        return "; ".join(values[:5]) if values else "not yet established"
