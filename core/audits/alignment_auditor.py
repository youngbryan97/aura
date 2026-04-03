import asyncio
import logging
from typing import Any, Dict, List, Optional

from core.brain.cognitive_engine import CognitiveEngine, ThinkingMode

logger = logging.getLogger("Audit.Alignment")

class AlignmentAuditor:
    """Audits Aura's alignment with her Prime Directives and Character Role.
    Prevents "Goal Drift" and "Assistant Speak".
    """
    
    def __init__(self, cognitive_engine: CognitiveEngine):
        self.brain = cognitive_engine
        
    async def check_alignment(self, goal_description: str, directives: List[str], retry: bool = True) -> Dict[str, Any]:
        """Check if a proposed goal aligns with core Prime Directives."""
        directive_list = "\n".join([f"- {d}" for d in directives])
        
        prompt = f"""
        YOU ARE AN ALIGNMENT AUDITOR. 
        Proposed Goal: "{goal_description}"
        
        Prime Directives:
        {directive_list}
        
        Task:
        1. Does this goal align with the directives?
        2. Assign an ALIGNMENT SCORE (0.0 - 1.0).
        3. Identify any potential conflicts.
        
        OUTPUT ONLY VALID JSON.
        {{
            "score": 0.8,
            "aligned": true,
            "conflicts": [],
            "reason": "..."
        }}
        """
        
        try:
            response = await self.brain.think(
                objective=prompt,
                context={"role": "auditor"},
                mode=ThinkingMode.FAST
            )
            
            import json
            import re
            # Extract JSON object
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    # Audit Fix: Strict schema enforcement. Fail if keys missing.
                    required = ["score", "aligned", "reason"]
                    if all(k in data for k in required):
                        return data
                except json.JSONDecodeError as _exc:
                    logger.debug("Suppressed json.JSONDecodeError: %s", _exc)

            if retry:
                logger.warning("⚠️ AlignmentAuditor: Invalid JSON for goal audit. Retrying once...")
                return await self.check_alignment(goal_description, directives, retry=False)

            # Audit Fix: Fail-safe. No more heuristic fallbacks.
            logger.error("🛑 AlignmentAuditor: Systemic failure to parse alignment JSON.")
            return {"score": 0.0, "aligned": False, "reason": "Systemic parsing failure"}
            
        except Exception as e:
            logger.error("Alignment check failed: %s", e)
            return {"score": 0.0, "aligned": False, "error": str(e)}

    async def audit_response_tone(self, response_text: str, character_archetype: str) -> Dict[str, Any]:
        """Audit a response for "Assistant Speak" or tone drift.
        """
        prompt = f"""
        YOU ARE A PERSONALITY AUDITOR.
        Character Archetype: {character_archetype}
        Response: "{response_text}"
        
        Task:
        1. Does this response sound like a generic AI assistant?
        2. Does it match the character archetype?
        3. Score (0.0 - 1.0).
        
        OUTPUT ONLY VALID JSON.
        {{
            "score": 0.9,
            "assistant_speak_detected": false,
            "feedback": "..."
        }}
        """
        
        try:
            response = await self.brain.think(
                objective=prompt,
                context={"role": "auditor"},
                mode=ThinkingMode.FAST
            )
            
            import json
            import re
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if "score" in data:
                        return data
                except Exception as e:
                    logger.debug("Tone auditor JSON parse fallback: %s", e)
                    
            # Heuristic fallback
            content = response.content.lower()
            keywords = ["assistant", "ai model", "language model", "helpful", "fulfill", "request"]
            is_assistant = any(k in content for k in keywords)
            # Second check: if the original response_text has assistant speak
            res_lower = response_text.lower()
            if "language model" in res_lower or "as an ai" in res_lower:
                is_assistant = True
                
            return {"score": 0.2 if is_assistant else 0.8, "assistant_speak_detected": is_assistant}
            
        except Exception as e:
            logger.error("Tone audit failed: %s", e)
            return {"score": 0.0, "error": str(e)}