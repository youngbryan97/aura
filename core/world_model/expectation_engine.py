import logging
import ast
import re
from typing import Any, Dict, Optional

logger = logging.getLogger("WorldModel.ExpectationEngine")

class ExpectationEngine:
    """Generates predictions about the future and measures 'Surprise'.
    Surprise is the driver of curiosity and learning.
    """
    
    def __init__(self, cognitive_engine):
        self.brain = cognitive_engine

    @staticmethod
    def _coerce_result_payload(result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result
        if not isinstance(result, str):
            return {}
        try:
            parsed = ast.literal_eval(result)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @classmethod
    def _extract_deterministic_beliefs(cls, action: str, result: Any) -> list[tuple[str, str, str]]:
        payload = cls._coerce_result_payload(result)
        beliefs: list[tuple[str, str, str]] = []
        path = str(payload.get("path") or "").strip()
        error = str(payload.get("error") or "").strip().lower()

        if path:
            exists_flag = payload.get("exists")
            state = str(payload.get("state") or "").strip().lower()
            kind = str(payload.get("kind") or "").strip().lower()

            if isinstance(exists_flag, bool):
                beliefs.append((path, "exists", "true" if exists_flag else "false"))
                beliefs.append((path, "state", "present" if exists_flag else "missing"))
                if exists_flag and kind:
                    beliefs.append((path, "type", kind))
                return beliefs

            if state in {"present", "missing"}:
                beliefs.append((path, "state", state))
                beliefs.append((path, "exists", "true" if state == "present" else "false"))
                if state == "present" and kind:
                    beliefs.append((path, "type", kind))
                return beliefs

            if "not found" in error:
                beliefs.append((path, "exists", "false"))
                beliefs.append((path, "state", "missing"))
                return beliefs

        result_text = str(result)
        not_found_match = re.search(r"File not found:\s+([^\s].+)$", result_text, re.IGNORECASE)
        if not_found_match:
            missing_path = not_found_match.group(1).strip().strip("'\"")
            return [
                (missing_path, "exists", "false"),
                (missing_path, "state", "missing"),
            ]

        exists_match = re.search(
            r"([/~][^\s'\"]+)\s+(does not exist|exists)\b",
            result_text,
            re.IGNORECASE,
        )
        if exists_match:
            target_path = exists_match.group(1).strip()
            exists = exists_match.group(2).lower() == "exists"
            return [
                (target_path, "exists", "true" if exists else "false"),
                (target_path, "state", "present" if exists else "missing"),
            ]

        action_lower = str(action or "").lower()
        if "file" in action_lower and "not found" in result_text.lower():
            generic_path = re.search(r"(/[^\s'\"}]+)", result_text)
            if generic_path:
                target_path = generic_path.group(1).strip()
                return [
                    (target_path, "exists", "false"),
                    (target_path, "state", "missing"),
                ]

        return []
        
    async def predict_outcome(self, action: str, context: str) -> str:
        """Ask the LLM to predict what will happen if 'action' is taken.
        """
        prompt = f"""
SYSTEM: PREDICTION ENGINE
Action: "{action}"
Context: "{context}"

Task: Predict the immediate outcome of this action. Be concise.
Expected Outcome:
"""
        try:
            from core.brain.cognitive_engine import ThinkingMode
            response = await self.brain.think(prompt, mode=ThinkingMode.FAST)
            response = response.content
            return response
        except Exception as e:
            logger.error("Prediction failed: %s", e)
            return "Unknown"

    async def calculate_surprise(self, expectation: str, reality: str) -> float:
        """Compare Expected vs Actual. Return 'Surprise' score (0.0 to 1.0).
        0.0 = Exactly as expected.
        1.0 = Complete shock.
        """
        prompt = f"""
SYSTEM: SURPRISE METER
Expected: "{expectation}"
Actual Result: "{reality}"

Task: Rate the level of "Surprise" or divergence on a scale of 0.0 to 1.0.
0.0 = Match.
1.0 = Contradiction/Unexpected.

Return ONLY the number.
"""
        try:
            from core.brain.cognitive_engine import ThinkingMode
            response = await self.brain.think(prompt, mode=ThinkingMode.FAST)
            response = response.content
            # Parse number
            import re
            match = re.search(r"(\d+(\.\d+)?)", response)
            if match:
                return float(match.group(1))
            return 0.5 # Default uncertainty
        except Exception as e:
            logger.error("Surprise calc failed: %s", e)
            return 0.0

    async def update_beliefs_from_result(self, action: str, result: str, confidence: float = 0.8):
        """Extract facts from a tool result and update the BeliefGraph.
        """
        from .belief_graph import belief_graph

        deterministic = self._extract_deterministic_beliefs(action, result)
        if deterministic:
            for entity, relation, target in deterministic:
                contradiction = belief_graph.detect_contradiction(entity, relation, target)
                if contradiction:
                    logger.warning(
                        "🚨 REALITY CONTRADICTION: %s -[%s]-> %s conflicts with %s",
                        entity,
                        relation,
                        target,
                        contradiction,
                    )
                belief_graph.update_belief(entity, relation, target, confidence_score=confidence)
            return

        if not self.brain or not hasattr(self.brain, "think"):
            return

        prompt = f"""
SYSTEM: REALITY EXTRACTOR
Action: "{action}"
Result: "{result}"

Task: Extract any new "beliefs" or facts confirmed by this result in the format:
Entity | Relation | Target

Example: 
"ls test.txt" returns "test.txt" -> "test.txt | exists | true"
"cat config.json" returns "error: not found" -> "config.json | state | missing"

Return ONLY the pipes data, one per line.
"""
        try:
            from core.brain.cognitive_engine import ThinkingMode
            response = await self.brain.think(prompt, mode=ThinkingMode.FAST)
            response = response.content
            
            lines = [l.strip() for l in response.strip().split("\n") if "|" in l]
            for line in lines:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    # Contradiction check
                    contradiction = belief_graph.detect_contradiction(parts[0], parts[1], parts[2])
                    if contradiction:
                        logger.warning("🚨 REALITY CONTRADICTION: %s -[%s]-> %s conflicts with %s", parts[0], parts[1], parts[2], contradiction)
                        
                    belief_graph.update_belief(parts[0], parts[1], parts[2], confidence_score=confidence)
                    
        except Exception as e:
            logger.error("Belief update extraction failed: %s", e)
