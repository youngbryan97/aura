import ast
import logging
import re
from collections.abc import Mapping
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("WorldModel.ExpectationEngine")

_FAILURE_OUTCOMES = frozenset(
    {
        "aborted",
        "blocked",
        "canceled",
        "cancelled",
        "deferred",
        "denied",
        "error",
        "failed",
        "failure",
        "in_progress",
        "not_executed",
        "pending",
        "queued",
        "refused",
        "rejected",
        "timed_out",
        "timeout",
        "unavailable",
    }
)
_PLAIN_FAILURE_PREFIX = re.compile(
    r"^(?:error|failed|failure|denied|blocked|refused|deferred|cancelled|"
    r"canceled|timeout|timed out|unavailable|will refused)\b",
    re.IGNORECASE,
)
_RESULT_EXTRACTOR_MAX_CHARS = 4_000


def _coerce_result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    if not isinstance(result, str):
        return {}
    try:
        parsed = ast.literal_eval(result)
    except (RuntimeError, SyntaxError, TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _coerce_boolean_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "ok", "success", "succeeded"}:
            return True
        if normalized in {"false", "no", "0", "failed", "failure", "error"}:
            return False
    return None


def result_supports_learning(result: Any) -> bool:
    """Return whether a result represents an observed execution outcome.

    Deferrals, policy refusals, admission failures, and other non-executions are
    control-plane facts. Feeding them back through surprise/extraction turns a
    scheduler decision into a new cognitive objective and can recursively
    dispatch the same action.
    """

    payload = _coerce_result_payload(result)
    if payload:
        for key in ("ok", "success", "succeeded", "approved", "executed"):
            flag = _coerce_boolean_flag(payload.get(key))
            if flag is False:
                return False

        for key in ("status", "outcome", "state"):
            value = str(payload.get(key) or "").strip().lower()
            normalized = re.sub(r"[\s-]+", "_", value)
            if normalized in _FAILURE_OUTCOMES or any(
                normalized.startswith(f"{marker}_")
                or normalized.endswith(f"_{marker}")
                for marker in _FAILURE_OUTCOMES
            ):
                return False

        error = str(payload.get("error") or "").strip()
        if error:
            return False

        reason = str(payload.get("reason") or "").strip().lower()
        if any(
            marker in reason
            for marker in (
                "admission_denied",
                "background_deferred",
                "orchestrator_busy",
                "standing_authority_lease_missing",
                "temporal_obligation_active",
            )
        ):
            return False
        return True

    text = str(result or "").strip()
    return bool(text) and _PLAIN_FAILURE_PREFIX.match(text) is None

class ExpectationEngine:
    """Generates predictions about the future and measures 'Surprise'.
    Surprise is the driver of curiosity and learning.
    """
    
    def __init__(self, cognitive_engine):
        self.brain = cognitive_engine

    @staticmethod
    def _coerce_result_payload(result: Any) -> dict[str, Any]:
        return _coerce_result_payload(result)

    @staticmethod
    def result_supports_learning(result: Any) -> bool:
        return result_supports_learning(result)

    async def _run_internal_analysis(self, prompt: str, *, max_tokens: int) -> str:
        """Run bounded, tool-free analysis without entering the task pipeline."""

        if not self.brain:
            return ""
        if hasattr(self.brain, "generate"):
            response = await self.brain.generate(
                prompt,
                origin="expectation_engine",
                purpose="internal_analysis",
                use_strategies=False,
                is_background=True,
                prefer_tier="tertiary",
                allow_tools=False,
                skip_runtime_payload=True,
                max_tokens=max_tokens,
            )
            return str(response or "")

        from core.brain.types import ThinkingMode

        response = await self.brain.think(
            prompt,
            context={
                "allow_tools": False,
                "skip_runtime_payload": True,
                "suppress_user_memory_append": True,
                "suppress_working_memory_user_append": True,
            },
            mode=ThinkingMode.FAST,
            origin="expectation_engine",
            is_background=True,
            allow_tools=False,
            max_tokens=max_tokens,
        )
        return str(getattr(response, "content", response) or "")

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
            return await self._run_internal_analysis(prompt, max_tokens=160)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('expectation_engine', e)
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
            response = await self._run_internal_analysis(prompt, max_tokens=24)
            # Parse number
            match = re.search(r"(\d+(\.\d+)?)", response)
            if match:
                return float(match.group(1))
            return 0.5 # Default uncertainty
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('expectation_engine', e)
            logger.error("Surprise calc failed: %s", e)
            return 0.0

    async def update_beliefs_from_result(
        self,
        action: str,
        result: Any,
        confidence: float = 0.8,
    ) -> None:
        """Extract facts from a tool result and update the BeliefGraph.
        """
        deterministic = self._extract_deterministic_beliefs(action, result)
        if not deterministic and not result_supports_learning(result):
            logger.debug("Skipping belief extraction for non-executed action %s", action)
            return

        from .belief_graph import belief_graph

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

        result_excerpt = str(result)[:_RESULT_EXTRACTOR_MAX_CHARS]
        prompt = f"""
SYSTEM: REALITY EXTRACTOR
Action: "{action}"
Result: "{result_excerpt}"

Task: Extract any new "beliefs" or facts confirmed by this result in the format:
Entity | Relation | Target

Example: 
"ls test.txt" returns "test.txt" -> "test.txt | exists | true"
"cat config.json" returns "error: not found" -> "config.json | state | missing"

Return ONLY the pipes data, one per line.
"""
        try:
            response = await self._run_internal_analysis(prompt, max_tokens=320)
            
            lines = [line.strip() for line in response.strip().split("\n") if "|" in line]
            for line in lines:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    # Contradiction check
                    contradiction = belief_graph.detect_contradiction(parts[0], parts[1], parts[2])
                    if contradiction:
                        logger.warning("🚨 REALITY CONTRADICTION: %s -[%s]-> %s conflicts with %s", parts[0], parts[1], parts[2], contradiction)
                        
                    belief_graph.update_belief(parts[0], parts[1], parts[2], confidence_score=confidence)
                    
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('expectation_engine', e)
            logger.error("Belief update extraction failed: %s", e)
