"""core/utils/json_utils.py
Robust JSON utilities for Sovereign local models.
"""
import ast
from core.utils.exceptions import capture_and_log
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Utils.JSON")


def extract_json(text: Optional[str], brain: Any = None) -> Dict[str, Any]:
    """Unified JSON extraction and repair."""
    repairer = SelfHealingJSON(brain=brain)
    import asyncio
    try:
        asyncio.get_running_loop()
        return repairer.parse_sync(text)
    except RuntimeError:
        return repairer.parse_sync(text)


class SelfHealingJSON:
    """Robust JSON Parser (The 'Optimizer').
    Pipeline: Standard -> Regex Heuristics -> LLM Reflection.
    """

    def __init__(self, brain=None):
        self.brain = brain

    @staticmethod
    def _coerce_text(raw_text: Optional[str]) -> str:
        if raw_text is None:
            return ""
        if isinstance(raw_text, str):
            return raw_text
        if hasattr(raw_text, "content") and not isinstance(raw_text, str):
            raw_text = getattr(raw_text, "content", "")
        return str(raw_text or "")

    def parse_sync(self, raw_text: Optional[str]) -> Dict[str, Any]:
        """Synchronous version of the repair pipeline."""
        clean_text = self._strip_markdown(raw_text)
        if not clean_text:
            return {}

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as exc:
            logger.debug("Ignored json.JSONDecodeError in json_utils.py: %s", exc)

        candidates = self._find_json_candidates(clean_text)
        for candidate in sorted(candidates, key=len, reverse=True):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return json.loads(self._heuristic_repair(candidate))
                except json.JSONDecodeError as exc:
                    logger.debug("Ignored json.JSONDecodeError in json_utils.py: %s", exc)
                parsed = self._parse_pythonish_dict(candidate)
                if parsed:
                    return parsed

        try:
            return json.loads(self._heuristic_repair(clean_text))
        except json.JSONDecodeError as exc:
            logger.debug("Ignored json.JSONDecodeError in json_utils.py: %s", exc)

        parsed = self._parse_pythonish_dict(clean_text)
        if parsed:
            return parsed

        return {}

    async def parse(self, raw_text: Optional[str]) -> Dict[str, Any]:
        """Full async repair pipeline including LLM reflection."""
        result = self.parse_sync(raw_text)
        if result:
            return result

        if self.brain:
            try:
                return await self._llm_repair(self._strip_markdown(raw_text))
            except Exception as e:
                capture_and_log(e, {"module": __name__})

        return {}

    def _strip_markdown(self, text: Optional[str]) -> str:
        normalized = self._coerce_text(text).strip()
        if not normalized:
            return ""
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", normalized, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if normalized.startswith("```"):
            normalized = re.sub(r"^```[a-zA-Z]*\n?", "", normalized)
            normalized = re.sub(r"```$", "", normalized)
        return normalized.strip()

    def _heuristic_repair(self, text: Optional[str]) -> str:
        normalized = self._coerce_text(text)
        normalized = normalized.replace("“", '"').replace("”", '"')
        normalized = normalized.replace("‘", "'").replace("’", "'")
        normalized = normalized.strip().rstrip(";")
        normalized = re.sub(r",\s*}", "}", normalized)
        normalized = re.sub(r",\s*]", "]", normalized)
        return normalized

    def _parse_pythonish_dict(self, text: Optional[str]) -> Dict[str, Any]:
        normalized = self._heuristic_repair(text)
        if not normalized:
            return {}
        try:
            parsed = ast.literal_eval(normalized)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _find_json_candidates(self, text: Optional[str]) -> List[str]:
        normalized = self._coerce_text(text)
        if not normalized:
            return []
        results: List[str] = []
        stack: List[str] = []
        start = -1
        in_string = False
        escape = False

        for i, char in enumerate(normalized):
            if char == '"' and not escape:
                in_string = not in_string

            if in_string:
                escape = (char == "\\" and not escape)
                continue

            if char in ['{', '[']:
                if not stack:
                    start = i
                stack.append(char)
            elif char in ['}', ']']:
                if stack:
                    opening = stack.pop()
                    if (opening == '{' and char == '}') or (opening == '[' and char == ']'):
                        if not stack:
                            results.append(normalized[start:i + 1])
                    else:
                        stack = []
        return results

    async def _llm_repair(self, broken_json: Optional[str]) -> Dict[str, Any]:
        normalized = self._strip_markdown(broken_json)
        if not normalized or not self.brain:
            return {}
        prompt = f"Fix this invalid JSON. Output ONLY valid JSON.\n\n{normalized}"
        thought = await self.brain.think(prompt)
        response = thought.content if hasattr(thought, "content") else self._coerce_text(thought)
        if not response or response.lower() == "none":
            return {}
        try:
            return json.loads(self._strip_markdown(response))
        except json.JSONDecodeError:
            return {}
