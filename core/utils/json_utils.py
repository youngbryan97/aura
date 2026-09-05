"""core/utils/json_utils.py
Robust JSON utilities for Sovereign local models.
"""
from core.runtime.errors import record_degradation
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


def extract_json_list(text: Optional[str]) -> List[Any]:
    """The list a model meant to write, or an empty one.

    ``extract_json`` returns an object, and some answers are a list: a plan is
    a sequence of steps. The caller that needed one was finding the first "["
    and the last "]" and hoping — which breaks on a thinking block, on prose
    that mentions a bracket, and on a fenced block that puts a comment above
    the JSON.

    LIVE 2026-08-29, from the autonomous planner's own log: 24 empty responses,
    15 "No JSON array in response", and a run of "Expecting ',' delimiter" at
    the same column. Each one fell through to a single generic step, and 102
    of 106 completed plans then read "Success=True (1/1 steps)" whatever the
    goal was.

    Same machinery as the object path — the markdown strip, the balanced-span
    scan, the smart-quote and trailing-comma repair — so a third parser does
    not drift away from the two that exist. A single object comes back as a
    one-element list, because a model asked for a list of one usually writes
    the one.
    """

    healer = SelfHealingJSON()
    body = healer._strip_markdown(text)
    if not body:
        return []
    for attempt in (body, healer._heuristic_repair(body)):
        try:
            found = json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(found, list):
            return found
        if isinstance(found, dict):
            return [found]
    # Longest balanced span first: a nested array inside the real one is a
    # candidate too, and the outer one is the answer.
    for candidate in sorted(healer._find_json_candidates(body), key=len, reverse=True):
        for attempt in (candidate, healer._heuristic_repair(candidate)):
            try:
                found = json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(found, list):
                return found
            if isinstance(found, dict):
                return [found]
    return []


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
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('json_utils', e)
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
        except (RuntimeError, AttributeError, TypeError, ValueError):
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
