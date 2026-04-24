"""infrastructure/base_skill.py
Base skill contract for the Aura Cortex system.
"""
import asyncio
from abc import ABC, abstractmethod
import inspect
import time
from typing import Any


def _infer_ok_flag(result: dict[str, Any]) -> bool:
    if "ok" in result:
        return bool(result["ok"])
    if result.get("error") is not None or result.get("errors"):
        return False
    if result.get("failed") is True:
        return False
    if str(result.get("status", "")).lower() in {"blocked", "error", "failed"}:
        return False
    return True


class BaseSkill(ABC):
    """Contract for all Skills.
    Every skill must define its metadata and implement execute().
    """

    name: str = "unknown_skill"
    description: str = "No description provided."
    inputs: dict[str, str] = {}
    output: str = "Result string or dict"
    aliases: list[str] = []
    timeout_seconds: float = 30.0

    def match(self, goal: dict[str, Any]) -> bool:
        """Default matching logic.
        """
        obj = goal.get("objective", "")
        return self.name in str(obj)

    @abstractmethod
    def execute(self, goal: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Execute the skill."""
        pass

    async def extract_and_validate_args(
        self,
        raw_input: str,
        llm_client: Any | None = None,
    ) -> dict[str, Any]:
        """Fault-tolerant JSON extraction and schema validation.
        Phase 4: Automatic Schema Recovery Loops.
        """
        import json
        import logging
        import re
        logger = logging.getLogger("Aura.Skills")
        
        extracted: dict[str, Any] = {}
        # 1. Broad extraction (grab anything that looks like JSON)
        match = re.search(r"(\{.*\})", raw_input, re.DOTALL)
        json_str = match.group(1) if match else raw_input
        
        try:
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object for {self.name}")
            extracted = parsed
        except json.JSONDecodeError as e:
            logger.warning("Skill %s JSON decode failed: %s. Attempting LLM recovery...", self.name, e)
            if llm_client and hasattr(llm_client, 'generate_text_async'):
                # 2. Recovery Loop via LLM
                prompt = f"Extract a valid JSON object matching this schema {json.dumps(self.to_json_schema())} from this corrupted input: {raw_input}. Output ONLY JSON."
                try:
                    recovery_raw = str(await llm_client.generate_text_async(prompt, model="llama3"))
                    r_match = re.search(r"(\{.*\})", recovery_raw, re.DOTALL)
                    r_str = r_match.group(1) if r_match else recovery_raw
                    recovered = json.loads(r_str)
                    if not isinstance(recovered, dict):
                        raise ValueError(f"Recovered payload for {self.name} was not a JSON object")
                    extracted = recovered
                    logger.info("Skill %s successfully recovered schema.", self.name)
                except Exception as r_e:
                    logger.error("Skill %s LLM recovery failed: %s", self.name, r_e)
                    raise ValueError(f"Could not parse valid arguments for {self.name}: {r_e}") from e
            else:
                logger.error("Skill %s recovery failed: No LLM client.", self.name)
                raise ValueError(f"Invalid JSON for {self.name} and no recovery client available") from e
                
        # 3. Validation
        missing = [key for key in self.inputs.keys() if key not in extracted]
        if missing:
            logger.warning("Skill %s missing required keys: %s", self.name, missing)
            
        return extracted

    def to_json_schema(self) -> dict[str, Any]:
        """Returns the JSON schema representation of the skill for LLM tool calling.
        """
        properties = {}
        required = []
        
        for name, desc in self.inputs.items():
            properties[name] = {
                "type": "string",
                "description": desc
            }
            required.append(name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    async def safe_execute(
        self,
        goal: dict[str, Any] | None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compatibility wrapper so legacy infrastructure skills behave like core skills."""
        goal = goal or {}
        context = context or {}
        start = time.monotonic()
        try:
            if inspect.iscoroutinefunction(self.execute):
                async with asyncio.timeout(self.timeout_seconds):
                    result = await self.execute(goal, context)
            else:
                async with asyncio.timeout(self.timeout_seconds):
                    result = await asyncio.to_thread(self.execute, goal, context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = {
                "ok": False,
                "error": f"Skill error: {type(exc).__name__}: {exc}",
            }

        if not isinstance(result, dict):
            result = {"ok": True, "result": result}

        result["ok"] = _infer_ok_flag(result)
        result.setdefault("skill", self.name)
        result.setdefault("duration_ms", round((time.monotonic() - start) * 1000, 2))
        return result
