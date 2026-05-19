"""Function Calling Adapter.
Bridges local LLMs to Aura's Skill Registry.
Ensures Mind/Body alignment: 'Aura says, Aura does.'
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("LLM.FunctionAdapter")


class FunctionCallingAdapter:
    def __init__(self, registry, router=None):
        self.registry = registry
        self.router = router  # Usually the same as registry now

    def get_tool_definitions(self) -> dict[str, dict[str, Any]]:
        """Converts Aura's skill registry to JSON tool descriptions."""
        # Use CapabilityEngine's native definitions if available
        from core.capability_engine import CapabilityEngine

        if isinstance(self.registry, CapabilityEngine):
            # Native CapabilityEngine tools are return as List[Dict] usually for OpenAI
            # This adapter expects Dict[name, Dict] for LocalAgentClient
            defs = {}
            for t in self.registry.get_tool_definitions():
                fn = t["function"]
                defs[fn["name"]] = fn
            return defs

        tools = {}
        # Legacy support
        skills_dict = getattr(self.registry, "skills", self.registry)
        if not skills_dict:
            return {}

        for name, skill in skills_dict.items():
            if hasattr(skill, "skill_class") and hasattr(skill.skill_class, "to_json_schema"):
                try:
                    tools[name] = skill.skill_class.to_json_schema()
                    continue
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation("function_calling_adapter", e)
                    logger.debug("Skill schema generation skipped for %s: %s", name, e)

            description = getattr(skill, "description", "")
            inputs = (
                getattr(skill, "inputs", {})
                if not hasattr(skill, "skill_class")
                else getattr(skill.skill_class, "inputs", {})
            )

            tools[name] = {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: {"type": "string", "description": v} for k, v in inputs.items()
                    },
                    "required": list(inputs.keys()),
                },
            }
        return tools

    def validate_tool_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Strictly validates args against the registered skill schema before execution."""

        # 1. Look up the capability directly via the container if it's the CapabilityEngine
        from core.capability_engine import CapabilityEngine

        if isinstance(self.registry, CapabilityEngine):
            # CapabilityEngine tools use Pydantic natively internally, but we can do a preliminary check here
            # Or we can let CapabilityEngine handle it and catch the ValidationError. We will pre-validate here if possible:
            skill = self.registry.skills.get(tool_name)
            if not skill:
                return {"valid": False, "error": f"Tool '{tool_name}' not found in registry."}

            # If the skill uses the v2 Pydantic inputs format:
            input_model = getattr(skill, "input_model", None)
            if input_model:
                try:
                    if hasattr(input_model, "model_validate"):
                        valid_data = input_model.model_validate(args)
                    else:
                        valid_data = input_model(**args)
                    dump = getattr(valid_data, "model_dump", None)
                    if callable(dump):
                        return {"valid": True, "args": dump()}
                    legacy_dump = getattr(valid_data, "dict", None)
                    if callable(legacy_dump):
                        return {"valid": True, "args": legacy_dump()}
                    return {"valid": True, "args": dict(valid_data)}
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation("function_calling_adapter", e)
                    return {"valid": False, "error": f"Pydantic Validation Error: {e}"}

            return {"valid": True, "args": args}

        # Legacy support
        skill = self.registry.load_skill(tool_name)
        if not skill:
            return {"valid": False, "error": f"Tool '{tool_name}' not found."}

        return {"valid": True, "args": args}

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Executes a tool via CapabilityEngine."""
        logger.info("⚙️ Mind using: %s", tool_name)
        try:
            from core.capability_engine import CapabilityEngine

            if isinstance(self.registry, CapabilityEngine):
                result = await self.registry.execute(
                    tool_name, args, {"source": "autonomous_brain"}
                )
            elif self.router:
                result = await self.router.execute(
                    {"tool": tool_name, "params": args}, {"source": "autonomous_brain"}
                )
            else:
                # Legacy direct execution
                skill = self.registry.load_skill(tool_name)
                if not skill:
                    return f"Error: {tool_name} not found"
                result = await skill.execute(args, {"source": "autonomous_brain"})

            return json.dumps(result, indent=2)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("function_calling_adapter", e)
            logger.error("Tool execution failed: %s", e)
            return f"Error: {str(e)}"


# Logic to be used in LocalAgentClient
