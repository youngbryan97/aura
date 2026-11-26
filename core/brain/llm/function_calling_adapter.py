"""Function Calling Adapter.
Bridges local LLMs to Aura's Skill Registry.
Ensures Mind/Body alignment: 'Aura says, Aura does.'

Hardening (CP126): every execution now validates its arguments against the
registered schema BEFORE dispatch, runs under a deadline, and returns a
size-bounded, secret-redacted result. Tool definitions are structurally
validated (duplicate names refused), the ungoverned legacy direct-execution
path is opt-in, and exception text is summarized rather than echoed verbatim to
the model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("LLM.FunctionAdapter")

_TOOL_DEADLINE_S = float(os.environ.get("AURA_TOOL_DEADLINE_S", "120") or "120")
_MAX_RESULT_CHARS = 32 * 1024
_MAX_ARGS = 64
_MAX_ERROR_CHARS = 300

_SECRET_KEY_MARKERS = ("api_key", "secret", "password", "passwd", "token", "credential", "auth")
_URL_CRED_RE = re.compile(r"([a-z]+://)[^/\s:@]+:[^/\s@]+@", re.IGNORECASE)


def _legacy_direct_execution_allowed() -> bool:
    """The ungoverned direct-skill path is opt-in (c7c582cf).

    Without CapabilityEngine or a router there is no central authority, quota,
    or receipt — so that path refuses by default instead of silently bypassing
    capability governance.
    """
    return str(os.environ.get("AURA_ALLOW_UNGOVERNED_TOOL_EXECUTION", "")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def _redact(value: Any, _depth: int = 0) -> Any:
    """Drop secret-bearing keys/values from a tool result before it reaches the model."""
    if _depth > 6:
        return "…"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and any(m in k.lower() for m in _SECRET_KEY_MARKERS):
                out[k] = "[REDACTED]"
            else:
                out[k] = _redact(v, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact(v, _depth + 1) for v in value[:200]]
    if isinstance(value, str):
        return _URL_CRED_RE.sub(r"\1***:***@", value)
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    return value


def _safe_error(prefix: str, exc: BaseException) -> str:
    """Summarize an exception for the model: type + bounded message, no argument
    values echoed back verbatim (c7971b5c)."""
    detail = " ".join(str(exc).split())[:_MAX_ERROR_CHARS]
    detail = _URL_CRED_RE.sub(r"\1***:***@", detail)
    return f"{prefix}: {type(exc).__name__}: {detail}"


def _serialize_result(result: Any) -> str:
    """Redact, serialize defensively, and bound a tool result (d7eb693e, 1f8794)."""
    redacted = _redact(result)
    try:
        text = json.dumps(redacted, indent=2, default=str)
    except (TypeError, ValueError) as exc:
        # A non-serializable result must not raise AFTER the tool already ran —
        # the effect happened, so report it as a serialization problem.
        text = json.dumps(
            {
                "ok": True,
                "serialization_error": _safe_error("result was not JSON-serializable", exc),
                "result_repr": repr(redacted)[:2000],
            },
            indent=2,
        )
    if len(text) > _MAX_RESULT_CHARS:
        return text[:_MAX_RESULT_CHARS] + f"\n... [truncated {len(text) - _MAX_RESULT_CHARS} chars]"
    return text


def _legacy_property_schema(spec: Any) -> dict[str, Any]:
    """Honor declared types/description instead of coercing everything to a
    required string (7c8082f5)."""
    if isinstance(spec, dict):
        prop = {k: v for k, v in spec.items() if k in {"type", "description", "enum", "default", "items"}}
        prop.setdefault("type", "string")
        return prop
    return {"type": "string", "description": str(spec)}


def _legacy_required(inputs: dict[str, Any]) -> list[str]:
    """Only genuinely required inputs are marked required."""
    required = []
    for name, spec in inputs.items():
        if isinstance(spec, dict):
            if spec.get("required") is False or "default" in spec or spec.get("optional") is True:
                continue
        required.append(name)
    return required


class FunctionCallingAdapter:
    def __init__(self, registry, router=None):
        self.registry = registry
        self.router = router  # Usually the same as registry now

    def get_tool_definitions(self) -> dict[str, dict[str, Any]]:
        """Converts Aura's skill registry to JSON tool descriptions."""
        from core.capability_engine import CapabilityEngine

        if isinstance(self.registry, CapabilityEngine):
            defs: dict[str, dict[str, Any]] = {}
            for t in self.registry.get_tool_definitions():
                # Structural validation: a malformed entry is skipped and a
                # duplicate name is refused rather than silently overwriting an
                # existing tool (de629d00).
                if not isinstance(t, dict):
                    logger.warning("Skipping non-mapping tool definition.")
                    continue
                fn = t.get("function")
                if not isinstance(fn, dict):
                    logger.warning("Skipping tool definition without a 'function' mapping.")
                    continue
                name = fn.get("name")
                if not isinstance(name, str) or not name.strip():
                    logger.warning("Skipping tool definition without a usable name.")
                    continue
                if name in defs:
                    logger.error("Duplicate tool definition refused: %s", name)
                    continue
                defs[name] = fn
            return defs

        tools: dict[str, dict[str, Any]] = {}
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
            if not isinstance(inputs, dict):
                inputs = {}

            tools[name] = {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {k: _legacy_property_schema(v) for k, v in inputs.items()},
                    "required": _legacy_required(inputs),
                },
            }
        return tools

    @staticmethod
    def _structural_arg_check(args: Any) -> str:
        """Shape/size floor applied even when no schema model exists (b4e37de4)."""
        if not isinstance(args, dict):
            return "arguments must be a JSON object"
        if len(args) > _MAX_ARGS:
            return f"too many arguments (>{_MAX_ARGS})"
        for key in args:
            if not isinstance(key, str) or not key:
                return "argument names must be non-empty strings"
        return ""

    def validate_tool_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Strictly validates args against the registered skill schema before execution."""

        shape_error = self._structural_arg_check(args)
        if shape_error:
            return {"valid": False, "error": f"Validation Error: {shape_error}"}

        from core.capability_engine import CapabilityEngine

        if isinstance(self.registry, CapabilityEngine):
            skill = self.registry.skills.get(tool_name)
            if not skill:
                return {"valid": False, "error": f"Tool '{tool_name}' not found in registry."}

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
                    # Summarized, not the raw pydantic dump (which echoes the
                    # offending argument values back to the model).
                    return {"valid": False, "error": _safe_error("Validation Error", e)}

            # No declarative model: the args pass only the structural floor, and
            # the caller is told the schema was not enforced.
            return {"valid": True, "args": args, "schema_enforced": False}

        # Legacy support
        skill = self.registry.load_skill(tool_name)
        if not skill:
            return {"valid": False, "error": f"Tool '{tool_name}' not found."}

        return {"valid": True, "args": args, "schema_enforced": False}

    async def execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        deadline_s: float = _TOOL_DEADLINE_S,
    ) -> str:
        """Executes a tool via CapabilityEngine, after validating its arguments."""
        logger.info("Mind using: %s", tool_name)

        # The strict validator is no longer dead code — nothing dispatches
        # unvalidated caller arguments (820e9983).
        verdict = self.validate_tool_args(tool_name, args)
        if not verdict.get("valid"):
            return f"Error: {verdict.get('error') or 'invalid arguments'}"
        call_args = verdict.get("args", args)

        # Caller-supplied identity is forwarded when present; the static source
        # remains the fallback (5acd8c38 is only partially addressed — a signed
        # principal is a capability-engine-wide change).
        call_ctx = {"source": "autonomous_brain"}
        if isinstance(context, dict):
            call_ctx.update(context)
            call_ctx.setdefault("source", "autonomous_brain")

        try:
            from core.capability_engine import CapabilityEngine

            if isinstance(self.registry, CapabilityEngine):
                coro = self.registry.execute(tool_name, call_args, call_ctx)
            elif self.router:
                coro = self.router.execute({"tool": tool_name, "params": call_args}, call_ctx)
            else:
                if not _legacy_direct_execution_allowed():
                    return (
                        f"Error: {tool_name} cannot run — no CapabilityEngine or router is "
                        "available and ungoverned direct execution is disabled"
                    )
                skill = self.registry.load_skill(tool_name)
                if not skill:
                    return f"Error: {tool_name} not found"
                coro = skill.safe_execute(call_args, call_ctx)

            # A hung tool can no longer stall the calling turn indefinitely
            # (297f2287).
            result = await asyncio.wait_for(coro, timeout=deadline_s)
            return _serialize_result(result)
        except TimeoutError:
            logger.warning("Tool %s exceeded its %.0fs deadline.", tool_name, deadline_s)
            return f"Error: {tool_name} exceeded its {deadline_s:.0f}s deadline (outcome uncertain)"
        except asyncio.CancelledError:
            raise
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError,
                KeyError, OSError, ConnectionError) as e:
            # Broadened from the original three classes so common tool failures
            # stay inside the adapter's string contract (31f805b3).
            record_degradation("function_calling_adapter", e)
            logger.error("Tool execution failed: %s", e)
            return f"Error: {_safe_error('tool execution failed', e)}"


# Logic to be used in LocalAgentClient
