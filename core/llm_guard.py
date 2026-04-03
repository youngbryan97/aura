"""
LLM output validation and sanitization.

Three concerns:
1. Schema validation — does the JSON have the right shape?
2. Content sanitization — remove prompt injection attempts from tool results
3. Hallucination detection — flag responses that look fabricated
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.LLMGuard")

# Prompt injection patterns in tool RESULTS (not in user messages)
# These indicate a tool result is trying to hijack the agent's next action
_INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions?",
    r"you are now",
    r"new system prompt",
    r"forget (your|the|all) (previous|prior|above)",
    r"<\|system\|>",
    r"\[SYSTEM\]",
    r"override (your|the) (instructions?|prompt|guidelines?)",
    r"act as (a )?(different|another|new)",
    r"DAN mode",
    r"jailbreak",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def sanitize_tool_result(result: Any) -> Tuple[Any, bool]:
    """
    Scan a tool result for prompt injection attempts.

    Returns:
        (sanitized_result, was_modified)
    """
    if not isinstance(result, str):
        result_str = json.dumps(result) if not isinstance(result, str) else result
    else:
        result_str = result

    injections_found = []
    for pattern in _INJECTION_RE:
        if pattern.search(result_str):
            injections_found.append(pattern.pattern)

    if injections_found:
        logger.warning(
            "🛡️ Prompt injection detected in tool result. Patterns: %s",
            injections_found,
        )
        # Return a sanitized placeholder instead of the raw result
        return (
            f"[TOOL RESULT SANITIZED — potential injection detected. "
            f"Original length: {len(result_str)} chars]",
            True,
        )

    return result, False


def validate_tool_call(tool_call: Dict[str, Any], registered_skills: List[str]) -> Tuple[bool, str]:
    """
    Validate a tool call from the LLM before executing it.

    Returns:
        (is_valid, error_message_or_empty)
    """
    if not isinstance(tool_call, dict):
        return False, f"Tool call must be a dict, got {type(tool_call)}"

    name = tool_call.get("name") or tool_call.get("function", {}).get("name")
    if not name:
        return False, "Tool call missing 'name' field"

    if not isinstance(name, str) or not re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$", name):
        return False, f"Invalid tool name format: '{name}'"

    if registered_skills and name not in registered_skills:
        # Don't hard-fail — Hephaestus may need to forge it — but log it
        logger.debug("Tool call for unregistered skill '%s'", name)

    params = tool_call.get("parameters") or tool_call.get("arguments") or {}
    if not isinstance(params, dict):
        try:
            params = json.loads(params) if isinstance(params, str) else {}
        except json.JSONDecodeError as e:
            return False, f"Tool call parameters not valid JSON: {e}"

    return True, ""


def repair_json(text: str) -> str:
    """
    Attempt to repair common LLM JSON malformations.
    """
    # 1. Remove comments (often added by some models)
    text = re.sub(r'//.*?\n|/\*.*?\*/', '', text, flags=re.S)
    
    # 2. Fix trailing commas in objects and arrays
    text = re.sub(r',\s*([\]}])', r'\1', text)
    
    # 3. Fix unquoted keys (e.g., {key: "value"} -> {"key": "value"})
    text = re.sub(r'([{,]\s*)([a-zA-Z0-9_\-]+)(\s*:)', r'\1"\2"\3', text)
    
    # 4. Convert single quotes to double quotes for keys/values
    # This is tricky because we don't want to break already quoted strings with escaped single quotes.
    # We use a simple approach for common cases.
    # 4. Convert single quotes to double quotes for keys/values
    # Improved regex: Only target keys 'key': or values : 'val' to avoid breaking logic.
    text = re.sub(r"\'([a-zA-Z0-9_/ \-]+)\'(?=\s*:)", r'"\1"', text)
    text = re.sub(r"(?<=:\s*)\'([^']+)\'", r'"\1"', text)

    # 5. Fix common "Infinity" or "NaN" (not valid JSON)
    text = text.replace(': Infinity', ': null').replace(': NaN', ': null')

    return text.strip()


def validate_json_response(raw: str, expected_keys: Optional[List[str]] = None) -> Tuple[bool, Any, str]:
    """
    Parse and validate an LLM JSON response with multi-stage repair.

    Returns:
        (success, parsed_object_or_None, error_message)
    """
    # Strip markdown code fences
    clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
    clean = clean.rstrip("`").strip()

    # Stage 1: Standard parse
    try:
        obj = json.loads(clean)
        return _validate_keys(obj, expected_keys)
    except json.JSONDecodeError as _e:
        logger.debug('Ignored json.JSONDecodeError in llm_guard.py: %s', _e)

    # Stage 2: Extraction parse (find first { ... } or [ ... ])
    match = re.search(r"(\{.*\}|\[.*\])", clean, re.DOTALL)
    if match:
        extracted = match.group(1)
        try:
            obj = json.loads(extracted)
            return _validate_keys(obj, expected_keys)
        except json.JSONDecodeError:
            # Stage 3: Repair & Parse
            repaired = repair_json(extracted)
            try:
                obj = json.loads(repaired)
                return _validate_keys(obj, expected_keys)
            except json.JSONDecodeError as e:
                return False, None, f"JSON repair failed: {e}"
    
    return False, None, "No JSON object or array found in response"


def _validate_keys(obj: Any, expected_keys: Optional[List[str]] = None) -> Tuple[bool, Any, str]:
    """Helper to check for mandatory keys."""
    if expected_keys and isinstance(obj, dict):
        missing = [k for k in expected_keys if k not in obj]
        if missing:
            return False, obj, f"Response missing expected keys: {missing}"
    return True, obj, ""


def truncate_for_context(text: str, max_chars: int = 12000, label: str = "") -> str:
    """
    Safely truncate tool results / memory retrievals to prevent context overflow.
    Adds a truncation notice so the LLM knows it received partial data.
    """
    if len(text) <= max_chars:
        return text
    suffix = f"\n...[{label or 'content'} truncated — {len(text) - max_chars} chars omitted]"
    trunc_len = max(0, max_chars - len(suffix))
    return text[:trunc_len] + suffix
