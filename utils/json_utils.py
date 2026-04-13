import json
import logging
import re
from typing import Any

logger = logging.getLogger("Utils.JSON")

JsonObject = dict[str, Any]
JsonValue = JsonObject | list[Any]


def extract_json(text: str | Any) -> JsonValue | None:
    """
    Robustly extract JSON from a string, handling markdown code blocks,
    surrounding text, and common LLM formatting issues.
    """
    text_str = text if isinstance(text, str) else str(text)

    # --- 🛑 PHASE 19.3: STALLING PHRASE TRAP ---
    if "Processing deeper reflections" in text_str or text_str.strip() == "":
        logger.warning("Caught non-JSON cognitive stalling phrase. Rejecting immediately.")
        return None 
    # ------------------------------------------

    # 1. Try simple parse first (fast path)
    try:
        parsed = json.loads(text_str)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code blocks
    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(pattern, text_str, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Find first { and last }
    start = text_str.find('{')
    end = text_str.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text_str[start:end+1]
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            # Try to fix common issues if needed, but for now just log
            pass
            
    logger.warning("Failed to extract JSON from text: %s...", text_str[:100])
    return None

def validate_with_schema(data: Any, schema: JsonObject | str) -> bool:
    """
    Very basic JSON schema validation. 
    In a full system, this would use 'jsonschema' library.
    For Aura's sovereign needs, we implement a lightweight recursive checker.
    """
    if not isinstance(data, dict) and not isinstance(data, list):
        return False
        
    # If schema is just a type hint (Ollama 'format' style)
    if isinstance(schema, str):
        if schema == "json":
            return isinstance(data, (dict, list))
        return True
        
    # Basic structural check
    if "type" in schema:
        expected_type = schema["type"]
        if expected_type == "object" and not isinstance(data, dict):
            return False
        if expected_type == "array" and not isinstance(data, list):
            return False
        
    if "properties" in schema and isinstance(data, dict):
        for prop, _prop_schema in schema["properties"].items():
            if prop in schema.get("required", []) and prop not in data:
                return False
            # Optional: recurse or check types
            
    return True
