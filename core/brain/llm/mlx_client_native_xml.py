"""A native XML tool call's parameters, read against the schema the tool advertised.

Lifted whole out of `mlx_client`, which imports them straight back: every caller and every patch that names them there
still finds them.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


def _tool_parameter_properties(
    tool_definitions: Mapping[str, Any] | None,
    function_name: str,
) -> dict[str, Any]:
    if not isinstance(tool_definitions, Mapping):
        return {}
    definition = tool_definitions.get(function_name)
    if not isinstance(definition, Mapping):
        return {}
    if isinstance(definition.get("function"), Mapping):
        definition = definition["function"]
    parameters = definition.get("parameters")
    if not isinstance(parameters, Mapping):
        return {}
    properties = parameters.get("properties")
    return dict(properties) if isinstance(properties, Mapping) else {}


def _strip_native_xml_framing(value: str) -> str:
    """Remove template framing newlines without changing parameter content."""

    result = value
    if result.startswith("\r\n"):
        result = result[2:]
    elif result.startswith("\n"):
        result = result[1:]
    if result.endswith("\r\n"):
        result = result[:-2]
    elif result.endswith("\n"):
        result = result[:-1]
    return result


def _coerce_native_xml_parameter(value: str, property_schema: Any) -> Any:
    """Convert one XML value only when the advertised schema names its type."""

    schema = property_schema if isinstance(property_schema, Mapping) else {}
    expected = schema.get("type")
    if expected == "string" or not isinstance(expected, str):
        return value
    compact = value.strip()
    if expected == "integer":
        if re.fullmatch(r"[-+]?\d+", compact) is None:
            raise ValueError("expected integer")
        return int(compact)
    if expected == "number":
        if re.fullmatch(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            compact,
        ) is None:
            raise ValueError("expected number")
        return float(compact) if any(char in compact for char in ".eE") else int(compact)
    if expected == "boolean":
        if compact.casefold() == "true":
            return True
        if compact.casefold() == "false":
            return False
        raise ValueError("expected boolean")
    if expected in {"array", "object"}:
        try:
            parsed = json.loads(compact)
        except json.JSONDecodeError as exc:
            raise ValueError(f"expected JSON {expected}") from exc
        if expected == "array" and not isinstance(parsed, list):
            raise ValueError("expected JSON array")
        if expected == "object" and not isinstance(parsed, dict):
            raise ValueError("expected JSON object")
        return parsed
    if expected == "null":
        if compact.casefold() != "null":
            raise ValueError("expected null")
        return None
    return value
