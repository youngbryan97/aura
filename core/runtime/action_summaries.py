"""What an action's parameters look like in a record, with secrets left out.

Lifted from core/runtime/action_executor.py, which receipts every action it
runs: a parameter whose name marks a secret is redacted, bulk text is reduced
to its type and length, and an address keeps its scheme, host and path only.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Mapping
from typing import Any

SENSITIVE_PARAM_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "secret",
    "session_id",
    "token",
)


def safe_action_summary(action_name: str, params: Mapping[str, Any]) -> str:
    summarized: dict[str, Any] = {}
    for key, value in params.items():
        key_text = str(key)
        if any(marker in key_text.casefold() for marker in SENSITIVE_PARAM_MARKERS):
            summarized[key_text] = "[REDACTED]"
        elif key_text in {"content", "payload", "script", "text"}:
            length = len(value) if hasattr(value, "__len__") else 0
            summarized[key_text] = f"<{type(value).__name__}:{length}>"
        elif isinstance(value, Mapping):
            summarized[key_text] = f"<mapping:{len(value)}>"
        elif isinstance(value, (list, tuple, set)):
            summarized[key_text] = safe_sequence_summary(value)
        elif key_text.casefold() in {"uri", "url"}:
            summarized[key_text] = safe_url_summary(str(value))
        else:
            summarized[key_text] = str(value)[:160]
    encoded = json.dumps(summarized, sort_keys=True, default=str)
    return f"{action_name} params={encoded}"[:1000]


def safe_sequence_summary(value: Any) -> list[Any]:
    summarized: list[Any] = []
    redact_next = False
    for raw_item in list(value)[:16]:
        if isinstance(raw_item, Mapping):
            summarized.append(safe_nested_mapping_summary(raw_item))
            continue
        item = str(raw_item)
        lowered = item.casefold()
        if redact_next:
            summarized.append("[REDACTED]")
            redact_next = False
            continue
        if lowered.startswith(("http://", "https://")):
            summarized.append(safe_url_summary(item))
            continue
        if any(marker in lowered for marker in SENSITIVE_PARAM_MARKERS):
            if "=" in item:
                summarized.append(item.split("=", 1)[0][:80] + "=[REDACTED]")
            elif item.lstrip().startswith("-"):
                summarized.append(item[:80])
                redact_next = True
            else:
                summarized.append("[REDACTED]")
            continue
        summarized.append(item[:80])
    return summarized


def safe_nested_mapping_summary(value: Mapping[Any, Any]) -> dict[str, Any]:
    summarized: dict[str, Any] = {}
    for raw_key, raw_value in list(value.items())[:32]:
        key = str(raw_key)[:80]
        lowered = key.casefold()
        if lowered == "value" or any(
            marker in lowered for marker in SENSITIVE_PARAM_MARKERS
        ):
            summarized[key] = "[REDACTED]"
        elif lowered in {"content", "payload", "script", "text"}:
            length = len(raw_value) if hasattr(raw_value, "__len__") else 0
            summarized[key] = f"<{type(raw_value).__name__}:{length}>"
        elif lowered in {"uri", "url"}:
            summarized[key] = safe_url_summary(str(raw_value))
        elif isinstance(raw_value, Mapping):
            summarized[key] = safe_nested_mapping_summary(raw_value)
        elif isinstance(raw_value, (list, tuple, set)):
            summarized[key] = safe_sequence_summary(raw_value)
        else:
            summarized[key] = str(raw_value)[:160]
    return summarized


def safe_url_summary(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return urllib.parse.urlunsplit(
            (parsed.scheme, host + port, parsed.path, "", "")
        )[:240]
    except ValueError:
        return "<invalid-url>"


__all__ = [
    "SENSITIVE_PARAM_MARKERS",
    "safe_action_summary",
    "safe_nested_mapping_summary",
    "safe_sequence_summary",
    "safe_url_summary",
]
