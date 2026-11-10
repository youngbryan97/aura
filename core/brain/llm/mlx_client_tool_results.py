"""What a tool result is cut to and serialized as for her model: the bounded evidence each call hands back.

Lifted whole out of `mlx_client`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _truncate_tool_result(result: Any, *, limit: int = 4000) -> str:
    """Bound a tool result WITHOUT cutting structured output mid-value.

    CP126 abd93abf: a raw character slice produced syntactically broken JSON
    that the model then reasoned over as if it were the real result.
    """
    text = result if isinstance(result, str) else str(result)
    if len(text) <= limit:
        return text
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            # Not a failure: a tool result that is not JSON is not JSON, which is what this is deciding.
            parsed = None
        if parsed is not None:
            from .tool_result_room import cut_the_longest_values

            fitted = cut_the_longest_values(parsed, original_chars=len(text), limit=limit)
            if fitted is not None:
                return fitted
            # Re-emit a VALID, explicitly-marked truncation envelope instead
            # of a broken fragment.
            preview = json.dumps(
                parsed,
                default=str,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )

            def envelope(candidate: str) -> str:
                return json.dumps(
                    {
                        "truncated": True,
                        "original_chars": len(text),
                        "note": "Result exceeded the context budget; preview only.",
                        "preview": candidate,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )

            if len(envelope("")) <= limit:
                low = 0
                high = len(preview)
                best = ""
                while low <= high:
                    middle = (low + high) // 2
                    candidate = envelope(preview[:middle])
                    if len(candidate) <= limit:
                        best = candidate
                        low = middle + 1
                    else:
                        high = middle - 1
                return best
    marker = "\n\n...[OUTPUT TRUNCATED FOR LENGTH]..."
    if limit <= len(marker):
        return marker[:limit]
    return text[: limit - len(marker)] + marker


def _serialize_tool_result_for_model(
    tool_name: str,
    raw_result: Any,
    *,
    limit: int = 4000,
) -> str:
    """Serialize the exact bounded evidence contract fed to the model."""
    from .mlx_client import (
        _truncate_tool_result,
    )


    model_result = raw_result
    if tool_name == "code_repl":
        from core.skills.code_repl import serialize_code_repl_model_result

        if not isinstance(raw_result, Mapping):
            raw_result = {
                "ok": False,
                "status": "error",
                "error": (
                    "code_repl returned a non-mapping result "
                    f"({type(raw_result).__name__})"
                ),
                "engine": "capability_engine",
            }
        return serialize_code_repl_model_result(raw_result, limit=limit)
    if isinstance(model_result, Mapping):
        serialized = json.dumps(
            model_result,
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        serialized = str(model_result)
    return _truncate_tool_result(serialized, limit=limit)


