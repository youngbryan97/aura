"""What a tool result is cut to and serialized as for her model: the bounded evidence each call hands back.

Lifted whole out of `mlx_client`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


#: What a result carries of the runtime's own receipts: whether authority closed after the call, the receipt ids, how
#: long it took and how often it was tried. The gateway that wrote them keeps them. Each of her goal's read_source
#: results opened with 200 characters of them, on every call of every pass (10 November 2026).
_RECEIPT_FIELDS = frozenset({
    "authority_closure", "authority_closure_effect_scope", "token_revoked", "standing_authority_closed",
    "intent_closed", "governance_receipt_id", "agency_receipt_id", "expectation_receipt_id",
    "deliberation_receipts", "duration_ms", "retries",
})


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
        # What the tool found, and whether it went: not the receipts of its going.
        model_result = {key: value for key, value in model_result.items() if key not in _RECEIPT_FIELDS}
        serialized = json.dumps(
            model_result,
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        serialized = str(model_result)
    return _truncate_tool_result(serialized, limit=limit)




def _prior_tool_turns(turns: Any) -> list[dict[str, Any]]:
    """Earlier turns of hers as the loop's own messages: each call with the tool turn that answered it, and what she
    said in words between them. A call without its answer is left out, since a template renders a call left
    unanswered as one still waiting, and so is anything else.

    Walked in order, not in pairs: her words are a single assistant message, and pairing every two turns by position
    misaligned every call after the first thing she said (10 November 2026, her goal pass given her own words).
    """
    if not isinstance(turns, (list, tuple)):
        return []
    kept: list[dict[str, Any]] = []
    at = 0
    while at < len(turns):
        turn = turns[at]
        if not isinstance(turn, Mapping):
            at += 1
            continue
        calls = turn.get("tool_calls")
        if turn.get("role") == "assistant" and not calls:
            words = turn.get("content")
            if isinstance(words, str) and words.strip():
                kept.append({"role": "assistant", "content": words})
            at += 1
            continue
        answer = turns[at + 1] if at + 1 < len(turns) else None
        if (turn.get("role") == "assistant" and isinstance(calls, list) and len(calls) == 1
                and isinstance(answer, Mapping) and answer.get("role") == "tool"
                and answer.get("tool_call_id") == calls[0].get("id")):
            kept.extend([dict(turn), dict(answer)])
            at += 2
            continue
        at += 1
    return kept


def _tool_loop_evidence_messages(evidence: Any) -> list[dict[str, Any]]:
    """Grounding blocks the turn already holds, as messages the loop can read.

    Only blocks a skill actually produced are carried: the conversational
    scaffold belongs to the reply, and wrapping a tool call in it produced an
    immediate end-of-turn. Bounded, because a document can be large and a tool
    prompt that outgrows its deadline answers nothing.
    """
    if not isinstance(evidence, (list, tuple)):
        return []
    try:
        from core.utils.injected_blocks import carries_read_evidence, stamp_grounding
    except ImportError:  # pragma: no cover - the module ships with the runtime
        return []
    from .mlx_client import _TOOL_LOOP_EVIDENCE_CHARS

    carried: list[dict[str, Any]] = []
    budget = _TOOL_LOOP_EVIDENCE_CHARS
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        if not carries_read_evidence(dict(item)):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        content = content[:budget]
        budget -= len(content)
        carried.append(stamp_grounding({"role": "system", "content": content}))
        if budget <= 0:
            break
    return carried
