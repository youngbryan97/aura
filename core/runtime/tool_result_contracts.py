from __future__ import annotations

from typing import Any


def _truncate_text(value: Any, limit: int = 1200) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "…[result truncated]"


def _compact_string_list(values: Any, *, limit: int = 3, item_limit: int = 240) -> list[str]:
    compact: list[str] = []
    for item in list(values or [])[:limit]:
        text = _truncate_text(item, limit=item_limit)
        if text:
            compact.append(text)
    return compact


def compact_result_payload(result: object) -> dict[str, object]:
    """Normalize tool/task outputs into a compact, prompt-safe payload."""
    if not isinstance(result, dict):
        text = _truncate_text(result)
        return {"result": text} if text else {}

    payload: dict[str, object] = {}
    for key in (
        "ok",
        "summary",
        "content",
        "result",
        "title",
        "source",
        "url",
        "message",
        "time",
        "readable",
        "error",
        "status",
        "task_id",
        "commitment_id",
        "objective",
        "requested_objective",
        "continued_from_task_id",
        "plan_id",
        "trace_id",
        "command",
        "phase",
        "active_step",
        "steps_completed",
        "steps_total",
        "duration_s",
        "verified",
        "verification",
        "verification_summary",
        "repair_count",
        "attempts",
        "succeeded",
        "return_code",
        "exit_code",
        "cwd",
        "stdout",
        "stderr",
    ):
        value = result.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            payload[key] = _truncate_text(value)
        else:
            payload[key] = value

    compact_files = _compact_string_list(result.get("files"), limit=5, item_limit=180)
    if compact_files:
        payload["files"] = compact_files

    compact_evidence = _compact_string_list(result.get("evidence"), limit=4, item_limit=260)
    if compact_evidence:
        payload["evidence"] = compact_evidence

    compact_results: list[dict[str, str]] = []
    for item in list(result.get("results") or [])[:3]:
        if not isinstance(item, dict):
            continue
        compact_item: dict[str, str] = {}
        for key in ("title", "snippet", "url"):
            value = item.get(key)
            if value in (None, ""):
                continue
            compact_item[key] = _truncate_text(value, limit=400)
        if compact_item:
            compact_results.append(compact_item)
    if compact_results:
        payload["results"] = compact_results

    if not payload:
        text = _truncate_text(result)
        if text:
            payload["result"] = text
    return payload
