from __future__ import annotations

from typing import Any


def tool_result_is_deferred(result: object) -> bool:
    """Recognize admission deferrals without treating them as executions."""
    if isinstance(result, dict):
        status = str(result.get("status") or "").strip().lower()
        return (
            result.get("deferred") is True
            or status in {"deferred", "deferred_by_executive"}
            or any(
                str(result.get(key) or "").strip().lower().startswith("background_deferred:")
                for key in ("error", "reason")
            )
        )
    return "background_deferred:" in str(result or "").lower()


def _truncate_text(value: Any, limit: int = 1200) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = text[:limit].rstrip()
    boundary = head.rfind(" ")
    if boundary > 0:
        head = head[:boundary].rstrip()
    return head + "…[result truncated]"


def _compact_string_list(values: Any, *, limit: int = 3, item_limit: int = 240) -> list[str]:
    compact: list[str] = []
    for item in list(values or [])[:limit]:
        text = _truncate_text(item, limit=item_limit)
        if text:
            compact.append(text)
    return compact


def _mark_tool_provenance(result: object) -> None:
    """Record that this turn ingested tool output.

    Contract-checking the SHAPE of a tool result says nothing about who wrote
    the text inside it: a search result, a fetched page summary, a file listing
    and a repository README all arrive here as validated dicts full of somebody
    else's prose. That is the whole of indirect prompt injection — the payload
    is well-formed and the content is an instruction.

    Never raises: this is a note about the turn, and failing to take the note
    must not fail the result. It is recorded instead, because a missing note
    means an action gate answers "trusted" for a turn that read a stranger's
    text.
    """
    try:
        from core.security.content_provenance import ProvenanceClass, record_ingest

        detail = ""
        if isinstance(result, dict):
            detail = str(result.get("source") or result.get("url") or "")[:120]
        record_ingest(ProvenanceClass.TOOL_OUTPUT, detail or "tool result")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation(
            "tool_result_contracts",
            exc,
            severity="warning",
            action="tool output provenance not recorded; action gates will treat this turn as untainted",
            enforce_failure_policy=False,
        )


def compact_result_payload(result: object) -> dict[str, object]:
    """Normalize tool/task outputs into a compact, prompt-safe payload."""
    _mark_tool_provenance(result)
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
        "deferred",
        "reason",
        "retryable",
        "retry_after_s",
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
        "action",
        "opened",
        "typed",
        "hotkey",
        "scrolled",
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
