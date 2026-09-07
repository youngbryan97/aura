"""Desktop objectives, the searches they need, and what the screen showed.

Lifted out of `interface/routes/chat.py`, which was 24,185 lines. These
functions read a desktop request, decide whether it can be executed without
first spending a foreground model call, gather the search evidence a request
requires, and say what a run actually changed. None of them touch the chat
module's own state — they take what they need and return an answer — which is
what made them the first ones that could leave.
"""
from __future__ import annotations

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.structured_input import (
    analyze_prompt_shape,
    answer_surface_token_floor,
)
from core.utils.completed_capability import make_completed_capability_evidence
from core.utils.intent_normalization import normalize_memory_intent_text
from fastapi import Request
from interface.routes import chat_capability_inventory as _chat_capability_inventory  # noqa: E402
from interface.routes import chat_desktop_objective as _chat_desktop_objective  # noqa: E402
from interface.routes import chat_desktop_repair as _chat_desktop_repair  # noqa: E402
from interface.routes import chat_memory_state as _chat_memory_state  # noqa: E402
from interface.routes import chat_preflight as _chat_preflight  # noqa: E402
from interface.routes import chat_runtime_proof as _chat_runtime_proof  # noqa: E402
from interface.routes.chat_common import _CHAT_RECOVERABLE_ERRORS, _SEARCH_SKILL_NAMES, logger
from interface.routes.chat_turn_evidence import _build_explicit_local_file_artifact
from pathlib import Path
from typing import Any
import asyncio
import os
import re
import time

# Lifted alongside this module; imported rather than re-derived.
from .chat_lane_bookkeeping import (
    _conversation_lane_needs_instant_social_contract,
    _resolve_chat_response_contract,
    _user_requested_research_memory_save,
)
from .chat_reply_shaping import (
    _build_runtime_fact_status_fastpath_reply,
)


def _extract_repo_probe_request(user_message: str) -> dict[str, str] | None:
    text = str(user_message or "").strip()
    if not text:
        return None

    patterns = (
        (
            r"^(?:read|open|inspect)\s+([A-Za-z0-9_./~-]+\.[A-Za-z0-9]+)\s+and\s+tell me\s+the\s+first\s+non-comment\s+dependency\s+line[.?!]*$",
            "first_non_comment_dependency_line",
        ),
        (
            r"^(?:read|open|inspect)\s+([A-Za-z0-9_./~-]+\.[A-Za-z0-9]+)\s+and\s+tell me\s+the\s+first\s+non-comment\s+line[.?!]*$",
            "first_non_comment_line",
        ),
        (
            r"^(?:read|open|inspect)\s+([A-Za-z0-9_./~-]+\.[A-Za-z0-9]+)\s+and\s+tell me\s+how many\s+lines(?:\s+it\s+has)?[.?!]*$",
            "line_count",
        ),
    )
    for pattern, mode in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return {"target": match.group(1), "mode": mode}
    return None


def _desktop_live_reply_token_budget(
    user_message: str,
    *,
    capability_inventory_contract: bool,
    bounded_planning_contract: bool,
    runtime_fact_status_contract: bool,
    memory_state_contract: bool,
    memory_state_contract_covers_turn: bool = True,
) -> int:
    """Allocate live reply capacity from semantic workload, not route name.

    The desktop lane intentionally uses a compact prompt, but "compact" must
    not imply a small completion for multi-step planning.  Keeping this policy
    beside the route classifiers also prevents backend and live UI calls from
    silently receiving different reasoning budgets for the same request.

    A memory-state turn gets the small budget only when the memory request is
    the whole turn — the same rule the engine's ladder applies. The engine
    floors currently rescue a compound turn that arrives here capped at 384,
    which makes this a latent duplicate of a defect already fixed downstream
    rather than a live one; two places deciding the same thing differently is
    how it comes back.
    """

    if runtime_fact_status_contract or (
        memory_state_contract and memory_state_contract_covers_turn
    ):
        return 384
    if capability_inventory_contract:
        return 384

    shape = analyze_prompt_shape(user_message)
    structural_floor = answer_surface_token_floor(user_message)
    question_parts = int(getattr(shape, "question_parts", 0) or 0)
    extended = bool(
        bounded_planning_contract
        or getattr(shape, "prefers_extended_answer", False)
        or getattr(shape, "requires_single_reply_coverage", False)
        or question_parts >= 2
    )
    if extended:
        return max(1536, structural_floor)
    if len(str(user_message or "")) > 600:
        return max(1280, structural_floor)
    return max(896, structural_floor)


def _desktop_cognitive_failure_repair_target(reason: str) -> str:
    """Choose the narrowest implementation surface implicated by a failed turn."""

    normalized = str(reason or "").lower()
    if any(marker in normalized for marker in ("timeout", "no_thought", "empty")):
        return "core/brain/llm/mlx_client.py"
    if any(marker in normalized for marker in ("quality", "unsafe", "failure_envelope")):
        return "core/phases/response_generation.py"
    return "core/brain/cognitive_engine.py"


def _status_represents_governed_action_result(status: str | None) -> bool:
    proof_status = str(status or "").strip()
    if proof_status.startswith(
        (
            "live_proof",
            "desktop_objective",
            "program_dna",
            "rsi_self_improvement",
            "web_interlocutor",
        )
    ):
        return True
    return proof_status in {
        "desktop_objective",
        "desktop_task",
        "computer_use",
        "file_operation",
        "improve_own_code",
        "program_dna_reconstruct",
        "web_interlocutor",
    }


def _governed_desktop_response_authority(
    *,
    desktop_result: Any,
    action_episode: Any,
) -> tuple[bool, str]:
    """Prove an exact action-result serialization from its governed source.

    Successful actions require observable effect receipts. Failed actions have
    a different truth condition: a governed executor authoritatively reported
    that the requested effect did not occur. The action episode records which
    condition was established, so the response contract must consult it before
    applying the success-only verifier.
    """

    if isinstance(action_episode, dict) and action_episode.get("authority_proven") is True:
        reason = str(action_episode.get("authority_reason") or "").strip()
        return True, reason or "governed_action_episode"
    if isinstance(desktop_result, dict):
        return _chat_desktop_objective._verified_desktop_task_result(desktop_result)
    return False, "desktop_result_missing"


def _collect_governed_action_lane_status(status: str) -> dict[str, Any]:
    """Return truthful lane status for a completed governed action response.

    Tool/action results should carry their own success evidence. They must not
    falsely mark inference healthy, but the desktop UI also must not treat a
    stale post-action generation timeout as proof that the completed action
    failed. Runtime heartbeat remains the authority for kernel/inference health.
    """
    lane = _chat_preflight._collect_conversation_lane_status()
    lane["governed_action_result"] = True
    lane["governed_action_status"] = str(status or "governed_action")
    lane["governed_action_completed_at"] = time.time()
    if not bool(lane.get("conversation_ready", False)):
        lane["governed_action_health_note"] = (
            "governed action completed; heartbeat/required probes remain authoritative "
            "for inference readiness"
        )
    return lane


def _desktop_required_bounded_reply_status(
    user_message: str,
    reply_text: Any,
    lane: dict[str, Any] | None,
) -> str:
    """Classify governed bounded desktop replies before labeling full cognition.

    `_run_cognitive_engine_chat_turn()` may return deterministic contracts for
    low-risk desktop turns when the foreground model lane is cold, busy, or
    unsafe to allocate. Those replies are valid live-runtime behavior, but they
    are not evidence that the heavy CognitiveEngine completed a foreground
    generation. Keep the wire status precise so the UI and health gates cannot
    accidentally treat a bounded contract as a fully warm Cortex turn.
    """

    reply = str(reply_text or "").strip()
    if not reply:
        return ""

    def _matches_bounded_contract(expected: str | None) -> bool:
        if not expected:
            return False
        return _chat_memory_state._normalize_user_message(
            reply
        ) == _chat_memory_state._normalize_user_message(expected)

    lane_status = dict(lane or {})
    if _chat_desktop_repair._is_low_risk_social_continuity_request(
        user_message
    ) and _conversation_lane_needs_instant_social_contract(lane_status):
        if _matches_bounded_contract(
            _chat_desktop_repair._build_social_continuity_repair_reply(user_message)
        ):
            return "desktop_social_presence_contract"
    if _chat_preflight._is_explicit_capability_inventory_request(user_message):
        return "cognitive_engine_capability_inventory"
    if _matches_bounded_contract(_chat_desktop_repair._build_bounded_planning_reply(user_message)):
        return "cognitive_engine_bounded_planning"
    if _matches_bounded_contract(
        _chat_desktop_repair._build_failure_mode_surface_reply(user_message)
    ):
        return "cognitive_engine_failure_mode_surface"
    if _chat_preflight._is_runtime_fact_status_request(user_message):
        expected = _build_runtime_fact_status_fastpath_reply(user_message, lane_status)
        if _matches_bounded_contract(expected):
            return "runtime_fact_status"
    return ""


def _extract_explicit_local_file_path(user_message: str) -> str | None:
    text = str(user_message or "")
    if not re.search(r"\b(?:create|write|save|generate|build|make)\b", text, re.IGNORECASE):
        return None
    match = re.search(
        r"(?:to|at|as|into|path)\s+([A-Za-z0-9_./-]+\.(?:html|js|css|py|md|txt|json|csv))\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    candidate = match.group(1).strip()
    if candidate.startswith(("/", "../")) or ".." in Path(candidate).parts:
        return None
    return candidate


async def _execute_explicit_local_file_objective(user_message: str) -> dict[str, Any] | None:
    if _chat_runtime_proof._is_live_runtime_proof_request(user_message):
        return None
    path = _extract_explicit_local_file_path(user_message)
    if not path:
        return None
    content = _build_explicit_local_file_artifact(user_message, path)
    if content is None:
        return None
    result = await _chat_capability_inventory._execute_governed_live_skill(
        "file_operation",
        {"action": "write", "path": path, "content": content},
        objective=str(user_message or ""),
        extra_context={
            "route": "chat.explicit_local_file_objective",
            "origin": "desktop_ui",
            "source": "desktop_ui",
            "explicit_local_file_objective": True,
        },
    )
    if not isinstance(result, dict):
        result = {"ok": bool(result), "result": result}
    if not result.get("ok"):
        return {
            "ok": False,
            "response": (
                "I routed the file objective through governed file_operation, "
                f"but the write did not complete: {result.get('error') or result}."
            ),
            "status": "file_operation",
            "data": {"path": path, "result": result},
        }
    abs_path = (Path.cwd() / path).resolve()
    exists = abs_path.exists()
    return {
        "ok": exists,
        "response": (
            f"I created `{path}` through the governed file_operation path"
            f"{' and verified it exists on disk' if exists else ', but verification did not find it on disk'}."
        ),
        "status": "file_operation",
        "data": {
            "path": path,
            "absolute_path": str(abs_path),
            "exists": exists,
            "bytes": len(content.encode("utf-8")),
            "result": result,
        },
    }


def _should_collect_desktop_required_search_evidence(
    user_message: str,
) -> tuple[bool, str, Any | None]:
    if not str(user_message or "").strip():
        return False, "", None
    if _chat_preflight._looks_like_desktop_objective(user_message):
        return False, "", None
    # A file on this disk is not a live-search question.
    #
    # LIVE 2026-08-17: "read the file CONTRIBUTING.md and tell me the first
    # rule it states" resolved to a contract with requires_search=True and was
    # dispatched to web_search, which failed with an empty error. The turn
    # ended "I attempted to read the file and it failed", offering to check
    # whether the file exists — it was in the repo root the whole time, and no
    # search result could ever have answered the question.
    #
    # A filename that RESOLVES inside her roots settles this: the bytes are
    # local, so the evidence is local.
    try:
        from core.conversation.filesystem_check import requested_file_read

        named_file = requested_file_read(user_message)
    except _CHAT_RECOVERABLE_ERRORS:
        named_file = None
    if named_file is not None and named_file.exists:
        return False, "", None
    # A DIRECTORY on this disk settles it for the same reason.
    #
    # The reader above finds a named file; "debug the project at <path>" names
    # a folder, so this fell through.
    #
    # LIVE 2026-08-27: a request to diagnose a project ran a web search for the
    # user's whole sentence, path included, and came back with GitHub issues
    # about disk usage under /private/tmp/claude-501. The search itself then
    # threw all five results away as irrelevant — correctly — having spent the
    # turn's tool budget on a question whose evidence was on the disk.
    try:
        from core.language.named_paths import first_existing_path

        if first_existing_path(user_message) is not None:
            return False, "", None
    except _CHAT_RECOVERABLE_ERRORS:
        pass
    # A document the person addressed directly is not a search question
    # either, for the same reason: the bytes are AT that address.
    #
    # LIVE 2026-08-20: "read https://api.open-meteo.com/v1/forecast?...
    # &latitude=64.15 and tell me the temperature it reports" ran a 33-second
    # web_search whose results were the API's documentation, whose example
    # uses New York. The fetch succeeded on the same turn, and she answered
    # from the search: "the result seems to be a link to an API page with
    # different coordinates (New York City)".
    try:
        from core.intent.opaque_spans import first_named_url

        if first_named_url(user_message):
            return False, "", None
    except _CHAT_RECOVERABLE_ERRORS:
        pass
    contract = _resolve_chat_response_contract(user_message)
    if not contract or not getattr(contract, "requires_search", False):
        return False, "", contract
    required_skill = str(getattr(contract, "required_skill", "") or "web_search").strip()
    if required_skill and required_skill not in _SEARCH_SKILL_NAMES:
        return False, "", contract
    query = str(getattr(contract, "search_query", "") or user_message or "").strip()
    return True, query[:240], contract


def _search_result_entries(result: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    raw_entries: list[Any] = []
    for key in ("results", "sources", "items"):
        value = result.get(key)
        if isinstance(value, list):
            raw_entries.extend(value)
    if not raw_entries and any(
        result.get(key) for key in ("url", "source", "title", "summary", "answer")
    ):
        raw_entries.append(result)
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        title = " ".join(
            str(raw.get("title") or raw.get("name") or raw.get("source_title") or "").split()
        )
        url = " ".join(str(raw.get("url") or raw.get("href") or raw.get("source") or "").split())
        snippet = " ".join(
            str(
                raw.get("snippet")
                or raw.get("summary")
                or raw.get("text")
                or raw.get("description")
                or ""
            ).split()
        )
        if not (title or url or snippet):
            continue
        entries.append({"title": title[:180], "url": url[:320], "snippet": snippet[:360]})
    return entries[:5]


def _filter_required_search_result_by_subject(
    user_message: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Keep only search material that addresses the authorizing request."""

    if not bool(result.get("ok")):
        return result
    from core.cognition.evidence_relevance import assess_evidence_alignment

    entries = _search_result_entries(result)
    kept: list[dict[str, str]] = []
    measurements: list[dict[str, Any]] = []
    for entry in entries:
        passage = " ".join(
            part
            for part in (
                str(entry.get("title") or "").strip(),
                str(entry.get("snippet") or "").strip(),
            )
            if part
        )
        verdict = assess_evidence_alignment(user_message, passage)
        measurements.append(
            {
                "title": str(entry.get("title") or "")[:120],
                "relevant": verdict.relevant,
                "measured": verdict.measured,
                "score": round(verdict.score, 4) if verdict.score is not None else None,
                "reason": verdict.reason,
            }
        )
        if verdict.relevant:
            kept.append(entry)

    summary = " ".join(
        str(
            result.get("summary")
            or result.get("answer")
            or result.get("synthesis")
            or ""
        ).split()
    )
    summary_verdict = assess_evidence_alignment(user_message, summary) if summary else None
    if kept or (summary_verdict is not None and summary_verdict.relevant):
        filtered = dict(result)
        if entries:
            for key in ("results", "sources", "items"):
                filtered.pop(key, None)
            filtered["results"] = kept
        filtered["evidence_relevance"] = measurements
        filtered["irrelevant_results_removed"] = max(0, len(entries) - len(kept))
        return filtered

    logger.warning(
        "Required search returned no subject-aligned evidence: request=%r candidates=%d",
        str(user_message or "")[:160],
        len(entries),
    )
    filtered = dict(result)
    for key in (
        "results",
        "sources",
        "items",
        "summary",
        "answer",
        "synthesis",
        "content",
        "result",
    ):
        filtered.pop(key, None)
    filtered.update(
        {
            "ok": False,
            "status": "required_search_subject_mismatch",
            "error": "search returned no evidence relevant to the request",
            "evidence_relevance": measurements,
            "irrelevant_results_removed": len(entries),
        }
    )
    return filtered


def _required_search_tool_query(query: str, user_message: str) -> str:
    cleaned = " ".join(str(query or user_message or "").strip().split())
    if not cleaned:
        return ""
    lowered_query = cleaned.lower()
    lowered_message = normalize_memory_intent_text(user_message)
    if re.search(r"\bfacts?\b", lowered_message) and "fact" not in lowered_query:
        cleaned = f"{cleaned} fact"
    return cleaned[:240]


def _render_desktop_required_search_evidence(
    *,
    query: str,
    result: dict[str, Any],
    contract: Any | None,
) -> str:
    ok = bool(result.get("ok"))
    lines = [
        f"query: {query}",
        f"ok: {str(ok).lower()}",
        f"skill: {result.get('skill') or result.get('tool') or 'web_search'}",
    ]
    summary = " ".join(
        str(
            result.get("summary")
            or result.get("answer")
            or result.get("synthesis")
            or result.get("message")
            or ""
        ).split()
    )
    if summary:
        lines.append(f"summary: {summary[:700]}")
    entries = _search_result_entries(result)
    if entries:
        lines.append("sources:")
        for index, entry in enumerate(entries, start=1):
            source = entry.get("url") or "no-url"
            title = entry.get("title") or "untitled"
            snippet = entry.get("snippet") or ""
            lines.append(f"{index}. {title} | {source} | {snippet}".strip())
    elif not ok:
        lines.append(
            f"error: {result.get('error') or result.get('status') or 'web_search returned no usable evidence'}"
        )
    if contract is not None:
        try:
            lines.append(f"contract_reason: {getattr(contract, 'reason', '')}")
        except _CHAT_RECOVERABLE_ERRORS:
            pass
    return "\n".join(lines).strip()


async def _store_desktop_required_search_memory(
    *,
    user_message: str,
    session_id: str,
    query: str,
    result: dict[str, Any],
    evidence_text: str,
) -> bool:
    if not _user_requested_research_memory_save(user_message):
        return False
    memory = ServiceContainer.get("memory_facade", default=None)
    if memory is None or not hasattr(memory, "commit_interaction"):
        return False
    try:
        await memory.commit_interaction(
            context=f"Desktop user requested provisional web research: {query}",
            action="execute_tool(web_search)",
            outcome=evidence_text[:1800],
            success=bool(result.get("ok")),
            emotional_valence=0.1 if result.get("ok") else -0.1,
            importance=0.72,
            metadata={
                "session_id": session_id,
                "source": "web_search",
                "provenance_source": "web_search",
                "intent_source": "autonomous_research",
                "confidence_tier": "provisional",
                "requires_reconciliation": True,
                "research_evidence": True,
                "tool_result_evidence": True,
                "runtime_evidence": True,
                "query": query,
            },
        )
        return True
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.required_search_memory", exc)
        logger.debug("Required-search provisional memory write failed: %s", exc)
        return False


def _recovered_search_result(query: str, exc: BaseException) -> dict[str, Any]:
    """Whatever the cancelled search had already fetched, as a usable result."""
    try:
        from core.search.gathered_sources import take_gathered
    except ImportError:
        take_gathered = None  # type: ignore[assignment]
    held = take_gathered() if take_gathered is not None else None
    if held is None or not held.sources:
        return {
            "ok": False,
            "status": "required_search_failed",
            "error": str(exc) or exc.__class__.__name__,
        }
    sources = list(held.sources)
    logger.info(
        "🔍 Search summary ran out of time; answering from the %d source(s) already fetched.",
        len(sources),
    )
    return {
        "ok": True,
        "status": "required_search_partial",
        "partial": True,
        "query": query,
        "error": "",
        "note": (
            f"The summary step ran out of time; these {len(sources)} source(s) "
            "were fetched and are quoted as found."
        ),
        "count": len(sources),
        "results": [
            {"title": item.title, "url": item.url, "snippet": item.snippet or item.text[:400]}
            for item in sources
        ],
        "citations": [
            {"title": item.title, "url": item.url} for item in sources if item.url
        ],
        "content": "\n\n".join(
            f"{item.title or item.url}\n{(item.text or item.snippet)[:1200]}" for item in sources
        ),
        "result": "\n\n".join(
            f"{item.title or item.url} — {(item.snippet or item.text)[:300]}" for item in sources
        ),
    }


async def _collect_desktop_required_search_evidence(
    user_message: str,
    *,
    session_id: str,
) -> dict[str, Any] | None:
    should_collect, query, contract = _should_collect_desktop_required_search_evidence(user_message)
    if not should_collect:
        return None
    tool_query = _required_search_tool_query(query, user_message)
    try:
        result = await asyncio.wait_for(
            _chat_capability_inventory._execute_governed_live_skill(
                "web_search",
                {
                    "query": tool_query or query or user_message,
                    "num_results": 5,
                    # The resident cortex is the one synthesis authority for
                    # this turn.  Deep search invoked it once here and chat
                    # invoked it again afterward, doubling model work and
                    # turning a retrieval deadline into a generation timeout.
                    "deep": False,
                    "retain": _user_requested_research_memory_save(user_message),
                    "force_refresh": True,
                },
                objective=user_message,
                extra_context={
                    "route": "chat.required_search_evidence",
                    "origin": "desktop_ui",
                    "source": "desktop_ui",
                    "effect_scope": "read_only_external_io",
                    "risk_level": "low",
                    "foreground_request": True,
                    "desktop_required_search_evidence": True,
                    "intent_source": "autonomous_research",
                    "confidence_tier": "provisional",
                    "requires_reconciliation": True,
                },
            ),
            timeout=35.0,
        )
    except (TimeoutError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        record_degradation("chat.required_search_evidence", exc)
        # The deadline covers gathering AND summarising, and cancelling the
        # task threw away pages that had already been fetched.
        #
        # LIVE, 2026-08-22: five sources were in hand when the summary ran
        # long. The turn reported REPLY PATH BLOCKED and served the canned
        # apology. Gathering and summarising fail differently, and what was
        # gathered is still evidence.
        result = _recovered_search_result(tool_query or query or user_message, exc)
    if not isinstance(result, dict):
        result = {"ok": bool(result), "result": result}
    result.setdefault("skill", "web_search")
    result.setdefault("query", tool_query or query or user_message)
    # Qwen3 evidence scoring is local model inference. Keep it off the event
    # loop so five source checks do not freeze heartbeats or text streaming.
    result = await asyncio.to_thread(
        _filter_required_search_result_by_subject,
        user_message,
        result,
    )
    evidence_text = _render_desktop_required_search_evidence(
        query=tool_query or query or user_message,
        result=result,
        contract=contract,
    )
    memory_saved = await _store_desktop_required_search_memory(
        user_message=user_message,
        session_id=session_id,
        query=tool_query or query or user_message,
        result=result,
        evidence_text=evidence_text,
    )
    # This search really happened, so record the receipt that entitles her to
    # say so.
    #
    # unfounded_tool_execution_claim exists because she once wrote "Python
    # code: 2 + 2 Output: 4" having run nothing. Its evidence is a receipt from
    # complete_tool_execution — and THIS path collects search evidence without
    # going through that seam, so a turn that genuinely searched had no
    # receipt. The guard then rewrote a true sentence into a confession:
    # "I said that as though it were done, and it isn't — the action didn't go
    # through." Punishing her for honestly reporting real work is worse than
    # the defect the guard was built to stop.
    try:
        from core.conversation.surface_disposition import record_tool_receipt

        record_tool_receipt(
            "web_search",
            action="web_search",
            object_ref=tool_query or query or user_message,
            ok=bool(result.get("ok")),
            effect_observed=bool(result.get("ok")),
            verification="result_received" if result.get("ok") else "failed",
            evidence=evidence_text,
        )
    except Exception as exc:  # bookkeeping must never break a collected search
        # `pass` here meant a receipt path broken forever looked exactly like
        # one that never needed to fire. The search still returns; the
        # bookkeeping failure is now on the record.
        record_degradation(
            "chat_routes",
            exc,
            severity="warning",
            action="returned the collected search after tool-receipt bookkeeping failed",
            enforce_failure_policy=False,
        )

    return make_completed_capability_evidence(
        _SEARCH_SKILL_NAMES,
        ok=bool(result.get("ok")),
        query=tool_query or query or user_message,
        result=result,
        evidence=evidence_text,
        memory_saved=memory_saved,
        contract=contract.to_dict() if hasattr(contract, "to_dict") else None,
    )


def _is_screen_perception_objective(user_message: str) -> bool:
    """A request whose desktop plan is nothing but LOOKING.

    Reading a screen has no effect to verify and produces no artifact — it
    produces evidence. That makes it the one desktop objective where the
    executor owns the gathering and something else owns the answer.
    """
    if _chat_desktop_objective._blocks_consequential_desktop_execution(user_message):
        return False
    if not _chat_preflight._looks_like_desktop_objective(user_message):
        return False
    try:
        from core.skills.desktop_task import DesktopTaskSkill

        steps = DesktopTaskSkill()._derive_steps_from_objective(str(user_message or "").strip(), {})
        return bool(steps) and DesktopTaskSkill._primitive_steps_are_only_observational(steps)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _asks_what_is_on_the_screen(user_message: str) -> bool:
    """A screen QUESTION, as distinct from any objective mentioning a screen.

    _is_screen_perception_objective also matches "put BUILD-42 on my clipboard"
    — a write, not a question — so using it here blocked a real desktop action
    from the executor path it belongs on. The registry's matcher is
    question-shaped and was already carrying this judgement.
    """

    try:
        from core.brain.observable_registry import _matches_screen

        return bool(_matches_screen(str(user_message or "")))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _screen_perception_needs_her_answer(user_message: str) -> bool:
    """A screen question that a description cannot answer.

    "What's on my screen" is served by the reading itself, natively and in
    about a second. "What was that repo?" is not: the reading is where the
    answer is found, not what the answer is.
    """
    if not _is_screen_perception_objective(user_message):
        return False
    try:
        from core.perception.observation_evidence import AnswerShape, answer_shape_for

        return answer_shape_for(user_message) is not AnswerShape.DESCRIBE
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _desktop_objective_self_sufficient_without_cognitive_text(user_message: str) -> bool:
    """Whether desktop_task can honestly complete without a model-composed body.

    This is deliberately narrower than "looks like desktop objective": it only
    admits objectives where the executor owns the missing prose through a
    canonical source (Aura self-summary, live research synthesis) or where the
    objective is primarily an observable desktop/file operation. Free-form
    essays, letters, and creative prose still require CognitiveEngine text.
    """
    if _chat_desktop_objective._blocks_consequential_desktop_execution(user_message):
        return False
    if _chat_capability_inventory._looks_like_program_dna_execution_request(user_message):
        return False
    if not _chat_preflight._looks_like_desktop_objective(user_message):
        return False
    if _asks_what_is_on_the_screen(user_message):
        # A screen question is never answered by a PROMISE to look.
        #
        # The self-sufficient path returns "I will execute this through the
        # governed desktop_task lane and report only receipt-verified effects"
        # without invoking the engine. For "what's on my screen right now?"
        # that placeholder was then refused by the authorship gate — it is not
        # her answer, because it is not an answer — and the turn failed closed
        # with "I couldn't get to an answer I'd stand behind on that one",
        # measured live 2026-08-17.
        #
        # The narrower predicate below admitted only questions a description
        # cannot answer ("what was that repo you saw?"), on the reasoning that
        # a plain "what's on my screen" IS served by the reading. That is true,
        # and it is served by the reading arriving as grounding — which now
        # happens through the observable registry — not by a sentence about
        # what the executor intends to do.
        return False
    if _screen_perception_needs_her_answer(user_message):
        # A specific question about the screen is not self-sufficient: the
        # executor can gather the evidence but cannot answer from it.
        #
        # Live 2026-08-04, "what was that repo you saw on my screen?" was
        # classified self-sufficient, so the turn returned a canned "I will
        # execute this through the governed desktop_task lane" WITHOUT ever
        # invoking the engine — and when the desktop lane declined to answer
        # a specific question with a generic description, the authorship
        # gate correctly refused the placeholder and the turn failed closed.
        # The read still happens; it just stops pretending to be the answer.
        return False
    text = str(user_message or "").strip()
    lowered = text.lower()
    try:
        from core.skills.desktop_task import DesktopTaskSkill

        steps = DesktopTaskSkill()._derive_steps_from_objective(text, {})
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False

    actions = {str(getattr(step, "action", "") or "") for step in steps}
    if not actions:
        return False
    # create_note carries a document body exactly as write_text_file does.
    # It arrived with the native Notes path and was in none of these sets, so
    # "open Notes and write a report about quantum mechanics" classified as
    # self-sufficient and skipped cognition — which is how a note ends up
    # holding the deterministic composer's "Notes on the requested subject:
    # The requested subject is the focus of this note."
    prose_actions = {
        "set_clipboard",
        "write_text_file",
        "render_text_pdf",
        "type",
        "write_in_app",
        "create_note",
    }
    if not (actions & prose_actions):
        return True
    literal_body_actions = {
        "create_folder",
        "open_app",
        "open_url",
        "set_clipboard",
        "wait",
        "hotkey",
        "type",
        "write_in_app",
        "create_note",
        "write_text_file",
        "render_text_pdf",
        "move_file",
    }
    if (
        actions <= literal_body_actions
        and DesktopTaskSkill._objective_supplies_literal_document_body(text)
    ):
        # Exact user-supplied text needs transcription, not a model draft.
        # Execution still traverses CapabilityEngine/Will and must return
        # effect evidence for every critical desktop step.
        return True
    local_artifact_actions = {
        "create_folder",
        "write_text_file",
        "render_text_pdf",
        "move_file",
        "read_menu_clock",
    }
    # A SELF-SUMMARY is the one body the executor owns canonically, and
    # writing it into an app is the same bounded artifact as writing it to a
    # file. Routed through cognition instead, it came back as a capability
    # denial that was then TYPED INTO THE APP by the hands it denied — "I
    # can describe myself, but I don't actually open apps or write notes."
    # Measured twice live in two different phrasings, which is why this is a
    # routing fix rather than another phrase in a regex.
    #
    # Deliberately narrower than the set above: a report about quantum
    # mechanics written into the same app is novel prose and still needs
    # cognition. What she is, is not novel prose.
    if actions <= (local_artifact_actions | {"write_in_app", "open_app"}) and (
        DesktopTaskSkill._objective_requests_self_summary(text)
    ):
        return True
    if actions <= local_artifact_actions and (
        DesktopTaskSkill._objective_requests_self_summary(text)
        or DesktopTaskSkill._objective_requests_written_artifact(text)
    ):
        # A bounded local artifact can be authored and verified inside
        # desktop_task without first asking the foreground model to narrate an
        # action that has not happened yet. Interactive app typing, research
        # synthesis, essays, and long-form creative writing still stay on the
        # full cognitive draft path below.
        return True
    # Original prose and source synthesis are cognitive work. They must not use
    # the pre-cognition mechanical shortcut merely because desktop_task has a
    # deterministic emergency body composer.
    if DesktopTaskSkill._objective_requests_self_summary(text):
        return False
    if DesktopTaskSkill._objective_requests_research_document(text):
        return False
    explicit_content_markers = (
        "essay",
        "letter",
        "poem",
        "story",
        "blog post",
        "article",
        "paragraph",
        "summary",
        "summarize",
        "in your own words",
        "opinion",
        "explain",
        "describe",
        "about",
    )
    if any(marker in lowered for marker in explicit_content_markers):
        return False
    if re.search(r"\b(?:write|draft|compose|create|make)\s+(?:a\s+|an\s+)?report\b", lowered):
        return False
    sourced_content_markers = (
        "copy ",
        "copy the",
        "clipboard",
        "selected text",
        "selection",
        "equation body",
        "from calculator",
        "from the page",
        "from chrome",
        "from safari",
        "from the article",
        "from the document",
        "from notes",
    )
    if any(marker in lowered for marker in sourced_content_markers):
        return True
    operational_report_markers = (
        "report the path",
        "report the paths",
        "report paths",
        "show me the path",
        "show me the paths",
        "where you saved",
        "saved path",
        "receipt",
        "what you did",
    )
    if any(marker in lowered for marker in operational_report_markers) and (
        "pdf" in lowered or "move" in lowered or "copy" in lowered
    ):
        return True
    return False
