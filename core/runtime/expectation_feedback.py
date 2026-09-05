"""Turn failed action expectations into bounded planning constraints.

Capability execution persists expectation verdicts as tool receipts. This
module selects only recent, goal-relevant failures and exposes a sanitized
summary that planners can use without replaying raw tool output or letting the
receipt backlog flood the prompt.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_SPACE_RE = re.compile(r"\s+")
_STOP_WORDS = frozenset(
    {
        "about",
        "action",
        "after",
        "before",
        "complete",
        "completed",
        "from",
        "into",
        "requested",
        "result",
        "that",
        "this",
        "through",
        "tool",
        "with",
    }
)
_EXPECTATION_SOURCE = "capability_engine.action_expectation"
_FAILED_EXPECTATION_STATUSES = frozenset(
    {
        "success_unverified",
        "partial_success",
        "failed_recoverable",
        "failed_unverified",
    }
)


def _bounded_text(value: Any, limit: int = 240) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())[:limit]


def _string_list(value: Any, *, limit: int = 8) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = ()
    normalized = []
    for item in values:
        text = _bounded_text(item, 160)
        if text and text not in normalized:
            normalized.append(text)
        if len(normalized) >= limit:
            break
    return tuple(normalized)


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(value or "").casefold())
        if token not in _STOP_WORDS
    }


@dataclass(frozen=True)
class ExpectationRepairSignal:
    receipt_id: str
    tool: str
    objective: str
    status: str
    missing_criteria: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    next_step: str
    created_at: float
    relevance: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "tool": self.tool,
            "objective": self.objective,
            "status": self.status,
            "missing_criteria": list(self.missing_criteria),
            "missing_evidence": list(self.missing_evidence),
            "next_step": self.next_step,
            "created_at": self.created_at,
            "relevance": self.relevance,
        }


def _signal_from_receipt(receipt: Any) -> ExpectationRepairSignal | None:
    metadata = getattr(receipt, "metadata", None)
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    evidence = getattr(receipt, "verification_evidence", None)
    evidence = dict(evidence) if isinstance(evidence, dict) else {}
    verdict = evidence.get("expectation_verdict")
    if not isinstance(verdict, dict):
        return None
    if metadata.get("source") != _EXPECTATION_SOURCE:
        return None
    if bool(verdict.get("passed", False)):
        return None

    status = _bounded_text(getattr(receipt, "status", ""), 64)
    if status not in _FAILED_EXPECTATION_STATUSES:
        return None

    receipt_id = _bounded_text(getattr(receipt, "receipt_id", ""), 120)
    if not receipt_id:
        return None
    objective = _bounded_text(
        metadata.get("expectation_objective") or getattr(receipt, "cause", ""),
        320,
    )
    return ExpectationRepairSignal(
        receipt_id=receipt_id,
        tool=_bounded_text(getattr(receipt, "tool", ""), 120),
        objective=objective,
        status=status,
        missing_criteria=_string_list(verdict.get("missing_criteria")),
        missing_evidence=_string_list(verdict.get("missing_evidence")),
        next_step=_bounded_text(
            verdict.get("next_step") or metadata.get("expectation_next_step"),
            240,
        ),
        created_at=float(getattr(receipt, "created_at", 0.0) or 0.0),
    )


def recent_expectation_repair_signals(
    goal: str,
    *,
    available_tools: Sequence[str] = (),
    receipt_store: Any = None,
    now: float | None = None,
    max_age_s: float = 7 * 24 * 60 * 60,
    scan_limit: int = 80,
    limit: int = 5,
) -> list[ExpectationRepairSignal]:
    """Return recent failed expectations that are relevant to ``goal``.

    Relevance requires lexical overlap with the failed objective or named tool;
    an unrelated receipt never enters the next planning prompt.
    """
    if limit <= 0 or not str(goal or "").strip():
        return []
    if receipt_store is None:
        from core.runtime.receipts import get_receipt_store

        receipt_store = get_receipt_store()

    try:
        receipts = receipt_store.query_recent(
            kinds=["tool_execution"],
            limit=max(1, int(scan_limit)),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return []

    goal_tokens = _tokens(goal)
    tool_allowlist = {
        str(tool or "").strip().casefold()
        for tool in available_tools
        if str(tool or "").strip()
    }
    current_time = float(time.time() if now is None else now)
    candidates: list[ExpectationRepairSignal] = []
    dedupe: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()

    for receipt in reversed(list(receipts)):
        signal = _signal_from_receipt(receipt)
        if signal is None:
            continue
        if max_age_s > 0 and signal.created_at < current_time - max_age_s:
            continue
        normalized_tool = signal.tool.casefold()
        if tool_allowlist and normalized_tool not in tool_allowlist:
            continue

        objective_tokens = _tokens(signal.objective)
        tool_tokens = _tokens(signal.tool.replace("_", " "))
        objective_overlap = goal_tokens & objective_tokens
        tool_overlap = goal_tokens & tool_tokens
        if not objective_overlap and not tool_overlap:
            continue
        relevance = len(objective_overlap) * 3 + len(tool_overlap) * 2
        key = (
            normalized_tool,
            signal.next_step,
            signal.missing_criteria,
            signal.missing_evidence,
        )
        if key in dedupe:
            continue
        dedupe.add(key)
        candidates.append(replace(signal, relevance=relevance))

    candidates.sort(key=lambda item: (item.relevance, item.created_at), reverse=True)
    return candidates[:limit]


def format_expectation_repair_guidance(
    signals: Sequence[ExpectationRepairSignal],
    *,
    max_chars: int = 2400,
) -> str:
    if not signals or max_chars <= 0:
        return ""
    lines = [
        "RECENT EXPECTATION FAILURES (causal planning constraints):",
        "Do not repeat these shallow completion patterns. Build the missing proof or repair step into the plan.",
    ]
    for signal in signals:
        missing = list(signal.missing_criteria) + list(signal.missing_evidence)
        missing_text = ", ".join(missing[:8]) or "unspecified acceptance evidence"
        next_step = signal.next_step or "collect effect evidence before completion"
        lines.append(
            f"- {signal.tool or 'unknown_tool'} [{signal.receipt_id}]: "
            f"missing {missing_text}; required next step: {next_step}."
        )
    return "\n".join(lines)[:max_chars]


def expectation_feedback_fingerprint(
    signals: Sequence[ExpectationRepairSignal],
) -> str:
    if not signals:
        return "none"
    payload = [signal.to_dict() for signal in signals]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


__all__ = [
    "ExpectationRepairSignal",
    "expectation_feedback_fingerprint",
    "format_expectation_repair_guidance",
    "recent_expectation_repair_signals",
]
