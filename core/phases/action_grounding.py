"""Action grounding — close the LLM-text-vs-real-action disconnect.

The critical bug: the LLM was emitting strings like

    [SKILL_RESULT:computer_use] ✅ I opened the Notes app...
    [ACTION:computer_use] terminal: echo '...'

...but nothing parsed those markers, so no skill ever executed. The user
saw a claim; the machine did nothing. This module detects those markers
in an outgoing response, dispatches the real skill via the capability
engine, and rewrites the response with the actual outcome.

If the skill executes successfully, the marker becomes the real summary.
If it fails (permission denied, unsupported, engine unavailable), the
response is rewritten to say so explicitly rather than pretending.

The grounding module also exposes a ``check_unverified_action_claims``
helper that detects first-person action assertions ("I just opened X",
"I typed Y", "I clicked Z") without any matching skill receipt in the
current turn's execution record. This makes memory/belief writes refuse
to encode hallucinated actions as truth.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Regexes that catch the most common marker forms Aura's LLM has been
# emitting. The match captures the skill name plus any tail text on the
# same line — the tail is discarded because the skill output will replace
# it.
_MARKER_RE = re.compile(
    r"\[(?:SKILL_RESULT|SKILL|ACTION|TOOL|SKILL_INVOCATION)\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]"
    r"\s*([^\n]*)",
    re.IGNORECASE,
)

# First-person action claims the model tends to hallucinate when no skill
# actually ran. These only fire the "unverified" audit path — they do not
# edit the user-visible text, because the user is the one demanding
# grounding evidence.
_ACTION_CLAIM_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bI\s+(just\s+)?(?:opened|launched|started)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 _.-]+?)(?:\.| app| application)?\b", re.I),
    re.compile(r"\bI\s+(?:just\s+)?(?:typed|wrote|entered)\s+['\"`]", re.I),
    re.compile(r"\bI\s+(?:just\s+)?(?:clicked|pressed)\s+", re.I),
    re.compile(r"\bI\s+(?:just\s+)?(?:created|wrote)\s+(?:a\s+)?(?:new\s+)?(?:note|file|document)", re.I),
    re.compile(r"\bI\s+(?:just\s+)?(?:ran|executed)\s+(?:the\s+)?(?:command|terminal|script)", re.I),
    re.compile(r"\bI\s+(?:just\s+)?(?:searched|looked\s+up|browsed)\s+(?:for\s+)?", re.I),
    re.compile(r"\bthe\s+note\s+is\s+there\b", re.I),
)


DEFAULT_SKILL_PARAMS: Dict[str, Dict[str, Any]] = {
    "computer_use": {"action": "read_screen_text"},
    "web_search": {"query": ""},
    "file_operation": {"action": "noop"},
    "os_manipulation": {"action": "noop"},
}


@dataclass
class GroundingResult:
    """Result of grounding a single response."""

    grounded_text: str
    marker_hits: List[Dict[str, Any]] = field(default_factory=list)
    claims_without_receipts: List[str] = field(default_factory=list)
    dispatched: int = 0
    dispatched_ok: int = 0
    replaced: int = 0

    @property
    def had_markers(self) -> bool:
        return bool(self.marker_hits)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "grounded_text": self.grounded_text,
            "marker_hits": list(self.marker_hits),
            "claims_without_receipts": list(self.claims_without_receipts),
            "dispatched": int(self.dispatched),
            "dispatched_ok": int(self.dispatched_ok),
            "replaced": int(self.replaced),
            "had_markers": self.had_markers,
        }


async def ground_response(
    response: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    capability_engine: Any = None,
    skill_receipts: Optional[Iterable[Dict[str, Any]]] = None,
    audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> GroundingResult:
    """Parse action markers, dispatch real skills, rewrite the response.

    ``capability_engine`` must expose an ``execute(skill_name, params,
    context)`` coroutine that returns a dict ``{ok: bool, summary: str,
    ...}``. When it is not provided the function still runs and reports
    every marker as unverified without executing anything — this keeps
    the honesty guarantee intact in test environments.
    """
    context = context or {}
    receipts: List[Dict[str, Any]] = list(skill_receipts or [])
    text = str(response or "")
    result = GroundingResult(grounded_text=text)

    if capability_engine is None:
        try:
            from core.container import ServiceContainer

            capability_engine = ServiceContainer.get("capability_engine", default=None)
        except Exception:
            capability_engine = None

    matches = list(_MARKER_RE.finditer(text))
    if matches:
        # Walk in reverse so we don't invalidate spans while replacing.
        for match in reversed(matches):
            skill_name = match.group(1).strip().lower()
            tail = (match.group(2) or "").strip()
            hit: Dict[str, Any] = {
                "skill": skill_name,
                "tail": tail,
                "span": [match.start(), match.end()],
                "status": "unverified",
                "replaced": False,
            }
            replacement = _unverified_text(skill_name, tail)

            if capability_engine is not None:
                params = _params_for_skill(skill_name, tail, context)
                try:
                    skill_result = await capability_engine.execute(
                        skill_name, params, context
                    )
                    result.dispatched += 1
                    ok = bool(skill_result.get("ok", skill_result.get("success", False)))
                    hit["result"] = {
                        "ok": ok,
                        "summary": str(skill_result.get("summary") or skill_result.get("result") or "")[:240],
                        "error": str(skill_result.get("error") or "")[:240],
                    }
                    if ok:
                        hit["status"] = "executed"
                        hit["replaced"] = True
                        result.dispatched_ok += 1
                        replacement = _success_text(skill_name, skill_result, tail)
                        receipts.append({
                            "skill": skill_name,
                            "at": time.time(),
                            "summary": hit["result"]["summary"],
                        })
                    else:
                        hit["status"] = "executed_failed"
                        hit["replaced"] = True
                        replacement = _failure_text(skill_name, skill_result)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Action grounding dispatch failed for %s: %s", skill_name, exc)
                    hit["status"] = "dispatch_error"
                    hit["error"] = repr(exc)
                    replacement = _failure_text(skill_name, {"error": repr(exc)})
                    hit["replaced"] = True
            else:
                # No engine available — replace the marker with an explicit
                # unverified-intent note so the user never sees the bare
                # marker, and so memory writers can refuse to promote it.
                hit["status"] = "unverified_no_engine"
                hit["replaced"] = True

            if hit["replaced"]:
                result.replaced += 1
                text = text[: match.start()] + replacement + text[match.end() :]
            result.marker_hits.append(hit)

    # Reverse so the list matches reading order.
    result.marker_hits.reverse()

    # Unverified first-person claims — only surface the audit, don't
    # rewrite the response. This is what keeps Aura honest in memory
    # without editing her voice.
    unverified = _unverified_claims(text, receipts)
    result.claims_without_receipts = unverified

    result.grounded_text = text

    if audit_callback is not None:
        try:
            audit_callback(result.as_dict())
        except Exception:
            pass

    return result


def receipts_from_context(context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract any skill receipts the pipeline already recorded."""
    if not context:
        return []
    receipts = context.get("skill_receipts") or context.get("skill_invocations")
    if not receipts:
        return []
    if isinstance(receipts, dict):
        receipts = [receipts]
    return [r for r in receipts if isinstance(r, dict)]


def check_unverified_action_claims(
    response: str,
    *,
    skill_receipts: Iterable[Dict[str, Any]] = (),
) -> List[str]:
    """Return every hallucinated-action phrase that lacks a matching receipt.

    This is the memory/belief gate: code that writes "I took action X" to
    the life ledger or belief graph should refuse if this list is
    non-empty, unless there's a matching receipt.
    """
    return _unverified_claims(response, list(skill_receipts))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _params_for_skill(
    skill_name: str,
    tail: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    defaults = DEFAULT_SKILL_PARAMS.get(skill_name, {}).copy()
    if not tail:
        return defaults
    lower = tail.lower()
    if skill_name == "computer_use":
        if any(k in lower for k in ("terminal", "shell", "bash", "zsh")):
            defaults.setdefault("action", "read_screen_text")
        elif any(k in lower for k in ("type ", "write ", "compose")):
            defaults["action"] = "type"
            defaults["text"] = tail
        elif "open" in lower:
            defaults["action"] = "open_app"
            defaults["target"] = tail.split("open", 1)[-1].strip(": ")
    if skill_name == "web_search":
        query = tail.strip().strip(":").strip()
        if query:
            defaults["query"] = query
    return defaults


def _unverified_text(skill_name: str, tail: str) -> str:
    return (
        f"(Note: I said I would invoke `{skill_name}` but the action "
        f"dispatcher was not available, so nothing actually ran. "
        "Treat this as intent, not completed action.)"
    )


def _success_text(skill_name: str, result: Dict[str, Any], tail: str) -> str:
    summary = str(result.get("summary") or result.get("result") or "").strip()
    if summary:
        return summary
    return f"({skill_name} completed.)"


def _failure_text(skill_name: str, result: Dict[str, Any]) -> str:
    err = str(result.get("error") or result.get("status") or "unknown failure").strip()
    return (
        f"(I attempted to run `{skill_name}` but it did not complete: {err}. "
        "I am not pretending this finished.)"
    )


def _unverified_claims(
    text: str, skill_receipts: List[Dict[str, Any]]
) -> List[str]:
    receipt_skills = {str(r.get("skill") or "").lower() for r in skill_receipts}
    lowered = text.lower()
    if receipt_skills:
        # If any receipts exist, only flag claims for different categories.
        if "computer_use" in receipt_skills or "os_manipulation" in receipt_skills:
            return []
    flagged: List[str] = []
    for pattern in _ACTION_CLAIM_PATTERNS:
        for match in pattern.finditer(lowered):
            flagged.append(match.group(0))
    return flagged
