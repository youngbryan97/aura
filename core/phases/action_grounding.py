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

import json
import logging
import re
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Regexes that catch the most common marker forms Aura's LLM has been
# emitting. The match captures the skill name plus any tail text on the
# same line — the tail is discarded because the skill output will replace
# it.
_MARKER_RE = re.compile(
    r"\[(?:SKILL_RESULT|SKILL|ACTION|TOOL|SKILL_INVOCATION)\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)(?:\(([^)]*)\))?\s*\]"
    r"\s*([^\n]*)",
    re.IGNORECASE,
)

# First-person action claims the model tends to hallucinate when no skill
# actually ran. These only fire the "unverified" audit path — they do not
# edit the user-visible text, because the user is the one demanding
# grounding evidence.
_ACTION_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bI\s+(just\s+)?(?:opened|launched|started)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 _.-]+?)(?:\.| app| application)?\b", re.I),
    re.compile(r"\bI\s+(?:just\s+)?(?:typed|wrote|entered)\s+['\"`]", re.I),
    re.compile(r"\bI\s+(?:just\s+)?(?:clicked|pressed)\s+", re.I),
    re.compile(r"\bI\s+(?:just\s+)?(?:created|wrote)\s+(?:a\s+)?(?:new\s+)?(?:note|file|document)", re.I),
    re.compile(r"\bI\s+(?:just\s+)?(?:ran|executed)\s+(?:the\s+)?(?:command|terminal|script)", re.I),
    re.compile(r"\bI\s+(?:just\s+)?(?:searched|looked\s+up|browsed)\s+(?:for\s+)?", re.I),
    re.compile(r"\bthe\s+note\s+is\s+there\b", re.I),
)


DEFAULT_SKILL_PARAMS: dict[str, dict[str, Any]] = {
    "computer_use": {"action": "read_screen_text"},
    "desktop_task": {"objective": "", "steps": []},
    "web_search": {"query": ""},
    "file_operation": {"action": "noop"},
    "os_manipulation": {"action": "noop"},
}

_CAPABILITY_ENGINE_UNSET = object()


@dataclass
class GroundingResult:
    """Result of grounding a single response."""

    grounded_text: str
    marker_hits: list[dict[str, Any]] = field(default_factory=list)
    claims_without_receipts: list[str] = field(default_factory=list)
    dispatched: int = 0
    dispatched_ok: int = 0
    replaced: int = 0

    @property
    def had_markers(self) -> bool:
        return bool(self.marker_hits)

    def as_dict(self) -> dict[str, Any]:
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
    context: dict[str, Any] | None = None,
    capability_engine: Any = _CAPABILITY_ENGINE_UNSET,
    skill_receipts: Iterable[dict[str, Any]] | None = None,
    audit_callback: Callable[[dict[str, Any]], None] | None = None,
) -> GroundingResult:
    """Parse action markers, dispatch real skills, rewrite the response.

    ``capability_engine`` must expose an ``execute(skill_name, params,
    context)`` coroutine that returns a dict ``{ok: bool, summary: str,
    ...}``. When it is not provided the function still runs and reports
    every marker as unverified without executing anything — this keeps
    the honesty guarantee intact in test environments.
    """
    context = context or {}
    receipts: list[dict[str, Any]] = list(skill_receipts or [])
    text = str(response or "")
    result = GroundingResult(grounded_text=text)

    if capability_engine is _CAPABILITY_ENGINE_UNSET:
        try:
            from core.container import ServiceContainer

            capability_engine = ServiceContainer.get("capability_engine", default=None)
        # not a failure: no service here, so the caller falls back to its own default.
        except (ImportError, AttributeError, RuntimeError):
            capability_engine = None

    matches = list(_MARKER_RE.finditer(text))
    if matches:
        # Walk in reverse so we don't invalidate spans while replacing.
        for match in reversed(matches):
            skill_name = match.group(1).strip().lower()
            marker_args = (match.group(2) or "").strip()
            tail = (match.group(3) or "").strip()
            if marker_args and not tail:
                tail = f"({marker_args})"
            hit: dict[str, Any] = {
                "skill": skill_name,
                "marker_args": marker_args,
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
                    hit["ok"] = ok
                    hit["summary"] = hit["result"]["summary"]
                    hit["error"] = hit["result"]["error"]
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
                except (sqlite3.Error, OSError) as exc:  # pragma: no cover - defensive
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug(
                "%s unavailable (%s: %s); the grounding audit callback raised and the turn carried on",
                "it",
                type(exc).__name__,
                exc,
            )

    return result


def receipts_from_context(context: dict[str, Any] | None) -> list[dict[str, Any]]:
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
    skill_receipts: Iterable[dict[str, Any]] = (),
) -> list[str]:
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
    context: dict[str, Any],
) -> dict[str, Any]:
    defaults = DEFAULT_SKILL_PARAMS.get(skill_name, {}).copy()
    if not tail:
        return defaults
    lower = tail.lower()
    parsed = _parse_json_tail(tail)
    if isinstance(parsed, dict):
        return parsed
    if skill_name == "computer_use":
        if any(k in lower for k in ("terminal", "shell", "bash", "zsh")):
            defaults.setdefault("action", "read_screen_text")
        elif any(k in lower for k in ("type ", "write ", "compose")):
            defaults["action"] = "type"
            defaults["target"] = tail
        elif "open" in lower:
            defaults["action"] = "open_app"
            defaults["target"] = tail.split("open", 1)[-1].strip(": ")
    if skill_name == "desktop_task":
        objective = str(context.get("objective") or context.get("message") or tail or "desktop task")
        defaults["objective"] = objective
        if isinstance(parsed, list):
            defaults["steps"] = parsed
        elif tail:
            defaults["steps"] = []
    if skill_name == "web_search":
        query = tail.strip().strip(":").strip()
        if query:
            defaults["query"] = query
    # ── Embodied action skills ─────────────────────────────────
    # Skills that take a single "action" parameter (keystroke, command, etc.)
    # The tail text IS the action. This covers execute_nethack_action and
    # any future embodied interface tools.
    if skill_name in ("execute_nethack_action",):
        # Zenith v47 Hardening: Support both [ACTION:nethack] key and [ACTION:nethack(key='y')]
        action_str = tail.strip().strip(":").strip()

        # If tail is empty but the skill name had parens (matched by updated _MARKER_RE),
        # they are currently lost because the regex only captures the name.
        # Preserve tail handling here until marker argument capture is widened.

        if action_str:
            # Handle Python-style args in tail if they leaked out: (key='y') -> y
            if action_str.startswith("(") and ")" in action_str:
                inner = action_str[1:action_str.find(")")].strip()
                if "key=" in inner:
                    action_key = inner.split("key=", 1)[-1].strip("'\" ")
                else:
                    action_key = inner.strip("'\" ")
            else:
                # Take just the first word/char as the action key
                # e.g. "l" → "l", "ESC to exit" → "ESC", "k (move up)" → "k"
                action_key = action_str.split()[0].strip("()").rstrip(",.;:)")

            defaults["action"] = action_key
    return defaults


def _parse_json_tail(tail: str) -> Any | None:
    text = str(tail or "").strip()
    if not text:
        return None
    if ":" in text and not text.startswith(("{", "[")):
        text = text.split(":", 1)[-1].strip()
    if not text.startswith(("{", "[")):
        return None
    try:
        return json.loads(text)
    # not a failure: text that is not the JSON this expects is not a record it can
    # read back.
    except json.JSONDecodeError:
        return None


def _unverified_text(skill_name: str, tail: str) -> str:
    return (
        f"(Note: I said I would invoke `{skill_name}` but the action "
        f"dispatcher was not available, so nothing actually ran. "
        "Treat this as intent, not completed action.)"
    )


def _success_text(skill_name: str, result: dict[str, Any], tail: str) -> str:
    summary = str(result.get("summary") or result.get("result") or "").strip()
    if summary:
        return summary
    return f"`{skill_name}` completed."


#: Grounding outcomes where a skill was really dispatched and did not succeed.
_FAILED_DISPATCHES: frozenset[str] = frozenset({"executed_failed", "dispatch_error"})


def remember_last_action(world: Any, result: GroundingResult) -> dict[str, Any] | None:
    """Tell the world model what she just did, in the shape it reads.

    `observe_cycle` shows the world model an action read off
    `world.facts["last_action"]`: whether there was one, whether it was
    verified, and whether she was the one who did it. Only the subject-core
    driver wrote that fact. In the running organism nothing did, so the world
    model was shown that she never acted, and what she did could not change
    what it predicted.

    The newest dispatched action counts, succeeded or failed. A marker that was
    never dispatched is an intention, not an action, and leaves the fact alone.
    """
    facts = getattr(world, "facts", None)
    if not isinstance(facts, dict):
        return None
    dispatched = [
        hit for hit in result.marker_hits
        if hit.get("status") == "executed" or hit.get("status") in _FAILED_DISPATCHES
    ]
    if not dispatched:
        return None
    hit = dispatched[-1]
    ok = hit.get("status") == "executed"
    record = {
        "intended": str(hit.get("skill") or ""),
        "verified": ok,
        "at": time.time(),
        "actor": "self",
        "kind": str(hit.get("skill") or ""),
        "outcome": "succeeded" if ok else "failed",
    }
    facts["last_action"] = record
    return record


def perceive_failed_actions(world: Any, result: GroundingResult) -> int:
    """Put each skill that ran and failed into the percept stream as an error.

    Her own action failing is something she perceives, the way a message
    arriving is. The affect phase maps an `error` percept to fear and
    frustration, and until this nothing in production emitted one, so that
    route was exercised only by the subject-core harness. A marker that
    dispatched nothing is not a failure she saw, and is left out.
    """
    from core.state.percepts import emit_percept

    emitted = 0
    for hit in result.marker_hits:
        if hit.get("status") not in _FAILED_DISPATCHES:
            continue
        record = emit_percept(
            world,
            "error",
            content=f"{hit.get('skill')} failed: {hit.get('error') or 'no reason given'}",
            source="action_grounding",
            skill=hit.get("skill"),
            status=hit.get("status"),
        )
        emitted += record is not None
    return emitted


def _failure_text(skill_name: str, result: dict[str, Any]) -> str:
    err = str(result.get("error") or result.get("status") or "unknown failure").strip()
    return (
        f"(I attempted to run `{skill_name}` but it did not complete: {err}. "
        "I am not pretending this finished.)"
    )


def _unverified_claims(
    text: str, skill_receipts: list[dict[str, Any]]
) -> list[str]:
    receipt_skills = {str(r.get("skill") or "").lower() for r in skill_receipts}
    lowered = text.lower()
    if receipt_skills:
        # If any receipts exist, only flag claims for different categories.
        if "computer_use" in receipt_skills or "os_manipulation" in receipt_skills:
            return []
    flagged: list[str] = []
    for pattern in _ACTION_CLAIM_PATTERNS:
        for match in pattern.finditer(lowered):
            flagged.append(match.group(0))
    return flagged
