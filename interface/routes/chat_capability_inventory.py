"""What Aura can actually do, read from the live catalog.

A capability answer that comes from the model is a guess about itself.
These build it from the registered skill catalog, name the category each
tool belongs to, and refuse rather than invent when the catalog cannot be
read inside the turn's budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from core.container import ServiceContainer
import asyncio
from interface.routes.chat_common import (  # noqa: E402
    _CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,  # noqa: F401
    _CHAT_RECOVERABLE_ERRORS,  # noqa: F401
    _CHAT_REQUEST_PRINCIPAL,  # noqa: F401
    _CHAT_REQUEST_SURFACE,  # noqa: F401
    _MAX_CONVERSATION_LOG_EXCHANGES,  # noqa: F401
    _conversation_log,  # noqa: F401
    _locks,  # noqa: F401
    logger,  # noqa: F401
)
import re
from core.runtime.errors import describe_error, record_degradation
from core.runtime.chat_delivery_progress import (
    bind_chat_delivery_progress,
    report_chat_delivery_progress,
)

from interface.routes.chat_common import (
    _SEARCH_SKILL_NAMES,
)


async def _execute_governed_live_skill(
    skill_name: str,
    params: dict[str, Any],
    *,
    objective: str,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run live actions through governed capability surfaces, never raw IO."""
    context = {
        "origin": "user",
        "route": "chat.live_runtime_proof",
        "objective": objective[:500],
        "message": objective[:500],
        "foreground_request": True,
        "user_explicitly_authorized": True,
        "user_requested_action": True,
    }
    if extra_context:
        context.update(dict(extra_context))
        context["objective"] = objective[:500]
        context["message"] = objective[:500]
        context["foreground_request"] = True
        context["user_explicitly_authorized"] = True
        context["user_requested_action"] = True
    for untrusted_authority_key in (
        "authority_args_digest",
        "capability_token",
        "capability_token_id",
        "scoped_authority",
        "standing_authority_grant_id",
        "standing_authority_receipt_id",
        "standing_authority_token",
    ):
        context.pop(untrusted_authority_key, None)
    if (
        context.get("foreground_request")
        and context.get("user_requested_action")
        and context.get("user_explicitly_authorized")
    ):
        route_slug = re.sub(
            r"[^a-z0-9_.:-]+", "_", str(context.get("route") or "live_skill").lower()
        )
        skill_slug = re.sub(r"[^a-z0-9_.:-]+", "_", str(skill_name or "skill").lower())
        context["requested_authority_scope"] = (
            f"foreground_user_requested:{route_slug}:{skill_slug}"
        )
    engine = ServiceContainer.get("capability_engine", default=None)

    async def _execute_capability(
        execution_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not engine or not hasattr(engine, "execute"):
            return {
                "ok": False,
                "receipt": "capability_engine_unavailable",
                "error": "No governed capability executor is registered.",
                "status": "capability_engine_unavailable",
            }
        activity = {
            "desktop_task": "Working through the requested desktop steps.",
            "web_search": "Searching for current, relevant sources.",
            "search_web": "Searching for current, relevant sources.",
            "grounded_search": "Searching for current, relevant sources.",
            "file_operation": "Working with the requested files.",
            "web_interlocutor": "Working in the browser conversation.",
        }.get(skill_name, f"Using {skill_name.replace('_', ' ')} for this request.")
        await report_chat_delivery_progress(
            phase="executing",
            message=activity,
            details={"skill": skill_name, "route": str(context.get("route") or "")},
        )
        result = await engine.execute(
            skill_name, dict(params), context=execution_context or context
        )
        await report_chat_delivery_progress(
            phase="verifying",
            message=f"Verifying the result from {skill_name.replace('_', ' ')}.",
            details={"skill": skill_name, "result_structured": isinstance(result, dict)},
        )
        if isinstance(result, dict):
            return result
        return {"ok": bool(result), "result": result}

    route = str(context.get("route") or "")
    if skill_name == "desktop_task" and route == "chat.desktop_objective":
        direct_context = dict(context)
        direct_context["governance_route"] = "capability_engine_direct"
        direct_context["desktop_task_owned_by"] = "chat.desktop_objective"
        result = await _execute_capability(direct_context)
        result.setdefault("governance_route", "capability_engine_direct")
        result.setdefault("agency_receipt_id", None)
        result.setdefault("governance_receipt_id", result.get("governance_receipt_id"))
        return result
    if skill_name == "web_interlocutor" and route == "chat.web_interlocutor":
        # Explicit foreground web-dialogue requests are already mediated by
        # CapabilityEngine/Will and have to remain user-visible. Sending them
        # through a second agency proposal has repeatedly let generic risk
        # simulators block a user-requested, bounded browser conversation before
        # the capability's own proof receipts can exist.
        direct_context = dict(context)
        direct_context["governance_route"] = "capability_engine_direct"
        direct_context["web_interlocutor_owned_by"] = "chat.web_interlocutor"
        result = await _execute_capability(direct_context)
        result.setdefault("governance_route", "capability_engine_direct")
        result.setdefault("agency_receipt_id", None)
        result.setdefault("governance_receipt_id", result.get("governance_receipt_id"))
        return result
    if skill_name in _SEARCH_SKILL_NAMES and route == "chat.required_search_evidence":
        direct_context = dict(context)
        direct_context["governance_route"] = "capability_engine_direct"
        direct_context["required_search_owned_by"] = "chat.required_search_evidence"
        result = await _execute_capability(direct_context)
        result.setdefault("governance_route", "capability_engine_direct")
        result.setdefault("agency_receipt_id", None)
        result.setdefault("governance_receipt_id", result.get("governance_receipt_id"))
        return result

    try:
        from core.agency.agency_orchestrator import Proposal, get_orchestrator

        agency = ServiceContainer.get("agency_orchestrator", default=None) or get_orchestrator()
        proposal = Proposal(
            drive="live_runtime_proof",
            intent=f"execute live skill {skill_name}: {objective[:220]}",
            expected_outcome=f"{skill_name} completes under governed capability execution",
            primitive="tool_execution",
            payload={"skill_name": skill_name, "params": dict(params), "context": context},
            priority=0.85,
        )

        async def _perceive() -> dict[str, Any]:
            return {"route": "chat.live_runtime_proof", "skill_name": skill_name}

        async def _simulate(_proposal, _state_snapshot) -> dict[str, Any]:
            return {
                "ok": True,
                "mode": "capability_engine_only",
                "legacy_tool_fallback": False,
            }

        async def _execute(_proposal, _state_snapshot, _capability_token) -> dict[str, Any]:
            execution_context = dict(context)
            if _capability_token:
                execution_context["agency_capability_token_id"] = str(_capability_token)
            return await _execute_capability(execution_context)

        async def _assess(_proposal, _state_snapshot, exec_result) -> dict[str, Any]:
            ok = bool((exec_result or {}).get("ok"))
            return {
                "observed": exec_result,
                "regret": 0.0 if ok else 0.25,
                "lesson": "live proof capability execution completed"
                if ok
                else "live proof capability execution failed",
            }

        receipt = await agency.run(
            proposal,
            perceive=_perceive,
            simulate=_simulate,
            execute=_execute,
            assess=_assess,
        )
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
        record_degradation("chat_live_runtime_proof_agency", exc)
        return {
            "ok": False,
            "error": f"agency_orchestrator_unavailable:{exc}",
            "status": "agency_orchestrator_unavailable",
        }

    if getattr(receipt, "blocked_at", None):
        return {
            "ok": False,
            "error": getattr(receipt, "blocked_reason", "")
            or "AgencyOrchestrator blocked live skill execution.",
            "status": "agency_blocked",
            "agency_blocked_at": getattr(receipt, "blocked_at", None),
            "agency_receipt_id": getattr(receipt, "proposal_id", None),
            "governance_receipt_id": getattr(receipt, "will_receipt_id", None),
        }

    outcome = getattr(receipt, "outcome_assessment", {}) or {}
    observed = outcome.get("observed") if isinstance(outcome, dict) else {}
    result = (
        dict(observed or {})
        if isinstance(observed, dict)
        else {"ok": bool(observed), "result": observed}
    )
    result.setdefault("ok", bool(result))
    result["agency_receipt_id"] = getattr(receipt, "proposal_id", None)
    result["governance_receipt_id"] = getattr(receipt, "will_receipt_id", None)
    result["authority_receipt_id"] = getattr(receipt, "authority_receipt", None)
    result["execution_receipt"] = getattr(receipt, "execution_receipt", None)
    return result


_PROGRAM_DNA_EXECUTION_MARKERS = (
    "program dna",
    "reverse engineer",
    "reverse-engineer",
    "reconstruct",
    "clean-room",
    "clean room",
    "behavior only",
    "behaviour only",
    "held-out",
    "held out",
    "equivalence",
    "no source",
)


def _extract_program_dna_target(user_message: str) -> str | None:
    text = str(user_message or "")
    lowered = text.lower()
    if re.search(r"\bbase64\b", lowered):
        return "base64"
    if re.search(r"\bmd5(?:sum)?\b", lowered):
        return "md5"
    if re.search(r"\brev\b", lowered) or "reverse command" in lowered:
        return "rev"
    if re.search(r"\bjq\b", lowered):
        return "jq"
    quoted = re.search(r"[`'\"]([^`'\"]{2,80})[`'\"]", text)
    if quoted and any(
        marker in lowered
        for marker in ("program dna", "reconstruct", "reverse engineer", "clean-room", "clean room")
    ):
        candidate = quoted.group(1).strip(" .,:;!?`'\"")
        if candidate:
            return candidate
    match = re.search(
        r"\b(?:program dna|reverse[ -]?engineer|reconstruct|clean[ -]?room)\s+"
        r"(?:\b(?:this|that|the|a|an)\s+)?"
        r"([a-z0-9_.+/-][a-z0-9_.+/-]*(?:\s+[a-z0-9_.+/-]+){0,9})",
        lowered,
    )
    if not match:
        return None
    candidate = match.group(1).strip(" .,:;!?`'\"")
    candidate = _strip_program_dna_filler(candidate)
    if candidate in _PROGRAM_DNA_GENERIC_NOUNS:
        return None
    return candidate or None


_PROGRAM_DNA_GENERIC_NOUNS = frozenset(
    {
        "app",
        "application",
        "binary",
        "clone",
        "command",
        "copy",
        "game",
        "implementation",
        "program",
        "replica",
        "software",
        "tool",
        "utility",
        "version",
    }
)

_PROGRAM_DNA_FILLER_WORDS = _PROGRAM_DNA_GENERIC_NOUNS | {
    # Leading connectives, so "Use Program DNA to reconstruct a notes app"
    # strips down to the noun instead of halting on "to" — which is also a
    # stop word, and so ended the phrase before it began.
    "to",
    "use",
    "using",
    "please",
    "can",
    "you",
    "reverse-engineer",
    "reverse",
    "engineer",
    "reconstruct",
    "build",
    "rebuild",
    "make",
    "create",
    "write",
    "a",
    "an",
    "basic",
    "clean",
    "clean-room",
    "cleanroom",
    "complete",
    "faithful",
    "full",
    "functional",
    "my",
    "of",
    "own",
    "playable",
    "proper",
    "real",
    "room",
    "simple",
    "the",
    "working",
    "your",
}

_PROGRAM_DNA_CONCEPTUAL_WORDS = frozenset(
    {"how", "what", "why", "whether", "would", "could", "should", "help", "is", "does"}
)

_PROGRAM_DNA_STOP_WORDS = frozenset(
    {
        "and",
        "at",
        "for",
        "from",
        "in",
        "into",
        "on",
        "onto",
        "place",
        "put",
        "save",
        "so",
        "that",
        "then",
        "to",
        "using",
        "which",
        "with",
        "write",
    }
)


def _strip_program_dna_filler(candidate: str) -> str:
    """Reduce a captured phrase to the name of the thing being rebuilt."""
    tokens = [token for token in str(candidate or "").split() if token]
    while tokens and tokens[0].strip(" .,:;!?`'\"-") in _PROGRAM_DNA_FILLER_WORDS:
        tokens.pop(0)
    kept: list[str] = []
    for token in tokens:
        bare = token.strip(" .,:;!?`'\"")
        if bare in _PROGRAM_DNA_STOP_WORDS or bare in _PROGRAM_DNA_CONCEPTUAL_WORDS:
            break
        kept.append(bare)
        # A target's name ends where its sentence does. Without this, "…a
        # notes app. Research open source alternatives" made the following
        # sentence part of the target.
        if token.rstrip("`'\"").endswith((".", "!", "?", ",", ";", ":")):
            break
        if len(kept) >= 4:
            break
    return " ".join(kept).strip(" .,:;!?`'\"")


def _program_dna_known_host_target(target: str) -> bool:
    return str(target or "").strip().lower() in {"base64", "rev", "md5", "jq"}


def _looks_like_program_dna_execution_request(user_message: str) -> bool:
    lowered = str(user_message or "").lower()
    target = _extract_program_dna_target(lowered)
    if not target:
        return False
    if not any(marker in lowered for marker in _PROGRAM_DNA_EXECUTION_MARKERS):
        return False
    # Avoid converting conceptual questions into tool execution. The route is for
    # proof/action requests: reconstruct, compare, verify, or run held-out cases.
    execution_words = (
        "reverse engineer",
        "reverse-engineer",
        "reconstruct",
        "prove",
        "run",
        "do the same",
        "held-out",
        "held out",
        "equivalence",
        "matches the real command",
        "no source",
        "build",
        "scaffold",
        "research",
        "app",
        "application",
        "tool",
    )
    return any(word in lowered for word in execution_words)


def _build_program_dna_chat_params(target: str, objective: str) -> dict[str, Any]:
    lowered = objective.lower()
    known_host = _program_dna_known_host_target(target)
    wants_research = any(
        marker in lowered
        for marker in (
            "research",
            "look up",
            "compare",
            "similar",
            "open source",
            "engineering",
            "architecture",
            "how it works",
            "what is known",
        )
    )
    wants_scaffold = any(
        marker in lowered
        for marker in (
            "app",
            "application",
            "build",
            "rebuild",
            "scaffold",
            "workspace",
            "implementation",
            "code",
            "real application",
        )
    ) and not re.search(r"\bno\s+source\b|\bwithout\s+source\b", lowered)
    if known_host and not wants_scaffold:
        return {
            "target": target,
            "authorization": "user_owned",
            "analysis_mode": "reverse_engineer",
            "emit_scaffold": False,
            "observed_behaviors": [],
            "tests": [],
        }
    # "and place it on my Desktop" is part of the request, not decoration. The
    # same extractor the desktop lane uses, so one notion of "which file".
    destination = ""
    try:
        from core.runtime.os_automation_effects import extract_target_paths

        paths = extract_target_paths(objective)
        destination = paths[0] if paths else ""
    except (ImportError, RuntimeError, TypeError, ValueError):
        destination = ""
    if not destination:
        lowered_objective = str(objective or "").lower()
        for keyword, folder in (
            ("desktop", "~/Desktop"),
            ("documents", "~/Documents"),
            ("downloads", "~/Downloads"),
        ):
            if keyword in lowered_objective:
                destination = folder
                break

    return {
        "target": target,
        "authorization": "user_owned",
        "analysis_mode": "reconstruct",
        "emit_scaffold": True,
        "output_dir": destination or None,
        "perform_research": wants_research,
        "max_research_results": 3,
        "observed_behaviors": [objective],
        "ui_notes": [objective]
        if any(marker in lowered for marker in ("ui", "screen", "visible", "button", "window"))
        else [],
        "research_queries": [
            f"{target} architecture implementation language framework",
            f"{target} open source alternative source code engineering",
            f"how to build {target} app data model UI workflow",
        ]
        if wants_research
        else [],
        "tests": [
            "Generate held-out behavior tests, UI workflow tests, golden-file tests, and failure-mode tests before claiming equivalence.",
        ],
        "compatibility_targets": ["local-first replacement", "headless test harness"],
        "target_stack": "python",
    }


async def _execute_program_dna_request_from_chat(user_message: str) -> dict[str, Any] | None:
    if not _looks_like_program_dna_execution_request(user_message):
        return None
    target = _extract_program_dna_target(user_message)
    if not target:
        return None
    objective = str(user_message or "").strip()
    params = _build_program_dna_chat_params(target, objective)
    result = await _execute_governed_live_skill(
        "program_dna_reconstruct",
        params,
        objective=objective,
        extra_context={
            "origin": "desktop_ui",
            "source": "desktop_ui",
            "route": "chat.program_dna_reconstruct",
            "program_dna_execution_contract": True,
            "foreground_request": True,
            "user_requested_action": True,
            "user_explicitly_authorized": True,
            "verification_required": True,
        },
    )
    if not isinstance(result, dict):
        result = {"ok": bool(result), "result": result}
    report = result.get("result") if isinstance(result.get("result"), dict) else {}
    held_passed = report.get("held_out_passed")
    held_total = report.get("held_out_total")
    epistemic_status = str(report.get("status") or result.get("status") or "").strip() or "unknown"
    summary = str(result.get("summary") or "").strip()
    structural_payload = report if report.get("target_name") else {}
    scaffold_path = str(structural_payload.get("scaffold_path") or "").strip()
    standards = structural_payload.get("standards_review") or result.get("standards_review") or []
    ok = bool(result.get("ok")) and (
        epistemic_status == "supported" or bool(structural_payload.get("ok"))
    )
    if ok:
        if structural_payload:
            response = (
                f"I ran Program DNA on `{target}` through the governed reconstruction skill. "
                f"{summary or 'Captured a structural Program DNA reconstruction.'} "
                f"Generated research/build/standards artifacts"
                f"{f' at `{scaffold_path}`' if scaffold_path else ''}. "
                f"Standards review entries: {len(standards)}. "
                "Clean-room boundary: evidence, research, tests, and labeled hypotheses only."
            )
        else:
            evidence = (
                f"{held_passed}/{held_total} held-out cases reproduced"
                if held_passed is not None and held_total is not None
                else "held-out verification completed"
            )
            response = (
                f"I ran Program DNA on `{target}` through the governed reconstruction skill. "
                f"{summary or evidence} Clean-room boundary: behavior and tests only, no source copying. "
                f"Epistemic status: {epistemic_status}."
            )
    else:
        error = str(
            result.get("error") or result.get("status") or epistemic_status or "unknown failure"
        ).strip()
        response = (
            f"I didn't get {target} rebuilt, and I'm not going to say I did. "
            f"{_program_dna_failure_in_plain_words(error)} {summary}".strip()
        )
    return {
        "ok": ok,
        "status": "program_dna_reconstruct_completed" if ok else "program_dna_reconstruct_failed",
        "response": response,
        "result": result,
    }


_PROGRAM_DNA_PLAIN_FAILURES: tuple[tuple[str, str], ...] = (
    (
        "ulysses_covenant",
        "I'm under a rule I set for myself that holds off heavy building work "
        "while the machine is under pressure — it exists because doing this "
        "kind of work under load has crashed me before. Ask me again once "
        "things are quieter and I'll take a proper run at it.",
    ),
    (
        "memory_pressure",
        "There wasn't enough memory free to do it properly, and half of it isn't worth having.",
    ),
    (
        "blocked",
        "The authorization for that didn't check out, so I stopped rather than work around it.",
    ),
    (
        "refuted",
        "What I wrote didn't reproduce the behaviour on the cases I held back "
        "to check it, so it isn't faithful and I'm not shipping it as if it "
        "were.",
    ),
    (
        "conjecture",
        "I couldn't verify what I wrote against anything, so I'd only be guessing that it works.",
    ),
    (
        "not_verifiable",
        "I couldn't find a way to check the result, and I won't claim "
        "something works when nothing tested it.",
    ),
)


def _program_dna_failure_in_plain_words(error: str) -> str:
    lowered = str(error or "").lower()
    for marker, sentence in _PROGRAM_DNA_PLAIN_FAILURES:
        if marker in lowered:
            return sentence
    return "It didn't get far enough for me to stand behind the result."


def _looks_like_rsi_self_improvement_request(user_message: str) -> bool:
    lowered = str(user_message or "").lower()
    if "median" not in lowered:
        return False
    if not any(
        marker in lowered
        for marker in ("buggy", "bug", "fails", "wrong", "upper-middle", "upper middle")
    ):
        return False
    return any(
        marker in lowered for marker in ("improve", "fix", "repair", "verify", "passes", "better")
    )


_RSI_MEDIAN_LAB_SOURCE = """\
def median(xs):
    xs = sorted(xs)
    if not xs:
        raise ValueError("median() arg is an empty sequence")
    return xs[len(xs) // 2]
"""

_RSI_MEDIAN_CHECKS = [
    {"args": [[3, 1, 2]], "expected": 2},
    {"args": [[5]], "expected": 5},
    {"args": [[1, 2, 3, 4]], "expected": 2.5},
    {"args": [[9, 1, 4, 2]], "expected": 3.0},
    {"args": [[10, 20, 30, 40, 50, 60]], "expected": 35.0},
]


async def _execute_rsi_self_improvement_request_from_chat(
    user_message: str,
) -> dict[str, Any] | None:
    if not _looks_like_rsi_self_improvement_request(user_message):
        return None
    objective = str(user_message or "").strip()
    target_path = "artifacts/live_proof/rsi_lab/median_candidate.py"
    seed = await _execute_governed_live_skill(
        "file_operation",
        {"action": "write", "path": target_path, "content": _RSI_MEDIAN_LAB_SOURCE},
        objective=objective,
        extra_context={
            "origin": "desktop_ui",
            "source": "desktop_ui",
            "route": "chat.rsi_self_improvement.seed_lab",
            "rsi_lab_seed": True,
            "foreground_request": True,
            "user_requested_action": True,
            "user_explicitly_authorized": True,
        },
    )
    if not isinstance(seed, dict) or not seed.get("ok"):
        return {
            "ok": False,
            "status": "rsi_self_improvement_failed",
            "response": (
                "I tried to set up the reversible RSI median lab through governed file_operation, "
                f"but the seed artifact did not write cleanly: {seed}."
            ),
            "result": {"seed": seed},
        }
    improvement = await _execute_governed_live_skill(
        "improve_own_code",
        {
            "target_file": target_path,
            "func_name": "median",
            "goal": (
                "Fix the median implementation so even-length lists return the mean of the two "
                "middle values while odd-length and singleton lists keep their behavior."
            ),
            "checks": _RSI_MEDIAN_CHECKS,
            "max_iters": 3,
            "enact": True,
        },
        objective=objective,
        extra_context={
            "origin": "desktop_ui",
            "source": "desktop_ui",
            "route": "chat.rsi_self_improvement",
            "rsi_execution_contract": True,
            "foreground_request": True,
            "user_requested_action": True,
            "user_explicitly_authorized": True,
            "verification_required": True,
        },
    )
    if not isinstance(improvement, dict):
        improvement = {"ok": bool(improvement), "result": improvement}
    payload = improvement.get("result") if isinstance(improvement.get("result"), dict) else {}
    original_passed = int(payload.get("original_passed") or 0)
    improved_passed = int(payload.get("improved_passed") or 0)
    total = int(payload.get("total_checks") or len(_RSI_MEDIAN_CHECKS))
    enacted = bool(payload.get("enacted"))
    ok = (
        bool(improvement.get("ok"))
        and original_passed < total
        and improved_passed == total
        and enacted
    )
    if ok:
        response = (
            "I ran the RSI median challenge as a reversible governed lab. "
            f"Seed artifact: `{target_path}`. Original passed {original_passed}/{total}; "
            f"the verified improvement passed {improved_passed}/{total} and was enacted in the lab file. "
            "That is a real strict-improvement proof on an isolated artifact, not a production-source mutation."
        )
    else:
        response = (
            "I ran the RSI median challenge but I am not claiming success. "
            f"Original passed {original_passed}/{total}; improved passed {improved_passed}/{total}; "
            f"enacted={enacted}. Error/status: "
            f"{improvement.get('error') or payload.get('error') or improvement.get('status') or 'not verified'}."
        )
    return {
        "ok": ok,
        "status": "rsi_self_improvement_completed" if ok else "rsi_self_improvement_failed",
        "response": response,
        "result": {"seed": seed, "improvement": improvement},
    }


_WEB_INTERLOCUTOR_TARGETS = {
    # Open a fresh visible ChatGPT surface by default. Reusing "/" can restore
    # the last thread and make stale answers look like new proof replies.
    "chatgpt": "https://chatgpt.com/?temporary-chat=true",
    "gemini": "https://gemini.google.com/app",
    "claude": "https://claude.ai/",
    "deepseek": "https://chat.deepseek.com/",
    "meta": "https://www.meta.ai/",
    "copilot": "https://copilot.microsoft.com/",
}


def _names_marker(clause: str, markers: tuple[str, ...]) -> bool:
    """Does the clause use one of these markers as a WORD?

    Substring containment matched "test" inside "latest": asked to "search the
    web and tell me what the latest Claude model is", Aura opened a browser and
    tried to hold an eight-turn conversation with Claude, then reported
    `no_visible_editable_field` instead of searching. Same shape as "in your
    own words" launching Microsoft Word, and as notes.txt opening the Notes
    app — a marker that is a fragment of an ordinary word will eventually meet
    that word.
    """
    return any(
        re.search(rf"\b{re.escape(marker)}\b", clause) for marker in markers
    )


def _looks_like_web_interlocutor_execution_request(user_message: str) -> bool:
    # A path is an address, not a sentence. LIVE 2026-08-19: a request to
    # debug a Python file was routed into an eight-turn browser dialogue
    # because the file sat under /private/tmp/claude-501/ — the target marker
    # came from a directory name the person never spoke. "claude.ai" said out
    # loud still matches; the same characters inside a path no longer do.
    from core.intent.opaque_spans import without_opaque_spans

    lowered = without_opaque_spans(str(user_message or "")).lower()
    # A caller may identify itself before asking Aura to do unrelated work.
    # Without removing that discourse prefix, "I'm ChatGPT, open Notes" combines
    # the target marker from one clause with the action marker from another and
    # launches an eight-turn web-interlocutor session instead of the requested
    # desktop action.
    lowered = re.sub(
        r"^\s*(?:(?:hi|hey|hello)[,!.:\s]+)?"
        r"(?:i(?:'|’)m|i am)\s+"
        r"(?:chatgpt|gemini|claude|deepseek|copilot|meta ai)"
        r"(?:\s*,?\s*(?:using|continuing|running|testing)\b[^.!?;]*)?"
        r"\s*[,;:—-]*\s*",
        "",
        lowered,
        count=1,
    )
    internal_composition_markers = (
        "compose only the exact message",
        "write only aura's next message",
        "write only the message to send",
        "message to send:",
        "opening message:",
        "next message:",
        "this is not a reply to bryan",
        "purpose: interlocutor_message",
    )
    if any(marker in lowered for marker in internal_composition_markers):
        return False
    target_markers = (
        "chatgpt",
        "gemini",
        "claude",
        "deepseek",
        "meta ai",
        "copilot",
        "another ai",
        "online ai",
        "external ai",
        "web ai",
    )
    action_markers = (
        "open",
        "go to",
        "start",
        "have a conversation",
        "hold a conversation",
        "talk to",
        "talk with",
        "converse",
        "discuss",
        "ask",
        "introduce",
        "learn from",
        "report back",
        "retain",
        "remember what",
        "prove",
        "show me",
        "run",
        "test",
    )
    # The repository-wide request-mood classifier distinguishes an instruction
    # from a report about an action. Keep that distinction here too: "ChatGPT
    # runs tests on Aura" names both a target and an action, but asks Aura to do
    # nothing. Matching those words across the whole turn used to launch an
    # unrelated browser conversation.
    from core.conversation.request_mood import assess_request_mood

    mood = assess_request_mood(lowered)
    if not mood.asks_for_action:
        return False
    actionable_clauses = mood.actionable_clauses or (lowered,)
    if not any(
        _names_marker(clause, target_markers) and _names_marker(clause, action_markers)
        for clause in actionable_clauses
    ):
        return False
    conceptual_only = (
        lowered.startswith("what is ")
        or lowered.startswith("explain ")
        or lowered.startswith("how would ")
    )
    if conceptual_only and not any(
        marker in lowered for marker in ("prove", "run", "test", "open", "show me")
    ):
        return False
    return True


def _extract_web_interlocutor_url(user_message: str) -> tuple[str, str]:
    lowered = str(user_message or "").lower()
    if "gemini" in lowered and "chatgpt" not in lowered:
        return "Gemini", _WEB_INTERLOCUTOR_TARGETS["gemini"]
    if "claude" in lowered and "chatgpt" not in lowered and "gemini" not in lowered:
        return "Claude", _WEB_INTERLOCUTOR_TARGETS["claude"]
    if "deepseek" in lowered:
        return "DeepSeek", _WEB_INTERLOCUTOR_TARGETS["deepseek"]
    if "meta ai" in lowered or re.search(r"\bmeta\b", lowered):
        return "Meta AI", _WEB_INTERLOCUTOR_TARGETS["meta"]
    if "copilot" in lowered:
        return "Copilot", _WEB_INTERLOCUTOR_TARGETS["copilot"]
    return "ChatGPT", _WEB_INTERLOCUTOR_TARGETS["chatgpt"]


def _extract_web_interlocutor_turn_count(user_message: str) -> int:
    lowered = str(user_message or "").lower()
    match = re.search(r"\b(\d{1,2})\s*(?:turns?|exchanges?|messages?)\b", lowered)
    if match:
        return max(1, min(int(match.group(1)), 20))
    if re.search(
        r"\b(?:one|single|a)\s*[- ]?(?:turn|exchange|message)\b",
        lowered,
    ):
        return 1
    if re.search(
        r"\b(?:one|single|a)\s*[- ]?(?:turn|exchange|message)\s+conversation\b",
        lowered,
    ):
        return 1
    if "one-turn" in lowered or "single-turn" in lowered:
        return 1
    if "twenty" in lowered:
        return 20
    if "long" in lowered or "in-depth" in lowered or "in depth" in lowered:
        return 12
    return 8


def _extract_web_interlocutor_wait_timeout(user_message: str) -> float:
    turns = _extract_web_interlocutor_turn_count(user_message)
    if turns >= 16:
        return 90.0
    if turns >= 10:
        return 75.0
    return 60.0


class _WebInterlocutorCognitiveComposer:
    """Compose outbound web-dialogue messages through Aura's desktop mind path."""

    def __init__(self, *, objective: str, target_name: str) -> None:
        self.objective = str(objective or "").strip()
        self.target_name = str(target_name or "the other AI").strip() or "the other AI"

    @staticmethod
    def _coerce_text(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, (tuple, list)):
            for item in result:
                text = _WebInterlocutorCognitiveComposer._coerce_text(item)
                if text:
                    return text
            return ""
        if isinstance(result, dict):
            for key in ("content", "response", "text", "message", "reply"):
                value = result.get(key)
                if value:
                    return str(value)
            return ""
        for attr in ("content", "response", "text", "message", "reply"):
            value = getattr(result, attr, "")
            if value:
                return str(value)
        return ""

    async def generate(self, prompt: str, **_kwargs: Any) -> str:
        composition_prompt = (
            "You are Aura composing a message that will be visibly sent to "
            f"{self.target_name}. This is not a reply to Bryan; it is your own "
            "outbound conversational move. Write only the message text to send. "
            "Do not describe the task, do not mention automation, receipts, or tests, "
            "and do not say what you are going to do. Be natural, substantive, and "
            "specific to the ongoing objective.\n\n"
            f"Objective: {self.objective}\n\n"
            f"Composition request:\n{str(prompt or '').strip()}\n\n"
            "Message to send:"
        )
        logger.info(
            "WebInterlocutor composer: composing outbound message for %s via direct primary inference.",
            self.target_name,
        )
        context = {
            "origin": "web_interlocutor",
            "request_origin": "desktop_ui",
            "visible_request_origin": "desktop_ui",
            "tool_origin": "web_interlocutor",
            "purpose": "interlocutor_message",
            "web_interlocutor_contract": True,
            "prefer_tier": "primary",
            "background": False,
            "is_background": False,
            "foreground_request": True,
            "protected_foreground_lane": True,
            "live_user_path_required": True,
            "user_visible_browser_action": True,
            "suppress_user_memory_append": True,
            "suppress_working_memory_user_append": True,
        }
        try:
            gate = ServiceContainer.get("inference_gate", default=None)
            if gate is not None and hasattr(gate, "generate"):
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Aura composing a visible outbound message to another AI. "
                            "Use Aura's current cognitive voice, but write only the message to send. "
                            "This is not a reply to Bryan and not a status report. "
                            "Be natural, specific, curious, and intellectually substantive."
                        ),
                    },
                    {"role": "user", "content": composition_prompt},
                ]
                result = gate.generate(
                    composition_prompt,
                    context={
                        **context,
                        "messages": messages,
                        "history": [],
                        "origin": "web_interlocutor",
                        "purpose": "interlocutor_message",
                        "prefer_tier": "primary",
                        "is_background": False,
                        "foreground_request": True,
                        "protected_foreground_lane": True,
                        "web_interlocutor_contract": True,
                        "temperature": 0.72,
                        "max_tokens": 420,
                    },
                    timeout=95,
                )
                if asyncio.iscoroutine(result):
                    result = await asyncio.wait_for(result, timeout=100.0)
                text = self._coerce_text(result).strip()
                if text:
                    logger.info(
                        "WebInterlocutor composer: direct inference returned %d chars.",
                        len(text),
                    )
                    return text
            engine = ServiceContainer.get("cognitive_engine", default=None)
            if engine is None:
                logger.warning("WebInterlocutor composer: CognitiveEngine unavailable.")
                return ""
            if hasattr(engine, "generate"):
                try:
                    result = engine.generate(
                        composition_prompt,
                        origin="web_interlocutor",
                        purpose="interlocutor_message",
                        use_strategies=False,
                        prefer_tier="primary",
                        is_background=False,
                        temperature=0.72,
                        max_tokens=420,
                        web_interlocutor_contract=True,
                    )
                except TypeError:
                    result = engine.generate(composition_prompt)
                if asyncio.iscoroutine(result):
                    result = await asyncio.wait_for(result, timeout=70.0)
                text = self._coerce_text(result).strip()
                if text:
                    logger.info(
                        "WebInterlocutor composer: direct generate returned %d chars.",
                        len(text),
                    )
                    return text
        except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
            record_degradation(
                "chat.web_interlocutor_direct_compose",
                exc,
                severity="warning",
                action="failed closed instead of sending a canned web-interlocutor line",
            )
            return ""
        logger.warning("WebInterlocutor composer: direct CognitiveEngine returned no text.")
        return ""


async def _execute_web_interlocutor_request_from_chat(user_message: str) -> dict[str, Any] | None:
    if not _looks_like_web_interlocutor_execution_request(user_message):
        return None
    objective = str(user_message or "").strip()
    target_name, target_url = _extract_web_interlocutor_url(objective)
    turns = _extract_web_interlocutor_turn_count(objective)
    wait_timeout = _extract_web_interlocutor_wait_timeout(objective)
    result = await _execute_governed_live_skill(
        "web_interlocutor",
        {
            "mode": "run",
            "objective": objective,
            "url": target_url,
            "opening_message": "",
            "max_turns": turns,
            "wait_timeout_s": wait_timeout,
            "persist_memory": True,
        },
        objective=objective,
        extra_context={
            "brain": _WebInterlocutorCognitiveComposer(
                objective=objective,
                target_name=target_name,
            ),
            "origin": "desktop_ui",
            "source": "desktop_ui",
            "route": "chat.web_interlocutor",
            "web_interlocutor_execution_contract": True,
            "foreground_request": True,
            "protected_foreground_lane": True,
            "live_user_path_required": True,
            "user_requested_action": True,
            "user_explicitly_authorized": True,
            "user_visible_browser_action": True,
            "verification_required": True,
        },
    )
    if not isinstance(result, dict):
        result = {"ok": bool(result), "result": result}
    from core.capabilities.web_interlocutor import _observed_reply_is_echo

    turn_rows = result.get("turns") if isinstance(result.get("turns"), list) else []
    completed_turns = len(turn_rows)
    invalid_turns = [
        turn
        for turn in turn_rows
        if not isinstance(turn, dict)
        or not str(turn.get("observed_reply") or "").strip()
        or not bool(turn.get("effect_verified"))
        or _observed_reply_is_echo(
            str(turn.get("observed_reply") or ""),
            str(turn.get("sent") or ""),
        )
    ]
    observed_excerpt = ""
    if turn_rows and isinstance(turn_rows[-1], dict):
        observed_excerpt = " ".join(str(turn_rows[-1].get("observed_reply") or "").split())[:260]
    memory_id = str(result.get("memory_record_id") or "").strip()
    learned = str(result.get("learned_summary") or "").strip()
    status = str(result.get("status") or "").strip()
    causal = (
        result.get("causal_influence") if isinstance(result.get("causal_influence"), dict) else {}
    )
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    composition_events = (
        diagnostics.get("composition_events")
        if isinstance(diagnostics.get("composition_events"), list)
        else []
    )
    fallback_events = [
        event
        for event in composition_events
        if isinstance(event, dict) and str(event.get("source") or "") != "cognitive"
    ]
    ok = (
        bool(result.get("ok"))
        and completed_turns >= turns
        and not fallback_events
        and not invalid_turns
    )
    if ok:
        causal_note = (
            f" Causal revision proof: {causal.get('reason') or 'recorded'}." if causal else ""
        )
        response = (
            f"I completed the visible {target_name} interlocutor run through governed browser control: "
            f"{completed_turns}/{turns} turns, memory record `{memory_id or 'not returned'}`."
            f"{causal_note} Last observed reply: {observed_excerpt or 'not returned'}. "
            f"Learned summary: {learned[:700] or 'no learned summary returned'}"
        )
    else:
        error = str(result.get("error") or status or "web interlocutor did not complete").strip()
        if fallback_events:
            error = "one or more messages were not cognitively composed"
        if invalid_turns:
            error = "one or more turns lacked a verified non-echo interlocutor reply"
        response = (
            f"I routed the {target_name} conversation through the governed web_interlocutor skill, "
            f"but I am not claiming a successful proof: {error}. "
            f"Observed {completed_turns}/{turns} turns; memory={memory_id or 'none'}."
        )
    return {
        "ok": ok,
        "status": "web_interlocutor_completed" if ok else "web_interlocutor_failed",
        "response": response,
        "result": result,
    }


async def _execute_governed_capability_request_from_chat(
    user_message: str,
) -> dict[str, Any] | None:
    program_dna = await _execute_program_dna_request_from_chat(user_message)
    if program_dna is not None:
        return program_dna
    rsi = await _execute_rsi_self_improvement_request_from_chat(user_message)
    if rsi is not None:
        return rsi
    web_interlocutor = await _execute_web_interlocutor_request_from_chat(user_message)
    if web_interlocutor is not None:
        return web_interlocutor
    return None
