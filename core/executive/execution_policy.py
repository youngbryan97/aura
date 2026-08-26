"""Canonical effect and risk classification for governed tool execution.

The capability engine, orchestrator, and authority gateway must reason about the
same operation.  This module resolves a concrete invocation into one effect
scope and one conservative risk class without treating an unknown tool as safe.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from core.skills.catalog_policy import resolve_skill_policy

#: The registry is absent in tests and in tools that never build a container.
_REGISTRY_LOOKUP_FAILURES = (ImportError, AttributeError, KeyError, RuntimeError, TypeError)

RISK_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

_MESSAGES_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_MESSAGES_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,240}$")
_MESSAGES_PRIVATE_CONTEXT_KEYS = frozenset(
    {
        "body",
        "content",
        "message",
        "objective",
        "original_request",
        "query",
        "reply",
        "response",
        "text",
        "user_objective",
    }
)

_GENERIC_EFFECT_SCOPES = {
    "browser": "external_io",
    "command": "privileged_mutation",
    "curiosity_web_search": "read_only",
    "edit_file": "read_write_artifacts",
    "execute": "privileged_mutation",
    # No generic "file_write" mapping: the canonical file skill is
    # file_operation (scoped per-invocation above). A bare unregistered
    # file_write stays "unknown" → classified critical by default —
    # presuming a write scope for an unowned tool under-gated it.
    "get_time": "status",
    "grep_search": "read_only",
    "list_dir": "read_only",
    "multi_replace_file_content": "read_write_artifacts",
    "notify_user": "external_io",
    "read_file": "read_only",
    "replace_file_content": "read_write_artifacts",
    "run_command": "privileged_mutation",
    "run_python": "sandboxed_compute",
    "search_web": "read_only",
    "self_diagnosis": "status",
    "sensory_motor_browser_research": "read_only",
    "status": "status",
    "subconscious_sandbox_probe": "sandboxed_compute",
    "swarm_debate": "pure_compute",
    "system_health": "status",
    "terminal": "privileged_mutation",
    "view_file": "read_only",
    "write_file": "read_write_artifacts",
    "write_to_file": "read_write_artifacts",
}

_CRITICAL_TOOLS = frozenset(
    {
        "auto_refactor",
        "command",
        "computer_use",
        "desktop_task",
        "execute",
        "install_package",
        "manage_abilities",
        "os_automation",
        "os_manipulation",
        "run_command",
        "self_evolution",
        "self_improvement",
        "self_modify",
        "self_repair",
        "shell",
        "sovereign_terminal",
        "terminal",
        "train_self",
        "web_interlocutor",
    }
)


def normalize_tool_name(value: Any) -> str:
    return str(value or "").strip().lower()


def canonical_authority_arguments(
    tool_name: Any,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact payload governance may persist and sign.

    Private Messages text is execution data, not governance telemetry. The
    authority chain binds its digest and byte/character bounds; the final sink
    recomputes that envelope from plaintext before accepting the capability.
    Unknown argument *names* remain visible so policy can reject scope
    expansion, while their potentially sensitive values never enter receipts.
    """

    arguments = dict(params or {})
    if normalize_tool_name(tool_name) != "messages":
        return arguments

    action = str(arguments.get("action") or "status").strip().lower()
    safe: dict[str, Any] = {"action": action}

    if arguments.get("alias_invalid") is True:
        safe["alias_invalid"] = True
        safe["alias_sha256"] = str(arguments.get("alias_sha256") or "")
    else:
        alias = str(arguments.get("alias") or "primary_operator").strip().lower()
        if _MESSAGES_ALIAS_RE.fullmatch(alias):
            safe["alias"] = alias
        else:
            safe.update(
                {
                    "alias_invalid": True,
                    "alias_sha256": hashlib.sha256(alias.encode("utf-8")).hexdigest(),
                }
            )

    if "body" in arguments and arguments.get("body") is not None:
        body = str(arguments.get("body") or "")
        encoded = body.encode("utf-8", errors="strict")
        safe.update(
            {
                "body_bytes": len(encoded),
                "body_chars": len(body),
                "body_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    elif any(key in arguments for key in ("body_bytes", "body_chars", "body_sha256")):
        safe.update(
            {
                "body_bytes": arguments.get("body_bytes"),
                "body_chars": arguments.get("body_chars"),
                "body_sha256": str(arguments.get("body_sha256") or ""),
            }
        )

    if arguments.get("idempotency_invalid") is True:
        safe["idempotency_invalid"] = True
        safe["idempotency_sha256"] = str(arguments.get("idempotency_sha256") or "")
    else:
        idempotency = str(arguments.get("idempotency_key") or "").strip()
        if idempotency:
            if _MESSAGES_IDEMPOTENCY_RE.fullmatch(idempotency):
                safe["idempotency_key"] = idempotency
            else:
                safe.update(
                    {
                        "idempotency_invalid": True,
                        "idempotency_sha256": hashlib.sha256(
                            idempotency.encode("utf-8")
                        ).hexdigest(),
                    }
                )

    canonical_keys = {
        "action",
        "alias",
        "alias_invalid",
        "alias_sha256",
        "body",
        "body_bytes",
        "body_chars",
        "body_sha256",
        "idempotency_invalid",
        "idempotency_key",
        "idempotency_sha256",
        "unexpected_argument_names",
    }
    prior_unexpected = arguments.get("unexpected_argument_names")
    if isinstance(prior_unexpected, (list, tuple, set, frozenset)):
        prior_names = tuple(str(key) for key in prior_unexpected)
    elif prior_unexpected:
        prior_names = (str(prior_unexpected),)
    else:
        prior_names = ()
    unexpected = sorted(
        {
            *(str(key) for key in set(arguments) - canonical_keys),
            *prior_names,
        }
    )
    if unexpected:
        safe["unexpected_argument_names"] = unexpected
    return safe


def canonical_authority_context(
    tool_name: Any,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hide private message prose while preserving authority provenance."""

    safe = dict(context or {})
    if normalize_tool_name(tool_name) != "messages":
        return safe
    for key in _MESSAGES_PRIVATE_CONTEXT_KEYS:
        if key not in safe or safe.get(key) is None:
            continue
        value = str(safe.pop(key))
        encoded = value.encode("utf-8", errors="strict")
        safe[f"{key}_chars"] = len(value)
        safe[f"{key}_sha256"] = hashlib.sha256(encoded).hexdigest()
    return safe


def normalize_risk(value: Any, *, default: str = "critical") -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in RISK_ORDER else default


def risk_at_most(actual: Any, maximum: Any) -> bool:
    return RISK_ORDER[normalize_risk(actual)] <= RISK_ORDER[normalize_risk(maximum)]


def _is_path_within_workspace(path: Any) -> bool:
    raw = str(path or "").strip()
    if not raw:
        return False
    try:
        root = Path.cwd().resolve()
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = root / target
        resolved = target.resolve()
        return os.path.commonpath([str(root), str(resolved)]) == str(root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _workspace_file_scope(params: dict[str, Any]) -> str | None:
    action = str((params or {}).get("action") or "").strip().lower()
    if action not in {"append", "copy", "exists", "list", "patch", "read", "write"}:
        return None
    if not _is_path_within_workspace((params or {}).get("path")):
        return None
    if action == "copy" and not _is_path_within_workspace((params or {}).get("destination")):
        return None
    return "workspace_file_io"


def _computer_use_scope(params: dict[str, Any]) -> str | None:
    action = str((params or {}).get("action") or "").strip().lower()
    if action in {"get_clipboard", "read_menu_clock", "read_screen_text", "wait"}:
        return "read_only"
    if action in {
        "click",
        "hotkey",
        "move_aura_bubble",
        "open_app",
        "open_url",
        "run_applescript",
        "scroll",
        "set_clipboard",
        "type",
    }:
        return "foreground_desktop_control"
    if action in {"move_file", "render_text_pdf", "write_text_file"}:
        return "desktop_file_io"
    if action != "run_command":
        return None
    try:
        argv = shlex.split(str((params or {}).get("target") or ""))
    except ValueError:
        return None
    if not argv:
        return None
    binary = Path(argv[0]).name
    if binary in {"cat", "echo", "find", "grep", "ls", "pwd", "tree"}:
        return "sandboxed_compute"
    if binary == "git":
        subcommand = argv[1] if len(argv) > 1 else ""
        if subcommand in {"branch", "diff", "log", "rev-parse", "show", "status"}:
            return "sandboxed_compute"
        return "subprocess"
    if binary in {"pip", "python3"} and len(argv) == 2 and argv[1] in {"--version", "-V"}:
        return "sandboxed_compute"
    return "subprocess"


def _auto_refactor_scope(params: dict[str, Any]) -> str:
    params = params or {}
    mode = str(params.get("mode") or params.get("action") or "scan").strip().lower()
    if bool(
        params.get("apply")
        or params.get("write")
        or params.get("commit")
        or params.get("promote")
        or params.get("allow_mutation")
        or mode in {"apply", "commit", "promote", "rewrite", "write"}
    ):
        return "privileged_mutation"
    return "sandboxed_compute" if bool(params.get("run_tests")) else "read_only"


def _test_generator_scope(params: dict[str, Any]) -> str:
    """A read-only request writes only to an ephemeral sandbox."""

    return (
        "sandboxed_compute"
        if bool((params or {}).get("read_only"))
        else "read_write_artifacts"
    )


def _self_evolution_scope(params: dict[str, Any]) -> str:
    """Proposal-only inspection is distinct from applying a code mutation."""

    arguments = params or {}
    action = str(arguments.get("action") or "propose").strip().lower()
    if action in {"", "propose"} and bool(arguments.get("read_only")):
        return "read_only"
    return "privileged_mutation"



# Effect scope belongs to what an invocation actually does, not to what its
# skill is capable of. desktop_task is the composite case: it can write files,
# drive AppleScript and read the screen, so its blanket declaration is the
# widest thing any of its steps might do. Governing every invocation at that
# width blocks a screen-reading objective as if it were a filesystem write, and
# governing at the parent's width also misses a child that does more. The
# honest scope of a plan is the widest scope among the steps it actually
# contains — computed here, from the steps, at the layer that knows them.
_DESKTOP_ACTION_SCOPES: dict[str, str] = {
    "get_clipboard": "read_only",
    "inspect_screen": "read_only",
    "read_menu_clock": "read_only",
    "read_screen_text": "read_only",
    "wait": "read_only",
    "click": "foreground_desktop_control",
    "hotkey": "foreground_desktop_control",
    "move_aura_bubble": "foreground_desktop_control",
    "open_app": "foreground_desktop_control",
    "open_url": "foreground_desktop_control",
    "run_applescript": "foreground_desktop_control",
    "scroll": "foreground_desktop_control",
    "set_clipboard": "foreground_desktop_control",
    "system_control": "foreground_desktop_control",
    "type": "foreground_desktop_control",
    "create_folder": "desktop_file_io",
    "fetch_topic_image": "desktop_file_io",
    "move_file": "desktop_file_io",
    "render_text_pdf": "desktop_file_io",
    "write_text_file": "desktop_file_io",
}
# Widest last: a plan is governed by the most consequential thing in it.
_DESKTOP_SCOPE_RANK: tuple[str, ...] = (
    "read_only",
    "foreground_desktop_control",
    "desktop_file_io",
)


def scope_is_within(presented: Any, granted: Any) -> bool:
    """Whether a scope is at most what was granted, on a known ordering.

    A plan is governed by the widest thing in it, so a lease issued for a
    plan is a lease for that width — and every step of the plan is at most
    that wide. Refusing a step because it is NARROWER than what was approved
    refuses something strictly safer than the thing already allowed.

    LIVE 2026-08-26: a desktop task approved at 'desktop_file_io' could not
    run its own file-writing step, which presented 'foreground_desktop_control'
    against that lease. She could not write a file at all.

    Only within one ordering. 'subprocess' and 'desktop_file_io' are not more
    or less than each other, and pretending they are would turn authority for
    one into authority for the other, so anything incomparable is refused.
    """
    here = str(presented or "").strip().lower()
    there = str(granted or "").strip().lower()
    if not here or not there:
        return False
    if here == there:
        return True
    if here not in _DESKTOP_SCOPE_RANK or there not in _DESKTOP_SCOPE_RANK:
        return False
    return _DESKTOP_SCOPE_RANK.index(here) <= _DESKTOP_SCOPE_RANK.index(there)


def _desktop_task_scope(params: dict[str, Any]) -> str | None:
    """The widest scope among a desktop plan's declared steps, or None."""
    raw_steps = (params or {}).get("steps")
    if isinstance(raw_steps, str):
        try:
            raw_steps = json.loads(raw_steps)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    widest = -1
    for step in raw_steps:
        action = ""
        if isinstance(step, dict):
            action = str(step.get("action") or "").strip().lower()
        elif isinstance(step, str):
            action = step.strip().lower()
        scope = _DESKTOP_ACTION_SCOPES.get(action)
        if scope is None:
            # An unrecognised step could do anything; refuse to narrow.
            return None
        widest = max(widest, _DESKTOP_SCOPE_RANK.index(scope))
    if widest < 0:
        return None
    return _DESKTOP_SCOPE_RANK[widest]


def resolve_execution_effect_scope(
    tool_name: Any,
    params: dict[str, Any] | None = None,
    *,
    declared_effect_scope: Any = "",
) -> str:
    """Resolve invocation-specific scope, returning ``unknown`` on ambiguity."""

    name = normalize_tool_name(tool_name)
    arguments = dict(params or {})
    if name == "file_operation":
        scoped = _workspace_file_scope(arguments)
        if scoped:
            return scoped
    elif name == "computer_use":
        scoped = _computer_use_scope(arguments)
        if scoped:
            return scoped
    elif name == "desktop_task":
        scoped = _desktop_task_scope(arguments)
        if scoped:
            return scoped
    elif name == "auto_refactor":
        return _auto_refactor_scope(arguments)
    elif name == "test_generator":
        return _test_generator_scope(arguments)
    elif name == "self_evolution":
        return _self_evolution_scope(arguments)
    elif name == "web_search":
        # Autonomous web research is READ-ONLY by contract: fetching and
        # summarizing never writes artifacts or posts. A broader scope
        # declared by a skill wrapper must not widen what autonomy may do
        # with a search (test_fictional_ai_runtime_contract pins this).
        return "read_only"
    elif name == "email_adapter":
        mode = str(arguments.get("mode") or "check").strip().lower()
        if mode in {"check", "read", "search"}:
            return "read_only"
    elif name == "messages":
        action = str(arguments.get("action") or "status").strip().lower()
        if action == "status":
            return "status"
        if action in {"pause", "resume"}:
            return "state_mutation"
    elif name == "reddit_adapter":
        mode = str(arguments.get("mode") or "browse").strip().lower()
        if mode in {
            "browse",
            "check_inbox",
            "check_shadowban",
            "read_post",
            "read_rules",
        }:
            return "read_only"

    declared = str(declared_effect_scope or "").strip().lower()

    # Every branch above is one skill's name. A skill that declares its own
    # per-action scopes needs no branch: ACTION_EFFECT_SCOPES on the class says
    # what each action costs, and the same table already drives which skills a
    # turn may be offered. Consulting it here is what makes the declaration
    # true at the moment of the call, so a reader added tomorrow is scoped as a
    # read without an edit to this chain.
    from core.skills.action_scope import (
        declared_action_name,
        declared_action_scopes,
        resolve_skill_target,
        skill_class_named,
    )

    target = resolve_skill_target(_skill_meta_for(name)) or skill_class_named(name)
    scopes = declared_action_scopes(target) or declared_action_scopes(skill_class_named(name))
    if scopes:
        scoped = scopes.get(declared_action_name(target, arguments))
        if scoped:
            return scoped

    policy = resolve_skill_policy(name, declared)
    if policy is not None:
        return policy.effect_scope
    return _GENERIC_EFFECT_SCOPES.get(name, "unknown")



def _skill_meta_for(name: str) -> Any:
    """The registry entry for a skill, or None when the registry is not up.

    Scope resolution runs in tests and in tools that never build a container,
    so a missing registry means "no declaration to read", not an error.
    """
    try:
        from core.container import get_container
        from core.service_names import ServiceNames

        engine = get_container().try_get(ServiceNames.CAPABILITY_ENGINE)
        skills = getattr(engine, "skills", None) or {}
        return skills.get(name)
    except _REGISTRY_LOOKUP_FAILURES:
        return None


def classify_execution_risk(
    tool_name: Any,
    params: dict[str, Any] | None = None,
    *,
    effect_scope: Any = "",
    metabolic_cost: int = 1,
) -> str:
    """Classify a concrete execution; unknown effects are critical by default."""

    name = normalize_tool_name(tool_name)
    arguments = dict(params or {})
    scope = str(effect_scope or "").strip().lower() or resolve_execution_effect_scope(
        name, arguments
    )
    if name == "run_code" or name == "run_python":
        stateful = bool(arguments.get("stateful", True))
        return "critical" if stateful else "high"
    if name == "auto_refactor":
        if scope == "privileged_mutation":
            return "critical"
        return "high" if scope == "sandboxed_compute" else "low"
    if name == "test_generator" and scope == "sandboxed_compute":
        return "high"
    if name == "diagnose_repo":
        # Running a project's own test suite is not the same act as running
        # code the model just wrote, which is what sandboxed_compute is rated
        # high for. The code here was on disk before the turn began, the
        # person named the directory, and nothing the model produces is
        # executed. It is medium rather than low because a test suite is still
        # somebody else's code running on this machine.
        #
        # LIVE, 2026-08-22: rated high by scope alone, it was refused with
        # "Requires user confirmation" on the very turn that asked for it,
        # after routing had finally found it.
        return "medium"
    if name == "self_evolution" and scope == "read_only":
        return "low"
    if name in _CRITICAL_TOOLS:
        if name == "computer_use" and scope == "read_only":
            return "low"
        return "critical"
    if scope in {"status", "read_only", "pure_compute"}:
        return "low"
    if scope == "sandboxed_compute":
        return "high"
    if scope in {"external_io", "state_mutation"}:
        return "medium"
    if scope in {
        "desktop_file_io",
        "foreground_browser_dialogue",
        "foreground_desktop_control",
        "read_write_artifacts",
        "workspace_file_io",
    }:
        return "high"
    if scope in {"privileged_mutation", "subprocess", "unknown"}:
        return "critical"
    if int(metabolic_cost or 1) >= 3:
        return "high"
    if int(metabolic_cost or 1) >= 2:
        return "medium"
    return "critical"


__all__ = [
    "RISK_ORDER",
    "canonical_authority_arguments",
    "canonical_authority_context",
    "classify_execution_risk",
    "normalize_risk",
    "normalize_tool_name",
    "resolve_execution_effect_scope",
    "risk_at_most",
]
