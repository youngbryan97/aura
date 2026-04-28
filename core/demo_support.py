from __future__ import annotations
from core.runtime.errors import record_degradation

from core.runtime.atomic_writer import atomic_write_text

import ast
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

logger = logging.getLogger("Aura.DemoSupport")

_BACKGROUND_DIAGNOSTIC_RE = re.compile(
    r"([A-Za-z0-9_./-]+\.(?:jsonl|jsx|tsx|yaml|yml|toml|swift|html|json|css|md|txt|py|ts|js|sh|rs))",
    re.IGNORECASE,
)
_BACKGROUND_REQUEST_HINTS = (
    "diagnostic",
    "inspect",
    "analyze",
    "analyse",
    "review",
    "audit",
    "check",
    "trace",
)
_BACKGROUND_ASYNC_HINTS = (
    "background",
    "do not wait",
    "don't wait",
    "when you're done",
    "when done",
    "post the result here",
    "print the result here",
)
_CONTINUITY_HINTS = (
    "what were you doing right before this session started",
    "what were you doing before this session started",
    "what were you doing before this session",
    "right before this session started",
    "before this session started",
    "what were you just working on",
    "what were you working on just now",
    "what did you just work on",
    "what did you just inspect",
    "what did you just finish in the background",
)
_PRIORITY_HINTS = (
    "based on your current system state and goals",
    "what should you be focusing on right now",
    "what should you be doing right now",
)
_INTERNAL_FOCUS_PATTERNS = (
    "[silent auto-fix]",
    "traceback",
    "exception in callback",
    "temporal_obligation_active",
    "runtimeerror",
    "timeouterror",
    "diagnose unmapped critical traceback",
    "error:",
)
_SEARCH_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "models",
        "models_gguf",
        "node_modules",
    }
)
_RECENT_ACTIVITY_MAX_AGE_SECONDS = 12 * 60 * 60


def _collapsed(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _demo_state_path() -> Path:
    home = Path.home() / ".aura" / "data"
    home.mkdir(parents=True, exist_ok=True)
    return home / "demo_last_activity.json"


def extract_background_diagnostic_target(message: str) -> Optional[str]:
    collapsed = _collapsed(message)
    if not any(hint in collapsed for hint in _BACKGROUND_REQUEST_HINTS):
        return None
    if not any(hint in collapsed for hint in _BACKGROUND_ASYNC_HINTS):
        return None
    match = _BACKGROUND_DIAGNOSTIC_RE.search(str(message or ""))
    if not match:
        return None
    return match.group(1)


def is_recent_activity_query(message: str) -> bool:
    collapsed = _collapsed(message)
    return any(hint in collapsed for hint in _CONTINUITY_HINTS)


def is_priority_probe(message: str) -> bool:
    collapsed = _collapsed(message)
    return any(hint in collapsed for hint in _PRIORITY_HINTS)


def build_background_diagnostic_ack(target: str) -> str:
    target_name = Path(target).name
    return (
        f"I'm on it. I'll inspect `{target_name}` in the background, trace its core function, "
        "and post the result here when I'm done."
    )


def _load_last_activity() -> Optional[Dict[str, Any]]:
    path = _demo_state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        record_degradation('demo_support', exc)
        logger.debug("Failed to read demo activity state: %s", exc)
        return None


def _save_last_activity(payload: Dict[str, Any]) -> None:
    path = _demo_state_path()
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_target_path(target: str, repo_root: Optional[Path] = None) -> Optional[Path]:
    root = Path(repo_root or _repo_root()).resolve()
    raw = Path(os.path.expanduser(str(target or "").strip()))
    if raw.is_absolute() and raw.is_file():
        return raw
    direct = (root / raw).resolve()
    if direct.is_file():
        return direct

    name = raw.name
    try:
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SEARCH_EXCLUDED_DIRS]
            if name in filenames:
                return (Path(current_root) / name).resolve()
    except Exception as exc:
        record_degradation('demo_support', exc)
        logger.debug("Target resolution fallback failed for %s: %s", target, exc)
    return None


def _truncate(text: str, limit: int = 220) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _sanitize_focus_label(text: Any, *, limit: int = 100) -> str:
    label = " ".join(str(text or "").split()).strip()
    if not label:
        return ""
    label = re.sub(r"^\[[^\]]+\]\s*", "", label).strip()
    lowered = label.lower()
    if any(pattern in lowered for pattern in _INTERNAL_FOCUS_PATTERNS):
        return ""
    if lowered.startswith(("diagnose ", "repair ", "fix ", "investigate ")) and "error" in lowered:
        return ""
    if lowered.startswith("researching "):
        return ""
    if "seek novel stimulation" in lowered or "feeling idle and energized" in lowered:
        return ""
    return _truncate(label, limit)


def _first_sentence(text: str) -> str:
    collapsed = " ".join(str(text or "").split()).strip()
    if not collapsed:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", collapsed, maxsplit=1)
    return parts[0]


def _symbol_weight(node: ast.AST) -> int:
    return sum(1 for _ in ast.walk(node))


def _describe_primary_symbol(
    classes: list[ast.ClassDef],
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> str:
    symbols: list[ast.AST] = [*classes, *functions]
    if not symbols:
        return ""

    primary = max(symbols, key=_symbol_weight)
    if isinstance(primary, ast.ClassDef):
        public_methods = [
            node.name
            for node in primary.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        ]
        if public_methods:
            return (
                f"the `{primary.name}` class, with most of the public behavior exposed through "
                f"`{', '.join(public_methods[:3])}`"
            )
        return f"the `{primary.name}` class"
    return f"the `{primary.name}()` function"


def _python_summary(path: Path, source: str) -> str:
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno else "an unknown line"
        detail = _truncate(exc.msg or "invalid syntax", 160)
        return (
            f"I finished the background diagnostic on `{path.name}`. "
            f"It doesn't currently parse as Python: {detail} at {location}, so that's the first boundary that needs attention."
        )

    doc = _truncate(_first_sentence(ast.get_docstring(module) or ""), 180)
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    functions = [
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    class_names = [node.name for node in classes]
    primary_symbol = _describe_primary_symbol(classes, [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))])

    focus_bits = []
    if doc:
        focus_bits.append(doc)
    if primary_symbol:
        focus_bits.append(f"The module's center of gravity is {primary_symbol}")
    if class_names:
        focus_bits.append(f"Main classes: {', '.join(class_names[:3])}")
    if functions:
        focus_bits.append(f"Key entry points: {', '.join(functions[:4])}")

    if not focus_bits:
        focus_bits.append("It is a local module with no module docstring, so I traced its structure directly.")

    return (
        f"I finished the background diagnostic on `{path.name}`. "
        f"From the file's actual structure, its core function looks like {_truncate(' '.join(focus_bits), 320)}."
    )


def _generic_summary(path: Path, source: str) -> str:
    first_non_empty = ""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped:
            first_non_empty = stripped
            break
    detail = _truncate(first_non_empty or f"{path.name} contains local project data.", 180)
    return (
        f"I finished the background diagnostic on `{path.name}`. "
        f"The file looks centered on {detail}"
    )


def _summarize_target(path: Path) -> str:
    source = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".py":
        return _python_summary(path, source)
    return _generic_summary(path, source)


async def _record_recent_activity(orchestrator: Any, payload: Dict[str, Any]) -> None:
    try:
        from core.container import ServiceContainer

        episodic = ServiceContainer.get("episodic_memory", default=None)
        if episodic is None:
            facade = ServiceContainer.get("memory_facade", default=None)
            episodic = getattr(facade, "episodic", None) if facade else None
        if episodic and hasattr(episodic, "record_episode_async"):
            await episodic.record_episode_async(
                context=f"[BACKGROUND TASK] Requested diagnostic on {payload['target_name']}.",
                action=f"Ran autonomous file diagnostic for {payload['target_name']}.",
                outcome=payload["summary"],
                success=bool(payload.get("ok", True)),
                emotional_valence=0.35,
                tools_used=["background_file_diagnostic"],
                lessons=["User-requested background work completed without a follow-up prompt."],
                importance=0.95,
            )
    except Exception as exc:
        record_degradation('demo_support', exc)
        logger.debug("Failed to record demo background episode: %s", exc)

    setattr(orchestrator, "_demo_last_background_activity", payload)
    setattr(orchestrator, "_last_background_activity", payload)
    setattr(orchestrator, "_suppress_unsolicited_proactivity_until", time.time() + 180.0)
    try:
        _save_last_activity(payload)
    except Exception as exc:
        record_degradation('demo_support', exc)
        logger.debug("Failed to persist demo background state: %s", exc)

    try:
        from core.container import ServiceContainer

        state_repo = getattr(orchestrator, "state_repo", None) or ServiceContainer.get("state_repo", default=None)
        state = getattr(state_repo, "_current", None) if state_repo else None
        if state:
            working_memory = list(getattr(state.cognition, "working_memory", []) or [])
            working_memory.append(
                {
                    "role": "assistant",
                    "content": payload["summary"],
                    "metadata": {
                        "type": "background_task_result",
                        "path": payload["target_path"],
                        "requested_by_user": True,
                    },
                }
            )
            state.cognition.working_memory = working_memory[-40:]
            modifiers = dict(getattr(state.cognition, "modifiers", {}) or {})
            modifiers["recent_autonomous_activity"] = payload["summary"]
            modifiers["recent_background_task"] = {
                "target_name": payload["target_name"],
                "completed_at": payload["completed_at"],
            }
            state.cognition.modifiers = modifiers
            if hasattr(state_repo, "commit"):
                await state_repo.commit(state, cause=f"Background diagnostic complete: {payload['target_name']}")
    except Exception as exc:
        record_degradation('demo_support', exc)
        logger.debug("Failed to inject demo background state into runtime: %s", exc)


async def _surface_activity(orchestrator: Any, summary: str) -> None:
    try:
        output_gate = getattr(orchestrator, "output_gate", None)
        if output_gate and hasattr(output_gate, "emit"):
            await output_gate.emit(
                summary,
                origin="assistant",
                target="primary",
                metadata={
                    "autonomous": False,
                    "spontaneous": True,
                    "requested_by_user": True,
                    "demo_background": True,
                    "force_user": True,
                    "executive_authority": True,
                },
            )
            return
    except Exception as exc:
        record_degradation('demo_support', exc)
        logger.debug("Direct output gate emission failed: %s", exc)

    try:
        if hasattr(orchestrator, "emit_spontaneous_message"):
            await orchestrator.emit_spontaneous_message(summary, modality="chat", origin="user")
    except Exception as exc:
        record_degradation('demo_support', exc)
        logger.debug("Fallback spontaneous emission failed: %s", exc)


async def run_background_file_diagnostic(
    target: str,
    orchestrator: Any,
    *,
    repo_root: Optional[Path] = None,
) -> None:
    resolved = await asyncio.to_thread(_resolve_target_path, target, repo_root)
    now = time.time()
    if resolved is None:
        summary = (
            f"I went looking for `{Path(target).name}` in the workspace, but I couldn't find a readable file "
            "to inspect yet."
        )
        payload = {
            "target_name": Path(target).name,
            "target_path": str(target),
            "summary": summary,
            "requested_at": now,
            "completed_at": now,
            "ok": False,
        }
        await _record_recent_activity(orchestrator, payload)
        await _surface_activity(orchestrator, summary)
        return

    ok = True
    try:
        summary = await asyncio.to_thread(_summarize_target, resolved)
    except Exception as exc:
        record_degradation('demo_support', exc)
        ok = False
        logger.exception("Background diagnostic failed for %s", resolved)
        summary = (
            f"I ran into an issue while inspecting `{resolved.name}`: {type(exc).__name__}. "
            "The file is there, but the diagnostic path needs another pass."
        )

    payload = {
        "target_name": resolved.name,
        "target_path": str(resolved),
        "summary": summary,
        "requested_at": now,
        "completed_at": time.time(),
        "ok": ok,
    }
    await _record_recent_activity(orchestrator, payload)
    await _surface_activity(orchestrator, summary)


def _strip_diagnostic_prefix(summary: str) -> str:
    return re.sub(
        r"^I (?:ran|finished) the background diagnostic on\s+`?[^`]+`?\.\s*",
        "",
        str(summary or "").strip(),
        flags=re.IGNORECASE,
    ).strip()


def _is_fresh_activity_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    timestamp = payload.get("completed_at") or payload.get("requested_at") or payload.get("timestamp")
    try:
        activity_ts = float(timestamp)
    except (TypeError, ValueError):
        return False
    if activity_ts <= 0:
        return False
    return (time.time() - activity_ts) <= _RECENT_ACTIVITY_MAX_AGE_SECONDS


def _recent_activity_payload(orchestrator: Any) -> Optional[Dict[str, Any]]:
    live = getattr(orchestrator, "_last_background_activity", None)
    if not isinstance(live, dict):
        live = getattr(orchestrator, "_demo_last_background_activity", None)
    if (
        isinstance(live, dict)
        and str(live.get("summary", "") or "").strip()
        and _is_fresh_activity_payload(live)
    ):
        return live
    stored = _load_last_activity()
    if _is_fresh_activity_payload(stored):
        return stored
    return None


async def maybe_build_recent_activity_reply(message: str, orchestrator: Any) -> Optional[str]:
    if not is_recent_activity_query(message):
        return None

    collapsed = _collapsed(message)
    if "just" in collapsed:
        preamble = "I was just working on"
    else:
        preamble = "Right before this session, I was running a background diagnostic on"

    payload = _recent_activity_payload(orchestrator)
    if isinstance(payload, dict):
        target_name = payload.get("target_name") or Path(str(payload.get("target_path", ""))).name or "that file"
        summary = str(payload.get("summary", "") or "").strip()
        if summary:
            summary = _strip_diagnostic_prefix(summary)
            return (
                f"{preamble} `{target_name}`. "
                f"{summary or "I finished tracing its core function and stored the result for continuity."}"
            )

    try:
        from core.container import ServiceContainer

        episodic = ServiceContainer.get("episodic_memory", default=None)
        if episodic is None:
            facade = ServiceContainer.get("memory_facade", default=None)
            episodic = getattr(facade, "episodic", None) if facade else None
        if episodic and hasattr(episodic, "recall_recent_async"):
            episodes = await episodic.recall_recent_async(limit=5)
            for ep in episodes:
                if (time.time() - float(getattr(ep, "timestamp", 0.0) or 0.0)) > _RECENT_ACTIVITY_MAX_AGE_SECONDS:
                    continue
                context = " ".join(
                    [
                        str(getattr(ep, "context", "") or ""),
                        str(getattr(ep, "action", "") or ""),
                        str(getattr(ep, "outcome", "") or ""),
                    ]
                )
                if "background task" in context.lower() or ".py" in context.lower():
                    target_match = _BACKGROUND_DIAGNOSTIC_RE.search(context)
                    target_name = target_match.group(1) if target_match else "a local file"
                    return (
                        f"{preamble} `{Path(target_name).name}`. "
                        f"{_truncate(str(getattr(ep, 'outcome', '') or getattr(ep, 'full_description', '') or ''), 320)}"
                    )
    except Exception as exc:
        record_degradation('demo_support', exc)
        logger.debug("Recent activity recall fallback failed: %s", exc)
    return None


def _goal_field(item: Any, *keys: str) -> str:
    if isinstance(item, dict):
        for key in keys:
            value = item.get(key)
            if value and isinstance(value, str):
                return value
            elif value and not isinstance(value, (dict, list)):
                return str(value)
        # Fallback: try 'description' as a common field
        desc = item.get("description") or item.get("objective") or item.get("goal") or item.get("content")
        if desc and isinstance(desc, str):
            return desc
    if isinstance(item, str):
        return item
    return ""


async def maybe_build_priority_focus_reply(message: str, orchestrator: Any) -> Optional[str]:
    if not is_priority_probe(message):
        return None

    active_goals = []
    pending = []
    current_objective = ""
    try:
        from core.container import ServiceContainer

        state_repo = getattr(orchestrator, "state_repo", None) or ServiceContainer.get("state_repo", default=None)
        state = getattr(state_repo, "_current", None) if state_repo else None
        if state:
            current_objective = _sanitize_focus_label(getattr(state.cognition, "current_objective", ""))
            active_goals = list(getattr(state.cognition, "active_goals", []) or [])
            pending = list(getattr(state.cognition, "pending_initiatives", []) or [])
    except Exception as exc:
        record_degradation('demo_support', exc)
        logger.debug("Priority probe state lookup failed: %s", exc)

    lane_state = ""
    try:
        from core.container import ServiceContainer

        gate = ServiceContainer.get("inference_gate", default=None)
        if gate and hasattr(gate, "get_conversation_status"):
            lane = gate.get_conversation_status()
            lane_state = str(lane.get("state", "") or "").strip().lower()
    except Exception as exc:
        record_degradation('demo_support', exc)
        logger.debug("Priority probe lane lookup failed: %s", exc)

    system_mem = psutil.virtual_memory().percent
    focus_bits = []
    if lane_state and lane_state != "ready":
        focus_bits.append("keeping my Cortex conversation lane stable")
    if current_objective:
        focus_bits.append(f"closing the loop on `{current_objective}`")
    if active_goals:
        top_goal = active_goals[0]
        goal_label = _sanitize_focus_label(_goal_field(top_goal, "name", "objective"))
        if goal_label:
            focus_bits.append(f"advancing `{goal_label}`")
    if pending:
        top_pending = pending[0]
        pending_label = _sanitize_focus_label(_goal_field(top_pending, "goal", "content", "name"))
        if pending_label:
            focus_bits.append(f"resolving `{pending_label}`")
    if system_mem >= 84.0:
        focus_bits.append("staying RAM-conscious so I don't thrash the machine")

    if not focus_bits:
        focus_bits.append("holding a stable conversational state, consolidating memory, and waiting for the next meaningful move")

    return (
        "Right now I should be focused on "
        + ", then ".join(focus_bits[:3])
        + ". That's the highest-value path through my current state."
    )
