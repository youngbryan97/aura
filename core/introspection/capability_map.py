"""core/introspection/capability_map.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Capability-aware decomposition grounding: when the user asks for real
action, hand the mind an honest map of which lanes are open so it
routes each sub-step to the cheapest granted path instead of declining
whole tasks.

Observed live (July 4): asked to write a note in Notes, create a
Desktop folder, and export there, she declined everything citing GUI
limits — but folder creation and file export are plain filesystem work
(always available), and Notes is scriptable through Automation (which
was GRANTED). Only raw mouse/keyboard control was actually blocked.

Lane order (cheapest first):
1. FILESYSTEM  — file_operation: create/read/write/export files and
   folders anywhere the user can. Needs no macOS permission.
2. SCRIPTING   — AppleScript/System Events via the Automation grant:
   Notes, Finder, Calendar, menu content.
3. GUI CONTROL — mouse/keyboard via Accessibility (+Screen for vision).
   Last resort; often the only blocked lane.

The block ends with the decomposition instruction: do every achievable
part now, and decline ONLY the residue that truly needs a missing
grant — naming the grant.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("Aura.Introspection.CapabilityMap")

_ACTION_REQUEST_RE = re.compile(
    r"(?:\b(?:create|make|write|save|export|move|copy|delete|rename|organize)\b"
    r".{0,80}\b(?:file|files|folder|folders|note|notes|document|directory|desktop|downloads)\b"
    r"|\b(?:open|use|launch|control)\b.{0,40}\b(?:app|application|notes app|finder|browser|calendar)\b"
    r"|\bon\s+my\s+(?:desktop|computer|mac|machine)\b"
    r"|\busing\s+my\s+\w+\s+app\b)",
    re.IGNORECASE,
)


def is_actionable_request(text: str | None) -> bool:
    """True when the user is asking for real action on their machine."""
    candidate = str(text or "").strip()
    if not candidate or len(candidate) > 800:
        return False
    return bool(_ACTION_REQUEST_RE.search(candidate))


def _permission_state() -> dict[str, bool]:
    """Best-effort cached permission truth; absent guard = pessimistic."""
    state = {"accessibility": False, "screen": False, "automation": False}
    try:
        from core.security.permission_guard import PermissionType, get_permission_guard

        guard = get_permission_guard()
        cached = getattr(guard, "_cache", {}) or {}
        for key, ptype in (
            ("accessibility", PermissionType.ACCESSIBILITY),
            ("screen", PermissionType.SCREEN),
            ("automation", PermissionType.AUTOMATION),
        ):
            entry = cached.get(ptype)
            if isinstance(entry, dict):
                state[key] = bool(entry.get("granted"))
    except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
        logger.debug("Permission state unavailable for capability map: %s", exc)
    return state


def _skill_available(name: str) -> bool:
    try:
        from core.container import ServiceContainer

        engine = ServiceContainer.get("capability_engine", default=None)
        if engine is None:
            return False
        active = getattr(engine, "active_skills", None)
        skills = getattr(engine, "skills", None)
        return bool(
            (active is None or name in active)
            and (skills is None or name in skills)
        )
    except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
        logger.debug(
            "the capability engine did not answer, so the skill counts as inactive (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return False


def build_capability_map_context(max_chars: int = 1800) -> str:
    """Compose the honest lane map + decomposition instruction."""
    perms = _permission_state()
    filesystem = _skill_available("file_operation")
    scripting = perms["automation"]
    gui = perms["accessibility"]

    def _mark(open_: bool) -> str:
        return "OPEN" if open_ else "CLOSED"

    lines = [
        f"- FILESYSTEM lane [{_mark(filesystem)}]: create/read/write/export "
        "files and folders (Desktop, Documents, anywhere) via file_operation. "
        "No macOS permission needed.",
        f"- SCRIPTING lane [{_mark(scripting)}]: drive Notes, Finder, and "
        "scriptable apps through AppleScript (Automation grant"
        f" {'granted' if scripting else 'NOT granted'}).",
        f"- GUI-CONTROL lane [{_mark(gui)}]: raw mouse/keyboard via "
        f"Accessibility ({'granted' if gui else 'NOT granted'})"
        + ("" if gui else " — the ONLY lane that is closed; name this grant "
           "if a step truly needs it"),
    ]
    return (
        "CAPABILITY MAP FOR THIS ACTION REQUEST (decompose the task and "
        "route every sub-step to the cheapest OPEN lane; do the achievable "
        "parts NOW; decline only the residue that genuinely requires a "
        "CLOSED lane, naming the missing grant — never decline the whole "
        "task because one lane is closed):\n" + "\n".join(lines)
    )[:max_chars]
