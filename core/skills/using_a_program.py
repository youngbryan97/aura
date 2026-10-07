"""Using a program she built to do something with it, the way a person uses one: its own controls, nothing else.

A program is proved by being used for what it is for. "Write a letter in it
and export it to my Desktop" is done here by her browser pursuit on the
program's own page (core/skills/sovereign_browser.py): she sees its menus and
toolbar and document as anyone would, types, formats, and exports, deciding
each step from what the page shows. Whatever the program saves or exports is
kept in the folder the person named (core/capabilities/where_downloads_go.py),
and the answer says which files are there, read back from disk.

Nothing here knows what any program is or what it is used for.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

__all__ = ["UseAProgramInput", "UseAProgramSkill", "the_folder_named_in", "use_what_she_built", "what_to_do_with_it"]

#: Keys that belong to the turn that asked, not to the pursuit it delegates.
_THE_TURN_S_OWN = (
    "action_expectation", "expectation", "acceptance_criteria", "criteria", "required_evidence",
    "evidence_required", "required_evidence_present", "semantic_predicates", "user_visible_effect",
    "visible_effect", "repair_hint", "rollback_hint", "allow_partial", "effect_scope", "risk_level",
)

#: Ways a request asks for what was built to be used: "prove it works by
#: writing...", "then use it to...", "and then write ... in it".
_USE_IT = (
    re.compile(r"\b(?:prove|show|demonstrate)\s+(?:that\s+|to\s+me\s+that\s+)?it\s+works\s+by\s+(.+)", re.IGNORECASE | re.S),
    re.compile(r"\buse\s+(?:it|this|that|the\s+[\w\s-]{1,40}?)\s+to\s+(.+)", re.IGNORECASE | re.S),
    re.compile(r"\b(?:then|and\s+then|after\s+that)\s+((?:write|type|draw|make|create|export|save|compose|draft|fill)\b.+)", re.IGNORECASE | re.S),
)


#: Folders a person names by their usual name: "to my Desktop".
_USUAL = {"desktop": "Desktop", "documents": "Documents", "downloads": "Downloads", "home folder": "", "home directory": ""}


def the_folder_named_in(asked: str) -> Path | None:
    """The folder a request names for a file: a path that is a folder, or one of the person's usual folders."""
    from core.language.named_paths import first_existing_path

    found = first_existing_path(asked or "")
    if found is not None and Path(found).is_dir():
        return Path(found)
    lowered = str(asked or "").lower()
    for said, folder in _USUAL.items():
        if re.search(rf"\b(?:to|in|into|on|onto)\s+(?:my|the)\s+{said}\b", lowered):
            return Path.home() / folder if folder else Path.home()
    return None


def what_to_do_with_it(asked: str) -> str:
    """What the request asks to be done with the program once it is built, in its own words, or ''."""
    for pattern in _USE_IT:
        found = pattern.search(asked or "")
        if found:
            return " ".join(found.group(1).split()).rstrip(" .!")
    return ""


async def use_what_she_built(page: Path, task: str, asked: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Do ``task`` with the program at ``page`` through her browser pursuit, keeping what it exports where ``asked`` says."""
    from core.capabilities.where_downloads_go import _default as downloads_go_to_default
    from core.capabilities.where_downloads_go import downloads_go_to, kept_downloads
    from core.container import ServiceContainer

    engine = ServiceContainer.get("capability_engine", default=None)
    if engine is None or not hasattr(engine, "execute"):
        return {"ok": False, "error": "no capability engine to drive her browser", "files": []}
    folder = the_folder_named_in(asked)
    # Writing something and keeping it is done with the program's own controls by code
    # (core/rebuilding/using_it_by_its_controls.py); her browser pursuit is for the rest.
    from core.rebuilding.her_model import ask_her_model, patiently
    from core.rebuilding.using_it_by_its_controls import write_it_in
    from core.skills.screen_pursuit import _tell

    used = await write_it_in(page, task, folder or downloads_go_to_default(), patiently(ask_her_model), visible=True, tell=_tell)
    if used is not None:
        shown = await _shown(used.files[0]) if used.ok and used.files else False
        return {"ok": used.ok, "files": [str(f) for f in used.files], "report": {"summary": used.summary(), "error": used.why_not},
                "folder": str(folder or ""), "shown": shown}
    since = len(kept_downloads())
    child = {k: v for k, v in dict(context or {}).items() if k not in _THE_TURN_S_OWN}
    goal = f"{task}. Do it in the program open on this page, with its own controls, as a person using it would."
    uri = await asyncio.to_thread(lambda: page.resolve().as_uri())
    if folder is not None:
        with downloads_go_to(folder):
            report = await engine.execute("sovereign_browser", {"mode": "pursue", "url": uri, "goal": goal}, context=child)
    else:
        report = await engine.execute("sovereign_browser", {"mode": "pursue", "url": uri, "goal": goal}, context=child)
    files = [f for f in kept_downloads(since) if await asyncio.to_thread(f.exists)]
    report = report if isinstance(report, dict) else {}
    return {"ok": bool(files) or bool(report.get("ok")), "files": [str(f) for f in files], "report": report, "folder": str(folder or "")}


async def _shown(file: str | Path) -> bool:
    """The file it saved, opened where the person can see it, in whatever their system opens that kind of file with."""
    import platform
    import subprocess

    if platform.system() != "Darwin":
        return False
    try:
        done = await asyncio.to_thread(subprocess.run, ["open", str(file)], check=False, timeout=30, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def how_the_use_went(used: dict[str, Any]) -> str:
    files = list(used.get("files") or [])
    report = used.get("report") or {}
    if files and used.get("ok") is False:
        # Saved, and not what was written: said so. LIVE 2026-10-06 a letter saved as "Der Future Self" was reported as done.
        why = str(report.get("error") or "").strip()
        return ("Then I used it as asked, and it saved " + ", ".join(files) + ", but what it saved is not what I wrote"
                + (f": {why[:240]}." if why else "."))
    if files:
        return ("Then I used it as asked, and it saved " + ", ".join(files)
                + (", which I opened so you can see it." if used.get("shown") else "."))
    why = str(used.get("error") or report.get("error") or report.get("summary") or "").strip()
    return "Then I tried to use it as asked, and nothing was saved" + (f": {why[:240]}" if why else ".")


class UseAProgramInput(BaseModel):
    task: str = Field("", description="What to do with the program. Read from the request when not given.")
    path: str = Field("", description="The build's folder. The one the request names, or her newest, when not given.")


class UseAProgramSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT

    name = "use_a_program"
    description = (
        "Use a program she built or rebuilt, with its own controls, to make something in it and export it: "
        "a document written and exported, a picture drawn and saved. The exported file is kept in the folder "
        "named, read back from disk."
    )
    input_model = UseAProgramInput
    timeout_seconds = 3600.0
    metabolic_cost = 2
    effect_scope = "read_write_artifacts"
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = UseAProgramInput(**{k: v for k, v in params.items() if k in ("task", "path")})
        elif not isinstance(params, UseAProgramInput):
            params = UseAProgramInput.model_validate(params)
        from core.conversation.session_scope import the_persons_own_words, the_request_in
        from core.rebuilding.rebuilding_a_program import the_build_meant
        from core.skills.changing_a_program import where_builds_are

        asked = the_persons_own_words(params.task) or the_request_in(context)
        folder = await asyncio.to_thread(the_build_meant, f"{params.path} {asked}", where_builds_are())
        if folder is None:
            return {"ok": False, "skill": self.name, "error": "no build of hers was found to use",
                    "summary": "I have not built a program yet that I could use for that."}
        used = await use_what_she_built(folder / "index.html", what_to_do_with_it(asked) or asked, asked, context)
        return {"ok": bool(used.get("files")), "skill": self.name, "path": str(folder / "index.html"),
                "files": used.get("files"), "summary": how_the_use_went(used)}
