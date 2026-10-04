"""Repairing a program someone else wrote: run it, read it, keep only the edits that make it behave.

The work is done by core/self_modification/repairing_by_behaviour.py. This is
the way in from a request: it finds the file the request names, says what she
is doing as she does it, and answers with what was wrong, what she changed and
what the program does now. Nothing here asks a model what the fix is.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

__all__ = ["RepairAProgramInput", "RepairAProgramSkill", "the_file_named_in"]

#: A file a request names: an absolute or home-relative path to something with a suffix.
_A_PATH = re.compile(r"(?:file://)?((?:~|/)[^\s'\"<>]+\.[A-Za-z0-9]{1,6})")

#: What the repair can read today: a page with its script in it, or a script.
_READABLE = (".html", ".htm", ".js", ".mjs")


def the_file_named_in(text: str) -> str:
    """The first path to a file in ``text``, without its file:// prefix or trailing punctuation."""
    found = _A_PATH.search(str(text or ""))
    return found.group(1).rstrip(".,;:)") if found else ""


class RepairAProgramInput(BaseModel):
    path: str = Field("", description="The program's file. Read from the request when not given.")


class RepairAProgramSkill(BaseSkill):
    #: What a caller gets back. The shared part only, as for every skill here.
    result_schema = THE_SHARED_RESULT

    name = "repair_a_program"
    description = (
        "Mend a program somebody else wrote that does not work as it should: run it and watch "
        "what it does, find the places in its code that look wrong, try their edits on copies, "
        "and keep only the ones that make it behave right. Keeps the original beside it."
    )
    input_model = RepairAProgramInput
    timeout_seconds = 1200.0
    metabolic_cost = 2
    effect_scope = "read_write_artifacts"
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = RepairAProgramInput(**{k: v for k, v in params.items() if k == "path"})
        elif not isinstance(params, RepairAProgramInput):
            params = RepairAProgramInput.model_validate(params)
        asked = " ".join(str((context or {}).get(key) or "") for key in ("objective", "message", "user_message", "goal"))
        named = params.path or the_file_named_in(asked)
        if not named:
            return {"ok": False, "skill": self.name, "error": "no file was named to repair",
                    "summary": "Which file is the program? I need its path to run it and read it."}
        path = await asyncio.to_thread(lambda: Path(named).expanduser().resolve())
        exists = await asyncio.to_thread(path.is_file)
        if not exists:
            return {"ok": False, "skill": self.name, "error": f"no file at {path}",
                    "summary": f"There is no file at {path}."}
        if path.suffix.lower() not in _READABLE:
            return {"ok": False, "skill": self.name, "error": f"cannot run a {path.suffix} file yet",
                    "summary": f"I can run and repair a web page or a script; {path.name} is neither."}

        from core.self_modification.repairing_by_behaviour import repair_by_behaviour

        repair = await repair_by_behaviour(path, say=_said)
        summary = repair.said[-1] if repair.said else "I could not run it."
        played: dict[str, Any] = {}
        if _asks_to_play(asked) and repair.kept and not repair.after:
            played = await _play_it(path.as_uri(), asked)
            summary += " " + _how_the_play_went(played)
        return {
            "ok": bool(repair.kept) and not repair.after,
            "skill": self.name,
            "path": str(path),
            "before": repair.before,
            "after": repair.after,
            "changes": repair.kept,
            "knowledge": repair.knowledge,
            "backup": repair.backup,
            "seconds": repair.seconds,
            "address": path.as_uri(),
            "played": played,
            "summary": summary,
        }


def _asks_to_play(request: str) -> bool:
    from core.language.how_a_game_ended import asks_to_win

    return asks_to_win(request) or bool(re.search(r"\bplay\b", request, re.IGNORECASE))


async def _play_it(address: str, request: str) -> dict[str, Any]:
    """Open the mended program in a browser window the person can watch, and play it as asked."""
    from core.capabilities.phantom_browser import PhantomBrowser
    from core.skills.sovereign_browser_drawing import played_on_the_drawing

    _said("Now I'll play it, in a window you can watch.")
    browser = PhantomBrowser(visible=True, browser_type="chromium", principal="owner")
    try:
        if not await browser.ensure_ready():
            return {"error": "her browser could not be opened"}
        await browser.page.goto(address, wait_until="load")
        await asyncio.sleep(0.5)
        return await played_on_the_drawing(browser, request, {"url": address})
    finally:
        await browser.close()


def _how_the_play_went(played: dict[str, Any]) -> str:
    if played.get("error"):
        return f"I could not play it afterwards: {played['error']}."
    runs = list(played.get("runs") or [])
    if played.get("won"):
        return f"Then I played it and won, on game {runs.index('won') + 1}." if "won" in runs else "Then I played it and won."
    if runs:
        return f"Then I played {len(runs)} game(s) and did not win one: " + ", ".join(runs) + "."
    return "Then I could not get a game going."


def _said(line: str) -> None:
    """Each step said where the person can see it, as she works."""
    from core.skills.screen_pursuit import _tell

    _tell(line)
