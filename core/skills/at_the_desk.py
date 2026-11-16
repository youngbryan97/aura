"""What she can do on the person's screen while she keeps them company at the desk.

Bryan, 15 November 2026, of her in the bubble: "she should also be able to use the computer from that mode. she's
allowed to physically act and react to things that are going on. highlight, show me something, literally anything OS
related. full companion. and yes im giving you permission to work with whatever permissions she needs".

LIVE 16 November 2026, 01:04, in the bubble: "I can't highlight or interact with any UI elements on your screen - I
don't have tools for that." Her turn at the desk was given no tools, so she said she had none, and the pointing she did
was chosen for her afterwards by matching her words against the screen. These are offered to that turn as data
through her model's own chat template (core/brain/her_unprompted_turn.py), the way her goal pass is offered its tools
(core/skills/her_draft.py). What she calls runs under the standing grant ``aura.desk-company``
(core/executive/standing_authority.py), from her desk turn's origin only. Pointing came first; showing a page and
showing a file are the "show me something" of what he asked.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

__all__ = ["DESK_TOOLS", "PointAtOnScreenSkill", "RevealFileSkill", "ShowPageSkill", "tool_definitions"]


class PointAtOnScreenInput(BaseModel):
    words: str = Field(..., description="Words on the person's screen to point at, as they appear there.")


class PointAtOnScreenSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "point_at_on_screen"
    description = ("Point at something on the person's screen: a ring drawn around the words for a few seconds. "
                   "Nothing is clicked or typed.")
    input_model = PointAtOnScreenInput
    # It draws on the person's screen, over whatever app is in front.
    effect_scope = "foreground_desktop_control"

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from core.perception.screen_highlight import highlight

        asked = params if isinstance(params, PointAtOnScreenInput) else PointAtOnScreenInput.model_validate(
            params if isinstance(params, dict) else dict(params or {}))
        words = asked.words.strip()
        if not words:
            return {"ok": False, "error": "nothing named to point at"}
        shown = await highlight(words, invited=True)
        if shown.shown:
            return {"ok": True, "summary": f"pointed at {shown.matched_text[:120]!r}", **shown.to_dict()}
        return {"ok": False, "error": shown.reason, **shown.to_dict()}


class ShowPageInput(BaseModel):
    url: str = Field(..., description="The http or https address of the page to show.")


class ShowPageSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "show_page"
    description = "Show the person a web page: it opens in a new tab of their browser."
    input_model = ShowPageInput
    # It opens a tab in the person's browser, which may come to the front.
    effect_scope = "foreground_desktop_control"

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from core.capabilities.browser_controller import BrowserNavigationRefused, canonical_navigable_url
        from core.runtime.service_registry import get_runtime_service

        asked = params if isinstance(params, ShowPageInput) else ShowPageInput.model_validate(
            params if isinstance(params, dict) else dict(params or {}))
        try:
            url = canonical_navigable_url(asked.url)
        except BrowserNavigationRefused as why:
            return {"ok": False, "error": f"not a page that can be shown: {why}"}
        browser = get_runtime_service("browser_controller", default=None)
        if browser is None or not hasattr(browser, "open_url"):
            return {"ok": False, "error": "no browser control in this runtime"}
        receipt = await browser.open_url(url, new_tab=True)
        if bool(getattr(receipt, "success", False)):
            return {"ok": True, "summary": f"opened {url} in a new tab", "url": url}
        return {"ok": False, "error": str(getattr(receipt, "error", "") or "the page did not open"), "url": url}


class RevealFileInput(BaseModel):
    path: str = Field(..., description="The file or folder to show, as a path on the person's Mac.")


class RevealFileSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "reveal_file"
    description = "Show the person a file or folder: Finder opens with it selected. Nothing is opened or run."
    input_model = RevealFileInput
    # It brings a Finder window forward on the person's screen.
    effect_scope = "foreground_desktop_control"

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from core.capabilities.file_broker import get_file_broker

        asked = params if isinstance(params, RevealFileInput) else RevealFileInput.model_validate(
            params if isinstance(params, dict) else dict(params or {}))
        where = asked.path.strip()
        if not where:
            return {"ok": False, "error": "nothing named to show"}
        # The broker reveals only inside the places it allows, and only by selecting it: open -R runs nothing.
        if await get_file_broker().reveal_in_finder(where):
            return {"ok": True, "summary": f"showed {where} in Finder", "path": where}
        return {"ok": False, "error": "it is not somewhere she may show, or Finder did not show it", "path": where}


#: Her tools at the desk.
DESK_TOOLS = (PointAtOnScreenSkill, ShowPageSkill, RevealFileSkill)


def tool_definitions() -> dict[str, dict[str, Any]]:
    """Her desk tools as her model is offered them: each one's name, what it does, and what it takes."""

    def plain(schema: dict[str, Any]) -> dict[str, Any]:
        properties = {name: {key: value for key, value in spec.items() if key != "title"}
                      for name, spec in schema.get("properties", {}).items()}
        return {"type": "object", "properties": properties, "required": list(schema.get("required", []))}

    return {skill.name: {"description": skill.description, "parameters": plain(skill.input_model.model_json_schema())}
            for skill in DESK_TOOLS}
