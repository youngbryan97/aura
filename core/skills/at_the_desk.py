"""What she can do on the person's screen while she keeps them company at the desk.

Bryan, 15 November 2026, of her in the bubble: "she should also be able to use the computer from that mode. she's
allowed to physically act and react to things that are going on. highlight, show me something, literally anything OS
related. full companion. and yes im giving you permission to work with whatever permissions she needs".

LIVE 16 November 2026, 01:04, in the bubble: "I can't highlight or interact with any UI elements on your screen - I
don't have tools for that." Her turn at the desk was given no tools, so she said she had none, and the pointing she did
was chosen for her afterwards by matching her words against the screen. These are offered to that turn as data
through her model's own chat template (core/brain/her_unprompted_turn.py), the way her goal pass is offered its tools
(core/skills/her_draft.py). What she calls runs under the standing grant ``aura.desk-company``
(core/executive/standing_authority.py), from her desk turn's origin only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

__all__ = ["DESK_TOOLS", "PointAtOnScreenSkill", "tool_definitions"]


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


#: Her tools at the desk.
DESK_TOOLS = (PointAtOnScreenSkill,)


def tool_definitions() -> dict[str, dict[str, Any]]:
    """Her desk tools as her model is offered them: each one's name, what it does, and what it takes."""

    def plain(schema: dict[str, Any]) -> dict[str, Any]:
        properties = {name: {key: value for key, value in spec.items() if key != "title"}
                      for name, spec in schema.get("properties", {}).items()}
        return {"type": "object", "properties": properties, "required": list(schema.get("required", []))}

    return {skill.name: {"description": skill.description, "parameters": plain(skill.input_model.model_json_schema())}
            for skill in DESK_TOOLS}
