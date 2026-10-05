"""Changing a program she built to what a person asks: a feature added, or one made to work differently.

The work is core/rebuilding/changing_what_was_built.py: the change is written
as features, checked before any code, and kept only when what it adds works
and nothing that worked before stops working. This is the way in from a
request: it finds the build meant, says what she does as she does it, and
answers with what changed.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

__all__ = ["ChangeAProgramInput", "ChangeAProgramSkill", "where_builds_are"]


def where_builds_are() -> list[Path]:
    """The folders her builds are kept in: rebuilt programs, and programs built to a specification."""
    root = Path(__file__).resolve().parents[2] / "artifacts"
    return [root / "rebuilt_programs", root / "live_apps"]


class ChangeAProgramInput(BaseModel):
    change: str = Field("", description="The change asked for. Read from the request when not given.")
    path: str = Field("", description="The build's folder. The one the request names, or her newest, when not given.")


class ChangeAProgramSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT

    name = "change_a_program"
    description = (
        "Change a program she built to do what is asked: add a feature, or make an existing one work or look "
        "differently. Writes the checks for the change before any code, changes only the parts it must, keeps the "
        "change only when it works and everything that worked before still works, and keeps the program as it was "
        "beside it."
    )
    input_model = ChangeAProgramInput
    #: A change is a few features written and checked by her own model.
    timeout_seconds = 5400.0
    metabolic_cost = 2
    effect_scope = "read_write_artifacts"
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ChangeAProgramInput(**{k: v for k, v in params.items() if k in ("change", "path")})
        elif not isinstance(params, ChangeAProgramInput):
            params = ChangeAProgramInput.model_validate(params)
        from core.conversation.session_scope import the_persons_own_words, the_request_in
        from core.rebuilding.changing_what_was_built import change_it
        from core.rebuilding.her_model import ask_her_model
        from core.rebuilding.rebuilding_a_program import the_build_meant
        from core.skills.program_dna_reconstruct import _open_for_the_person
        from core.skills.screen_pursuit import _tell

        asked = the_persons_own_words(params.change) or the_request_in(context)
        folder = await asyncio.to_thread(the_build_meant, f"{params.path} {asked}", where_builds_are())
        if folder is None:
            return {"ok": False, "skill": self.name, "error": "no build of hers was found to change",
                    "summary": "I have not built a program yet that I could change. Ask me to build or rebuild one first."}
        changed = await change_it(folder, asked, ask_her_model, tell=_tell)
        kept = [o for o in changed.outcomes if o.kept]
        opened = await _open_for_the_person(folder / "index.html") if kept else ""
        return {
            "ok": bool(kept) and not changed.why_not,
            "skill": self.name,
            "path": str(folder / "index.html"),
            "changed": [o.feature.name for o in kept],
            "not_changed": [o.feature.name for o in changed.outcomes if not o.kept],
            "summary": changed.summary() + (f" {opened}" if opened else ""),
        }
