"""Improve-own-code skill (recursive self-improvement, verified + enacted).

Aura improves one function in her OWN source: researches how to do it better,
generates a fix with the un-steered code model, VERIFIES it passes behavioral
checks the current function fails (and breaks none it passed), then enacts the
change in the real file. Narrow-waisted and honest — only verified improvements
are applied.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT


class ImproveOwnCodeInput(BaseModel):
    target_file: str = Field(..., description="Path to the source file containing the function.")
    func_name: str = Field(..., description="Name of the function to improve.")
    goal: str = Field(..., description="What to improve, in plain language.")
    checks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Behavioral checks: each {'args': [...], 'expected': <value>}.",
    )
    max_iters: int = Field(3, description="Max research/generate/verify iterations.")
    enact: bool = Field(True, description="Apply the verified fix to the file.")


class ImproveOwnCodeSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill here
    #: returns `ok`, and a schema claiming to be complete would be wrong
    #: for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "improve_own_code"
    description = (
        "Recursively improve one function in Aura's own source code: research it, generate a "
        "fix, VERIFY it passes behavioral checks the current code fails (regressing none), and "
        "enact the change in the real file. Only verified improvements are applied."
    )
    input_model = ImproveOwnCodeInput
    timeout_seconds = 600.0
    metabolic_cost = 3
    effect_scope = "read_write_artifacts"
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ImproveOwnCodeInput(**params)
        elif not isinstance(params, ImproveOwnCodeInput):
            params = ImproveOwnCodeInput.model_validate(params)

        from core.capabilities.self_code_improver import improve_function

        result = await improve_function(
            target_file=params.target_file,
            func_name=params.func_name,
            goal=params.goal,
            checks=params.checks,
            max_iters=max(1, min(int(params.max_iters or 3), 5)),
            enact=bool(params.enact),
        )
        return {
            "ok": bool(result.ok),
            "skill": self.name,
            "func": result.func_name,
            "enacted": result.enacted,
            "result": result.to_dict(),
            "summary": (
                f"Improved {result.func_name}: passed {result.improved_passed}/{result.total_checks} "
                f"checks (original passed {result.original_passed}/{result.total_checks}); "
                f"enacted={result.enacted} in {result.iterations} iteration(s). Lesson retained."
                if result.ok
                else f"Could not verify an improvement to {result.func_name}: {result.error}"
            ),
        }


__all__ = ["ImproveOwnCodeInput", "ImproveOwnCodeSkill"]
