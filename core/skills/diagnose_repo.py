"""Diagnose a project by running it.

The runtime finds the project's test runner, runs it, and reads the failure
the runner reports, along with the source around the failing line and what the
project says it is supposed to do. The language model's part is to explain that
finding; it is not asked to guess at one.
"""
from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill


class DiagnoseRepoInput(BaseModel):
    path: str = Field(..., description="Directory of the project to run.")
    command: str = Field(
        "", description="How to run its tests; empty discovers the runner."
    )


class DiagnoseRepoSkill(BaseSkill):
    name = "diagnose_repo"
    description = (
        "Run a project's own tests and report what actually failed: the failing test, the "
        "assertion the runner printed, the file and line, the source around it, what that "
        "line calls, and what the project's README says it should do. Use for any question "
        "about why code in a directory is failing."
    )
    input_model = DiagnoseRepoInput

    # Running someone's tests reads their tree and writes only pytest's own
    # scratch; it is not a desktop action and not an external effect.
    timeout_seconds = 180.0
    metabolic_cost = 1
    effect_scope = "sandboxed_compute"
    requires_approval = False

    @staticmethod
    def available_here() -> bool:
        return True

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = DiagnoseRepoInput(**params)
        elif not isinstance(params, DiagnoseRepoInput):
            params = DiagnoseRepoInput.model_validate(params)

        from core.diagnosis.repository import describe_diagnosis, diagnose_repository

        argv = params.command.split() if params.command.strip() else None
        diagnosis = await asyncio.to_thread(
            diagnose_repository, params.path, argv=argv
        )
        described = describe_diagnosis(diagnosis)
        if diagnosis.error:
            return {
                "ok": False,
                "skill": self.name,
                "error": diagnosis.error,
                "path": diagnosis.root,
                "summary": described,
            }
        return {
            "ok": True,
            "skill": self.name,
            "path": diagnosis.root,
            "ran": diagnosis.ran,
            "passed": diagnosis.passed,
            "failed": diagnosis.failed,
            "failures": [
                {
                    "test": item.test,
                    "assertion": item.assertion,
                    "file": item.file,
                    "line": item.line,
                }
                for item in diagnosis.failures
            ],
            "source": diagnosis.source,
            "stated_intent": diagnosis.stated_intent,
            "summary": described,
        }


__all__ = ["DiagnoseRepoInput", "DiagnoseRepoSkill"]
