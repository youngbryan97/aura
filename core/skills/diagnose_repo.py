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

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.skills.base_skill import BaseSkill


class DiagnoseRepoInput(BaseModel):
    path: str = Field(..., description="Directory of the project to run.")
    command: str = Field(
        "", description="How to run its tests; empty discovers the runner."
    )


class DiagnoseRepoSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "diagnose_repo"
    # What the skill DOES, which the router matches a request against. It had
    # said "why code in a directory is failing", and the engine grew past that:
    # when the suite passes it runs the project the way a person runs it, under
    # a tracer, and reports state that outlives the call that touches it.
    #
    # LIVE, 2026-08-28: "Something's off in <path> ... No error, nothing
    # crashes, the tests such as they are pass. But when I build two invoices
    # in a row the second one comes out wrong." — a request this skill answers
    # completely, and the only description of it on file began by requiring a
    # failure. The router offered code_repl and file_operation instead, and the
    # turn ended with no answer. The engine had invoice.py:4, the mutable
    # default, the surrounding source and the remedy.
    description = (
        "Find out what a project actually does by running it. Reports the failing test, the "
        "assertion the runner printed, the file and line, the source around it, what that "
        "line calls, and what the project's README says it should do. When nothing fails, "
        "runs the project the way a person runs it and reports what survives a call: state "
        "left behind by one call that the next one starts from. Use to investigate, debug or "
        "explain any question about why code in a directory misbehaves, has a bug, gives the "
        "wrong answer, or is not doing what it should — including when there is no error and "
        "the tests pass."
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
        await _remember_the_failure(diagnosis, params.path)
        # What was observed is the answer.
        #
        # LIVE, 2026-08-22: this skill ran in 428ms and returned the failing
        # test, the assertion, the line and the project's stated invariant —
        # and the turn served "I couldn't get to an answer I'd stand behind",
        # because the model's draft of that finding was rejected for missing
        # the very numbers the finding contains. The runtime had the answer
        # and was asking the model to reproduce it.
        try:
            from core.conversation.session_scope import record_solved_answer

            if described and not diagnosis.error:
                record_solved_answer("repo_diagnosis", described)
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
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


async def _remember_the_failure(diagnosis: Any, path: str) -> None:
    """Put what failed into the store the pattern analyser reads.

    A failure seen once in one project and again in another is what
    ErrorPatternAnalyzer exists to notice, and it could not see any of them:
    the store only accepted exceptions, which is right for her own faults and
    cannot describe somebody else's test run.

    This is the difference between a tool that answers one question and a
    capability that gets better at answering the next one.
    """
    if getattr(diagnosis, "error", "") or not getattr(diagnosis, "failures", ()):
        return
    store = None
    try:
        # The live system if the runtime has one, so the finding lands in the
        # same store its analyser is already reading, rather than beside it.
        from core.container import ServiceContainer

        live = ServiceContainer.get("error_intelligence", default=None)
        store = getattr(live, "logger_system", None)
    except (ImportError, AttributeError, RuntimeError, LookupError):
        store = None
    if store is None:
        try:
            from core.self_modification.error_intelligence import StructuredErrorLogger

            store = StructuredErrorLogger()
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return
    first = diagnosis.failures[0]
    assertion = str(getattr(first, "assertion", "") or "")
    kind = assertion.split(":", 1)[0].strip() if ":" in assertion else "AssertionError"
    try:
        await store.log_observed_failure(
            error_type=kind or "AssertionError",
            error_message=assertion or str(getattr(first, "message", "")),
            context={
                "project": str(path),
                "ran": str(getattr(diagnosis, "ran", "")),
                "test": str(getattr(first, "test", "")),
                "passed": int(getattr(diagnosis, "passed", 0)),
                "failed": int(getattr(diagnosis, "failed", 0)),
                "stated_intent": str(getattr(diagnosis, "stated_intent", ""))[:400],
                "reason": "a project's own tests were run and reported a failure",
                "classification": "observed_project_failure",
            },
            skill_name="diagnose_repo",
            goal=f"why {getattr(first, 'test', 'a test')} fails",
            file_path=str(getattr(first, "file", "")) or None,
            line_number=int(getattr(first, "line", 0)) or None,
            detail=str(getattr(diagnosis, "source", ""))[:2000],
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        # Bookkeeping must never break a diagnosis that already succeeded.
        from core.runtime.errors import record_degradation

        record_degradation(
            "diagnosis.remember_failure",
            RuntimeError("could not record the observed failure"),
            severity="debug",
            action="returned the diagnosis without recording the failure",
            enforce_failure_policy=False,
        )


__all__ = ["DiagnoseRepoInput", "DiagnoseRepoSkill"]
