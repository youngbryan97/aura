"""Build-an-app skill.

Aura builds a runnable single-file web app from a description, checks it
works, and writes it where it can be opened.

The language model plans; the runtime builds. That division is the whole
point of this skill: a plan is typed data a few hundred tokens long, and
everything after it — repairing the plan, compiling the page, running its
state machine against the runtime's own model of the same operations — is
work the system does and can check. Nothing here executes model output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.skills.base_skill import BaseSkill


class BuildAppInput(BaseModel):
    spec: str = Field(..., description="What app to build, e.g. 'a tally counter'.")
    # Empty means the runtime's own place for built apps. Naming the default
    # here made it a relative path that then nested under itself.
    out_dir: str = Field("", description="Subdirectory for the app file; empty uses the standard one.")
    # Kept for callers that still pass them. The build is deterministic after
    # the plan, so neither a token budget nor an iteration count changes it.
    max_tokens: int = Field(0, description="Unused: the plan has its own small budget.")
    max_iters: int = Field(1, description="Unused: the build is checked, not retried blindly.")


class BuildAppSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "build_app"
    description = (
        "Build a real, runnable program (an app, a tool, a tracker, a game) to a person's specification: "
        "say what it must do as features, write the checks a person would make before any code, write it "
        "part by part keeping each part only when its checks hold, measure and mend how finished it is, "
        "then open it to use."
    )
    input_model = BuildAppInput

    @staticmethod
    def available_here() -> bool:
        """Whether this host can run a build.

        It always can. The old builder asked a 21.5GB code model for a
        finished document, was refused beside a 25.3GB resident cortex, and
        spent forty to seventy seconds failing on every request. Building is
        now the runtime's own work, and the only model call is a plan that
        fits any lane.
        """
        return True

    #: Written part by part by her own model and checked as it goes
    #: (core/rebuilding); a small program takes minutes, a large one an hour.
    timeout_seconds = 10800.0
    metabolic_cost = 1
    effect_scope = "read_write_artifacts"
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = BuildAppInput(**params)
        elif not isinstance(params, BuildAppInput):
            params = BuildAppInput.model_validate(params)

        # Where the file goes, confined.
        #
        # LIVE, 2026-08-21: PermissionError: [Errno 13] Permission denied:
        # '/Users/user'. A model-authored payload named a home directory that
        # does not exist on this machine, and the path was used as given.
        from core.runtime.payload_values import payload_path

        root = (Path(__file__).resolve().parents[2] / "artifacts" / "live_apps").resolve()
        out_dir = payload_path(
            {"out_dir": params.out_dir}, "out_dir", root=root, default=root
        )

        from core.construction.build_app_system import build_app
        from core.conversation.session_scope import the_persons_own_words

        # Built to the specification by the same engine that rebuilds a named
        # program, checked feature by feature; the compiled plan below is what
        # is left when her model cannot say what the program is to do.
        built = await _built_to_specification(the_persons_own_words(params.spec), Path(str(out_dir or root)))
        if built is not None:
            return built
        # The requirement is what the person asked for; `spec` is the model's
        # restatement of it. Reading a requirement from a paraphrase is how a
        # six-slide request became a three-section deck reported as finished.
        result = await build_app(
            the_persons_own_words(params.spec), out_dir=str(out_dir or root)
        )
        payload = result.to_dict()
        if not result.ok:
            return {
                "ok": False,
                "skill": self.name,
                "error": "; ".join(result.problems) or "no workable plan",
                "spec": result.request,
                "path": result.path,
                "result": payload,
                "summary": result.summary(),
            }
        return {
            "ok": True,
            "skill": self.name,
            "spec": result.request,
            "path": result.path,
            "title": result.title,
            "checks": list(result.checks),
            "result": payload,
            "summary": result.summary(),
        }


async def _built_to_specification(asked: str, where: Path) -> dict[str, Any] | None:
    """The program the person specified, built and checked by core/rebuilding, or None when nothing could be said of it."""
    from core.rebuilding.her_model import ask_her_model
    from core.rebuilding.rebuilding_a_program import rebuild
    from core.skills.program_dna_reconstruct import _open_for_the_person
    from core.skills.screen_pursuit import _tell

    rebuilt = await rebuild("", ask_her_model, where, tell=_tell, asked=asked)
    if rebuilt.built is None:
        return None
    working = rebuilt.built.working()
    opened = await _open_for_the_person(rebuilt.built.path) if working else ""
    return {
        "ok": bool(working),
        "skill": "build_app",
        "spec": asked,
        "path": str(rebuilt.built.path),
        "title": rebuilt.genome.name if rebuilt.genome else "",
        "features": [o.feature.name for o in working],
        "not_working": [o.feature.name for o in rebuilt.built.outcomes if not o.kept],
        "summary": rebuilt.summary() + (f" {opened}" if opened else ""),
    }


__all__ = ["BuildAppInput", "BuildAppSkill"]
