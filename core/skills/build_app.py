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
    name = "build_app"
    description = (
        "Build a real, runnable single-file web app (tool, tracker, or toy) from a natural "
        "description: plan it, compile it, check every control and view is wired and that its "
        "logic matches the runtime's own model of it, then write it to disk to open and use."
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

    # Planning is one short call; the build after it is measured in
    # milliseconds. The old ceiling was 1500 seconds for a loop that
    # generated, tested and repaired whole documents.
    timeout_seconds = 180.0
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


__all__ = ["BuildAppInput", "BuildAppSkill"]
