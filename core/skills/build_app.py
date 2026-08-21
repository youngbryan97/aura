"""Build-an-app skill.

Live capability surface: Aura builds a real, runnable single-file web app from a
natural spec, validates that it actually works, and writes it to disk so it can
be opened and used. General across app kinds (games, tools, toys).
"""
from __future__ import annotations

from typing import Any

from pathlib import Path

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill


class BuildAppInput(BaseModel):
    spec: str = Field(..., description="What app to build, e.g. 'a playable checkers game'.")
    # Empty means the runtime's own place for built apps, the same way
    # max_tokens=0 means the code lane's own ceiling. Naming the default here
    # made it a relative path that then nested under itself.
    out_dir: str = Field("", description="Subdirectory for the app file; empty uses the standard one.")
    # 0 means "whatever the code lane allows". A number here that the lane
    # refuses is a skill that cannot run: 9000 against a policy ceiling of
    # 2048 raised local_code_model_max_tokens_out_of_policy on every call.
    max_tokens: int = Field(0, description="Generation budget; 0 uses the code lane's own ceiling.")
    max_iters: int = Field(3, description="Max research/build/test iterations (1-6).")


class BuildAppSkill(BaseSkill):
    name = "build_app"
    description = (
        "Build a real, runnable single-file web app (game, tool, or toy) from a natural "
        "description: research it, build it, functionally TEST that it actually works, "
        "persist until it does, and retain the lesson. Writes it to disk to open and use."
    )
    input_model = BuildAppInput
    # The verified loop researches + generates + tests + repairs across several
    # iterations on the 32B, so it needs a generous budget.
    timeout_seconds = 1500.0
    metabolic_cost = 3
    effect_scope = "read_write_artifacts"
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = BuildAppInput(**params)
        elif not isinstance(params, BuildAppInput):
            params = BuildAppInput.model_validate(params)

        # Self-taught, test-driven build: recall prior lessons, research the task
        # (concepts + reference code), generate, FUNCTIONALLY test that it works,
        # feed the exact failure back, persist, and retain the general lesson.
        from core.capabilities.self_taught_builder import build_app_verified

        try:
            from core.brain.llm.local_code_model import max_code_tokens

            ceiling = int(max_code_tokens())
        except (ImportError, AttributeError, TypeError, ValueError):
            ceiling = 2048
        requested_tokens = int(params.max_tokens or 0)
        budget = min(requested_tokens, ceiling) if requested_tokens > 0 else ceiling

        # Where the file goes, confined.
        #
        # LIVE, 2026-08-21: PermissionError: [Errno 13] Permission denied:
        # '/Users/user'. The model filled out_dir with a home directory that
        # does not exist on this machine, and the path was used as given. A
        # path from a model-authored payload is confined to a root the
        # runtime chose, which is the rule core/runtime/payload_values.py
        # already states for every other skill.
        from core.runtime.payload_values import payload_path

        root = (Path(__file__).resolve().parents[2] / "artifacts" / "live_apps").resolve()
        out_dir = payload_path(
            {"out_dir": params.out_dir}, "out_dir", root=root, default=root
        )

        result = await build_app_verified(
            params.spec,
            out_dir=str(out_dir or root),
            max_tokens=budget,
            max_iters=max(1, min(int(params.max_iters or 3), 6)),
        )
        payload = result.to_dict()
        if not result.ok:
            # The result carries `error` and `status` and neither was passed
            # on, so a build that ran for seventy-three seconds and failed
            # came back as "build_app reported failure without a cause".
            # Whatever went wrong, the person and the retry both need it.
            reason = (
                str(result.error or "").strip()
                or str(result.status or "").strip()
                or "the build finished without producing a working file"
            )
            return {
                "ok": False,
                "skill": self.name,
                "error": reason,
                "spec": result.spec,
                "path": result.path,
                "result": payload,
                "summary": (
                    f"Could not build '{result.spec}': {reason} "
                    f"(after {result.iterations} iteration(s))."
                ),
            }
        return {
            "ok": True,
            "skill": self.name,
            "spec": result.spec,
            "path": result.path,
            "playable": result.playable,
            "result": payload,
            "summary": (
                f"Built '{result.spec}' -> {result.path}: "
                f"playable={result.playable} after {result.iterations} test-driven iteration(s); "
                f"researched {len(result.research_used)} source(s), recalled "
                f"{len(result.recalled_lessons)} prior lesson(s). Lesson retained."
            ),
        }


__all__ = ["BuildAppInput", "BuildAppSkill"]
