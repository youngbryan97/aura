"""Her tools for a goal she set herself (core/agency/her_goal_at_work.py): read her source, search it, change a draft
of it, check the draft, and offer it to Bryan.

Each acts in her workspace for the goal: a git worktree of her source on a branch of its own. A draft changes nothing
she runs on until Bryan takes it. A path outside the workspace, inside its .git, or naming keys or secrets is refused.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

#: Names a draft may not read or write, whatever directory they are in.
_SECRET = re.compile(r"(^|/)(\.env[^/]*|.*\.pem|.*\.key|id_rsa[^/]*|id_ed25519[^/]*|\.netrc|credentials[^/]*)$", re.I)
#: The most lines one read returns, and the most matches one search returns: what fits one tool result
#: (core/brain/llm/mlx_client.py truncates a result at 4,000 characters).
READ_LINES, FOUND_LINES = 120, 60
#: How long one check may take, in seconds: a focused pytest file runs in well under this on her machine; the smoke
#: suite of 164 tests took 47 s on 8 November 2026.
CHECK_SECONDS = 300.0


class Refused(ValueError):
    """A path or a request her workspace does not allow."""


def _workspace(context: dict[str, Any] | None) -> Path:
    from core.agency.her_goal_at_work import workspaces_dir

    raw = str((context or {}).get("her_goal_workspace") or "")
    if not raw:
        raise Refused("these tools work on a goal of hers, and no goal was given")
    path = Path(raw).resolve()
    if workspaces_dir().resolve() not in path.parents or not (path / ".git").exists():
        raise Refused("that is not one of her workspaces")
    return path


def _inside(workspace: Path, relative: str) -> Path:
    name = str(relative or "").strip().lstrip("/")
    if not name:
        raise Refused("no path was given")
    path = (workspace / name).resolve()
    if workspace != path and workspace not in path.parents:
        raise Refused(f"{name} is outside her workspace")
    rel = path.relative_to(workspace).as_posix()
    if rel == ".git" or rel.startswith(".git/") or _SECRET.search(rel):
        raise Refused(f"{name} is not hers to read or change")
    return path


async def _run(*args: str, cwd: Path, timeout: float, env: dict[str, str] | None = None) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(*args, cwd=str(cwd), env=env, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return 124, f"stopped after {int(timeout)} s"
    return int(process.returncode or 0), out.decode("utf-8", errors="replace")


def _goal_of(context: dict[str, Any] | None) -> str:
    return str((context or {}).get("her_goal") or "")


def _params(model: type[BaseModel], params: Any) -> Any:
    if isinstance(params, model):
        return params
    return model.model_validate(params if isinstance(params, dict) else dict(params or {}))


class ReadSourceInput(BaseModel):
    path: str = Field(..., description="File path, from the top of her source, e.g. core/memory/cold_store.py.")
    from_line: int = Field(1, description="First line to read, counting from 1.")


class ReadSourceSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "read_source"
    description = "Read up to 120 lines of a file in her source, numbered, from a given line."
    input_model = ReadSourceInput
    effect_scope = "read_only"
    timeout_seconds = 30.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            asked = _params(ReadSourceInput, params)
            path = _inside(_workspace(context), asked.path)
            lines = (await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")).splitlines()
        except (Refused, OSError) as why:
            return {"ok": False, "error": str(why)}
        start = max(1, int(asked.from_line or 1))
        shown = lines[start - 1:start - 1 + READ_LINES]
        text = "\n".join(f"{start + offset}: {line}" for offset, line in enumerate(shown))
        more = len(lines) - (start - 1 + len(shown))
        return {"ok": True, "path": asked.path, "lines": len(lines), "text": text,
                "summary": f"{asked.path} has {len(lines)} lines; {more} more after these" if more > 0 else
                f"{asked.path} has {len(lines)} lines"}


class FindInSourceInput(BaseModel):
    pattern: str = Field(..., description="A regular expression to look for.")
    where: str = Field(".", description="A directory or file in her source to look in, e.g. core/memory.")


class FindInSourceSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "find_in_source"
    description = "Find lines in her source matching a regular expression; up to 60, as path:line: text."
    input_model = FindInSourceInput
    effect_scope = "read_only"
    timeout_seconds = 60.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            asked = _params(FindInSourceInput, params)
            workspace = _workspace(context)
            where = _inside(workspace, asked.where or ".") if (asked.where or ".") != "." else workspace
        except Refused as why:
            return {"ok": False, "error": str(why)}
        code, out = await _run("git", "-c", "core.fsmonitor=false", "grep", "-n", "-I", "-E", asked.pattern, "--",
                               str(where.relative_to(workspace)) if where != workspace else ".",
                               cwd=workspace, timeout=60.0)
        if code not in (0, 1):
            return {"ok": False, "error": out.strip()[:500]}
        found = [line[:200] for line in out.splitlines() if not _SECRET.search(line.split(":", 1)[0])]
        return {"ok": True, "matches": len(found), "text": "\n".join(found[:FOUND_LINES]),
                "summary": f"{len(found)} line(s) match" + (f"; the first {FOUND_LINES} shown" if len(found) > FOUND_LINES else "")}


class EditDraftInput(BaseModel):
    path: str = Field(..., description="File path, from the top of her source.")
    old_text: str = Field(..., description="Text in the file to replace, exactly as it is there and found once. "
                                           "Empty to make a new file.")
    new_text: str = Field(..., description="What goes in its place, or the new file's whole text.")


class EditDraftSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "edit_draft"
    description = "Change her draft: replace text found once in a file with new text, or make a new file."
    input_model = EditDraftInput
    effect_scope = "read_write_artifacts"
    timeout_seconds = 30.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from core.agency.her_goal_at_work import load, save
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        try:
            asked = _params(EditDraftInput, params)
            path = _inside(_workspace(context), asked.path)
            exists = path.exists()
            if not asked.old_text:
                if exists:
                    raise Refused(f"{asked.path} exists; give the text in it to replace")
                text = asked.new_text
            else:
                if not exists:
                    raise Refused(f"{asked.path} does not exist")
                before = await asyncio.to_thread(path.read_text, encoding="utf-8")
                found = before.count(asked.old_text)
                if found != 1:
                    raise Refused(f"the text to replace is in {asked.path} {found} times, not once")
                text = before.replace(asked.old_text, asked.new_text, 1)
        except (Refused, OSError, UnicodeDecodeError) as why:
            return {"ok": False, "error": str(why)}
        gateway = get_file_write_gateway()
        with local_internal_governed_scope("her_draft.edit", domain="state_mutation", receipt_prefix="her-draft",
                                           constraints={"artifact": "aura.her_draft.v1", "operation": "edit"}):
            await gateway.ensure_directory_async(path.parent, source="her_draft")
            await gateway.write_text_async(path, text, source="her_draft")
        record = load(_goal_of(context))
        if record is not None:
            record.last_edit_at = time.time()
            await save(record)
        return {"ok": True, "path": asked.path, "made": not exists,
                "summary": f"{'made' if not exists else 'changed'} {asked.path} ({len(text.splitlines())} lines)"}


class CheckDraftInput(BaseModel):
    tests: list[str] = Field(default_factory=list, description="Test files to run besides any her draft changed, "
                                                                "e.g. tests/test_cold_store.py.")


class CheckDraftSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "check_draft"
    description = ("Check her draft: compile and lint the Python files it changed, and run its changed test files "
                   "and the ones named.")
    input_model = CheckDraftInput
    effect_scope = "sandboxed_compute"
    timeout_seconds = 420.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from core.agency.her_goal_at_work import load, save

        try:
            asked = _params(CheckDraftInput, params)
            workspace = _workspace(context)
            named = [_inside(workspace, test).relative_to(workspace).as_posix() for test in asked.tests]
        except Refused as why:
            return {"ok": False, "error": str(why)}
        record = load(_goal_of(context))
        base = (record.base if record is not None else "") or "HEAD"
        _, changed_out = await _run("git", "-c", "core.fsmonitor=false", "diff", "--name-only", base, cwd=workspace,
                                    timeout=60.0)
        _, new_out = await _run("git", "-c", "core.fsmonitor=false", "ls-files", "--others", "--exclude-standard",
                                cwd=workspace, timeout=60.0)
        changed = sorted({line.strip() for line in (changed_out + "\n" + new_out).splitlines() if line.strip()})
        python = [name for name in changed if name.endswith(".py") and (workspace / name).exists()]
        tests = sorted({*named, *(name for name in python if Path(name).name.startswith("test_"))})
        env = {**os.environ, "AURA_LOG_DIR": tempfile.mkdtemp(prefix="her-draft-logs-"),
               "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        report: dict[str, Any] = {"changed": changed, "tests": tests}
        passed = True
        if python:
            code, out = await _run(sys.executable, "-m", "py_compile", *python, cwd=workspace, timeout=120.0, env=env)
            report["compile"] = "ok" if code == 0 else out.strip()[-1500:]
            passed &= code == 0
            code, out = await _run(sys.executable, "-m", "ruff", "check", "--select", "E9,F", "--output-format",
                                   "concise", *python, cwd=workspace, timeout=120.0, env=env)
            report["lint"] = "ok" if code == 0 else out.strip()[-1500:]
            passed &= code == 0
        if tests:
            code, out = await _run("nice", "-n", "10", sys.executable, "-m", "pytest", "-q", "-x", "-p",
                                   "no:cacheprovider", *tests, cwd=workspace, timeout=CHECK_SECONDS, env=env)
            report["tests_output"] = out.strip()[-2500:]
            passed &= code == 0
        report["passed"] = bool(passed and (python or tests))
        if record is not None:
            record.last_check = {"passed": report["passed"], "at": time.time(), "tests": tests, "changed": changed}
            await save(record)
        said = ("passes" if report["passed"] else "does not pass") if (python or tests) else "changes no Python file"
        return {"ok": True, **report, "summary": f"her draft {said}: {len(changed)} file(s) changed, "
                                                  f"{len(tests)} test file(s) run"}


class OfferDraftInput(BaseModel):
    summary: str = Field(..., description="What the draft does and why, in her words, for Bryan.")


class OfferDraftSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "offer_draft"
    description = ("Offer her checked draft to Bryan. Only his approval puts it into the source she runs on; "
                   "until he answers, the goal waits.")
    input_model = OfferDraftInput
    effect_scope = "read_write_artifacts"
    timeout_seconds = 60.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from core.agency.her_goal_at_work import OFFERED, load, save

        try:
            asked = _params(OfferDraftInput, params)
            workspace = _workspace(context)
        except Refused as why:
            return {"ok": False, "error": str(why)}
        record = load(_goal_of(context))
        if record is None:
            return {"ok": False, "error": "no record of this goal"}
        check = record.last_check or {}
        if not check.get("passed") or float(check.get("at") or 0) < float(record.last_edit_at or 0):
            return {"ok": False, "error": "her draft has not passed a check since it last changed"}
        await _run("git", "add", "-A", cwd=workspace, timeout=60.0)
        message = f"{asked.summary.strip()}\n\nDrafted by Aura toward her goal: {record.goal}"
        code, out = await _run("git", "-c", "core.fsmonitor=false", "-c", "user.name=Aura", "-c",
                               "user.email=aura@localhost", "commit", "-q", "-m", message, cwd=workspace, timeout=60.0)
        if code != 0:
            return {"ok": False, "error": out.strip()[:500]}
        _, head = await _run("git", "rev-parse", "HEAD", cwd=workspace, timeout=30.0)
        record.status = OFFERED
        record.offer = {"summary": asked.summary.strip(), "commit": head.strip(), "at": time.time(),
                        "files": check.get("changed") or []}
        await save(record)
        import logging

        logging.getLogger("Aura.HerGoalAtWork").warning(
            "🛠️ Aura offers a draft toward '%s' (%s): %s. Answer with tools/her_goals.py approve|decline %s",
            record.goal[:80], head.strip()[:9], asked.summary.strip()[:200], record.slug)
        return {"ok": True, "commit": head.strip(), "summary": "offered to Bryan; the goal waits for his answer"}


#: Her tools, in the order a goal is worked.
HER_TOOLS = (ReadSourceSkill, FindInSourceSkill, EditDraftSkill, CheckDraftSkill, OfferDraftSkill)


def tool_definitions() -> dict[str, dict[str, Any]]:
    """Her tools as her model is offered them: each one's name, what it does, and what it takes."""

    def plain(schema: dict[str, Any]) -> dict[str, Any]:
        properties = {name: {key: value for key, value in spec.items() if key != "title"}
                      for name, spec in schema.get("properties", {}).items()}
        return {"type": "object", "properties": properties, "required": list(schema.get("required", []))}

    return {skill.name: {"description": skill.description, "parameters": plain(skill.input_model.model_json_schema())}
            for skill in HER_TOOLS}


__all__ = ["CheckDraftSkill", "EditDraftSkill", "FindInSourceSkill", "HER_TOOLS", "OfferDraftSkill",
           "ReadSourceSkill", "Refused", "tool_definitions"]
