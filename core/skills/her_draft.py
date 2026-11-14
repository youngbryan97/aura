"""Her tools for a goal she set herself (core/agency/her_goal_at_work.py): read her source, search it, change a draft
of it, check the draft, and offer it to Bryan.

Each acts in her workspace for the goal: a git worktree of her source on a branch of its own. A draft changes nothing
she runs on until Bryan takes it. A path outside the workspace, inside its .git, or naming keys or secrets is refused.
"""
from __future__ import annotations

import asyncio
import json
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
#: What one read or one search may give back: the room of one tool result, less what the gateway's receipt on it takes
#: (214 characters measured on 10 November 2026) and as much again for whatever is added to it later. At a fixed 120
#: lines a read of real source ran to 5,000 to 7,000 characters and came back cut and escaped twice; 80 of one goal's
#: 161 calls did.
def _room_for_what_she_reads() -> int:
    from core.brain.llm.tool_result_room import TOOL_RESULT_CHARS

    return TOOL_RESULT_CHARS - 2 * 214


def _what_fits(rows: list[str], fields: dict[str, Any]) -> list[str]:
    """The first rows that fit, with the result's other fields, in what one read may give back."""
    room, shown = _room_for_what_she_reads(), []
    for row in rows:
        if shown and len(json.dumps({**fields, "text": "\n".join([*shown, row])})) > room:
            break
        shown.append(row)
    return shown
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


#: The governed domain of each kind of effect her tools have; a call with none of these only reads.
_EFFECTS = {"commit": "file_write", "compute": "tool_execution"}


async def _run(*args: str, cwd: Path, timeout: float, env: dict[str, str] | None = None,
               effect: str = "") -> tuple[int, str]:
    """A program run for her tools through the subprocess gateway: read-only, or an effect inside a governed scope.

    An effect is a commit in her workspace or running code (her tests, a compile). Its output and errors come back
    together, as they did when these ran outside the gateway (Master Todo Q06).
    """
    import subprocess

    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    gateway, source = get_subprocess_gateway(), f"her_draft.{effect or 'read'}"
    try:
        if not effect:
            done = await gateway.run_async(list(args), cwd=cwd, env=env, timeout=timeout, read_only=True,
                                           source=source, accelerator_capability="none")
        else:
            with local_internal_governed_scope(source, domain=_EFFECTS[effect], receipt_prefix="her-draft",
                                               constraints={"artifact": "aura.her_draft.v1", "operation": effect}):
                done = await gateway.run_async(list(args), cwd=cwd, env=env, timeout=timeout, source=source,
                                               accelerator_capability="none")
    except subprocess.TimeoutExpired:
        return 124, f"stopped after {int(timeout)} s"
    return int(done.returncode or 0), str(done.stdout or "") + str(done.stderr or "")


def _goal_of(context: dict[str, Any] | None) -> str:
    return str((context or {}).get("her_goal") or "")


def _params(model: type[BaseModel], params: Any) -> Any:
    if isinstance(params, model):
        return params
    return model.model_validate(params if isinstance(params, dict) else dict(params or {}))


def _named(workspace: Path, name: str, most: int = 10) -> list[str]:
    """The paths in her workspace of files called ``name``, nearest the top first."""
    found = []
    for path in workspace.rglob(name):
        rel = path.relative_to(workspace)
        if path.is_file() and not any(part.startswith(".") or part == "node_modules" for part in rel.parts):
            found.append(rel.as_posix())
    return sorted(found, key=lambda rel: (rel.count("/"), rel))[:most]


class ReadSourceInput(BaseModel):
    path: str = Field(..., description="File path, from the top of her source, e.g. core/memory/cold_store.py.")
    from_line: int = Field(1, description="First line to read, counting from 1.")


class ReadSourceSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "read_source"
    description = ("Read a file in her source, numbered, from a given line: as many lines as one result holds, "
                   "and the line to read on from.")
    input_model = ReadSourceInput
    effect_scope = "read_only"
    timeout_seconds = 30.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            asked = _params(ReadSourceInput, params)
            workspace = _workspace(context)
            path = _inside(workspace, asked.path)
            lines = (await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")).splitlines()
        except FileNotFoundError:
            # A file she names that is not there is an answer, as it is to a person: where files of that name are.
            # LIVE 10 November 2026 her goal pass asked for core/governance/permission_model.py, which is in
            # core/capabilities, and it came back as a skill error and a surprise of 1.00.
            found = await asyncio.to_thread(_named, workspace, Path(asked.path).name)
            where = f"; a file of that name is at {', '.join(found)}" if found else "; no file of that name is in it"
            return {"ok": True, "path": asked.path, "exists": False, "same_name": found,
                    "summary": f"{asked.path} is not in her source{where}"}
        except (Refused, OSError) as why:
            return {"ok": False, "error": str(why)}
        start = max(1, int(asked.from_line or 1))
        numbered = [f"{start + offset}: {line}" for offset, line in enumerate(lines[start - 1:])]
        summary = f"{asked.path} has {len(lines)} lines; read on from line {len(lines)}"
        shown = _what_fits(numbered, {"ok": True, "path": asked.path, "lines": len(lines), "summary": summary})
        after = start + len(shown)
        return {"ok": True, "path": asked.path, "lines": len(lines), "text": "\n".join(shown),
                "summary": f"{asked.path} has {len(lines)} lines; read on from line {after}" if after <= len(lines)
                else f"{asked.path} has {len(lines)} lines, and these are the last"}


class FindInSourceInput(BaseModel):
    pattern: str = Field(..., description="A regular expression to look for.")
    where: str = Field(".", description="A directory or file in her source to look in, e.g. core/memory.")


class FindInSourceSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "find_in_source"
    description = ("Find lines in her source matching a regular expression, as path:line: text: as many as one "
                   "result holds, and how many there were.")
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
        # Perl-compatible, the dialect she writes: "(?i)quantum error correction|qec\b" is fatal to -E (13 November 2026).
        code, out = await _run("git", "-c", "core.fsmonitor=false", "grep", "-n", "-I", "-P", asked.pattern, "--",
                               str(where.relative_to(workspace)) if where != workspace else ".",
                               cwd=workspace, timeout=60.0)
        if code not in (0, 1):
            return {"ok": False, "error": out.strip()[:500]}
        found = [line[:200] for line in out.splitlines() if not _SECRET.search(line.split(":", 1)[0])]
        summary = f"{len(found)} line(s) match; the first {len(found)} shown"
        shown = _what_fits(found, {"ok": True, "matches": len(found), "summary": summary})
        return {"ok": True, "matches": len(found), "text": "\n".join(shown),
                "summary": f"{len(found)} line(s) match" + (f"; the first {len(shown)} shown" if len(shown) < len(found)
                                                             else "")}


class EditDraftInput(BaseModel):
    path: str = Field(..., description="File path, from the top of her source.")
    # Optional and None-tolerant: the capability engine drops an empty string before a skill sees it, so "" (a new
    # file) arrived as nothing and failed validation (day run 4, 9 November 2026).
    old_text: str | None = Field("", description="Text in the file to replace, exactly as it is there and found once. "
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
            code, out = await _run(sys.executable, "-m", "py_compile", *python, cwd=workspace, timeout=120.0, env=env,
                                   effect="compute")
            report["compile"] = "ok" if code == 0 else out.strip()[-1500:]
            passed &= code == 0
            code, out = await _run(sys.executable, "-m", "ruff", "check", "--select", "E9,F", "--output-format",
                                   "concise", *python, cwd=workspace, timeout=120.0, env=env)
            report["lint"] = "ok" if code == 0 else out.strip()[-1500:]
            passed &= code == 0
        if tests:
            code, out = await _run("nice", "-n", "10", sys.executable, "-m", "pytest", "-q", "-x", "-p",
                                   "no:cacheprovider", *tests, cwd=workspace, timeout=CHECK_SECONDS, env=env,
                                   effect="compute")
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
    description = ("Offer her checked draft. When the pass ends it goes into the source she runs on if it passes "
                   "the tests of every module it changed, the smoke suite, lint, layering and the writing gate, unless "
                   "it changes her own governance or a run is measuring her; then it waits for Bryan.")
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
        await _run("git", "add", "-A", cwd=workspace, timeout=60.0, effect="commit")
        message = f"{asked.summary.strip()}\n\nDrafted by Aura toward her goal: {record.goal}"
        code, out = await _run("git", "-c", "core.fsmonitor=false", "-c", "user.name=Aura", "-c",
                               "user.email=aura@localhost", "commit", "-q", "-m", message, cwd=workspace, timeout=60.0,
                               effect="commit")
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
        # Her own yes runs the gates when this pass ends (her_goal_at_work.the_gates_her_yes_asks), outside this
        # call: they take minutes, and a tool call has one.
        return {"ok": True, "commit": head.strip(), "taken": False,
                "summary": "offered; when this pass ends her own yes runs the gates, and takes it if they pass and it "
                           "does not change her own governance"}


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
