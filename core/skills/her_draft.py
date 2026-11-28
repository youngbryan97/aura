"""Her tools for a goal she set herself (core/agency/her_goal_at_work.py): read what has gone wrong in her running, read
her source, search it, look up what she needs to know in her library or on the web, change a draft of it, check the
draft, and offer it; or end a goal that needs no change, saying what came of it.

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


def _where_she_reads(context: dict[str, Any] | None) -> Path:
    """Her goal's workspace, or with no goal the source she runs on: reading it changes nothing. LIVE 18 November 2026
    these were offered on a question about her own code and refused it for having no goal."""
    if (context or {}).get("her_goal_workspace"):
        return _workspace(context)
    from core.agency.her_goal_at_work import source_root

    return Path(source_root()).resolve()


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
            workspace = _where_she_reads(context)
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
        except IsADirectoryError:
            # A folder she names is read as a person reads one: what is in it. LIVE 21 November 2026 her goal pass asked
            # to read her workspace's own folder, and it came back as a skill error.
            return await _what_is_in(workspace, path, asked.path)
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


async def _what_is_in(workspace: Path, folder: Path, asked: str) -> dict[str, Any]:
    """What her source keeps in ``folder``, one level down, a folder's name ending in "/": as many as one result holds."""
    where = folder.relative_to(workspace).as_posix() if folder != workspace else "."
    code, out = await _run("git", "-c", "core.fsmonitor=false", "ls-files", "--", where, cwd=workspace, timeout=30.0)
    if code != 0:
        return {"ok": False, "error": out.strip()[:500]}
    inside = {}
    for line in out.splitlines():
        rest = line if where == "." else line[len(where) + 1:]
        first, _, deeper = rest.partition("/")
        if first and not _SECRET.search(first):
            inside[first + ("/" if deeper else "")] = True
    entries = sorted(inside)
    summary = f"{asked} is a folder of {len(entries)} entries"
    shown = _what_fits(entries, {"ok": True, "path": asked, "folder": True, "entries": len(entries), "summary": summary})
    return {"ok": True, "path": asked, "folder": True, "entries": len(entries), "text": "\n".join(shown),
            "summary": summary + (f"; the first {len(shown)} shown" if len(shown) < len(entries) else "")}


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
            workspace = _where_she_reads(context)
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


class MyGoalInput(BaseModel):
    pass


class MyGoalSkill(BaseSkill):
    """The goal a pass of hers works toward, as she set it, and where it stands: what her pass reads first, as a call of
    hers. Put to her model as a person's message instead, a goal she had set herself read as somebody's request, and
    a bare one ("Running a self-integrity scan") as somebody who said almost nothing: LIVE 27 November 2026, after 354
    passes, her model wrote "Your messages have been empty the whole time" and "you haven't asked me anything"."""

    result_schema = THE_SHARED_RESULT
    name = "my_goal"
    description = "The goal she set herself that this pass works toward, when she set it, and where it stands."
    input_model = MyGoalInput
    effect_scope = "read_only"
    timeout_seconds = 30.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from core.agency.her_goal_at_work import load, what_she_set_out_to_do

        record = await asyncio.to_thread(load, _goal_of(context), require_readable=True)
        if record is None:
            return {"ok": False, "error": "no record of this goal"}
        return what_she_set_out_to_do(record)


class WhatWentWrongInput(BaseModel):
    subsystem: str = Field("", description="Only what this subsystem recorded, or the start of its name, e.g. memory.")


#: What a record of her runtime at these severities says is not a fault: expected waits and notes (CLAUDE.md, "log at
#: info and only record a degradation when the condition is persistent").
_NOT_A_FAULT = frozenset({"debug", "info"})


def faults_recorded(subsystem: str = "") -> tuple[list[str], int]:
    """What has gone wrong in her own running since it started, from the faults her runtime recorded: each kind once, the
    most frequent first, as a line; and how many there were in all."""
    from core.runtime.errors import get_degradation_tracker

    kinds: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in get_degradation_tracker().recent(limit=10_000):
        if str(record.severity) in _NOT_A_FAULT or not str(record.subsystem).startswith(subsystem.strip()):
            continue
        call = record.failed_call
        where = (f"{call.file}:{call.line} in {call.qualname}" if getattr(call, "file", "")
                 else ".".join(record.raised_in) if record.raised_in else "where it was raised was not kept")
        kind = kinds.setdefault((record.subsystem, record.error_type, where), {"times": 0})
        kind.update(times=kind["times"] + 1, severity=record.severity, error=record.error_message[:200],
                    action=record.action[:160], last=record.timestamp)
    ranked = sorted(kinds.items(), key=lambda item: (-item[1]["times"], -item[1]["last"]))
    rows = [f"{kind['times']}x {kind['severity']} {subsystem}: {error_type} at {where}: {kind['error']}"
            + (f" (then: {kind['action']})" if kind["action"] else "")
            for (subsystem, error_type, where), kind in ranked]
    return rows, sum(kind["times"] for kind in kinds.values())


class WhatWentWrongSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "what_went_wrong"
    description = ("What has gone wrong in her own running since it started, from the faults her runtime recorded: "
                   "each kind once, how many times, where in her source it was raised, the error, and what was done "
                   "about it, the most frequent first.")
    input_model = WhatWentWrongInput
    effect_scope = "read_only"
    timeout_seconds = 30.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Her runtime's own record, read in her own process: LIVE 21 November 2026 her goal "Find something productive
        to work on" had made 2,022 calls over 290 passes with no edit, searching her source for TODO and FIXME, while
        her causal world model's every save was being refused in the same process, sixty times that afternoon."""
        try:
            asked = _params(WhatWentWrongInput, params)
        except Refused as why:
            return {"ok": False, "error": str(why)}
        rows, times = faults_recorded(asked.subsystem)
        summary = f"{len(rows)} kind(s) of fault recorded, {times} in all"
        shown = _what_fits(rows, {"ok": True, "kinds": len(rows), "summary": summary})
        return {"ok": True, "kinds": len(rows), "text": "\n".join(shown),
                "summary": summary + (f"; the first {len(shown)} shown" if len(shown) < len(rows) else "")}


class HowIAmRunningInput(BaseModel):
    """Nothing to say: what is read is how she is running now."""


class HowIAmRunningSkill(BaseSkill):
    result_schema = THE_SHARED_RESULT
    name = "how_i_am_running"
    description = ("How her own running stands now, as her runtime measures it: the memory her own process and her workers "
                   "hold, each sample her memory governor kept over its last stretch and the change across it, what each "
                   "worker holds and runs now, and what her runtime's health contract says of itself.")
    input_model = HowIAmRunningInput
    effect_scope = "read_only"
    timeout_seconds = 30.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Read in her own process, where her runtime measures itself. LIVE 27 November 2026 her goal of steadying herself
        read her lane control's source for a leak for eleven passes and stopped short of a change each time: "I have no
        live RSS or process probe", and she asked Bryan for a sample."""
        from core.container import ServiceContainer

        governor = ServiceContainer.get("memory_governor", default=None)
        measured = governor.what_it_measured() if hasattr(governor, "what_it_measured") else None
        rows: list[str] = []
        if measured and measured["samples"]:
            first, last = measured["samples"][0], measured["samples"][-1]
            said = (f"over the last {(last[0] - first[0]) / 60.0:.0f} min her own process went from {first[1]:.0f} to "
                    f"{last[1]:.0f} MB and her workers from {first[2]:.0f} to {last[2]:.0f} MB; her memory stands "
                    f"{measured['state']}")
            rows += [f"worker {pid} holds {mb:.0f} MB" + (f": {name}" if name else "") for mb, pid, name in measured["workers_now"]]
            rows += [f"{time.strftime('%H:%M:%S', time.localtime(at))}: her process {core:.0f} MB, her workers {runtime:.0f} MB"
                     for at, core, runtime in reversed(measured["samples"])]
        else:
            said = "her memory governor has measured nothing yet" if governor is not None else "her memory governor is not running"
        try:
            from core.runtime.health_contract import runtime_health_report

            report = await asyncio.to_thread(runtime_health_report)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as why:
            report = {}
            rows.append(f"her health contract could not be read: {why}")
        if report:
            said += f"; her runtime reads itself {report.get('status')}"
            failing = {name: probe for name, probe in dict(report.get("required_probes") or {}).items()
                       if isinstance(probe, dict) and not probe.get("ok", True)}
            rows[:0] = [f"probe {name} failing: {json.dumps(probe, default=str)[:200]}" for name, probe in failing.items()]
            for name, value in dict(report.get("integrity") or {}).items():
                if value not in (None, [], {}, ""):
                    rows.insert(len(failing), f"integrity {name}: {json.dumps(value, default=str)[:240]}")
        shown = _what_fits(rows, {"ok": True, "summary": said})
        return {"ok": True, "text": "\n".join(shown),
                "summary": said + (f"; the first {len(shown)} of {len(rows)} lines shown" if len(shown) < len(rows) else "")}


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
        with local_internal_governed_scope("her_draft.edit", domain="state_mutation", receipt_prefix="her-draft",
                                           constraints={"artifact": "aura.her_draft.v1", "operation": "edit"}):
            await gateway.write_text_async(path, text, source="her_draft")
        record = await asyncio.to_thread(load, _goal_of(context), require_readable=True)
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
        record = await asyncio.to_thread(load, _goal_of(context), require_readable=True)
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
        record = await asyncio.to_thread(load, _goal_of(context), require_readable=True)
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
            "Aura offers a draft toward '%s' (%s): %s. Answer with tools/her_goals.py approve|decline %s",
            record.goal[:80], head.strip()[:9], asked.summary.strip()[:200], record.slug)
        # Her own yes runs the gates when this pass ends (her_goal_at_work.the_gates_her_yes_asks), outside this
        # call: they take minutes, and a tool call has one.
        return {"ok": True, "commit": head.strip(), "taken": False,
                "summary": "offered; when this pass ends her own yes runs the gates, and takes it if they pass and it "
                           "does not change her own governance"}


class NothingToChangeInput(BaseModel):
    what_came_of_it: str = Field(..., description="What she found or did toward the goal, in her words, and why her "
                                                  "source needs no change for it.")


class NothingToChangeSkill(BaseSkill):
    """A goal of hers that needs no change to her source, ended: a check that found nothing to mend, a question she has
    answered. LIVE 27 November 2026 "Running a self-integrity scan" had been worked 352 passes and 1,054 calls; her model
    said there was nothing left to verify, and a goal ended only by offering a draft, so it was worked again each time
    her source moved."""

    result_schema = THE_SHARED_RESULT
    name = "nothing_to_change"
    description = ("End this goal with no change to her source, saying what came of it. Refused while her workspace "
                   "holds a change: that is checked and offered, or taken back, instead.")
    input_model = NothingToChangeInput
    effect_scope = "read_write_artifacts"
    timeout_seconds = 60.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from core.agency.her_goal_at_work import DONE, WORKING, load, save

        try:
            asked = _params(NothingToChangeInput, params)
            workspace = _workspace(context)
        except Refused as why:
            return {"ok": False, "error": str(why)}
        record = await asyncio.to_thread(load, _goal_of(context), require_readable=True)
        if record is None:
            return {"ok": False, "error": "no record of this goal"}
        if record.status != WORKING:
            return {"ok": False, "error": f"this goal is {record.status}"}
        _, changed = await _run("git", "-c", "core.fsmonitor=false", "status", "--porcelain", cwd=workspace,
                                timeout=60.0)
        _, ahead = await _run("git", "-c", "core.fsmonitor=false", "rev-list", "--count", f"{record.base}..HEAD",
                              cwd=workspace, timeout=60.0) if record.base else (0, "0")
        if changed.strip() or ahead.strip() not in ("", "0"):
            return {"ok": False, "error": "her workspace holds a change toward this goal: check and offer it, or take "
                                          "it back, rather than end the goal without one"}
        record.status = DONE
        record.came_of_it = {"words": asked.what_came_of_it.strip(), "at": time.time()}
        await save(record)
        return {"ok": True, "summary": "the goal is done, with nothing in her source changed"}


#: Her tools, in the order a goal is worked.
class LookUpInput(BaseModel):
    query: str = Field(..., description="What to look up: an error, a function or library, a protocol, an idea.")


class LookUpSkill(BaseSkill):
    """Bryan, 26 November 2026: "Does Aura ever consult either the cortex or brainstem's llms ... Or the internet? Or
    internal Wikipedia? Not just on whole problem sets but individual pieces of code and problems and issues." Her goal
    passes read her source and her faults and nothing else; her mend of _coerce_argv narrowed a refusal whose purpose a
    reference would have given her."""

    result_schema = THE_SHARED_RESULT
    name = "look_up"
    description = ("Look something up in her reference library on this machine (Wikipedia, textbooks and manuals): "
                   "what an error, a function, a protocol or an idea is, each passage with where it comes from.")
    input_model = LookUpInput
    effect_scope = "read_only"
    timeout_seconds = 30.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from core.knowledge.local_corpus import get_local_corpus_store

        try:
            asked = _params(LookUpInput, params)
        except Refused as why:
            return {"ok": False, "error": str(why)}
        query = " ".join(str(asked.query or "").split())
        if not query:
            return {"ok": False, "error": "nothing was given to look up"}
        hits = await asyncio.to_thread(get_local_corpus_store().search, query, 5)
        rows = [f"{hit.title} ({hit.source}): {' '.join(str(hit.snippet).split())}" for hit in hits]
        summary = f"{len(rows)} passage(s) in her library" if rows else "nothing in her library on this"
        shown = _what_fits(rows, {"ok": True, "found": len(rows), "summary": summary})
        return {"ok": True, "found": len(rows), "text": "\n".join(shown), "summary": summary}


class SearchTheWebInput(BaseModel):
    query: str = Field(..., description="What to search the web for: documentation, an error message, how a problem "
                                        "was solved elsewhere.")


class SearchTheWebSkill(BaseSkill):
    """For what her library does not hold, where she is online and her Will lets her reach out (Bryan: "Internet if she
    has it of course. Not requirement for functionality"). The call is governed as this tool, under her standing grant
    for her own goal work; the search it makes is her research pipeline's (core/skills/web_search.py)."""

    result_schema = THE_SHARED_RESULT
    name = "search_the_web"
    description = ("Search the web for what her library does not hold: documentation, an error message, how a "
                   "problem was solved elsewhere, each result with its address.")
    input_model = SearchTheWebInput
    effect_scope = "external_io"
    timeout_seconds = 180.0

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from core.skills.web_search import EnhancedWebSearchSkill

        try:
            asked = _params(SearchTheWebInput, params)
        except Refused as why:
            return {"ok": False, "error": str(why)}
        query = " ".join(str(asked.query or "").split())
        if not query:
            return {"ok": False, "error": "nothing was given to search for"}
        found = await EnhancedWebSearchSkill().execute({"query": query, "num_results": 5, "retain": False},
                                                     {"origin": "her_goal_background"})
        results = [item for item in (found or {}).get("results") or [] if isinstance(item, dict)]
        rows = [f"{item.get('title') or ''} <{item.get('url') or item.get('link') or ''}>: "
                f"{' '.join(str(item.get('snippet') or item.get('content') or '').split())}" for item in results]
        if not rows:
            why = str((found or {}).get("error") or (found or {}).get("summary") or "no results")
            return {"ok": False, "error": f"the web gave nothing back: {why}"[:400]}
        summary = f"{len(rows)} result(s) from the web"
        shown = _what_fits(rows, {"ok": True, "found": len(rows), "summary": summary})
        return {"ok": True, "found": len(rows), "text": "\n".join(shown), "summary": summary}


HER_TOOLS = (WhatWentWrongSkill, HowIAmRunningSkill, ReadSourceSkill, FindInSourceSkill, LookUpSkill, SearchTheWebSkill,
             NothingToChangeSkill, EditDraftSkill, CheckDraftSkill, OfferDraftSkill)


def tool_definitions() -> dict[str, dict[str, Any]]:
    """Her tools as her model is offered them: each one's name, what it does, and what it takes."""

    def plain(schema: dict[str, Any]) -> dict[str, Any]:
        properties = {name: {key: value for key, value in spec.items() if key != "title"}
                      for name, spec in schema.get("properties", {}).items()}
        return {"type": "object", "properties": properties, "required": list(schema.get("required", []))}

    return {skill.name: {"description": skill.description, "parameters": plain(skill.input_model.model_json_schema())}
            for skill in HER_TOOLS}


__all__ = ["CheckDraftSkill", "EditDraftSkill", "FindInSourceSkill", "HER_TOOLS", "LookUpSkill", "MyGoalSkill",
           "NothingToChangeSkill", "OfferDraftSkill",
           "ReadSourceSkill", "Refused", "SearchTheWebSkill", "WhatWentWrongSkill", "tool_definitions"]
