"""A fault from her own running, mended by her own loop (Master Todo F15).

A fault she can make again (``core/self_modification/faults_she_can_reproduce.py``) becomes a goal of hers with a
workspace of its own (``core/agency/her_goal_at_work.py``). Its test is written and run there, and only a test that
raises what she raised goes on: that is the defect repair by selection works to. Candidates come from her own Python
first (``proposals_without_a_model``, at the line the fault was raised) and from her model only when one is given; the
fault's test and the tests that name the file judge each. A candidate counts only when tests that passed before were
run on it and held, so a fault in a file no test touches is not mended this way, and the goal says so. A mend is
committed on the goal's branch with its test, as a draft of hers, and offered; her own yes then runs her gates and
takes it into the source she runs on unless it changes her governance (``take_it_herself``).
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

#: A test's verdict in pytest's ``-rA`` summary.
_RESULT = re.compile(r"^(PASSED|FAILED|ERROR|XPASS|XFAIL|SKIPPED)\s+(\S+)", re.MULTILINE)

#: As long as one run of tests is given: as long as one of her gates (her_goal_at_work.GATE_SECONDS).
_RUN_SECONDS = 2400.0


def goal_for(fault: Any) -> str:
    """The goal a fault becomes, in words her goal record keeps."""
    call = fault.call
    return (f"Mend {call.module}.{call.qualname}, which raised {call.error_type.rsplit('.', 1)[-1]} while she ran: "
            f"{call.error_message[:120]}")


async def _governed_run(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None, timeout: float,
                        effect: str) -> tuple[int, str]:
    """A program run in her workspace through the subprocess gateway, inside a governed scope of its own."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    domain = {"commit": "file_write", "compute": "tool_execution"}[effect]
    source = f"mending_a_fault_of_hers.{effect}"
    with local_internal_governed_scope(source, domain=domain, receipt_prefix="her-mend",
                                       constraints={"artifact": "aura.her_mend.v1", "operation": effect}):
        done = await get_subprocess_gateway().run_async(list(argv), cwd=cwd, env=dict(env) if env else None,
                                                        timeout=timeout, capture_output=True, source=source,
                                                        accelerator_capability="none")
    return done.returncode, ((done.stdout or "") + (done.stderr or ""))


class InHerWorkspace:
    """Runs tests in her workspace on a candidate's sources, and puts the originals back after."""

    def __init__(self, workspace: Path, original: Mapping[str, str]) -> None:
        self.workspace = Path(workspace)
        self.original = dict(original)

    async def _write(self, sources: Mapping[str, str]) -> None:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        with local_internal_governed_scope("mending_a_fault_of_hers.candidate", domain="file_write",
                                           receipt_prefix="her-mend-candidate",
                                           constraints={"artifact": "aura.her_mend_candidate.v1",
                                                        "operation": "replace"}):
            for path, text in sources.items():
                await gateway.write_text_async(self.workspace / path, text, source="mending_a_fault_of_hers")

    async def run(self, tests: Sequence[str], sources: Mapping[str, str]) -> tuple[dict[str, bool], str]:
        import os
        import tempfile

        if not tests:
            return {}, ""
        env = {**os.environ, "AURA_LOG_DIR": tempfile.mkdtemp(prefix="her-mend-logs-"),
               "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        await self._write(sources)
        try:
            code, text = await _governed_run(
                ["nice", "-n", "10", sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", *tests],
                cwd=self.workspace, env=env, timeout=_RUN_SECONDS, effect="compute")
        finally:
            await self._write(self.original)
        verdicts = {node: word in {"PASSED", "XFAIL"} for word, node in _RESULT.findall(text)}
        results = {}
        for test in tests:
            inside = [ok for node, ok in verdicts.items() if node == test or node.split("::")[0] == test]
            results[test] = bool(inside) and all(inside)
        failures = text[text.find("= FAILURES ="):] if "= FAILURES =" in text else text[-4000:]
        return results, failures


def the_lines_it_ran(source: str, path: str, line: int) -> list[tuple[tuple[str, int], float]]:
    """The line a fault was raised at first, then every other line of the function it was raised in.

    What caused a raise is often a line above it: the condition before a ``raise``, the index computed before the
    subscript. The function is the place a fault's evidence points to; the raise is the most suspect line in it.
    """
    import ast

    ranked = [((path, line), 1.0)]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ranked
    holding = [node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.lineno <= line <= (node.end_lineno or node.lineno)]
    if not holding:
        return ranked
    innermost = min(holding, key=lambda node: (node.end_lineno or node.lineno) - node.lineno)
    body_from = innermost.body[0].lineno if innermost.body else innermost.lineno
    return ranked + [((path, other), 0.5) for other in range(body_from, (innermost.end_lineno or line) + 1)
                     if other != line]


def her_own_python_first(workspace: Path, at: tuple[str, int],
                         then: Callable[..., Awaitable[list[tuple[str, str]]]] | None = None, error_type: str = ""):
    """Her own Python's exact changes first, at the lines of the function the fault was raised in, then ``then``'s
    (her model's)."""
    from core.self_modification.proposals_without_a_model import proposals_without_a_model

    pending: dict[str, Any] = {}

    async def propose(defect: Any, start: Mapping[str, str], parent: Any, n: int) -> list[tuple[str, str]]:
        key = parent.id if parent is not None else ""
        if key not in pending:
            failure = parent.fitness.failures if parent is not None and parent.fitness else defect.seen
            path, line = at
            ranked = the_lines_it_ran(start.get(path, ""), path, line)
            pending[key] = proposals_without_a_model(failure, start, workspace, ranked, error_type)
        exact = next(pending[key], None)
        if exact is not None:
            return [exact]
        return await then(defect, start, parent, n) if then is not None else []

    return propose


async def mend(fault: Any, *, proposals: int, seconds: float, model: Any = None) -> dict[str, Any]:
    """Make the fault again in a workspace of hers, search for a change that mends it, and offer the one that does."""
    from core.agency import her_goal_at_work as goals
    from core.self_modification.faults_she_can_reproduce import as_defect, reproduce
    from core.self_modification.repair_by_selection import applied, repair_by_selection

    goal = goal_for(fault)
    slug = goals.slug_of(goal)
    record = goals.load(slug) or goals.GoalRecord(goal=goal, slug=slug)
    if record.status in (goals.OFFERED, goals.MERGED):
        return {"slug": slug, "status": record.status, "why": "already mended or offered"}
    if goals.governs_her(fault.call.file):
        # LIVE 16 November 2026, 01:11 to 01:35, her loop searched until its deadline for a change to
        # require_governance, the check that refused an ungoverned save. A mend there loosens the check, and her own
        # yes refuses the file anyway.
        return await _stop(record, f"it was raised in {fault.call.file}, which governs her; her own yes cannot take a "
                                   "change there, so her loop drafts none", fault.call.error_message)
    workspace = await asyncio.to_thread(goals.ensure_workspace, record)
    await goals.save(record)

    reproduction = await reproduce(fault, workspace)
    if not reproduction.reproduced:
        return await _stop(record, "its test did not raise what she raised", reproduction.output)
    defect = as_defect(reproduction)
    file = fault.call.file
    sources = {file: (Path(workspace) / file).read_text(encoding="utf-8")}
    runner = InHerWorkspace(workspace, sources)

    async def neighbours() -> list[str] | None:
        named = await asyncio.to_thread(goals.tests_that_touch, [file], Path(workspace))
        named = [test for test in named if test != reproduction.test]
        if not named:
            return []
        passed, _ = await runner.run(named, sources)
        return [test for test in named if passed.get(test)]

    lineage = await repair_by_selection(
        defect, sources=sources, neighbours=neighbours, run=runner.run,
        propose=her_own_python_first(Path(workspace), (file, fault.call.line), model, fault.call.error_type),
        proposals=proposals, deadline=time.time() + seconds,
    )
    best = lineage.best
    if best is None or best.fitness is None or not best.fitness.mends:
        why = lineage.stopped or "no candidate mended it"
        if best is not None and best.fitness is not None and best.fitness.fixed and not best.fitness.guarded:
            why = f"a candidate fixed its test, and no test that passed before guards {file}"
        return await _stop(record, why, lineage.to_json())
    changed, refused = applied(sources, best.edits)
    if refused:
        return await _stop(record, refused, lineage.to_json())
    await runner._write(changed)
    offered = await _offer(record, workspace, [file, reproduction.test], best, fault)
    if offered.get("status") == "merged":
        offered["loaded"] = _into_her_running_process(fault, sources[file], changed[file], record.slug)
    return offered


def _into_her_running_process(fault: Any, before: str, after: str, slug: str) -> dict[str, Any]:
    """Her taken mend loaded into the process she is running (core/self_modification/into_the_running_process.py).

    What cannot be loaded waits for her next start and says why."""
    from core.governance_context import local_internal_governed_scope
    from core.self_modification.into_the_running_process import load_into_this_process

    with local_internal_governed_scope("mending_a_fault_of_hers.load", domain="state_mutation",
                                       receipt_prefix="her-mend-load",
                                       constraints={"artifact": "aura.her_mend_load.v1", "operation": "load"}):
        loaded = load_into_this_process(fault.call.module, before, after, goal=slug, mended_call=fault.call)
    return {"functions": [swap.qualname for swap in loaded.swaps], "waits": list(loaded.waits)}


async def _stop(record: Any, why: str, evidence: Any) -> dict[str, Any]:
    from core.agency import her_goal_at_work as goals

    record.status = goals.STALLED
    # Not tried again on the same source: a change to her source is what can change the answer.
    record.stalled_on = await asyncio.to_thread(goals.source_head)
    record.said.append({"words": f"Not mended: {why}.", "at": time.time()})
    record.last_check = {"passed": False, "at": time.time(), "why": why,
                         "evidence": str(evidence)[-2000:]}
    await goals.save(record)
    return {"slug": record.slug, "status": record.status, "why": why}


async def _offer(record: Any, workspace: Path, files: list[str], best: Any, fault: Any) -> dict[str, Any]:
    """Commit the mend and its test on the goal's branch, offer it, and let her own yes run her gates."""
    from core.agency import her_goal_at_work as goals

    call = fault.call
    message = (f"Mend {call.module}.{call.qualname}: it raised {call.error_type.rsplit('.', 1)[-1]} while she ran\n\n"
               f"{call.error_message[:300]}\n\nFound by her own running, made again by {files[1]}, mended by "
               f"{best.origin} (repair by selection). Drafted by Aura toward her goal: {record.goal}")
    code, out = await _governed_run(["git", "add", "--", *files], cwd=workspace, env=None, timeout=60.0,
                                    effect="commit")
    if code == 0:
        code, out = await _governed_run(
            ["git", "-c", "core.fsmonitor=false", "-c", "user.name=Aura", "-c", "user.email=aura@localhost",
             "commit", "-q", "-m", message], cwd=workspace, env=None, timeout=60.0, effect="commit")
    if code != 0:
        return await _stop(record, f"her mend could not be committed: {out.strip()[:200]}", out)
    _, head = await _governed_run(["git", "rev-parse", "HEAD"], cwd=workspace, env=None, timeout=30.0,
                                  effect="compute")
    record.status = goals.OFFERED
    record.offer = {"summary": message.splitlines()[0], "commit": head.strip(), "at": time.time(), "files": files,
                    "origin": best.origin}
    await goals.save(record)
    try:
        taken = await asyncio.to_thread(goals.take_it_herself, record.slug)
    except (OSError, RuntimeError, ValueError) as exc:
        record_degradation("mending_a_fault_of_hers", exc, severity="warning",
                           action="left her mend offered for Bryan; her own yes did not run")
        taken = {"ok": False, "why": f"her own yes did not run: {type(exc).__name__}"}
    return {"slug": record.slug, "status": "merged" if taken.get("ok") else goals.OFFERED, "commit": head.strip(),
            "origin": best.origin, "taken": taken}


async def mend_the_faults_she_heard(*, seconds: float = _RUN_SECONDS) -> list[dict[str, Any]]:
    """The most frequent fault she heard and has not tried on this source, mended: one a cycle, in her idle time.

    Her own Python's proposals are finite, and ``seconds`` bounds the search; a fault that stalled is tried again once
    her source has changed.
    """
    from core.agency import her_goal_at_work as goals
    from core.self_modification.faults_she_can_reproduce import (
        faults_heard,
        keep_what_was_heard,
        mends_undone,
    )

    await keep_what_was_heard()
    # A mend undone in her running process (it faulted after it was loaded) comes back out of her source first.
    taken_back = []
    for slug in mends_undone():
        done = await asyncio.to_thread(goals.take_it_back, slug, "it faulted in her running process after it was "
                                                                  "loaded, so it was undone there")
        taken_back.append({"slug": slug, "status": "taken back" if done.get("ok") else "not taken back",
                           "why": done.get("why") or done.get("revert", "")})
    if taken_back:
        return taken_back
    heard = faults_heard()
    if not heard:
        return []
    head = await asyncio.to_thread(goals.source_head)
    for fault in heard:
        record = goals.load(goals.slug_of(goal_for(fault)))
        if record is not None and (record.status in (goals.OFFERED, goals.MERGED)
                                   or (record.status == goals.STALLED and record.stalled_on == head)):
            continue
        return [await mend(fault, proposals=sys.maxsize, seconds=seconds)]
    return []


__all__ = ["InHerWorkspace", "goal_for", "her_own_python_first", "mend", "mend_the_faults_she_heard"]
