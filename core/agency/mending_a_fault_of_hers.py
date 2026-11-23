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
import json
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
#: The code that finds and judges a repair: when it changes, a fault it could not mend may now be mended.
_HER_MEANS_OF_REPAIR = ("core/self_modification/", "core/agency/mending_a_fault_of_hers.py")


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

    async def run(self, tests: Sequence[str], sources: Mapping[str, str], *,
                  forcings: Path | None = None) -> tuple[dict[str, bool], str]:
        """``tests`` run on ``sources``; with ``forcings``, once for each of its plans in one session
        (core/self_modification/forcing_session.py), whose verdicts are in its own file."""
        import os
        import tempfile

        if not tests:
            return {}, ""
        env = {**os.environ, "AURA_LOG_DIR": tempfile.mkdtemp(prefix="her-mend-logs-"),
               "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        plugin: list[str] = []
        if forcings is not None:
            from core.self_modification.forcing_session import pytest_invocation

            plugin, more = pytest_invocation(forcings)
            env.update(more)
        await self._write(sources)
        try:
            code, text = await _governed_run(
                ["nice", "-n", "10", sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", *plugin,
                 *tests],
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
                         then: Callable[..., Awaitable[list[tuple[str, str]]]] | None = None, error_type: str = "", *,
                         run: Callable[..., Awaitable[tuple[dict[str, bool], str]]] | None = None,
                         passing: Callable[[], Awaitable[Sequence[str] | None]] | None = None,
                         deadline: float | None = None):
    """Her own Python's changes first, at the lines of the function the fault was raised in: the exact ones, then
    semantic repair from her program's values there when ``run`` is given, then the fix templates; then ``then``'s
    (her model's)."""
    import tempfile

    from core.self_modification.semantic_repair import her_own_python

    async def rank(start: Mapping[str, str]) -> list[tuple[tuple[str, int], float]]:
        path, line = at
        return the_lines_it_ran(start.get(path, ""), path, line)

    async def passing_at(start: Mapping[str, str], ranked: Sequence[Any]) -> Sequence[str] | None:
        return await passing() if passing is not None else []

    async def run_forcings(tests: Sequence[str], sources: Mapping[str, str], spec: Path) -> None:
        await run(tests, sources, forcings=spec)

    return her_own_python(repo=workspace, rank=rank, run=run, passing=passing_at,
                          workdir=Path(tempfile.mkdtemp(prefix="her-semantic-repair-")),
                          deadline=(lambda: deadline) if deadline is not None else None, error_type=error_type,
                          then=then, run_forcings=run_forcings if run is not None else None)


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
        why = (f"it was raised in {fault.call.file}, which governs her; her own yes cannot take a change there, so her "
               "loop drafts none")
        if record.status == goals.STALLED and (record.last_check or {}).get("why") == why:
            # Said once: no change to her source makes it hers to change. LIVE 21 November 2026 one refused save
            # heard on every surprise of her mind tick was said again each time, some sixty times in an afternoon.
            return {"slug": slug, "status": record.status, "why": why}
        return await _stop(record, why, fault.call.error_message)
    if record.status == goals.STALLED:
        # Taken up again: worked again, so its workspace is brought up to her source before anything runs in it.
        # LIVE 16 November 2026 her drill's retry ran in the workspace of the try before, on the old source.
        record.status = goals.WORKING
    workspace = await asyncio.to_thread(goals.ensure_workspace, record)
    await goals.save(record)
    again = await _her_draft_asked_again(record, workspace, fault)
    if again is not None:
        return again

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

    held: list[str] | None = None

    async def guarding() -> list[str] | None:
        """The tests that touch the file and pass now, found once: the guard and semantic repair's passing runs."""
        nonlocal held
        if held is None:
            held = await neighbours()
        return held

    deadline = time.time() + seconds
    lineage = await repair_by_selection(
        defect, sources=sources, neighbours=guarding, run=runner.run,
        propose=her_own_python_first(Path(workspace), (file, fault.call.line), model, fault.call.error_type,
                                     run=runner.run, passing=guarding, deadline=deadline),
        proposals=proposals, deadline=deadline,
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
    return await _offer(record, workspace, [file, reproduction.test], best, fault, (sources[file], changed[file]))


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


async def _offer(record: Any, workspace: Path, files: list[str], best: Any, fault: Any,
                 mended: tuple[str, str]) -> dict[str, Any]:
    """Commit the mend and its test on the goal's branch, offer it, and let her own yes run her gates.

    A mend her yes takes is loaded into her running process in the same call, on the same thread, so a mend that
    landed is a mend that was loaded, whatever happens to the coroutine waiting on it."""
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
    if (record.offer or {}).get("commit"):
        # Kept, not replaced: LIVE 17 November 2026 her drill's mend landed, was taken back out, and vanished from her
        # page when the goal's next draft was offered over it.
        record.offers_before = [*record.offers_before, dict(record.offer)]
    kept = {key: value for key, value in (record.offer or {}).items() if key in ("must_keep", "kept_in")}
    record.offer = {"summary": message.splitlines()[0], "commit": head.strip(), "at": time.time(), "files": files,
                    "origin": best.origin, **kept}
    await goals.save(record)
    taken = await _taken_and_loaded(record, fault, mended)
    return {"slug": record.slug, "status": "merged" if taken.get("ok") else goals.OFFERED, "commit": head.strip(),
            "origin": best.origin, "taken": taken, "loaded": taken.get("loaded") or {}}


async def _taken_and_loaded(record: Any, fault: Any, mended: tuple[str, str]) -> dict[str, Any]:
    """Her own yes on her offered draft, and the draft loaded into her running process when it lands."""
    from core.agency import her_goal_at_work as goals

    def take_and_load() -> dict[str, Any]:
        taken = goals.take_it_herself(record.slug)
        if taken.get("ok"):
            taken["loaded"] = _into_her_running_process(fault, *mended, record.slug)
            goals.note_on_offer(record.slug, loaded={**taken["loaded"], "at": time.time()})
        return taken

    try:
        return await asyncio.to_thread(take_and_load)
    except (OSError, RuntimeError, ValueError) as exc:
        record_degradation("mending_a_fault_of_hers", exc, severity="warning",
                           action="left her mend offered for Bryan; her own yes did not run")
        return {"ok": False, "why": f"her own yes did not run: {type(exc).__name__}"}


async def _her_draft_asked_again(record: Any, workspace: Path, fault: Any) -> dict[str, Any] | None:
    """Her draft for the goal, carried onto her source as it is now, asked of her gates again when it was her gates
    that refused it; None when there is no such draft.

    LIVE 17 November 2026, 04:16 to 04:17: her drill's mend passed every gate but governance-lint, which main had
    failed since 1dec3d5b7 for a reason outside her draft. Taken up again, her loop made the fault again in a workspace
    that held her draft, the draft mended it, and she was told "its test did not raise what she raised". Her draft is
    the change; what has to be asked again is her gates, on the source as it is."""
    from core.agency import her_goal_at_work as goals

    offer = record.offer or {}
    refused = any(not (gate or {}).get("passed") for gate in (offer.get("gates") or {}).values())
    if not offer.get("commit") or offer.get("merged_at") or offer.get("reverted") or not refused:
        return None
    judged = (offer.get("gates") or {}).get("outside-check") or {}
    if judged and not judged.get("passed"):
        # The outside check refused the draft itself, not her source around it: asked again, the same draft gets the
        # same verdict every cycle. So it is set aside and searched for again, keeping what the refusal said it lost.
        return await _search_again_keeping_what_was_lost(record, workspace, judged)
    _, ahead = await _governed_run(["git", "-c", "core.fsmonitor=false", "rev-list", "--count",
                                    f"{record.base}..HEAD"], cwd=workspace, env=None, timeout=30.0, effect="compute")
    if ahead.strip() in ("", "0"):
        return None
    _, head = await _governed_run(["git", "rev-parse", "HEAD"], cwd=workspace, env=None, timeout=30.0,
                                  effect="compute")
    file = fault.call.file
    _, before = await _governed_run(["git", "show", f"{record.base}:{file}"], cwd=workspace, env=None, timeout=30.0,
                                    effect="compute")
    after = (Path(workspace) / file).read_text(encoding="utf-8")
    record.status = goals.OFFERED
    record.offer = {**offer, "commit": head.strip(), "at": time.time(), "asked_again_on": record.base}
    record.said.append({"words": f"My draft is on my source as it is now ({record.base[:9]}); I asked my gates again.",
                        "at": time.time()})
    await goals.save(record)
    taken = await _taken_and_loaded(record, fault, (before, after))
    return {"slug": record.slug, "status": "merged" if taken.get("ok") else goals.OFFERED, "commit": head.strip(),
            "origin": str(offer.get("origin") or ""), "taken": taken, "loaded": taken.get("loaded") or {},
            "why": "her draft asked of her gates again"}


async def _search_again_keeping_what_was_lost(record: Any, workspace: Path, judged: Mapping[str, Any]) -> dict[str, Any] | None:
    """Her refused draft set aside, and a test of each input it lost committed in its place, so her next search keeps
    them (core/self_modification/what_her_change_must_keep.py); None to search again now, or why the goal stalls.

    Nothing to keep is no new evidence: a search from the same place finds the same draft, so the goal waits for her
    source to change, as a stalled one does."""
    from core.agency import her_goal_at_work as goals
    from core.self_modification.what_her_change_must_keep import keeping_test_source, path_of_what_it_must_keep

    offer = dict(record.offer or {})
    known = {json.dumps(one, sort_keys=True): one for one in [*(offer.get("must_keep") or []),
                                                               *(judged.get("must_keep") or [])]}
    must_keep = list(known.values())
    source = keeping_test_source(must_keep, goal=record.goal, root=workspace)
    record.offers_before = [*record.offers_before, {**offer, "set_aside": "the outside check refused it"}]
    if not source:
        record.offer = {"must_keep": must_keep}
        return await _stop(record, "the outside check refused my draft, and nothing it said it lost can be kept as a "
                                   "test, so a search from the same place would find the same draft",
                           str(judged.get("tail") or "")[-1500:])
    code, out = await _governed_run(["git", "-c", "core.fsmonitor=false", "reset", "-q", "--hard", record.base],
                                    cwd=workspace, env=None, timeout=60.0, effect="commit")
    path = path_of_what_it_must_keep(record.slug)
    if code == 0:
        await InHerWorkspace(workspace, {})._write({path: source})
        code, out = await _governed_run(["git", "add", "--", path], cwd=workspace, env=None, timeout=60.0,
                                        effect="commit")
    if code == 0:
        code, out = await _governed_run(
            ["git", "-c", "core.fsmonitor=false", "-c", "user.name=Aura", "-c", "user.email=aura@localhost", "commit",
             "-q", "-m", f"What my change for this goal must keep: {len(must_keep)} input(s) the outside check said a "
                         f"draft of mine lost\n\nGoal: {record.goal}"],
            cwd=workspace, env=None, timeout=60.0, effect="commit")
    if code != 0:
        return await _stop(record, f"what my refused draft lost could not be committed as a test: {out.strip()[:200]}",
                           out)
    record.offer = {"must_keep": must_keep, "kept_in": path}
    record.status = goals.WORKING
    record.said.append({"words": f"The outside check refused my draft: it lost {len(must_keep)} input(s) that worked. I "
                                 f"set it aside, wrote them as tests in {path}, and searched again keeping them.",
                        "at": time.time()})
    await goals.save(record)
    return None


def _what_it_stalled_on_has_moved(record: Any, file: str, head: str) -> bool:
    """Whether trying a stalled fault again could end differently: the file it was raised in, a test naming that file,
    or her means of repair changed since it stalled. LIVE, 17 and 18 November 2026: one locked database was tried 35
    times, once for each commit to her source, none of which touched what it stood on."""
    from core.agency import her_goal_at_work as goals

    if record.stalled_on == head:
        return False
    if not record.stalled_on or not head:
        return True
    root = goals.source_root()
    shown = goals._git("diff", "--name-only", record.stalled_on, head, cwd=root)
    if shown.returncode != 0:
        return True
    changed = set(shown.stdout.splitlines())
    return file in changed or any(name.startswith(_HER_MEANS_OF_REPAIR) for name in changed) \
        or bool(changed & set(goals.tests_that_touch([file], root)))


async def _keep_her_recent_turns() -> None:
    """Her finished turns and a reading of her memory, kept off her event loop for probation to measure
    (core/agency/her_turns_on_probation.py)."""
    try:
        from core.agency.her_turns_on_probation import keep_her_recent_turns

        await asyncio.to_thread(keep_her_recent_turns)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation("mending_a_fault_of_hers", exc, severity="warning",
                           action="her turns were not kept this cycle, so probation measures fewer of them")


async def _keep_her_real_calls_of_what_she_may_change() -> None:
    """Her real calls watched for each function she may change next, and what was kept so far written to her state."""
    from core.self_modification.her_real_calls import keep_her_real_calls, watch_what_she_may_change

    try:
        await asyncio.to_thread(watch_what_she_may_change)
        await keep_her_real_calls()
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        record_degradation("her_real_calls", exc, severity="warning",
                           action="her real calls were not kept this cycle; her changes are judged without them")


async def _start_an_outside_comparison_when_due() -> None:
    """The outside benchmarks asked again while a change of hers is on probation, when nobody waits on her."""
    from core.agency import her_goal_at_work as goals
    from core.agency.measuring_her_on_probation import start_a_comparison_when_due

    if goals.change_on_probation() is None:
        return
    try:
        # It says when it starts one; why it did not is in what it returns.
        await asyncio.to_thread(start_a_comparison_when_due)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        record_degradation("measuring_her_on_probation", exc, severity="warning",
                           action="no outside comparison was started; her change stays on probation")


async def mend_the_faults_she_heard(*, seconds: float = _RUN_SECONDS) -> list[dict[str, Any]]:
    """The most frequent fault she heard and has not tried as things now stand, mended: one a cycle, in her idle time.

    Her own Python's proposals are finite, and ``seconds`` bounds the search; a fault that stalled is tried again once
    what it stands on has moved (``_what_it_stalled_on_has_moved``).
    """
    from core.agency import her_goal_at_work as goals
    from core.self_modification.faults_she_can_reproduce import (
        faults_heard,
        keep_what_was_heard,
        mends_undone,
    )

    await keep_what_was_heard()
    await _keep_her_real_calls_of_what_she_may_change()
    # A mend undone in her running process (it faulted after it was loaded) comes back out of her source first.
    taken_back = []
    for slug in dict.fromkeys([*mends_undone(), *goals.mends_undone_on_record()]):
        done = await asyncio.to_thread(goals.take_it_back, slug, "it faulted in her running process after it was "
                                                                  "loaded, so it was undone there")
        taken_back.append({"slug": slug, "status": "taken back" if done.get("ok") else "not taken back",
                           "why": done.get("why") or done.get("revert", "")})
    # Off her loop: settling can run the outside check over her real calls, which takes minutes.
    await _keep_her_recent_turns()
    taken_back += await asyncio.to_thread(goals.settle_probation)
    if taken_back:
        return taken_back
    await _start_an_outside_comparison_when_due()
    heard = faults_heard()
    head = await asyncio.to_thread(goals.source_head) if heard else ""
    for fault in heard:
        record = goals.load(goals.slug_of(goal_for(fault)))
        if record is not None and (record.status in (goals.OFFERED, goals.MERGED) or (
                record.status == goals.STALLED
                and not await asyncio.to_thread(_what_it_stalled_on_has_moved, record, fault.call.file, head))):
            continue
        # Her model proposes too, last, once her own Python has nothing left, and its mend is judged as any other is
        # (Bryan, 22 November 2026: "as a last resort ... so long as its checked and verified the same way").
        from core.self_modification.a_mend_her_model_proposes import her_model_as_a_last_resort

        return [await mend(fault, proposals=sys.maxsize, seconds=seconds,
                           model=her_model_as_a_last_resort(fault.call.file, fault.call.line))]
    # Nothing heard to mend: a draft of hers that waits on her own yes is asked again, once her source has moved.
    asked = await _ask_again_of_a_draft_that_waits()
    if asked is not None:
        return [asked]
    # Then a change for a cost her running measures, or the judgement of one she made (F16, part 3).
    from core.agency.improving_by_measurement import improve_by_measurement

    return await improve_by_measurement()


async def _ask_again_of_a_draft_that_waits() -> dict[str, Any] | None:
    """The draft of hers that has waited longest on her own yes, asked of her gates again on her source as it now is,
    when that source has moved since it was last asked: one a cycle, and none that changes her own governance.

    A draft is asked again where what made it comes back: a fault heard again, a cost measured again. Where it does not,
    it waited for good. LIVE 22 November 2026: her mend of the subprocess gateway's argument check had waited since 17
    November, after tests that were not its own failed, and her young collection in the MLX worker's lifecycle since
    20 November, after a restart stopped its outside check, both said on her changes page to wait for Bryan.
    """
    import time

    from core.agency import her_goal_at_work as goals

    head = await asyncio.to_thread(goals.source_head)
    if not head:
        return None
    waiting = [record for record in await asyncio.to_thread(goals.all_records)
               if record.status == goals.OFFERED and record.offer.get("asked_on") != head
               and not any(goals.governs_her(name) for name in record.offer.get("files") or [])]
    if not waiting:
        return None
    record = min(waiting, key=lambda each: float(each.offer.get("asked_at") or each.offer.get("at") or 0.0))
    record.offer = {**record.offer, "asked_on": head, "asked_at": time.time()}
    await goals.save(record)
    try:
        await asyncio.to_thread(goals.ensure_workspace, record)
    except (RuntimeError, OSError) as exc:
        record_degradation("her_goal_at_work", exc, severity="warning",
                           action="her waiting draft was not asked again: its workspace could not be brought up")
        return {"slug": record.slug, "status": record.status, "why": f"{type(exc).__name__}: {exc}"[:240]}
    done = await asyncio.to_thread(goals.take_it_herself, record.slug)
    after = goals.load(record.slug)
    return {"slug": record.slug, "status": after.status if after is not None else record.status,
            "why": str(done.get("why") or ("taken" if done.get("ok") else ""))}


__all__ = ["InHerWorkspace", "goal_for", "her_own_python_first", "mend", "mend_the_faults_she_heard"]
