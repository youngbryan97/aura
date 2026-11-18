#!/usr/bin/env python3
"""Search for a change to her code that makes failing tests pass and breaks none that passed.

The search is core/self_modification/repair_by_selection.py. This gives it a
copy of the repository to run tests in (a detached git worktree at HEAD, never
the checkout her app runs from), the tests that import the files a change may
touch as the ones it must not break, and proposals: first the exact changes her
own Python writes (core/self_modification/proposals_without_a_model.py), then semantic repair from her program's
values at the suspect lines (core/self_modification/semantic_repair.py), then the fix templates, and only when those
are spent her code lane's model, or a file of proposals. It writes every candidate, its parent and its fitness to
``OUT/lineage.json`` and the patch genealogy graph in ``OUT``, and the fittest
change to ``OUT/best.diff``. It applies nothing to her code.

Proposals come from her code lane (core/brain/llm/local_code_model.py), which
loads the named model under the model lane's ownership, so it refuses while
her app holds the lane, and samples at the model's own published temperature.

Usage:
    repair_by_selection.py --test NODE [--test NODE ...] --file PATH [--file PATH ...]
        [--model-path DIR | --proposals-from FILE] --proposals N --minutes M --out DIR
        [--plant PATH OLD NEW] [--at COMMIT --tests-from COMMIT]

``--plant`` makes a defect on the copy before the search starts, for proving the
search on a known answer: the change it finds should undo the plant. ``--at`` and
``--tests-from`` start from a fix commit's parent with the fix's tests in, for the
defects tools/defects_from_her_history.py finds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = sys.executable
#: The most suspicious lines semantic repair looks at (core/self_modification/semantic_repair.semantic_repairs).
SUSPECT_LINES = 15
#: Passing tests run with the probe: those that run the most suspect lines. Each is examples of what a condition
#: there must keep deciding, and each is time in the run that records them.
PASSING_KEPT = 40
_RESULT = re.compile(r"^(PASSED|FAILED|ERROR|XPASS|XFAIL|SKIPPED)\s+(\S+)", re.MULTILINE)


def neighbours_of(files: Sequence[str], repo: Path) -> list[str]:
    """Test files that import a module the change may touch: what a change to it can break."""
    modules = [f[:-3].replace("/", ".") for f in files if f.endswith(".py")]
    found = []
    for test in sorted((repo / "tests").rglob("test_*.py")):
        text = test.read_text(encoding="utf-8", errors="ignore")
        if any(re.search(rf"(?:from|import)\s+{re.escape(m)}\b", text) for m in modules):
            found.append(str(test.relative_to(repo)))
    return found


def guarding_groups(files: Sequence[str], repo: Path) -> list[list[str]]:
    """neighbours_of each file on its own, the file fewest tests import first: the tests nearest a change, before
    those of a module half her suite imports (a702a104eb27's two files: 2 test files and 123)."""
    return sorted((neighbours_of([f], repo) for f in files), key=len)


class Copy:
    """A detached worktree at HEAD whose files are set for each run and put back after it."""

    def __init__(self, repo: Path, where: Path, at: str = "HEAD") -> None:
        self.repo, self.path = repo, where / "copy"
        subprocess.run(["git", "-C", str(repo), "-c", "core.fsmonitor=false", "worktree", "add", "--detach",
                        str(self.path), at], check=True, capture_output=True)
        self.state = where / "state"
        self.logs = where / "logs"

    def read(self, files: Sequence[str]) -> dict[str, str]:
        return {f: (self.path / f).read_text(encoding="utf-8") for f in files}

    def write(self, sources: Mapping[str, str]) -> None:
        for path, text in sources.items():
            (self.path / path).write_text(text, encoding="utf-8")

    async def run(self, tests: Sequence[str], sources: Mapping[str, str], *, original: Mapping[str, str],
                  seconds: float, forcings: Path | None = None,
                  first_failure: bool = False) -> tuple[dict[str, bool], str]:
        """``tests`` run on the copy holding ``sources``; with ``forcings``, once for each of its plans in one session
        (core/self_modification/forcing_session.py), whose verdicts are in its own file. With ``first_failure`` the
        session stops at the first test that fails, and the tests it did not reach are left out of the verdicts."""
        self.write(sources)
        env = {k: v for k, v in os.environ.items() if k not in {"AURA_TESTING"}}
        env.update(AURA_STATE_ROOT=str(self.state), AURA_LOG_DIR=str(self.logs), PYTHONDONTWRITEBYTECODE="1",
                   VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
        plugin: list[str] = []
        if forcings is not None:
            from core.self_modification.forcing_session import pytest_invocation

            plugin, more = pytest_invocation(forcings)
            env.update(more)
        try:
            proc = await asyncio.create_subprocess_exec(
                PYTHON, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", *(["-x"] if first_failure else []),
                *plugin, *tests,
                cwd=self.path, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=seconds)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return {t: False for t in tests}, f"the tests ran past {seconds:.0f} s"
        finally:
            self.write(original)
        text = out.decode("utf-8", errors="replace")
        verdicts: dict[str, bool] = {}
        for word, node in _RESULT.findall(text):
            verdicts[node] = word in {"PASSED", "XFAIL"}
        results = {}
        for t in tests:
            if t in verdicts:
                results[t] = verdicts[t]
            else:
                # A whole file passes when every test in it that ran passed.
                inside = [ok for node, ok in verdicts.items() if node.split("::")[0] == t]
                if inside or not first_failure:
                    results[t] = bool(inside) and all(inside)
        failures = text[text.find("= FAILURES ="):] if "= FAILURES =" in text else text[-4000:]
        return results, failures

    async def passing_in(self, files: Sequence[str], sources: Mapping[str, str], *,
                         seconds: float) -> list[str] | None:
        """Every test in ``files`` that passes on the copy as it is, or None when they run past ``seconds``."""
        if not files:
            return []
        self.write(sources)
        env = {k: v for k, v in os.environ.items() if k not in {"AURA_TESTING"}}
        env.update(AURA_STATE_ROOT=str(self.state), AURA_LOG_DIR=str(self.logs), PYTHONDONTWRITEBYTECODE="1",
                   VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
        proc = await asyncio.create_subprocess_exec(
            PYTHON, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", *files,
            cwd=self.path, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=seconds)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return None
        return [node for word, node in _RESULT.findall(out.decode("utf-8", errors="replace")) if word == "PASSED"]

    async def lines_each_test_runs(self, tests: Sequence[str], files: Sequence[str], sources: Mapping[str, str], *,
                                   original: Mapping[str, str], seconds: float,
                                   groups: Sequence[Sequence[str]] = ()) -> tuple[dict[str, set], dict[str, set], int]:
        """For each test run (``tests``, the rest of their files, and each group of test files in ``groups``), the
        lines of ``files`` it ran: failing, passing, and how many of ``groups`` were run whole.

        coverage.py's per-test contexts (dynamic_context = test_function) say which test ran which line.
        """
        try:
            return await asyncio.to_thread(self._lines_each_test_runs, tests, files, sources, seconds=seconds,
                                           groups=groups)
        finally:
            self.write(original)

    def _lines_each_test_runs(self, tests, files, sources, *, seconds, groups=()):
        self.write(sources)
        where = self.path.parent / "coverage"
        where.mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k not in {"AURA_TESTING"}}
        env.update(AURA_STATE_ROOT=str(self.state), AURA_LOG_DIR=str(self.logs), PYTHONDONTWRITEBYTECODE="1",
                   VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
        # The failing tests alone first: seconds, and enough to rank. Then the rest of their files, then each group
        # of the files that guard the change, each measured on its own so that one that runs out of time loses
        # nothing measured before it. A group is run when the seconds a test file has taken so far say it fits in
        # half of what is left: a702a104eb27's second group took 4 min 45 s alone. Without passing tests every line
        # the failing tests run scores the same, and the lines are tried in the order written: LIVE 16 November
        # 2026, f5da1f5da653's faulty line held her own fix first, and its turn never came.
        own = list(dict.fromkeys(t.split("::")[0] for t in tests))
        phases = [list(tests), own]
        done = set(own)
        for group in groups:
            phases.append([f for f in group if f not in done])
            done.update(group)
        started = time.time()
        verdicts: dict[str, bool] = {}
        ran: dict[str, set] = {}
        whole = 0
        spent, files_run = 0.0, 0
        for n, targets in enumerate(phases):
            if not targets:
                whole += n >= 2
                continue
            left = seconds - (time.time() - started)
            if n >= 2 and files_run and spent / files_run * len(targets) > left / 2:
                break
            began = time.time()
            rc = where / f"rc.{n}"
            data = where / f"data.{n}"
            data.unlink(missing_ok=True)
            rc.write_text("[run]\ndynamic_context = test_function\ndata_file = " + str(data) + "\ninclude =\n"
                          + "".join(f"    {self.path / f}\n" for f in files))
            try:
                out = subprocess.run(
                    [PYTHON, "-m", "coverage", "run", f"--rcfile={rc}", "-m", "pytest", "-q", "-rA", "-p",
                     "no:cacheprovider", *targets], cwd=self.path, env=env, capture_output=True, text=True,
                    timeout=max(1.0, left if n == 0 else left / 2)).stdout
            except subprocess.TimeoutExpired:
                break
            verdicts.update({node: word == "PASSED" for word, node in _RESULT.findall(out)})
            whole += n >= 2
            if n >= 1:
                spent, files_run = spent + time.time() - began, files_run + len(targets)
            report = where / f"report.{n}.json"
            subprocess.run([PYTHON, "-m", "coverage", "json", f"--rcfile={rc}", "--show-contexts", "-o", str(report)],
                           cwd=self.path, env=env, capture_output=True)
            if not report.exists():
                continue
            for path, measured in json.loads(report.read_text())["files"].items():
                relative = (str(Path(path).resolve().relative_to(self.path.resolve())) if Path(path).is_absolute()
                            else path)
                for line, contexts in (measured.get("contexts") or {}).items():
                    for context in contexts:
                        name = context.split("|")[0].rsplit(".", 1)[-1]
                        if name:
                            ran.setdefault(name, set()).add((relative, int(line)))
        failing, passing = {}, {}
        for node, passed in verdicts.items():
            name = node.split("::")[-1].split("[")[0]
            (passing if passed else failing)[node] = ran.get(name, set())
        return failing, passing, whole

    def remove(self) -> None:
        subprocess.run(["git", "-C", str(self.repo), "-c", "core.fsmonitor=false", "worktree", "remove", "--force",
                        str(self.path)], capture_output=True)


def request_for(defect, start: Mapping[str, str], parent) -> str:
    """What her model is asked: the failure, the code it passed through, and the form a change is written in."""
    from core.self_modification.repair_by_selection import the_code_a_failure_passes_through

    failure = parent.fitness.failures if parent is not None and parent.fitness else defect.seen
    shown = the_code_a_failure_passes_through(failure, start)
    code = "\n\n".join(f"{path}\n```python\n{part}\n```" for path, parts in shown.items() for part in parts)
    return (
        f"These tests fail:\n{chr(10).join(defect.tests)}\n\nWhat they printed:\n```\n{failure[-6000:]}\n```\n\n"
        f"The code they ran through:\n\n{code}\n\n"
        "Change the code so these tests pass without breaking others. Write each change as the file's path on "
        "its own line, then\n<<<<<<< SEARCH\n(lines exactly as they are now)\n=======\n(the lines that replace "
        "them)\n>>>>>>> REPLACE"
    )


def code_lane(model_path: str, out: Path):
    """A proposer on her code lane (core/brain/llm/local_code_model.py): a model trusted by digest, loaded
    under the lane's ownership, no persona steering, a receipt for every answer and a refusal for one cut
    off at its token limit. It samples at the model's own published temperature, a new seed each time."""
    import os

    from core.brain.llm.local_code_model import LocalCodeModel, write_model_trust_manifest

    # The operator named this model on the command line; the run trusts it from its own folder, never the
    # live instance's trust root.
    os.environ["AURA_CODE_MODEL_TRUST_MANIFEST"] = str(
        write_model_trust_manifest(model_path, out / "trust" / "local_code_model.json"))
    lane = LocalCodeModel(model_path)
    settings_file = Path(model_path) / "generation_config.json"
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    temperature = float(settings.get("temperature", 1.0))
    receipts = out / "receipts.jsonl"
    asked = 0

    async def propose(defect, start, parent, n):
        nonlocal asked
        asked += 1
        from core.brain.llm.local_code_model import LocalCodeModelError

        try:
            result = await lane.think_with_receipt(request_for(defect, start, parent), temperature=temperature,
                                                   seed=asked)
        except LocalCodeModelError as exc:
            # The lane refuses an answer cut off at its token limit, and other answers it cannot vouch for:
            # a refused candidate, and the search goes on.
            with receipts.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"asked": asked, "parent": parent.id if parent else "", "refused": str(exc)}) + "\n")
            return [(f"code_lane_refused:{exc}", "")]
        text = result.text.split("</think>")[-1]
        with receipts.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"asked": asked, "parent": parent.id if parent else "", "text": text,
                                "receipt": {k: str(v) for k, v in asdict(result.receipt).items()}}) + "\n")
        # One proposal a call, so the next is written from whichever candidate is then fittest.
        return [("code_lane", text)]

    return propose


#: A comment line that is the writer thinking aloud: a question, or an opening that talks the problem through.
_THINKING_ALOUD = re.compile(r"#\s*(?:(?:let'?s|let me|wait|hmm|better approach)\b|(?:actually|no|yes),)|\?(?:\s|$)", re.I)


def thinking_aloud(text: str) -> int:
    """Comment lines in ``text`` that think aloud. A model's patch on 19 October left fourteen such lines in a
    function ("...from the worktree? No, that shows the worktree top."), and the writing gate found none."""
    return sum(1 for line in text.splitlines() if "#" in line and _THINKING_ALOUD.search("#" + line.split("#", 1)[1]))


def findings(sources: Mapping[str, str], repo: Path) -> int:
    """What her standards report in these files: ruff's findings, the writing gate's in their comments, and comment
    lines that think aloud."""
    count = 0
    with tempfile.TemporaryDirectory() as folder:
        for path, text in sources.items():
            if not path.endswith(".py"):
                continue
            done = subprocess.run([PYTHON, "-m", "ruff", "check", "--quiet", "--output-format", "concise",
                                   "--stdin-filename", path, "-"], input=text, capture_output=True, text=True,
                                  cwd=repo)
            count += sum(1 for line in done.stdout.splitlines() if line.strip())
            copy = Path(folder) / Path(path).name
            copy.write_text(text, encoding="utf-8")
            done = subprocess.run([PYTHON, str(repo / "tools" / "lint_ai_writing.py"), str(copy)],
                                  capture_output=True, text=True, cwd=repo)
            said = re.search(r"(\d+) findings across", done.stdout)
            count += int(said.group(1)) if said else 0
            count += thinking_aloud(text)
    return count


def from_file(path: Path):
    texts = list(json.loads(path.read_text()))

    async def propose(defect, start, parent, n):
        return [("file", texts.pop(0))] if texts else []

    return propose


def _carry_what_the_fix_wrote_to_check_itself(copy: Path, fix: str, files: list[str], came_with: list[str],
                                               tests: list[str]) -> list[str]:
    """The invariants the fix added to the files under repair that its tests name, put into those modules from the
    copy's conftest, outside the code a candidate can change (core/self_modification/a_fixs_own_specification.py)."""
    from core.self_modification.a_fixs_own_specification import injected, module_of, specification_in

    said = "\n".join((copy / path).read_text(encoding="utf-8") for path in
                     dict.fromkeys([*(t.split("::")[0] for t in tests), *came_with]) if (copy / path).is_file())
    named = set(re.findall(r"\w+", said))
    carried: list[str] = []
    for path in files:
        after = subprocess.run(["git", "-C", str(ROOT), "-c", "core.fsmonitor=false", "show", f"{fix}:{path}"],
                               capture_output=True, text=True).stdout
        specification, names = specification_in((copy / path).read_text(encoding="utf-8"), after, named)
        if specification:
            with (copy / "tests" / "conftest.py").open("a", encoding="utf-8") as conftest:
                conftest.write(injected(module_of(path), specification))
            carried.extend(names)
    return carried


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--file", action="append", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--proposals-from", type=Path)
    parser.add_argument("--proposals", type=int, required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plant", nargs=3, metavar=("PATH", "OLD", "NEW"))
    parser.add_argument("--at", default="HEAD", help="the commit the copy is made at")
    parser.add_argument("--tests-from", default="", help="a commit whose versions of the --test files are put in")
    parser.add_argument("--only-model", action="store_true",
                        help="ask only the model, to measure what it adds beyond her own Python's exact changes")
    parser.add_argument("--without-semantic", action="store_true",
                        help="leave semantic repair out, to measure what it adds beyond the exact changes and templates")
    parser.add_argument("--one-forcing-a-run", action="store_true",
                        help="force each place in a pytest run of its own, as before sessions of many forcings, to "
                             "measure what the sessions add")
    args = parser.parse_args()
    if args.model_path and args.proposals_from:
        parser.error("give at most one of --model-path and --proposals-from")

    from core.self_modification.patch_genealogy import PatchGenealogyGraph, PatchNode
    from core.self_modification.repair_by_selection import Defect, repair_by_selection

    args.out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.minutes * 60
    where = Path(tempfile.mkdtemp(prefix="repair-by-selection-"))
    copy = Copy(ROOT, where, args.at)
    carried: list[str] = []
    try:
        if args.tests_from:
            # Every test file the fixing commit came with, not only the failing tests' own: a commit that changes
            # what a test expects changes it with the fix, and the parent's copy would hold the old behaviour against
            # every right repair (a702a104eb27 rewrote test_control_trials_use_the_observed_cadence.py; 11 of the
            # 81 defects changed test files beyond their failing tests').
            came_with = subprocess.run(
                ["git", "-C", str(ROOT), "-c", "core.fsmonitor=false", "diff", "--name-only", "--diff-filter=d",
                 args.at, args.tests_from, "--", "tests"], check=True, capture_output=True, text=True).stdout.split()
            for path in dict.fromkeys([*(t.split("::")[0] for t in args.test), *came_with]):
                shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{args.tests_from}:{path}"], check=True,
                                       capture_output=True, text=True).stdout
                (copy.path / path).parent.mkdir(parents=True, exist_ok=True)
                (copy.path / path).write_text(shown, encoding="utf-8")
            carried = _carry_what_the_fix_wrote_to_check_itself(copy.path, args.tests_from, args.file, came_with, args.test)
        if args.plant:
            path, old, new = args.plant
            text = (copy.path / path).read_text(encoding="utf-8")
            if text.count(old) != 1:
                raise SystemExit(f"the planted text occurs {text.count(old)} times in {path}")
            (copy.path / path).write_text(text.replace(old, new, 1), encoding="utf-8")
        sources = copy.read(args.file)
        # Each run may take what the whole search has left.

        async def run(tests, s):
            """Tests on the copy holding ``s``. The guard runs a group of files at a time, nearest the change first,
            and stops at the first group that breaks: the change does not mend, and a group of 2,129 tests is not
            run to say so; within a group it stops at the first test that breaks (a702a104eb27's widest group broke
            2,068 tests after 4 min 45 s). What it did not run is left out of what it reports, not counted as held."""
            seconds = lambda: max(1.0, deadline - time.time())  # noqa: E731 - each run may take what is left
            if not found or list(tests) != found or len(groups) < 2:
                return await copy.run(tests, s, original=sources, seconds=seconds())
            results: dict[str, bool] = {}
            said: list[str] = []
            left = list(tests)
            for group in [*groups, None]:
                within = set(group) if group is not None else None
                mine = [t for t in left if within is None or t.split("::")[0] in within]
                if not mine:
                    continue
                taken = set(mine)
                left = [t for t in left if t not in taken]
                held, failures = await copy.run(mine, s, original=sources, seconds=seconds(), first_failure=True)
                results.update(held)
                said.append(failures)
                if not all(held.get(t) for t in mine):
                    break
            return results, "\n".join(part for part in said if part)

        async def run_forcings(tests, s, spec) -> None:
            await copy.run(tests, s, original=sources, seconds=max(1.0, deadline - time.time()), forcings=spec)
        # What must not break is each test, not each file: a file with one failing test still guards the rest.
        # Read from the spectrum of the defect's own code when its run held them, else run once a candidate first
        # fixes something (repair_by_selection.py).
        files = neighbours_of(args.file, copy.path)
        groups = guarding_groups(args.file, copy.path)
        found: list[str] = []

        async def neighbours() -> list[str] | None:
            # The spectrum of the defect's own code ran every guarding file already, when they fitted its time.
            if spectrum.get("whole"):
                found[:] = [t for t in spectrum["original"] if t.split("::")[0] in files and t not in args.test]
                return found
            passing = await copy.passing_in(files, sources, seconds=max(1.0, deadline - time.time()))
            if passing is None:
                return None
            found[:] = [n for n in passing if n not in args.test]
            return found
        graph = PatchGenealogyGraph(args.out / "genealogy.sqlite3")

        def record(candidate) -> None:
            graph.add_node(PatchNode(
                patch_id=candidate.id, trigger_id="|".join(args.test), target_files=tuple(args.file),
                bug_fingerprint="|".join(args.test), risk_tier="offline",
                status="refused" if candidate.refused else ("mends" if candidate.fitness.mends else "scored"),
                validation={"refused": candidate.refused} if candidate.refused else {
                    k: v for k, v in vars(candidate.fitness).items() if k != "failures"}))
            if candidate.parent:
                graph.add_edge(candidate.parent, candidate.id, "child_of")

        model = (code_lane(args.model_path, args.out) if args.model_path
                 else from_file(args.proposals_from) if args.proposals_from else None)
        spectrum: dict[str, dict[str, set]] = {}

        async def rank(start):
            from core.self_modification.fix_templates import ochiai

            failing, passing, whole = await copy.lines_each_test_runs(
                args.test, args.file, start, original=sources, groups=groups, seconds=max(1.0, deadline - time.time()))
            failing = {t: lines for t, lines in failing.items() if t in args.test} or failing
            spectrum["passing"] = passing
            ranked = ochiai(failing, passing)
            if start == sources:
                spectrum["original"], spectrum["whole"] = passing, whole == len(groups)
                spectrum["ranked"] = ranked
            return ranked

        async def passing_at(start, ranked):
            """The tests that pass now and run the suspect lines, those that run most of them first."""
            top = {line for line, score in ranked[:SUSPECT_LINES] if score > 0}
            ran = {t: len(lines & top) for t, lines in spectrum.get("passing", {}).items() if t not in args.test}
            return [t for t, count in sorted(ran.items(), key=lambda item: -item[1]) if count][:PASSING_KEPT]

        from core.self_modification.semantic_repair import Evidence, her_own_python

        evidence = Evidence()
        propose = model if args.only_model and model is not None else her_own_python(
            repo=copy.path, rank=rank, run=None if args.without_semantic else run, passing=passing_at,
            workdir=where / "semantic", deadline=lambda: deadline, then=model, evidence=evidence,
            run_forcings=None if args.one_forcing_a_run else run_forcings)
        before = findings(sources, copy.path)

        async def standards(changed) -> int:
            return max(0, await asyncio.to_thread(findings, changed, copy.path) - before)

        lineage = asyncio.run(repair_by_selection(
            Defect(tuple(args.test), tuple(args.file)), sources=sources, neighbours=neighbours, run=run,
            propose=propose, proposals=args.proposals, deadline=deadline, record=record, standards=standards))
        (args.out / "lineage.json").write_text(json.dumps(
            {**lineage.to_json(), "guarding_files": files, "guarding_tests": found, "planted": args.plant,
             "specification_carried": carried}, indent=1))
        (args.out / "semantic.json").write_text(json.dumps(evidence.to_json(), indent=1, default=str))
        # Where the search looked: the lines of the defect's own code, most suspicious first, and what ranked them.
        (args.out / "ranked.json").write_text(json.dumps({
            "groups": [len(g) for g in groups], "whole": bool(spectrum.get("whole")),
            "passing": len(spectrum.get("original") or {}),
            "lines": [[list(line), round(score, 4)] for line, score in spectrum.get("ranked", [])[:SUSPECT_LINES * 2]]},
            indent=1))
        if lineage.best is not None:
            import difflib

            from core.self_modification.repair_by_selection import applied

            after, _ = applied(sources, lineage.best.edits)
            (args.out / "best.diff").write_text("".join(
                "".join(difflib.unified_diff(sources[p].splitlines(True), after[p].splitlines(True),
                                             fromfile=f"a/{p}", tofile=f"b/{p}")) for p in after))
        print(json.dumps({"stopped": lineage.stopped, "candidates": len(lineage.candidates),
                          "best": lineage.best.id if lineage.best else "",
                          "best_fitness": ({k: v for k, v in vars(lineage.best.fitness).items() if k != "failures"}
                                           if lineage.best and lineage.best.fitness else None)}, indent=1))
        return 0 if lineage.stopped in {"mended", "the defect's tests already pass"} else 1
    finally:
        copy.remove()


if __name__ == "__main__":
    raise SystemExit(main())
