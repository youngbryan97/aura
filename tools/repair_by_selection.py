#!/usr/bin/env python3
"""Search for a change to her code that makes failing tests pass and breaks none that passed.

The search is core/self_modification/repair_by_selection.py. This gives it a
copy of the repository to run tests in (a detached git worktree at HEAD, never
the checkout her app runs from), the tests that import the files a change may
touch as the ones it must not break, and proposals: first the exact changes her
own Python writes (core/self_modification/proposals_without_a_model.py), and
only when those are spent her code lane's model, or a file of proposals. It writes every candidate, its parent and its fitness to
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = sys.executable
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
                  seconds: float) -> tuple[dict[str, bool], str]:
        self.write(sources)
        env = {k: v for k, v in os.environ.items() if k not in {"AURA_TESTING"}}
        env.update(AURA_STATE_ROOT=str(self.state), AURA_LOG_DIR=str(self.logs), PYTHONDONTWRITEBYTECODE="1",
                   VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
        try:
            proc = await asyncio.create_subprocess_exec(
                PYTHON, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", *tests,
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
                results[t] = bool(inside) and all(inside)
        failures = text[text.find("= FAILURES ="):] if "= FAILURES =" in text else text[-4000:]
        return results, failures

    async def passing_in(self, files: Sequence[str], sources: Mapping[str, str], *, seconds: float) -> list[str]:
        """Every test in ``files`` that passes on the copy as it is."""
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
            raise SystemExit(f"the neighbouring tests ran past {seconds:.0f} s; nothing searched") from None
        return [node for word, node in _RESULT.findall(out.decode("utf-8", errors="replace")) if word == "PASSED"]

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
        result = await lane.think_with_receipt(request_for(defect, start, parent), temperature=temperature,
                                               seed=asked)
        with receipts.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"asked": asked, "parent": parent.id if parent else "",
                                "receipt": {k: str(v) for k, v in vars(result.receipt).items()}}) + "\n")
        # One proposal a call, so the next is written from whichever candidate is then fittest.
        return [("code_lane", result.text.split("</think>")[-1])]

    return propose


def first_without_a_model(then, repo: Path):
    """Her own Python's exact changes for the current code first (proposals_without_a_model.py), then ``then``."""
    from core.self_modification.proposals_without_a_model import proposals_without_a_model

    pending: dict[str, object] = {}

    async def propose(defect, start, parent, n):
        key = parent.id if parent is not None else ""
        if key not in pending:
            failure = parent.fitness.failures if parent is not None and parent.fitness else defect.seen
            pending[key] = proposals_without_a_model(failure, start, repo)
        exact = next(pending[key], None)
        if exact is not None:
            return [exact]
        return await then(defect, start, parent, n) if then is not None else []

    return propose


def from_file(path: Path):
    texts = list(json.loads(path.read_text()))

    async def propose(defect, start, parent, n):
        return [("file", texts.pop(0))] if texts else []

    return propose


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
    args = parser.parse_args()
    if args.model_path and args.proposals_from:
        parser.error("give at most one of --model-path and --proposals-from")

    from core.self_modification.patch_genealogy import PatchGenealogyGraph, PatchNode
    from core.self_modification.repair_by_selection import Defect, repair_by_selection

    args.out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.minutes * 60
    where = Path(tempfile.mkdtemp(prefix="repair-by-selection-"))
    copy = Copy(ROOT, where, args.at)
    try:
        if args.tests_from:
            for path in dict.fromkeys(t.split("::")[0] for t in args.test):
                shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{args.tests_from}:{path}"], check=True,
                                       capture_output=True, text=True).stdout
                (copy.path / path).parent.mkdir(parents=True, exist_ok=True)
                (copy.path / path).write_text(shown, encoding="utf-8")
        if args.plant:
            path, old, new = args.plant
            text = (copy.path / path).read_text(encoding="utf-8")
            if text.count(old) != 1:
                raise SystemExit(f"the planted text occurs {text.count(old)} times in {path}")
            (copy.path / path).write_text(text.replace(old, new, 1), encoding="utf-8")
        sources = copy.read(args.file)
        # Each run may take what the whole search has left.
        run = lambda tests, s: copy.run(tests, s, original=sources, seconds=max(1.0, deadline - time.time()))  # noqa: E731
        # What must not break is each test, not each file: a file with one failing test still guards the rest.
        files = neighbours_of(args.file, copy.path)
        passed_before = asyncio.run(copy.passing_in(files, sources, seconds=max(1.0, deadline - time.time())))
        neighbours = [n for n in passed_before if n not in args.test]
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
        propose = first_without_a_model(model, copy.path)
        lineage = asyncio.run(repair_by_selection(
            Defect(tuple(args.test), tuple(args.file)), sources=sources, neighbours=neighbours, run=run,
            propose=propose, proposals=args.proposals, deadline=deadline, record=record))
        (args.out / "lineage.json").write_text(json.dumps(
            {**lineage.to_json(), "neighbours": neighbours, "planted": args.plant}, indent=1))
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
